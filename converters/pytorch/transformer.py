from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import compute_padding_1d_pytorch, compute_padding_2d_pytorch, parse_data_format
from hls4ml.converters.pytorch.core import parse_linear_layer
from hls4ml.converters.pytorch_to_hls import layer_handlers
import math

def _normalize_reduction_cfg(cfg: dict):
    # print(f"[DEBUG] _normalize_reduction_cfg 接收到的 cfg: {cfg}")
    # print(f"[DEBUG] cfg 的類型: {type(cfg)}")
    
    cfg = cfg or {}
    loc = cfg.get('reduction_loc', [])
    kr  = cfg.get('keep_rate', [])
    use_clc = bool(cfg.get('use_clc', False))
    method = cfg.get('method', 'topk')

    valid_methods = ['topk', 'evit']
    if method not in valid_methods:
        raise ValueError(f"Unsupported token reduction method: {method}. Use {valid_methods}")
    
    # print(f"[DEBUG] reduction_loc: {loc}")
    # print(f"[DEBUG] keep_rate: {kr}")
    # print(f"[DEBUG] use_clc: {use_clc}")
    
    # broadcast keep_rate if single value
    if isinstance(kr, (int, float)):
        kr = [float(kr)]
    if len(kr) == 1 and len(loc) > 1:
        kr = kr * len(loc)
    
    kr_map = {int(l): float(k) for l, k in zip(loc, kr)} if loc and kr else {}
    # print(f"[DEBUG] 最終的 kr_map: {kr_map}")
    # print(f"[DEBUG] 最終的 use_clc: {use_clc}")
    
    return kr_map, use_clc, method

def _encoder_layer_index(layer_name: str):
    try:
        return int(layer_name.split('_')[-1])
    except Exception:
        return None

@pytorch_handler('LayerNorm')
def parse_layernorm_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'LayerNorm' in operation
    layer = {}
    layer['embed_dim'] = input_shapes[0][-1]
    layer['seq_len'] = input_shapes[0][-2]
    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['scale_data'] = class_object.weight.data.numpy()
    layer['bias_data'] = class_object.bias.data.numpy()
    layer['class_name'] = 'LayerNorm'
    layer['data_format'] = 'channels_first'
    #only implemented for in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
    #TODO: implement for other weights and biases

    output_shapes = input_shapes   
    return layer, output_shapes

@pytorch_handler('MultiheadAttention')
def parse_mha_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'MultiheadAttention' in operation
    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names.copy()
    layer['class_name'] = 'MultiheadAttention'
    layer['data_format'] = 'channels_first'
    #only implemented for in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
    #TODO: implement for other weights and biases
    layer['num_heads'] = class_object.num_heads
    layer['head_dim'] = class_object.head_dim
    layer['embed_dim'] = class_object.embed_dim
    layer['seq_len'] = input_shapes[0][-2]
    layer['in_proj_weight_data'] = class_object.in_proj_weight.data.numpy()
    layer['in_proj_bias_data'] = class_object.in_proj_bias.data.numpy()
    layer['out_proj_weight_data'] = class_object.__dict__['_modules']['out_proj'].weight.data.numpy()
    layer['out_proj_bias_data'] = class_object.__dict__['_modules']['out_proj'].bias.data.numpy()

    output_shapes = input_shapes   
    return layer, output_shapes

@pytorch_handler('FeedForwardNetwork')
def parse_ffn_layer(operation, layer_name, input_names, input_shapes, node, class_object1, class_object2, data_reader, config):
    assert 'FeedForwardNetwork' in operation
    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names.copy()
    layer['class_name'] = 'FeedForwardNetwork'
    layer['embed_dim'] = input_shapes[0][-1]
    layer['hidden_dim'] = class_object1.out_features
    assert class_object1.out_features == class_object2.in_features
    layer['seq_len'] = input_shapes[0][-2]
    layer['in_proj_weight_data'] = class_object1.weight.data.numpy()
    layer['in_proj_bias_data'] = class_object1.bias.data.numpy()
    layer['out_proj_weight_data'] = class_object2.weight.data.numpy()
    layer['out_proj_bias_data'] = class_object2.bias.data.numpy()

    output_shapes = input_shapes   
    return layer, output_shapes

@pytorch_handler('TransformerEncoderLayer')
def parse_transenc_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):

    assert 'TransformerEncoderLayer' in operation
    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names.copy()
    layer['class_name'] = 'LayerGroup'
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)
    layer_list = []
    
    prev_layer_name = input_names.copy()
    if class_object.__dict__['norm_first']:
        subclass_object = class_object.__dict__['_modules']['norm1']
        sublayer, _= layer_handlers['LayerNorm']('LayerNorm', layer_name+'_norm1', prev_layer_name, input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        subclass_object = class_object.__dict__['_modules']['self_attn']
        sublayer, _= layer_handlers['MultiheadAttention']('MultiheadAttention', layer_name+'_self_attn', [layer_name+'_norm1'], input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        sublayer, _= layer_handlers['add']('add', layer_name+'_add1', [layer_name+'_self_attn', prev_layer_name[0]], input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        # print(f"[DEBUG] {layer_name} input_shapes: {input_shapes}")
        # print(f"[DEBUG] {layer_name} input_shapes[0]: {input_shapes[0]}")

        kr_map, use_clc, method = _normalize_reduction_cfg(config.get('HLSConfig', {}).get('Model', {}).get('Reduction', {}))
        enc_idx = _encoder_layer_index(layer_name)
        # print(f"[DEBUG] layer_name: {layer_name}, enc_idx: {enc_idx}")
        # print(f"[DEBUG] kr_map: {kr_map}, use_clc: {use_clc}")
        keep_len = None
        next_input_name = layer_name + '_add1'  # default, if not pruned here
        
        if enc_idx is not None and enc_idx in kr_map:
            keep_rate = kr_map[enc_idx]
            prune_name = layer_name + '_prune'  # consistent suffix for later passes

            if method == 'topk':
                prune_layer = {
                    'name': prune_name,
                    'inputs': [layer_name + '_add1'],
                    'class_name': 'TopKPruning',
                    'data_format': 'channels_first',
                    'keep_rate': keep_rate,
                    'use_clc': use_clc,
                    'embed_dim': input_shapes[0][-1],
                    'seq_len_in': input_shapes[0][-2],
                    'seq_len_out': math.ceil((input_shapes[0][-2] - 1) * keep_rate) + 1,
                }
            elif method == 'evit':
                prune_layer = {
                    'name': prune_name,
                    'inputs': [layer_name + '_add1'],
                    'class_name': 'EViTPruning',
                    'data_format': 'channels_first',
                    'keep_rate': keep_rate,
                    'use_clc': use_clc,
                    'embed_dim': input_shapes[0][-1],
                    'seq_len_in': input_shapes[0][-2],
                    'seq_len_out': math.ceil((input_shapes[0][-2] - 1) * keep_rate) + 2,
                }
            layer_list.append(prune_layer)
            keep_len = prune_layer['seq_len_out']
            input_shapes[0] = (input_shapes[0][0], keep_len, input_shapes[0][2])
            next_input_name = prune_name

        subclass_object = class_object.__dict__['_modules']['norm2']
        sublayer, _= layer_handlers['LayerNorm']('LayerNorm', layer_name+'_norm2', [next_input_name], input_shapes, node, subclass_object, data_reader, config)
        if keep_len is not None:
            sublayer['seq_len'] = int(keep_len)
        layer_list.append(sublayer)

        subclass_object1 = class_object.__dict__['_modules']['linear1']
        subclass_object2 = class_object.__dict__['_modules']['linear2']
        sublayer, _= layer_handlers['FeedForwardNetwork']('FeedForwardNetwork', layer_name+'_ffn', [layer_name+'_norm2'], input_shapes, node, subclass_object1, subclass_object2, data_reader, config)
        sublayer["activation"] = class_object.activation
        if keep_len is not None:
            sublayer['seq_len'] = int(keep_len)
        layer_list.append(sublayer)

        sublayer, _= layer_handlers['add']('add', layer_name+'_add2', [layer_name+'_ffn', next_input_name], input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        next_input_name = layer_name + '_add2'

        if use_clc:
            if enc_idx in kr_map:
                # layer 3,6,9 都是CLC_RecoverAndEmpty3 (恢復6個tokens)
                reduction_locations = sorted(kr_map.keys())
                group_idx = reduction_locations.index(enc_idx)
                group_id = f'g{group_idx}'
                
                recover_layer = {
                    'name': f'{layer_name}_clc_recover',
                    'class_name': 'CLC_RecoverAndEmpty3',
                    'inputs': [next_input_name],
                    'embed_dim': input_shapes[0][-1],
                    'N_pruned': input_shapes[0][-2],
                    'group_id': group_id,
                    'seq_len_out': input_shapes[0][-2] + 6,
                }
                layer_list.append(recover_layer)
                keep_len = recover_layer['seq_len_out']
                input_shapes[0] = (input_shapes[0][0], keep_len, input_shapes[0][2])
                # input_shapes[0][-2] += 6  # 恢復6個tokens
                next_input_name = f'{layer_name}_clc_recover'
            
            elif enc_idx == 10:
                # layer 10: CLC_RecoverAndEmpty1 (恢復2個tokens)
                recover_layer = {
                    'name': f'{layer_name}_clc_recover',
                    'class_name': 'CLC_RecoverAndEmpty1',
                    'inputs': [next_input_name],
                    'embed_dim': input_shapes[0][-1],
                    'N_pruned': input_shapes[0][-2],  # 應該是14
                    'group_id': 'g3',
                    'seq_len_out': input_shapes[0][-2] + 2,
                }
                layer_list.append(recover_layer)
                keep_len = recover_layer['seq_len_out']
                input_shapes[0] = (input_shapes[0][0], keep_len, input_shapes[0][2])
                # input_shapes[0][-2] += 2  # 恢復2個tokens
                next_input_name = f'{layer_name}_clc_recover'
        
        if use_clc and enc_idx < 10:
            group_id = f'g{enc_idx // 3}'
            
            cache_push_layer = {
                'name': f'{layer_name}_clc_push',
                'class_name': 'CLC_CachePush',
                'inputs': [next_input_name],
                'seq_len': input_shapes[0][-2],
                'embed_dim': input_shapes[0][-1],
                'group_id': group_id,
            }
            layer_list.append(cache_push_layer)
            # next_input_name = f'{layer_name}_clc_push'
    else:
        subclass_object = class_object.__dict__['_modules']['self_attn']
        sublayer, _= layer_handlers['MultiheadAttention']('MultiheadAttention', layer_name+'_self_attn', prev_layer_name, input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        sublayer, _= layer_handlers['add']('add', layer_name+'_add1', [layer_name+'_self_attn', prev_layer_name[0]], input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        kr_map, use_clc, method = _normalize_reduction_cfg(config.get('Model', {}).get('Reduction', {}))
        enc_idx = _encoder_layer_index(layer_name)
        keep_len = None
        next_input_name = layer_name + '_add1'  # default, if not pruned here
        if enc_idx is not None and enc_idx in kr_map:
            keep_rate = kr_map[enc_idx]
            prune_name = layer_name + '_prune'  # consistent suffix for later passes

            if method == 'topk':
                prune_layer = {
                    'name': prune_name,
                    'inputs': [layer_name + '_add1'],
                    'class_name': 'TopKPruning',
                    'data_format': 'channels_first',
                    'keep_rate': keep_rate,
                    'use_clc': use_clc,
                    'embed_dim': input_shapes[0][-1],
                    'seq_len_in': input_shapes[0][-2],
                    'seq_len_out': math.ceil((input_shapes[0][-2] - 1) * keep_rate) + 1,
                }
            elif method == 'evit':
                prune_layer = {
                    'name': prune_name,
                    'inputs': [layer_name + '_add1'],
                    'class_name': 'EViTPruning',
                    'data_format': 'channels_first',
                    'keep_rate': keep_rate,
                    'use_clc': use_clc,
                    'embed_dim': input_shapes[0][-1],
                    'seq_len_in': input_shapes[0][-2],
                    'seq_len_out': math.ceil((input_shapes[0][-2] - 1) * keep_rate) + 2,
                }
            layer_list.append(prune_layer)
            keep_len = prune_layer['seq_len_out']
            next_input_name = prune_name

        subclass_object = class_object.__dict__['_modules']['norm1']
        sublayer, _= layer_handlers['LayerNorm']('LayerNorm', layer_name+'_norm1', [next_input_name], input_shapes, node, subclass_object, data_reader, config)
        if keep_len is not None:
            sublayer['seq_len'] = int(keep_len)
        layer_list.append(sublayer)

        subclass_object1 = class_object.__dict__['_modules']['linear1']
        subclass_object2 = class_object.__dict__['_modules']['linear2']
        sublayer, _= layer_handlers['FeedForwardNetwork']('FeedForwardNetwork', layer_name+'_ffn', [layer_name+'_norm2'], input_shapes, node, subclass_object1, subclass_object2, data_reader, config)
        sublayer["activation"] = class_object.activation
        if keep_len is not None:
            sublayer['seq_len'] = int(keep_len)
        layer_list.append(sublayer)

        sublayer, _= layer_handlers['add']('add', layer_name+'_add2', [layer_name+'_ffn', next_input_name], input_shapes, node, subclass_object, data_reader, config)
        if keep_len is not None:
            sublayer['seq_len'] = int(keep_len)
        layer_list.append(sublayer)

        subclass_object = class_object.__dict__['_modules']['norm2']
        sublayer, _= layer_handlers['LayerNorm']('LayerNorm', layer_name+'_norm2', [layer_name+'_add2'], input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)

        next_input_name = layer_name + '_norm2'

        if use_clc:
            # 判斷是否需要CLC_RecoverAndEmpty3 (有pruning的layers)
            if enc_idx in kr_map:
                # layer 3,6,9 都是CLC_RecoverAndEmpty3 (恢復6個tokens)
                reduction_locations = sorted(kr_map.keys())
                group_idx = reduction_locations.index(enc_idx)
                group_id = f'g{group_idx}'
                
                recover_layer = {
                    'name': f'{layer_name}_clc_recover',
                    'class_name': 'CLC_RecoverAndEmpty3',
                    'inputs': [next_input_name],
                    'embed_dim': input_shapes[0][-1],
                    'N_pruned': input_shapes[0][-2],
                    'group_id': group_id,
                }
                layer_list.append(recover_layer)
                input_shapes[0][-2] += 6  # 恢復6個tokens
                next_input_name = f'{layer_name}_clc_recover'
            
            # 判斷是否需要CLC_RecoverAndEmpty1 (layer 10的特殊情況)
            elif enc_idx == 10:
                # layer 10: CLC_RecoverAndEmpty1 (恢復2個tokens)
                recover_layer = {
                    'name': f'{layer_name}_clc_recover',
                    'class_name': 'CLC_RecoverAndEmpty1',
                    'inputs': [next_input_name],
                    'embed_dim': input_shapes[0][-1],
                    'N_pruned': input_shapes[0][-2],  # 應該是14
                    'group_id': 'g3',
                }
                layer_list.append(recover_layer)
                input_shapes[0][-2] += 2  # 恢復2個tokens
                next_input_name = f'{layer_name}_clc_recover'
        
        # 9. 新增：CLC_CachePush (每個layer最後都有)
        if use_clc:
            group_id = f'g{enc_idx // 3}'
            
            cache_push_layer = {
                'name': f'{layer_name}_clc_push',
                'class_name': 'CLC_CachePush',
                'inputs': [next_input_name],
                'seq_len': input_shapes[0][-2],
                'embed_dim': input_shapes[0][-1],
                'group_id': group_id,
            }
            layer_list.append(cache_push_layer)
            # next_input_name = f'{layer_name}_clc_push'
        
    layer['output_shape'] = input_shapes
    layer['layer_list'] = layer_list
    layer['input_layers'] = []
    layer['output_layers'] = []
    layer['data_reader'] = data_reader
    output_shapes = input_shapes
    return layer, output_shapes

@pytorch_handler('ModuleList')
def parse_layers(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'ModuleList' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names.copy()
    layer['class_name'] = 'LayerGroup'
    layer_list = []
    prev_layer_name = input_names.copy()
    for key, subclass_object in class_object.__dict__['_modules'].items():
        sublayer_name = layer_name + '_' + key
        class_name = subclass_object.__class__.__name__
        sublayer, _= layer_handlers[class_name](class_name, sublayer_name, prev_layer_name, input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)
        prev_layer_name = [sublayer_name]

    # LayerGroup info
    layer['output_shape'] = input_shapes
    layer['layer_list'] = layer_list
    layer['input_layers'] = []
    layer['output_layers'] = []
    layer['data_reader'] = data_reader

    output_shape = input_shapes  # Channel first as default

    return layer, output_shape

@pytorch_handler('TransformerEncoder')
def parse_transenc(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'TransformerEncoder' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names.copy()
    layer['class_name'] = 'LayerGroup'
    layer_list = []
    prev_layer_name = input_names.copy()
    for key, subclass_object in class_object.__dict__['_modules'].items():
        class_name = subclass_object.__class__.__name__
        sublayer, _= layer_handlers[class_name](class_name, key, prev_layer_name, input_shapes, node, subclass_object, data_reader, config)
        layer_list.append(sublayer)
        prev_layer_name = [key]
        #if key == 'layers':
        #    print("input_names = ", input_names)
        #    sublayer, _= parse_layers('Layers', key, ['src'], input_shapes, node, subclass_object, data_reader, config)
        #    layer_list.append(sublayer)
        #elif key == 'norm':
        #    print("input_names = ", input_names)
        #    sublayer, _= parse_layernorm_layer('LayerNorm', key, ['layers'], input_shapes, node, subclass_object, data_reader, config)
        #    layer_list.append(sublayer)

    # LayerGroup info
    layer['output_shape'] = input_shapes
    layer['layer_list'] = layer_list
    layer['input_layers'] = []
    layer['output_layers'] = []
    layer['data_reader'] = data_reader

    output_shape = input_shapes  # Channel first as default

    return layer, output_shape


#@pytorch_handler('Conv2d')
def parse_conv2d_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Conv2d' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'Conv2D'
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)

    layer['weight_data'] = class_object.weight.data.numpy()
    if class_object.bias is not None:
        layer['bias_data'] = class_object.bias.data.numpy()
    else:
        layer['bias_data'] = None

    # Input info
    (layer['in_height'], layer['in_width'], layer['n_chan']) = parse_data_format(
        input_shapes[0], 'channels_first'
    )  # Keras's default is channels_last

    # Additional parameters
    layer['n_filt'] = class_object.out_channels
    layer['filt_height'] = class_object.kernel_size[0]
    layer['filt_width'] = class_object.kernel_size[1]
    layer['stride_height'] = class_object.stride[0]
    layer['stride_width'] = class_object.stride[1]
    layer['dilation'] = class_object.dilation[0]
    layer['pad_top'] = layer['pad_bottom'] = class_object.padding[0]
    layer['pad_left'] = layer['pad_right'] = class_object.padding[1]

    if all(x == 0 for x in class_object.padding):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else:  # Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'

    # Ouput info
    (layer['out_height'], layer['out_width'], _, _, _, _) = compute_padding_2d_pytorch(
        class_object.padding,
        layer['in_height'],
        layer['in_width'],
        layer['stride_height'],
        layer['stride_width'],
        layer['filt_height'],
        layer['filt_width'],
        class_object.dilation[0],
        class_object.dilation[1],
    )

    output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape
