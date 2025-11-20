from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    MultiheadAttention,
    LayerNorm,
    FeedForwardNetwork,
    TopKPruning,
    EViTPruning,
    CLC_CachePush,
    CLC_RecoverAndEmpty3,
    CLC_RecoverAndEmpty1,
)

mha_template = """struct config{index} : nnet::mha_config {{
    static const unsigned n_head = {num_heads};
    static const unsigned head_dim = {head_dim};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned seq_len = {seq_len};
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {tiling_factor};
    static const unsigned inv_table_size = {inv_table_size};
    static const unsigned exp_table_size = {exp_table_size};
    typedef {out_proj_bias_t.name} out_proj_bias_t;
    typedef {out_proj_weight_t.name} out_proj_weight_t;
    typedef {in_proj_bias_t.name} in_proj_bias_t;
    typedef {in_proj_weight_t.name} in_proj_weight_t;
    typedef {mask_t.name} mask_t;
    typedef {exp_table_t.name} exp_table_t;
    typedef {inv_table_t.name} inv_table_t;
    typedef {scale_t.name} scale_t;
    typedef {accum_t.name} accum_t;
    typedef {in_proj_out_t.name} in_proj_out_t;
    typedef {out_proj_in_t.name} out_proj_in_t;
    typedef {row_sum_t.name} row_sum_t;
    static const unsigned inv_range = {inv_table_range};
    static const unsigned exp_range = {exp_table_range};
    static const bool     enable_topk = {enable_topk};
    static const unsigned topk        = {topk};
    
}};\n"""

ffn_template = """struct config{index} : nnet::ffn_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned hidden_dim = {hidden_dim};
    static const unsigned in_ram_style = nnet::{in_ram_style};
    static const unsigned out_ram_style = nnet::{out_ram_style};
    static const bool activation_gelu = {activation_gelu};
    static constexpr unsigned tiling_factor[3] = {tiling_factor};
    typedef {out_proj_bias_t.name} out_proj_bias_t;
    typedef {out_proj_weight_t.name} out_proj_weight_t;
    typedef {in_proj_bias_t.name} in_proj_bias_t;
    typedef {in_proj_weight_t.name} in_proj_weight_t;
    typedef {hidden_t.name} hidden_t;
    typedef {accum_t.name} accum_t;
    typedef {cdf_table_t.name} cdf_table_t;
    static const unsigned cdf_table_size = {cdf_table_size};
    static const unsigned cdf_table_range = {cdf_table_range};
}};\n"""

layernorm_template = """struct config{index} : nnet::layernorm_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned table_size = {var_table_size};
    static constexpr float table_range = {var_table_range};
    static constexpr unsigned tiling_factor[3] = {tiling_factor};
    typedef {sum_sqr_t.name} sum_sqr_t;
    typedef {mean_t.name} mean_t;
    typedef {sum_t.name} sum_t;   
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    typedef {var_table_t.name} var_table_t;
    typedef {accum_t.name} accum_t;
}};\n"""

prune_config_template = """struct config{index} : nnet::pruning_config {{
    static const unsigned N = {seq_len_in};
    static const unsigned keep_tokens = {keep_tokens};
}};\n"""

clc_cache_push_config_template = """struct config{index} : nnet::clc_base_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
}};\n"""

clc_recover3_config_template = """struct config{index} : nnet::clc_recover_base {{
    static const unsigned embed_dim = {embed_dim};
    static const unsigned N_pruned = {N_pruned};
    static const unsigned n_carriers = 6;
}};\n"""

clc_recover1_config_template = """struct config{index} : nnet::clc_recover_base {{
    static const unsigned embed_dim = {embed_dim};
    static const unsigned N_pruned = {N_pruned};
    static const unsigned n_carriers = 2;
}};\n"""


mha_function_template = '''hls::stream<int> {topk_idx}("layer{index}_topk_idx");
    #pragma HLS STREAM variable={topk_idx} depth={seq_len}
    nnet::MultiHeadAttention<{input_t}, {output_t}, {config}>({input}, {output}, {topk_idx}, {iprj_w}, {iprj_b}, {oprj_w}, {oprj_b}, {mask});'''
mha_include_list = ["nnet_utils/nnet_multiheadattention_stream.h"]

layernorm_function_template = 'nnet::LayerNormalize<{input_t}, {output_t}, {config}>({input}, {output}, {s}, {b});'
layernorm_include_list = ["nnet_utils/nnet_layernorm_stream.h"]

ffn_function_template = 'nnet::FeedForwardNetwork<{input_t}, {output_t}, {config}>({input}, {output}, {iprj_w}, {iprj_b}, {oprj_w}, {oprj_b});'
ffn_include_list = ["nnet_utils/nnet_feedforwardnetwork_stream.h"]

prune_function_template = "nnet::PruningLayer<{input_t}, {output_t}, {config}>({input}, {output}, {topk_idx});"
prune_include_list = ["nnet_utils/nnet_topk_pruning_stream.h"]
eviT_pruning_include_list = ["nnet_utils/nnet_evit_pruning_stream.h"]

clc_cache_push_function_template = """hls::stream<token_t> {cache_stream}("cache_stream");
    #pragma HLS BIND_STORAGE variable={cache_stream} type=FIFO impl=lutram
    #pragma HLS STREAM variable={cache_stream} depth={embed_dim}*2
    nnet::CLC_CachePush<{input_t}, token_t, {config}>({input}, {output}, {cache_stream});"""
clc_recover3_function_template = """nnet::CLC_RecoverAndEmpty3<{input_t}, token_t, {output_t}, {config}>({input}, {cache_stream0}, {cache_stream1}, {cache_stream2}, {output});"""
clc_recover1_function_template = """nnet::CLC_RecoverAndEmpty1<{input_t}, token_t, {output_t}, {config}>({input}, {cache_stream}, {output});"""
clc_include_list = ["nnet_utils/nnet_clc_stream.h"]

class MHAConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((MultiheadAttention))
        self.mha_template  = mha_template 

    def format(self, node):
        params = self._default_config_params(node)
        try:
            params['seq_len'] = node.get_input_variable().shape[0][1]
        except Exception:
            pass
        params['tiling_factor'] = '{'+','.join([str(x) for x in params['tiling_factor']])+'}'
        # print(f"[DEBUG] {node.name} seq_len: {params['seq_len']}")

        enable_topk = False
        topk = 0

        # 嘗試找「同一個 encoder、接在 _add1 後」的 TopKPruning 來決定是否開啟
        layers = list(node.model.get_layers())
        mypos = layers.index(node)
        enc_prefix = node.name.replace('_self_attn', '')  # ex: layers_5
        target_input = f'{enc_prefix}_add1'

        # print(f"[DEBUG] {node.name}: mypos={mypos}, enc_prefix='{enc_prefix}', target_input='{target_input}'")
        
        # 檢查附近的層
        search_range = layers[mypos: min(mypos+8, len(layers))]
        # print(f"[DEBUG] {node.name}: 搜索範圍的層:")
        # for i, l in enumerate(search_range):
        #     class_name = l.__class__.__name__
        #     inputs = getattr(l, 'inputs', None)
        #     print(f"  [{mypos+i}] {l.name} - 類型: {class_name} - inputs: {inputs}")
            
        #     # 檢查是否為 TopKPruning
        #     if 'TopKPruning' in class_name:
        #         print(f"    >>> 找到 TopKPruning: {l.name}")
        #         if inputs and len(inputs) > 0:
        #             print(f"    >>> inputs[0]: '{inputs[0]}' vs target: '{target_input}'")
        #             if inputs[0] == target_input:
        #                 print(f"    >>> 匹配成功！")
        #             else:
        #                 print(f"    >>> 輸入不匹配")

        keep_tokens = None
        for l in search_range:
            class_name = l.__class__.__name__
            if ('TopKPruning' in class_name or 'Pruning' in class_name) and l.inputs and l.inputs[0] == target_input:
                # print(class_name)
                try:
                    keep_tokens = int(l.get_attr('seq_len_out'))
                    # print(f"[DEBUG] {node.name}: 找到匹配的 pruning 層 {l.name}, keep_tokens={keep_tokens}")
                    break
                except Exception as e:
                    print(f"[DEBUG] {node.name}: 無法獲取 keep_tokens: {e}")
        
        if keep_tokens is not None:
            enable_topk = True
            if ('EViTPruning' in class_name):
                topk = keep_tokens - 2
            else:
                topk = keep_tokens - 1

        params['enable_topk'] = 'true' if enable_topk else 'false'
        params['topk'] = str(topk)
        # print(f"[DEBUG] {node.name} enable_topk: {params['enable_topk']}, topk: {params['topk']}")
        # print(f"[DEBUG] {node.name} ===========================")

        return self.mha_template.format(**params)

class MHAFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((MultiheadAttention), include_header=mha_include_list)
        self.templates = mha_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['iprj_w'] = node.get_weights('in_proj_weight').name
        params['iprj_b'] = node.get_weights('in_proj_bias').name
        params['oprj_w'] = node.get_weights('out_proj_weight').name
        params['oprj_b'] = node.get_weights('out_proj_bias').name
        params['mask'] = node.get_weights('mask').name
        params['topk_idx'] = f"layer{node.index}_topk_idx"
        params['seq_len']  = node.get_attr('seq_len')
        params['embed_dim']= node.get_attr('embed_dim')
        return self.templates.format(**params)

class LayerNormConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((LayerNorm))
        self.layernorm_template  = layernorm_template 

    def format(self, node):
        params = self._default_config_params(node)
        try:
            params['seq_len'] = node.get_input_variable().shape[0][1]
        except Exception:
            pass
        params['tiling_factor'] = '{'+','.join([str(x) for x in params['tiling_factor']])+'}'
        layernorm_config = self.layernorm_template.format(**params)
        return layernorm_config

class LayerNormFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((LayerNorm), include_header=layernorm_include_list)
        self.templates = layernorm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['s'] = node.get_weights('scale').name
        params['b'] = node.get_weights('bias').name

        return self.templates.format(**params)

class FFNConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((FeedForwardNetwork))
        self.ffn_template  = ffn_template 

    def format(self, node):
        params = self._default_config_params(node)
        try:
            params['seq_len'] = node.get_input_variable().shape[0][1]
        except Exception:
            pass
        params['activation_gelu'] = 'true' if node.get_attr('activation').__name__ == 'gelu' else 'false'
        params['tiling_factor'] = '{'+','.join([str(x) for x in params['tiling_factor']])+'}'
        ffn_config = self.ffn_template.format(**params)
        return ffn_config

class FFNFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((FeedForwardNetwork), include_header=ffn_include_list)
        self.templates = ffn_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['iprj_w'] = node.get_weights('in_proj_weight').name
        params['iprj_b'] = node.get_weights('in_proj_bias').name
        params['oprj_w'] = node.get_weights('out_proj_weight').name
        params['oprj_b'] = node.get_weights('out_proj_bias').name
        return self.templates.format(**params)
    
class PruneConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((TopKPruning))
        self.template = prune_config_template

    def format(self, node):
        params = self._default_config_params(node)
        # 直接用 parser 在 step 2 塞進來的屬性
        try:
            params['seq_len_in']  = int(node.get_input_variable().shape[0][-1])
        except Exception:
            params['seq_len_in']  = int(node.get_attr('seq_len_in'))
        params['keep_tokens'] = int(node.get_attr('seq_len_out'))
        # print(f"[DEBUG] Pruning {node.name}: N={params['seq_len_in']}, keep_tokens={params['keep_tokens']}")
        return self.template.format(**params)

class PruneFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((TopKPruning), include_header=prune_include_list)
        self.templates = prune_function_template

    def format(self, node):
        params = self._default_function_params(node)

        # 找到「就近在這個剪枝層前面的那個 MHA」的 index，
        # 以便取得在第 1 步宣告的 `layer{mha_idx}_topk_idx` 變數名
        layers = list(node.model.get_layers())
        me_pos = layers.index(node)
        prev_mha = None

        for L in reversed(layers[:me_pos]):
            # 類名比對（避免直接 import 類別造成循環）
            if (getattr(L, 'class_name', '') == 'MultiheadAttention' or
                L.__class__.__name__ == 'VitisMultiheadAttention' or
                'self_attn' in L.name):
                prev_mha = L
                break

        assert prev_mha is not None, f"TopKPruning {node.name} 找不到對應的 MHA 層"

        params['topk_idx']  = f"layer{prev_mha.index}_topk_idx"

        return self.templates.format(**params)

class EViTPruningConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((EViTPruning))
        self.template = prune_config_template

    def format(self, node):
        params = self._default_config_params(node)
        # 直接用 parser 在 step 2 塞進來的屬性
        try:
            params['seq_len_in']  = int(node.get_input_variable().shape[0][-1])
        except Exception:
            params['seq_len_in']  = int(node.get_attr('seq_len_in'))
        params['keep_tokens'] = int(node.get_attr('seq_len_out'))
        # print(f"[DEBUG] Pruning {node.name}: N={params['seq_len_in']}, keep_tokens={params['keep_tokens']}")
        return self.template.format(**params)

class EViTPruningFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((EViTPruning), include_header=eviT_pruning_include_list)
        self.templates = prune_function_template

    def format(self, node):
        params = self._default_function_params(node)

        # 找到「就近在這個剪枝層前面的那個 MHA」的 index，
        # 以便取得在第 1 步宣告的 `layer{mha_idx}_topk_idx` 變數名
        layers = list(node.model.get_layers())
        me_pos = layers.index(node)
        prev_mha = None

        for L in reversed(layers[:me_pos]):
            # 類名比對（避免直接 import 類別造成循環）
            if (getattr(L, 'class_name', '') == 'MultiheadAttention' or
                L.__class__.__name__ == 'VitisMultiheadAttention' or
                'self_attn' in L.name):
                prev_mha = L
                break

        assert prev_mha is not None, f"TopKPruning {node.name} 找不到對應的 MHA 層"

        params['topk_idx']  = f"layer{prev_mha.index}_topk_idx"

        return self.templates.format(**params)

class CLC_CachePushConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((CLC_CachePush))
        self.template = clc_cache_push_config_template

    def format(self, node):
        params = self._default_config_params(node)
        
        params['group_id'] = node.get_attr('group_id')
        # 以圖上實際 shape 為準（避免靜態 shape 沒跟上）
        try:
            params['seq_len'] = int(node.get_input_variable().shape[0][-1])
        except Exception:
            params['seq_len'] = int(node.get_attr('seq_len'))
        params['embed_dim'] = int(node.get_attr('embed_dim'))

        return self.template.format(**params)


class CLC_CachePushFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((CLC_CachePush), include_header=clc_include_list)
        self.template = clc_cache_push_function_template

    def format(self, node):
        params = self._default_function_params(node)
        # I/O 與型別
        input_var  = node.get_input_variable()
        output_var = node.get_output_variable()

        layers = list(node.model.get_layers())
        me_pos = layers.index(node) % 3

        params['input'] = input_var.name
        params['output'] = output_var.name
        params['group_id'] = node.get_attr('group_id')
        
        layer_name = node.name  # 例如 "layers_5_clc_push"
        
        # 提取 encoder layer index
        if '_' in layer_name:
            parts = layer_name.split('_')
            if len(parts) >= 2 and parts[0] == 'layers':
                try:
                    enc_idx = int(parts[1])  # 從 "layers_5_clc_push" 提取 5
                    
                    # 根據 encoder index 在 group 內的位置決定 bucket
                    bucket_idx = enc_idx % 3  # 0, 1, 2 對應 b0, b1, b2
                    params['cache_stream'] = f"clc_cache_{params['group_id']}_b{bucket_idx}"
                except (ValueError, IndexError):
                    # 如果無法解析，使用預設值
                    params['cache_stream'] = f"clc_cache_{params['group_id']}_b0"
            else:
                params['cache_stream'] = f"clc_cache_{params['group_id']}_b0"
        else:
            params['cache_stream'] = f"clc_cache_{params['group_id']}_b0"

        return self.template.format(**params)


class CLC_RecoverAndEmpty3ConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((CLC_RecoverAndEmpty3))
        self.template = clc_recover3_config_template

    def format(self, node):
        params = self._default_config_params(node)

        params['group_id'] = node.get_attr('group_id')
        params['embed_dim'] = int(node.get_attr('embed_dim'))

        # N_pruned 以圖上實際輸入長度為主；若要嚴格用屬性可改回 get_attr('N_pruned')
        try:
            params['N_pruned'] = int(node.get_input_variable().shape[0][-1])
        except Exception:
            params['N_pruned'] = int(node.get_attr('N_pruned'))

        return self.template.format(**params)


class CLC_RecoverAndEmpty3FunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((CLC_RecoverAndEmpty3), include_header=clc_include_list)
        self.template = clc_recover3_function_template

    def format(self, node):
        params = self._default_function_params(node)

        input_var  = node.get_input_variable()
        output_var = node.get_output_variable()

        gid = node.get_attr('group_id')

        params['input'] = input_var.name
        params['output'] = output_var.name
        params['group_id'] = gid
        params['cache_stream0'] = f"clc_cache_{gid}_b0"
        params['cache_stream1'] = f"clc_cache_{gid}_b1"
        params['cache_stream2'] = f"clc_cache_{gid}_b2"

        return self.template.format(**params)


class CLC_RecoverAndEmpty1ConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((CLC_RecoverAndEmpty1))
        self.template = clc_recover1_config_template

    def format(self, node):
        params = self._default_config_params(node)

        params['group_id'] = node.get_attr('group_id')
        params['embed_dim'] = int(node.get_attr('embed_dim'))

        try:
            params['N_pruned'] = int(node.get_input_variable().shape[0][-1])
        except Exception:
            params['N_pruned'] = int(node.get_attr('N_pruned'))

        return self.template.format(**params)


class CLC_RecoverAndEmpty1FunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((CLC_RecoverAndEmpty1), include_header=clc_include_list)
        self.template = clc_recover1_function_template

    def format(self, node):
        params = self._default_function_params(node)

        input_var  = node.get_input_variable()
        output_var = node.get_output_variable()

        gid = node.get_attr('group_id')

        params['input'] = input_var.name
        params['output'] = output_var.name
        params['group_id'] = gid
        params['cache_stream'] = f"clc_cache_{gid}_b0"

        return self.template.format(**params)
