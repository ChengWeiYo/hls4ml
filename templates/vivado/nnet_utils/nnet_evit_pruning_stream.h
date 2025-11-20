#ifndef NNET_PRUNING_H_
#define NNET_PRUNING_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include "hls_streamofblocks.h"
#include <math.h>
#include <cmath>
#include <iostream>

namespace nnet {

struct pruning_config {
    static const unsigned N = 17;
    static const unsigned embed_dim = 192;
    static const unsigned keep_tokens = 17;
};

template<typename CONFIG_T>
void ComputeComplementIndices(
    const int* topk_indices,
    int* complement_indices
    // const int total_tokens,
    // const int keep_tokens
) {
    #pragma HLS PIPELINE II=1
    
    // 建立標記陣列
    bool is_selected[CONFIG_T::N];
    // #pragma HLS ARRAY_PARTITION variable=is_selected complete dim=1
    
    // 初始化所有為false
    for (int i = 0; i < CONFIG_T::N; i++) {
        #pragma HLS UNROLL
        is_selected[i] = false;
    }
    
    // 標記TopK索引
    for (int i = 0; i < CONFIG_T::keep_tokens-2; i++) {
        #pragma HLS UNROLL
        int idx = topk_indices[i];
        if (idx >= 0 && idx < CONFIG_T::N) {
            is_selected[idx] = true;
        }
    }
    
    // 計算Complement索引
    int compl_idx = 0;
    for (int i = 0; i < CONFIG_T::N-1; i++) {
        #pragma HLS PIPELINE II=1
        if (!is_selected[i]) {
            complement_indices[compl_idx] = i;
            compl_idx++;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void PruningLayer(
    hls::stream<data_T> &data_in,
    hls::stream<res_T> &data_out,
    hls::stream<int>    &topk_idx_in
) {
    #pragma HLS INLINE off
    const unsigned int D = CONFIG_T::embed_dim;

    // 可視需要保留或移除
    // #pragma HLS DATAFLOW

    // 1) 準備 keep mask
    bool keep_mask[CONFIG_T::N];
    // #pragma HLS ARRAY_PARTITION variable=keep_mask complete dim=1

    init_mask_loop: for (int i = 0; i < CONFIG_T::N; i++) {
        #pragma HLS UNROLL
        keep_mask[i] = (i == 0); // 保留 CLS
    }

    read_topk_loop: for (int i = 0; i < CONFIG_T::keep_tokens - 2; i++) {
        #pragma HLS PIPELINE II=1
        int idx = topk_idx_in.read();
        if (idx >= 1 && idx < CONFIG_T::N) {
            keep_mask[idx] = true;
        }
    }

    // 2) 邊讀邊處理：保留的 token 直接寫出，被 prune 的累加成 fusion_acc
    data_T fusion_acc[D];
    // #pragma HLS ARRAY_PARTITION variable=fusion_acc complete dim=1

    // 清零累加器
    clear_acc_loop: for (int j = 0; j < D; j++) {
        #pragma HLS UNROLL
        for (int k = 0; k < data_T::size; k++) {
            #pragma HLS UNROLL
            fusion_acc[j][k] = 0;
        }
    }

    int pruned_count = 0;

    token_loop: for (int i = 0; i < CONFIG_T::N; i++) {
        // 每個 token 的 D 個字
        embed_loop: for (int j = 0; j < D; j++) {
            #pragma HLS PIPELINE II=1
            data_T inw = data_in.read();

            if (keep_mask[i]) {
                // 直接輸出保留 token
                res_T outw;
                cast_keep_parallel: for (int k = 0; k < res_T::size; k++) {
                    #pragma HLS UNROLL
                    outw[k] = (typename res_T::value_type) inw[k];
                }
                data_out.write(outw);
            } else {
                // 累加被 prune 的 token
                acc_parallel: for (int k = 0; k < data_T::size; k++) {
                    #pragma HLS UNROLL
                    fusion_acc[j][k] += inw[k];
                }
            }
        }
        if (!keep_mask[i]) {
            pruned_count++;
        }
    }

    // 3) 輸出融合 token：一次乘以倒數，避免在內層做除法
    if (pruned_count == 0) pruned_count = 1; // 避免除以 0
    typename res_T::value_type inv = typename res_T::value_type(1) / typename res_T::value_type(pruned_count);

    output_fusion_loop: for (int j = 0; j < D; j++) {
        #pragma HLS PIPELINE II=1
        res_T out_fusion;
        cast_out_fus_parallel: for (int k = 0; k < res_T::size; k++) {
            #pragma HLS UNROLL
            out_fusion[k] = (typename res_T::value_type)(fusion_acc[j][k] * inv);
        }
        data_out.write(out_fusion);
    }
}

} // namespace nnet

#endif
