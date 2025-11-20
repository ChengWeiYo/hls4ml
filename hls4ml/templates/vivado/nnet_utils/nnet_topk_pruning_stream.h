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
    static const unsigned N = 197;
    static const unsigned embed_dim = 192;
    static const unsigned keep_tokens = 197;
};

template<class data_T, class res_T, typename CONFIG_T>
void PruningLayer(
    hls::stream<data_T> &data_in,
    hls::stream<res_T> &data_out,
    hls::stream<int>    &topk_idx_in
    // const float keep_rate
    // const unsigned int N
) {
    const unsigned int D = CONFIG_T::embed_dim;
    // const unsigned int keep_tokens = static_cast<unsigned int>(std::ceil((CONFIG_T::N-1) * keep_rate));

    struct token_buf_t {
        data_T data[CONFIG_T::embed_dim];
    };

    hls::stream<token_buf_t> token_buffer_stream;
    #pragma HLS STREAM variable=token_buffer_stream depth=1

    hls::stream<token_buf_t> pruned_token_buffer_stream;
    #pragma HLS STREAM variable=pruned_token_buffer_stream depth=1

    #pragma HLS DATAFLOW

    // 建立一個 keep_mask[CONFIG_T::N]，預設 false
    bool keep_mask[CONFIG_T::N];
    // #pragma HLS ARRAY_PARTITION variable=keep_mask complete dim=1
    for (int i = 0; i < CONFIG_T::N; i++) {
        #pragma HLS UNROLL
        keep_mask[i] = false;
    }
    // std::cout << "finish keep_mask reset" << std::endl;

    // 讀取 Top-K 索引，將 keep_mask[idx] = true
    for (int i = 0; i < CONFIG_T::keep_tokens-1; i++) {
        #pragma HLS PIPELINE II=1
        int idx = topk_idx_in.read();
        // std::cout << "Read Top-K index: " << idx << std::endl;
        if (idx >= 0 && idx < CONFIG_T::N) {
            keep_mask[idx] = true;
        }
    }
    // std::cout << "finish keep_mask update" << std::endl;

    // Stage 1: Read input data and store in token buffer stream
    read_input:
    for (int i = 0; i < CONFIG_T::N; i++) {
        token_buf_t token_buffer;
        for (int j = 0; j < D; j++) {
            #pragma HLS PIPELINE II=1
            data_T data = data_in.read();
            for (int k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                token_buffer.data[j][k] = data[k];
            }
        }
        token_buffer_stream.write(token_buffer);
    }
    // std::cout << "finish read_input" << std::endl;

    // Stage 2: Prune tokens (excluding cls_token)
    // prune_tokens:
    // for (int i = 0; i < CONFIG_T::N; i++) {
    //     token_buf_t token_buffer = token_buffer_stream.read();
    //     if (i == CONFIG_T::keep_tokens-1) {
    //         for (int j = 0; j < CONFIG_T::keep_tokens; j++) {
    //             pruned_token_buffer_stream.write(token_buffer); // Pass cls_token directly
    //         }
    //     }
    // }
    prune_tokens:
    for (int i = 0; i < CONFIG_T::N; i++) {
        token_buf_t token_buffer = token_buffer_stream.read();
        if (i == 0 || keep_mask[i]) {
            pruned_token_buffer_stream.write(token_buffer); // Pass cls_token directly
        }
    }

    // Stage 3: Process sorted token buffer stream and write output data
    process_tokens:
    for (int i = 0; i < CONFIG_T::keep_tokens; i++) { // include cls_token
        token_buf_t token_buffer = pruned_token_buffer_stream.read();
        for (int j = 0; j < D; j++) {
            #pragma HLS PIPELINE II=1
            res_T out_data;
            for (int k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = token_buffer.data[j][k];
            }
            data_out.write(out_data);
        }
    }
}
}

#endif
