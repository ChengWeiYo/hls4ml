#ifndef NNET_TOME_MERGING_H_
#define NNET_TOME_MERGING_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <algorithm>

namespace nnet {

struct merging_config {
    static const unsigned seq_len = 197;
    static const unsigned embed_dim = 192;
    static const unsigned n_merge = 49;  // r value
};

// ToMe Merging Layer - Simplified version without token size tracking
template<class data_T, class res_T, typename CONFIG_T>
void ToMeMergingLayer(
    hls::stream<data_T> &data_in,
    hls::stream<res_T> &data_out,
    hls::stream<int> &merge_src_idx,
    hls::stream<int> &merge_dst_idx
)
{
    #pragma HLS INLINE OFF
    
    // Buffer to store all input tokens
    typename data_T::value_type token_buffer[CONFIG_T::seq_len * CONFIG_T::embed_dim];
    #pragma HLS ARRAY_PARTITION variable=token_buffer cyclic factor=CONFIG_T::embed_dim
    
    // Flags to mark which tokens are merged (src tokens)
    bool is_merged[CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=is_merged complete
    
    // Store merge indices
    int src_indices[CONFIG_T::n_merge];
    int dst_indices[CONFIG_T::n_merge];
    #pragma HLS ARRAY_PARTITION variable=src_indices complete
    #pragma HLS ARRAY_PARTITION variable=dst_indices complete
    
    // Initialize flags
    INIT_FLAGS:
    for (int i = 0; i < CONFIG_T::seq_len; i++) {
        #pragma HLS UNROLL
        is_merged[i] = false;
    }
    
    // Read merge indices
    READ_INDICES:
    for (int i = 0; i < CONFIG_T::n_merge; i++) {
        #pragma HLS PIPELINE
        src_indices[i] = merge_src_idx.read();
        dst_indices[i] = merge_dst_idx.read();
        is_merged[src_indices[i]] = true;
    }
    
    // Read all input tokens into buffer
    READ_TOKENS:
    for (int i = 0; i < CONFIG_T::seq_len; i++) {
        for (int j = 0; j < CONFIG_T::embed_dim; j++) {
            #pragma HLS PIPELINE II=1
            data_T data_pack;
            if (j == 0) {
                data_pack = data_in.read();
            }
            token_buffer[i * CONFIG_T::embed_dim + j] = data_pack[j];
        }
    }
    
    // Merge tokens: add src tokens to dst tokens (average)
    MERGE_TOKENS:
    for (int m = 0; m < CONFIG_T::n_merge; m++) {
        int src_idx = src_indices[m];
        int dst_idx = dst_indices[m];
        
        for (int j = 0; j < CONFIG_T::embed_dim; j++) {
            #pragma HLS PIPELINE II=1
            // Average merge: (dst + src) / 2
            token_buffer[dst_idx * CONFIG_T::embed_dim + j] = 
                (token_buffer[dst_idx * CONFIG_T::embed_dim + j] + 
                 token_buffer[src_idx * CONFIG_T::embed_dim + j]) / 2;
        }
    }
    
    // Output unmerged and merged tokens
    OUTPUT_TOKENS:
    for (int i = 0; i < CONFIG_T::seq_len; i++) {
        if (!is_merged[i]) {
            for (int j = 0; j < CONFIG_T::embed_dim; j++) {
                #pragma HLS PIPELINE II=1
                res_T res_pack;
                res_pack[j] = token_buffer[i * CONFIG_T::embed_dim + j];
                if (j == CONFIG_T::embed_dim - 1) {
                    data_out.write(res_pack);
                }
            }
        }
    }
}

} // namespace nnet

#endif
