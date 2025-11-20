// nnet_clc_stream.h
#ifndef NNET_CLC_STREAM_H_
#define NNET_CLC_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {

struct clc_base_config {
    static const unsigned seq_len   = 18;
    static const unsigned embed_dim = 192;
    static const unsigned n_clr     = 1;
    static const bool include_gap   = true;   // 每個 block 快取 GAP
    static const bool pool_cls      = false;  // GAP 不含 CLS
    static const bool pool_clr      = false;  // GAP 不含 CLR
    typedef ap_fixed<80,32> accum_t;          // GAP 累加器
};

// 主線透傳 + 把 GAP / CLR 寫入 cache_out
template <typename data_T, typename cache_T, typename CONFIG_T>
void CLC_CachePush(hls::stream<data_T>  &data_in,
                   hls::stream<data_T>  &data_out,
                   hls::stream<cache_T> &cache_out) {
#pragma HLS INLINE off
    const unsigned S    = CONFIG_T::seq_len;
    const unsigned D    = CONFIG_T::embed_dim;
    const unsigned NCLR = CONFIG_T::n_clr;

    typename CONFIG_T::accum_t gap_acc[D][data_T::size];
#pragma HLS ARRAY_PARTITION variable=gap_acc complete dim=2
init_gap:
    for (unsigned j=0;j<D;j++){
#pragma HLS UNROLL
        for (unsigned k=0;k<data_T::size;k++){ gap_acc[j][k]=0; }
    }
    struct token_buf_t { cache_T data[D]; };
    token_buf_t clr_buf[(NCLR>0?NCLR:1)];
#pragma HLS ARRAY_PARTITION variable=clr_buf complete dim=1

    const unsigned pool_start = CONFIG_T::pool_cls ? 0 : 1;
    const unsigned pool_end   = CONFIG_T::pool_clr ? S : (S - NCLR); // [start,end)
    const unsigned pool_len   = (pool_end>pool_start) ? (pool_end-pool_start) : 0;

read_seq:
    for (unsigned i=0;i<S;i++){
    read_tok:
        for (unsigned j=0;j<D;j++){
#pragma HLS PIPELINE II=1
            data_T v = data_in.read();
            data_out.write(v);

            if (CONFIG_T::include_gap && i>=pool_start && i<pool_end){
            acc_lane:
                for (unsigned k=0;k<data_T::size;k++){
#pragma HLS UNROLL
                    gap_acc[j][k] += (typename CONFIG_T::accum_t)v[k];
                }
            }

            if (NCLR>0 && i>=S-NCLR){
                const unsigned idx = i - (S - NCLR);
            copy_clr_lane:
                for (unsigned k=0;k<data_T::size;k++){
#pragma HLS UNROLL
                    clr_buf[idx].data[j][k] = (typename cache_T::value_type)v[k];
                }
            }
        }
    }

    // GAP token（1 個）
    if (CONFIG_T::include_gap && pool_len>0){
    write_gap_tok:
        for (unsigned j=0;j<D;j++){
#pragma HLS PIPELINE II=1
            cache_T c;
        norm_lane:
            for (unsigned k=0;k<data_T::size;k++){
#pragma HLS UNROLL
                auto tmp = gap_acc[j][k] / (typename CONFIG_T::accum_t)pool_len;
                c[k] = (typename cache_T::value_type)tmp;
            }
            cache_out.write(c);
        }
    }

    if (NCLR>0){
    write_clr_tokens:
        for (unsigned t=0;t<NCLR;t++){
        write_one:
            for (unsigned j=0;j<D;j++){
#pragma HLS PIPELINE II=1
                cache_out.write(clr_buf[t].data[j]);
            }
        }
    }
}

// 剪枝後主線先透傳，再把 cache 接回（或只清空）
struct clc_recover_base {
    static const unsigned embed_dim = 192;
    static const unsigned N_pruned  = 10; // 當層主線長度（剪枝後）
    static const unsigned n_carriers = 6; // 這次要接回多少 token
    static const bool recover_to_stream = true;
};

// 3-cache 版本：一個 group 內三個 block 各自的 cache（每條各 2 tokens）
template <typename data_T, typename cache_T, class res_T, typename CONFIG_T>
void CLC_RecoverAndEmpty3(
    hls::stream<data_T>  &pruned_in,
    hls::stream<cache_T> &cache_b0,
    hls::stream<cache_T> &cache_b1,
    hls::stream<cache_T> &cache_b2,
    hls::stream<res_T>  &data_out
){
#pragma HLS INLINE off
    const unsigned D = CONFIG_T::embed_dim;

    // 1) 先把剪枝後主線透傳
forward_pruned:
    for (unsigned i = 0; i < CONFIG_T::N_pruned; i++) {
    forward_pruned_tok:
        for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
            data_T v = pruned_in.read();
            res_T v_out;
            for (unsigned k = 0; k < res_T::size; k++) {
                v_out[k] = (typename res_T::value_type)v[k];
            }
            data_out.write(v_out);
        }
    }

    // 定義一個小工具內聯函式，複製一個 token 的每個 lane
    auto copy_and_write = [&](const cache_T &c){
#pragma HLS INLINE
        if (CONFIG_T::recover_to_stream) {
            res_T v;
        copy_lane:
            for (unsigned k = 0; k < res_T::size; k++) {
#pragma HLS UNROLL
                v[k] = (typename res_T::value_type)c[k];
            }
            data_out.write(v);
        }
    };

    // 2) 依序讀三條 cache；每條固定讀 2 個 token（每 token 有 D 個 chunk）
drain_b0_tok0:
    for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
        cache_T c = cache_b0.read();
        copy_and_write(c);
    }
drain_b0_tok1:
    for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
        cache_T c = cache_b0.read();
        copy_and_write(c);
    }

drain_b1_tok0:
    for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
        cache_T c = cache_b1.read();
        copy_and_write(c);
    }
drain_b1_tok1:
    for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
        cache_T c = cache_b1.read();
        copy_and_write(c);
    }

drain_b2_tok0:
    for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
        cache_T c = cache_b2.read();
        copy_and_write(c);
    }
drain_b2_tok1:
    for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
        cache_T c = cache_b2.read();
        copy_and_write(c);
    }
}

// 1-cache 版本（最後 group：第 10 層接回 2 個）
template <typename data_T, typename cache_T, class res_T, typename CONFIG_T>
void CLC_RecoverAndEmpty1(
    hls::stream<data_T>  &pruned_in,
    hls::stream<cache_T> &cache_b0,
    hls::stream<res_T>  &data_out
){
#pragma HLS INLINE off
    const unsigned D = CONFIG_T::embed_dim;

    for (unsigned i=0;i<CONFIG_T::N_pruned;i++){
        for (unsigned j=0;j<D;j++){
#pragma HLS PIPELINE II=1
            data_T v = pruned_in.read();
            res_T v_out;
            for (unsigned k = 0; k < res_T::size; k++) {
                v_out[k] = (typename res_T::value_type)v[k];
            }
            data_out.write(v_out);
        }
    }

    for (int t=0;t<2;t++){
        for (unsigned j=0;j<D;j++){
#pragma HLS PIPELINE II=1
            cache_T c = cache_b0.read();
            if (CONFIG_T::recover_to_stream){
                res_T v;
                for (unsigned k=0;k<res_T::size;k++) 
                    v[k] = (typename res_T::value_type)c[k];
                data_out.write(v);
            }
        }
    }
}


// 只把 cache 清空（不接回主線）
struct clc_empty_base {
    static const unsigned embed_dim = 192;
    static const unsigned n_carriers = 2;
};

template <typename cache_T, typename CONFIG_T>
void CLC_Empty(hls::stream<cache_T> &cache_in) {
#pragma HLS INLINE off
    const unsigned D = CONFIG_T::embed_dim;
drain:
    for (unsigned t=0;t<CONFIG_T::n_carriers;t++){
    drain_tok:
        for (unsigned j=0;j<D;j++){
#pragma HLS PIPELINE II=1
            (void)cache_in.read();
        }
    }
}

} // namespace nnet
#endif
