// nnet_tap_cls.h
#ifndef NNET_TAP_CLS_H_
#define NNET_TAP_CLS_H_

#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {

struct tapcls_config_default {
    static const unsigned seq_len   = 16;  // 這個 layer 所在位置的序列長度（ex: 16/15/14/16）
    static const unsigned embed_dim = 192; // 通道數
};

// TapCLS：讀一條輸入流 → 主線原封不動送到 out_main；同時把第 0 個 token (CLS) 寫到 cls_out
template <typename in_T, typename out_main_T, typename out_cls_T, typename CONFIG_T>
void TapCLS(hls::stream<in_T>      &in,
            hls::stream<out_main_T> &out_main,
            hls::stream<out_cls_T>  &cls_out) {
#pragma HLS INLINE off
    const unsigned S = CONFIG_T::seq_len;
    const unsigned D = CONFIG_T::embed_dim;

    // 逐 token 讀入，token 0 → 也寫到 cls_out
read_seq:
    for (unsigned i = 0; i < S; i++) {
    read_tok:
        for (unsigned j = 0; j < D; j++) {
#pragma HLS PIPELINE II=1
            in_T v = in.read();

            // 主線照常前送
            out_main_T m;
        copy_main:
            for (unsigned k = 0; k < in_T::size; k++) {
#pragma HLS UNROLL
                m[k] = (typename out_main_T::value_type) v[k];
            }
            out_main.write(m);

            // 只有 i==0（CLS）時，複寫到 side-stream
            if (i == 0) {
                out_cls_T c;
            copy_cls:
                for (unsigned k = 0; k < in_T::size; k++) {
#pragma HLS UNROLL
                    c[k] = (typename out_cls_T::value_type) v[k];
                }
                cls_out.write(c);
            }
        }
    }
}

} // namespace nnet
#endif
