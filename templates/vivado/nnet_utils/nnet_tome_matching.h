#ifndef NNET_TOME_MATCHING_H_
#define NNET_TOME_MATCHING_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <cmath>

namespace nnet {

// Helper function: Compute L2 norm of a vector
template<typename T, int DIM>
T compute_l2_norm(T vec[DIM]) {
    #pragma HLS INLINE
    T sum = 0;
    L2_NORM:
    for (int i = 0; i < DIM; i++) {
        #pragma HLS UNROLL
        sum += vec[i] * vec[i];
    }
    return hls::sqrt(sum);
}

// Helper function: Normalize a vector
template<typename T, int DIM>
void normalize_vector(T vec[DIM], T norm) {
    #pragma HLS INLINE
    NORMALIZE:
    for (int i = 0; i < DIM; i++) {
        #pragma HLS UNROLL
        vec[i] = vec[i] / (norm + 1e-6);  // Add epsilon to avoid division by zero
    }
}

// Helper function: Compute dot product (cosine similarity after normalization)
template<typename T, int DIM>
T dot_product(T a[DIM], T b[DIM]) {
    #pragma HLS INLINE
    T sum = 0;
    DOT_PROD:
    for (int i = 0; i < DIM; i++) {
        #pragma HLS UNROLL
        sum += a[i] * b[i];
    }
    return sum;
}

// Structure to hold similarity score and index for sorting
struct SimilarityPair {
    float score;
    int src_idx;
    int dst_idx;
};

// Comparison function for sorting
inline bool compare_similarity(const SimilarityPair &a, const SimilarityPair &b) {
    return a.score > b.score;  // Descending order
}

// Simple insertion sort for small arrays (hardware-friendly)
template<int N>
void insertion_sort(SimilarityPair arr[N]) {
    SORT_OUTER:
    for (int i = 1; i < N; i++) {
        #pragma HLS PIPELINE OFF
        SimilarityPair key = arr[i];
        int j = i - 1;
        
        SORT_INNER:
        while (j >= 0 && arr[j].score < key.score) {
            #pragma HLS PIPELINE OFF
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// ToMe Bipartite Soft Matching
// Computes which tokens to merge based on K (attention keys)
template<typename CONFIG_T>
void compute_tome_matching(
    typename CONFIG_T::in_proj_out_t K[CONFIG_T::seq_len * CONFIG_T::n_head * CONFIG_T::head_dim],
    hls::stream<int> &merge_src_idx,
    hls::stream<int> &merge_dst_idx,
    int r  // Number of tokens to merge
)
{
    #pragma HLS INLINE OFF
    
    const int SEQ_LEN = CONFIG_T::seq_len;
    const int HEAD_DIM = CONFIG_T::head_dim;
    const int N_HEAD = CONFIG_T::n_head;
    
    // Average K across all heads
    typename CONFIG_T::in_proj_out_t K_avg[SEQ_LEN * HEAD_DIM];
    #pragma HLS ARRAY_PARTITION variable=K_avg cyclic factor=HEAD_DIM
    
    // Average across heads
    AVG_HEADS:
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            #pragma HLS PIPELINE II=1
            typename CONFIG_T::in_proj_out_t sum = 0;
            for (int h = 0; h < N_HEAD; h++) {
                #pragma HLS UNROLL
                sum += K[i * N_HEAD * HEAD_DIM + h * HEAD_DIM + d];
            }
            K_avg[i * HEAD_DIM + d] = sum / N_HEAD;
        }
    }
    
    // Normalize K_avg
    NORMALIZE_K:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE OFF
        typename CONFIG_T::in_proj_out_t vec[HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=vec complete
        
        // Load vector
        for (int d = 0; d < HEAD_DIM; d++) {
            #pragma HLS UNROLL
            vec[d] = K_avg[i * HEAD_DIM + d];
        }
        
        // Compute norm and normalize
        typename CONFIG_T::in_proj_out_t norm = compute_l2_norm<typename CONFIG_T::in_proj_out_t, HEAD_DIM>(vec);
        normalize_vector<typename CONFIG_T::in_proj_out_t, HEAD_DIM>(vec, norm);
        
        // Store back
        for (int d = 0; d < HEAD_DIM; d++) {
            #pragma HLS UNROLL
            K_avg[i * HEAD_DIM + d] = vec[d];
        }
    }
    
    // Bipartite partition: A = odd indices (1,3,5,...), B = even indices (0,2,4,...)
    // Skip CLS token (index 0), so A starts from index 1
    const int N_A = (SEQ_LEN - 1) / 2;  // Number of tokens in set A
    const int N_B = (SEQ_LEN - 1) - N_A;  // Number of tokens in set B
    
    // Store similarity scores and indices
    SimilarityPair similarities[SEQ_LEN];  // Over-allocate for simplicity
    #pragma HLS ARRAY_PARTITION variable=similarities cyclic factor=8
    
    int pair_count = 0;
    
    // For each token in A, find the most similar token in B
    MATCH_TOKENS:
    for (int a_idx = 0; a_idx < N_A; a_idx++) {
        #pragma HLS PIPELINE OFF
        int global_a_idx = 2 * a_idx + 1;  // 1, 3, 5, 7, ...
        
        typename CONFIG_T::in_proj_out_t a_vec[HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=a_vec complete
        
        // Load A token
        for (int d = 0; d < HEAD_DIM; d++) {
            #pragma HLS UNROLL
            a_vec[d] = K_avg[global_a_idx * HEAD_DIM + d];
        }
        
        // Find best match in B
        typename CONFIG_T::in_proj_out_t best_score = -1000.0;
        int best_b_idx = 0;
        
        FIND_BEST_MATCH:
        for (int b_idx = 0; b_idx < N_B; b_idx++) {
            #pragma HLS PIPELINE OFF
            int global_b_idx = 2 * b_idx + 2;  // 2, 4, 6, 8, ... (skip CLS at 0)
            
            if (global_b_idx >= SEQ_LEN) break;
            
            typename CONFIG_T::in_proj_out_t b_vec[HEAD_DIM];
            #pragma HLS ARRAY_PARTITION variable=b_vec complete
            
            // Load B token
            for (int d = 0; d < HEAD_DIM; d++) {
                #pragma HLS UNROLL
                b_vec[d] = K_avg[global_b_idx * HEAD_DIM + d];
            }
            
            // Compute similarity
            typename CONFIG_T::in_proj_out_t score = dot_product<typename CONFIG_T::in_proj_out_t, HEAD_DIM>(a_vec, b_vec);
            
            if (score > best_score) {
                best_score = score;
                best_b_idx = global_b_idx;
            }
        }
        
        // Store the match
        similarities[pair_count].score = best_score;
        similarities[pair_count].src_idx = global_a_idx;  // Source (will be merged)
        similarities[pair_count].dst_idx = best_b_idx;    // Destination (merge target)
        pair_count++;
    }
    
    // Sort by similarity score (descending) and take top r pairs
    insertion_sort<SEQ_LEN>(similarities);
    
    // Output top r merge pairs
    OUTPUT_INDICES:
    for (int i = 0; i < r; i++) {
        #pragma HLS PIPELINE
        merge_src_idx.write(similarities[i].src_idx);
        merge_dst_idx.write(similarities[i].dst_idx);
    }
}

} // namespace nnet

#endif
