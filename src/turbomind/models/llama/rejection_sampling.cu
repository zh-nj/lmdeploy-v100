// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/rejection_sampling.h"

#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace turbomind {

// Greedy rejection sampling kernel: one block per batch element.
// Each block processes K+1 logit vectors sequentially, computing argmax for each,
// then compares against draft tokens to find the first mismatch.
template<typename T>
__global__ void GreedyRejectKernel(int*       num_accepted,   // [batch]
                                   int*       bonus_tokens,   // [batch]
                                   const T*   logits,         // [batch, K+1, vocab_size]
                                   const int* draft_tokens,   // [batch, K]
                                   int        K,
                                   int        vocab_size)
{
    const int b = blockIdx.x;

    const T*   batch_logits = logits + (size_t)b * (K + 1) * vocab_size;
    const int* batch_drafts = draft_tokens + (size_t)b * K;

    // Shared memory for block-level argmax reduction
    __shared__ float s_max_val[32];
    __shared__ int   s_max_idx[32];
    // Shared memory for broadcasting argmax result to all threads
    __shared__ int s_argmax_token;

    const int warp_id   = threadIdx.x / 32;
    const int lane_id   = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    int accepted = K;  // assume all accepted, will be overwritten on mismatch
    int bonus    = 0;

    for (int pos = 0; pos <= K; ++pos) {
        const T* pos_logits = batch_logits + (size_t)pos * vocab_size;

        // Each thread finds local max over its strided elements
        float max_val = -1e30f;
        int   max_idx = 0;
        for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            float val = static_cast<float>(pos_logits[i]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        // Warp-level reduction
        for (int mask = 16; mask > 0; mask >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, max_val, mask);
            int   other_idx = __shfl_xor_sync(0xffffffff, max_idx, mask);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        // Block-level reduction via shared memory
        if (lane_id == 0) {
            s_max_val[warp_id] = max_val;
            s_max_idx[warp_id] = max_idx;
        }
        __syncthreads();

        if (threadIdx.x < num_warps) {
            max_val = s_max_val[threadIdx.x];
            max_idx = s_max_idx[threadIdx.x];
        }
        else {
            max_val = -1e30f;
            max_idx = 0;
        }

        if (threadIdx.x < 32) {
            for (int mask = 16; mask > 0; mask >>= 1) {
                float other_val = __shfl_xor_sync(0xffffffff, max_val, mask);
                int   other_idx = __shfl_xor_sync(0xffffffff, max_idx, mask);
                if (other_val > max_val) {
                    max_val = other_val;
                    max_idx = other_idx;
                }
            }
        }

        // Thread 0 broadcasts the argmax token
        if (threadIdx.x == 0) {
            s_argmax_token = max_idx;
        }
        __syncthreads();

        int target_token = s_argmax_token;

        // Compare with draft token (positions 0..K-1)
        // Position K is the "bonus" position — no draft to compare against
        if (pos < K) {
            int draft = batch_drafts[pos];
            if (draft != target_token && accepted == K) {
                // First mismatch found
                accepted = pos;
                bonus    = target_token;
            }
        }
        else {
            // pos == K: this is the bonus token position
            if (accepted == K) {
                // All K drafts matched; bonus = argmax(logits[K])
                bonus = target_token;
            }
        }

        // Early exit: once we found a mismatch and computed the bonus token,
        // no need to process remaining positions
        if (accepted < K) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        num_accepted[b] = accepted;
        bonus_tokens[b] = bonus;
    }
}

RejectionResult GreedyReject(const void*  verification_logits,
                              const int*   draft_tokens,
                              int          batch_size,
                              int          K,
                              int          vocab_size,
                              DataType     dtype,
                              cudaStream_t stream)
{
    RejectionResult result;
    result.num_accepted = Buffer_<int>(batch_size, kDEVICE);
    result.bonus_tokens = Buffer_<int>(batch_size, kDEVICE);

    if (batch_size == 0 || K == 0) {
        return result;
    }

    constexpr int block = 256;
    const int     grid  = batch_size;

    int* d_num_accepted = result.num_accepted.data();
    int* d_bonus_tokens = result.bonus_tokens.data();

    if (dtype == DataType::kFloat16) {
        GreedyRejectKernel<<<grid, block, 0, stream>>>(
            d_num_accepted, d_bonus_tokens, (const half*)verification_logits, draft_tokens, K, vocab_size);
    }
#ifdef ENABLE_BF16
    else if (dtype == DataType::kBfloat16) {
        GreedyRejectKernel<<<grid, block, 0, stream>>>(
            d_num_accepted, d_bonus_tokens, (const __nv_bfloat16*)verification_logits, draft_tokens, K, vocab_size);
    }
#endif
    else {
        // Fallback: treat as float
        GreedyRejectKernel<<<grid, block, 0, stream>>>(
            d_num_accepted, d_bonus_tokens, (const float*)verification_logits, draft_tokens, K, vocab_size);
    }

    return result;
}

}  // namespace turbomind
