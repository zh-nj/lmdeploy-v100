// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"

#include <cstdint>

#include <cuda_runtime.h>
namespace turbomind {

void invokeGatherOutput(int*         output_ids,
                        const int*   ids,
                        const int*   context_length,
                        int          max_context_len,
                        int          max_gen_step,
                        int          max_output_len,
                        int          batch_size,
                        cudaStream_t stream);

void invokeUpdateOutput(int**        request_output_ids_ptrs,
                        int**        request_seqlen_ptrs,
                        const int*   output_ids,
                        const int*   sequence_lengths,
                        const int*   request_output_ids_lens,
                        int          max_session_len,
                        bool         token_generated,
                        int          batch_size,
                        cudaStream_t stream);

// [aaa, bbbb, cc, ddd] -> [aaabbbbccddd]
void invokeCompactOutputIds(int*         cu_output_ids,
                            const int*   output_ids,
                            const int*   sequence_lengths,
                            int          max_session_len,
                            bool         token_generated,
                            int          batch_size,
                            cudaStream_t stream);

void invokeIndexedCopy(void**       h_src_ptr,
                       void**       h_dst_ptr,
                       const int*   h_elem_sz,
                       const int*   h_src_idx,
                       const int*   h_dst_idx,
                       int          count,
                       int          n_copys,
                       cudaStream_t st);

void invokeBatchedCopy(void** src_ptr, void** dst_ptr, int* size, int count, cudaStream_t st);

// ABCDe            ABCDe     e
// ABCDEFGHIJk      ABCDEFGHIJk
// ABCDEFGHi    ->  ABCDEFGHi i
// ABCDEFGh         ABCDEFGh  h
// ABCd             ABCd      d
void invokePadLastTokenIds(
    int* token_ids, const int* context_length, int max_context_len, int batch_size, cudaStream_t stream);

void invokeGetFeatureOfLastToken(
    uint16_t* output, const uint16_t* input, const int* cu_seqlens, int dims, int batch_size, cudaStream_t stream);

template<typename T>
void invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream);

void invokeCastFloat2D(const core::Tensor& src, core::Tensor& dst, cudaStream_t stream);

void CollectHiddenStates(const Tensor& src, const Buffer_<int>& idxs, Ref<Tensor> dst, cudaStream_t st);

void BatchPrefixSum(const int** srcs, const int* ns, int** dsts, int count, cudaStream_t st);

inline void PrefixSum(const int* src, int n, int* dst, cudaStream_t st)
{
    return BatchPrefixSum(&src, &n, &dst, 1, st);
}

void AppendTokenIds(int**        token_ids_ptrs,  //
                    const int*   output_ids,
                    const int*   positions,
                    int          batch_size,
                    cudaStream_t stream);

/// Per-row argmax over logits: output_ids[i] = argmax(logits[i, :vocab_size])
void invokeArgmax(int*         output_ids,
                  const void*  logits,
                  int          batch_size,
                  int          vocab_size,
                  DataType     dtype,
                  cudaStream_t st);

/// Build selected_token_pos for speculative decoding steady-state:
/// output[i*(K+1)+j] = q_offsets[i] + j, for i in [0,bsz), j in [0,K]
void invokeBuildSpecSelectedPos(int*         output,
                                const int*   q_offsets,
                                int          bsz,
                                int          K,
                                cudaStream_t st);

/// Gather hidden states by num_accepted index for MTP draft:
/// dst[i] = src[i*(K+1) + num_accepted[i]], row-wise copy of hidden_size elements
void invokeGatherHiddenByAccepted(void*        dst,
                                  const void*  src,
                                  const int*   num_accepted,
                                  int          bsz,
                                  int          K,
                                  int          hidden_size,
                                  DataType     dtype,
                                  cudaStream_t st);

/// Scatter rows from contiguous src to strided positions in dst:
/// dst[(N*i + t) * row_size ... +row_size] = src[i * row_size ... +row_size]
/// Used by multi-token decode to place per-call output into interleaved layout.
void invokeScatterRows(void*        dst,
                       const void*  src,
                       int          bsz,
                       int          N,
                       int          t,
                       int          row_size_bytes,
                       cudaStream_t st);

/// Build cu_k_len for multi-token decode call t:
/// new_cu_k_len[i] = k_offsets_orig[i] - i * (N - 1 - t)
/// This adjusts context_len so that history_len = original_history + t (correct RoPE position).
void invokeBuildMtdCuKLen(int*         cu_k_len_out,
                          const int*   k_offsets_orig,
                          int          bsz,
                          int          delta,
                          cudaStream_t st);

/// Fill buffer with [0, 1, 2, ..., n-1] (iota)
void invokeIota(int* output, int n, cudaStream_t st);

}  // namespace turbomind
