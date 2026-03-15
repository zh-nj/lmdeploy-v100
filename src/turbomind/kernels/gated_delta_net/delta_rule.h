// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// Fused Recurrent Gated Delta Rule — Decode (single-step update)
// with inline gating computation (eliminates separate gating kernel).
//
// Gating is computed inline from raw inputs:
//   g    = exp(-exp(A_log[h]) * softplus(alpha[t,h] + dt_bias[h]))
//   beta = sigmoid(beta_raw[t,h])
//
// Recurrence (per head):
//   q_n  = q / ||q||_2 * scale
//   k_n  = k / ||k||_2
//   S    = g * S
//   kv_mem = S^T @ k_n
//   delta  = beta * (v - kv_mem)
//   S    = S + k_n @ delta^T
//   out  = S^T @ q_n
//
// Parameters:
//   output           [batch, num_heads, head_v_dim]  fp16
//   recurrent_state  [batch, num_heads, head_k_dim, head_v_dim]  fp32, updated in-place
//   q                [batch, ?, head_k_dim]  fp16, strided
//   k                [batch, ?, head_k_dim]  fp16, strided
//   v                [batch, ?, head_v_dim]  fp16, strided (from conv_out)
//   A_log            [num_heads]  fp16, log decay factor
//   dt_bias          [num_heads]  fp16, time step bias
//   alpha_ptr        [batch, alpha_stride]  fp16, raw alpha (in proj_all)
//   beta_raw_ptr     [batch, beta_stride]   fp16, raw beta  (in proj_all)
//   batch, num_heads, q_num_heads, kv_ratio, q_stride, k_stride, v_stride,
//   head_k_dim, head_v_dim, stream, recurrent_state_batch_stride: same as before
//   alpha_stride     stride between adjacent batch rows for alpha (in half elements)
//   beta_stride      stride between adjacent batch rows for beta_raw (in half elements)
//   alpha_col_offset column offset within each row to reach alpha[h]
//   beta_col_offset  column offset within each row to reach beta_raw[h]
void invokeRecurrentDeltaRuleDecode(half*        output,
                                    float*       recurrent_state,
                                    const half*  q,
                                    const half*  k,
                                    const half*  v,
                                    const half*  A_log,
                                    const half*  dt_bias,
                                    const half*  alpha_ptr,
                                    const half*  beta_raw_ptr,
                                    int          batch,
                                    int          num_heads,
                                    int          q_num_heads,
                                    int          kv_ratio,
                                    int          q_stride,
                                    int          k_stride,
                                    int          v_stride,
                                    int          head_k_dim,
                                    int          head_v_dim,
                                    cudaStream_t stream,
                                    int          recurrent_state_batch_stride,
                                    int          alpha_stride,
                                    int          beta_stride,
                                    int          alpha_col_offset,
                                    int          beta_col_offset);

// Chunk Gated Delta Rule — Prefill with inline gating computation.
//
// Same gating formulas as decode, applied per-token.
void invokeChunkDeltaRulePrefill(half*        output,
                                 float*       recurrent_state,
                                 const half*  q,
                                 const half*  k,
                                 const half*  v,
                                 const half*  A_log,
                                 const half*  dt_bias,
                                 const half*  alpha_ptr,
                                 const half*  beta_raw_ptr,
                                 const int*   cu_seqlens,
                                 int          batch,
                                 int          token_num,
                                 int          num_heads,
                                 int          q_num_heads,
                                 int          kv_ratio,
                                 int          q_stride,
                                 int          k_stride,
                                 int          v_stride,
                                 int          head_k_dim,
                                 int          head_v_dim,
                                 cudaStream_t stream,
                                 int          recurrent_state_batch_stride,
                                 int          alpha_stride,
                                 int          beta_stride,
                                 int          alpha_col_offset,
                                 int          beta_col_offset);

}  // namespace turbomind
