// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "src/turbomind/kernels/gated_delta_net/delta_rule.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// Inline gating computation (replaces separate gating kernel)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void computeGating(float& g_out,
                                              float& beta_out,
                                              float  a_log_val,
                                              float  dt_bias_val,
                                              float  alpha_val,
                                              float  beta_raw_val)
{
    // softplus(x) = log(1 + exp(x)), numerically stable
    const float sp_input = alpha_val + dt_bias_val;
    float sp;
    if (sp_input > 20.0f) {
        sp = sp_input;
    } else if (sp_input < -20.0f) {
        sp = expf(sp_input);
    } else {
        sp = log1pf(expf(sp_input));
    }
    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    const float log_decay = -expf(a_log_val) * sp;
    g_out = expf(log_decay);
    // beta = sigmoid(beta_raw)
    beta_out = 1.0f / (1.0f + expf(-beta_raw_val));
}

// ---------------------------------------------------------------------------
// Common reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warpReduceSum(float val)
{
    const unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val, float* smem)
{
    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp_id   = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    val = warpReduceSum(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    float sum = 0.f;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.f;
        sum = warpReduceSum(sum);
        if (lane == 0) {
            smem[0] = sum;
        }
    }
    __syncthreads();
    return smem[0];
}


// ---------------------------------------------------------------------------
// Decode Kernel with inline gating
// ---------------------------------------------------------------------------

__global__ void recurrentDeltaRuleDecodeKernel(half*        output,
                                               float*       recurrent_state,
                                               const half*  q,
                                               const half*  k,
                                               const half*  v,
                                               const half*  A_log,
                                               const half*  dt_bias,
                                               const half*  alpha_ptr,
                                               const half*  beta_raw_ptr,
                                               int          num_heads,
                                               int          q_num_heads,
                                               int          kv_ratio,
                                               int          q_stride,
                                               int          k_stride,
                                               int          v_stride,
                                               int          head_k_dim,
                                               int          head_v_dim,
                                               int          split_v,
                                               int          v_chunk,
                                               int          recurrent_state_batch_stride,
                                               int          alpha_stride,
                                               int          beta_stride,
                                               int          alpha_col_offset,
                                               int          beta_col_offset)
{
    const int bhv        = blockIdx.x;
    const int bh         = bhv / split_v;
    const int v_split_id = bhv - bh * split_v;
    const int tid        = threadIdx.x;
    const int b          = bh / num_heads;
    const int h          = bh % num_heads;
    const int vi_begin   = v_split_id * v_chunk;
    const int vi_end     = min(head_v_dim, vi_begin + v_chunk);

    if (tid >= head_k_dim) return;
    if (vi_begin >= vi_end) return;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* shared_vec = smem + head_k_dim;

    // Inline gating computation (thread 0 computes, then broadcast via shared mem)
    float g_val, beta_val;
    if (tid == 0) {
        float a_log_val   = __half2float(A_log[h]);
        float dt_bias_val = __half2float(dt_bias[h]);
        float alpha_val   = __half2float(alpha_ptr[b * alpha_stride + alpha_col_offset + h]);
        float beta_raw    = __half2float(beta_raw_ptr[b * beta_stride + beta_col_offset + h]);
        computeGating(g_val, beta_val, a_log_val, dt_bias_val, alpha_val, beta_raw);
        g_val    = isfinite(g_val) ? fminf(fmaxf(g_val, 0.f), 1.f) : 0.f;
        beta_val = isfinite(beta_val) ? fminf(fmaxf(beta_val, 0.f), 1.f) : 0.f;
        shared_vec[0] = g_val;
        shared_vec[1] = beta_val;
    }
    __syncthreads();
    g_val    = shared_vec[0];
    beta_val = shared_vec[1];
    __syncthreads();

    const float scale = rsqrtf((float)head_k_dim);

    const int qk_head    = kv_ratio > 1 ? (h / kv_ratio) : h;
    const int q_offset   = b * q_stride + qk_head * head_k_dim;
    const int k_offset   = b * k_stride + qk_head * head_k_dim;
    const int v_in_offset  = b * v_stride + h * head_v_dim;
    const int out_offset = bh * head_v_dim;
    const int s_offset  = b * recurrent_state_batch_stride + h * head_k_dim * head_v_dim;

    float q_val = __half2float(q[q_offset + tid]);
    float k_val = __half2float(k[k_offset + tid]);

    // L2 normalise q, then apply scale
    float q_sq_sum   = blockReduceSum(q_val * q_val, reduce_buf);
    float q_norm_den = q_sq_sum + 1e-6f;
    if (isfinite(q_norm_den) && q_norm_den > 0.f) {
        q_val *= rsqrtf(q_norm_den) * scale;
    } else {
        q_val = 0.f;
    }

    // L2 normalise k
    float k_sq_sum   = blockReduceSum(k_val * k_val, reduce_buf);
    float k_norm_den = k_sq_sum + 1e-6f;
    if (isfinite(k_norm_den) && k_norm_den > 0.f) {
        k_val *= rsqrtf(k_norm_den);
    } else {
        k_val = 0.f;
    }

    float* S_row = recurrent_state + s_offset + tid * head_v_dim;

    // Step 1: Decay state
    for (int vi = vi_begin; vi < vi_end; ++vi) {
        float s = S_row[vi] * g_val;
        S_row[vi] = isfinite(s) ? s : 0.f;
    }
    __syncthreads();

    // Step 2: kv_mem + delta
    for (int vi = vi_begin; vi < vi_end; ++vi) {
        float partial = k_val * S_row[vi];
        float kv_mem_vi = blockReduceSum(partial, reduce_buf);
        if (tid == 0) {
            float v_val = __half2float(v[v_in_offset + vi]);
            float delta = beta_val * (v_val - kv_mem_vi);
            shared_vec[vi - vi_begin] = isfinite(delta) ? delta : 0.f;
        }
    }
    __syncthreads();

    // Step 3: rank-1 update
    for (int vi = vi_begin; vi < vi_end; ++vi) {
        float s = S_row[vi] + k_val * shared_vec[vi - vi_begin];
        S_row[vi] = isfinite(s) ? s : 0.f;
    }
    __syncthreads();

    // Step 4: output
    for (int vi = vi_begin; vi < vi_end; ++vi) {
        float partial = q_val * S_row[vi];
        float out_vi = blockReduceSum(partial, reduce_buf);
        if (tid == 0) {
            out_vi = isfinite(out_vi) ? out_vi : 0.f;
            output[out_offset + vi] = __float2half(out_vi);
        }
    }
}

inline int chooseSplitV(int batch_heads, int head_v_dim)
{
    if (batch_heads <= 0 || head_v_dim <= 0) return 1;
    int split_v = 1;
    if (batch_heads < 128 && head_v_dim >= 64) {
        split_v = 128 / batch_heads;
        if (split_v < 1) split_v = 1;
        if (split_v > 8) split_v = 8;
        if (split_v > head_v_dim) split_v = head_v_dim;
        while (split_v > 1) {
            const int vc = (head_v_dim + split_v - 1) / split_v;
            if (vc >= 16) break;
            split_v >>= 1;
        }
    }
    return split_v;
}

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
                                    int          beta_col_offset)
{
    if (q_num_heads <= 0) q_num_heads = num_heads;
    kv_ratio = kv_ratio > 0 ? kv_ratio : 1;
    if (q_stride == 0) q_stride = q_num_heads * head_k_dim;
    if (k_stride == 0) k_stride = q_num_heads * head_k_dim;
    if (v_stride == 0) v_stride = num_heads * head_v_dim;

    const int batch_heads = batch * num_heads;
    const int split_v = chooseSplitV(batch_heads, head_v_dim);
    const int v_chunk = (head_v_dim + split_v - 1) / split_v;
    const int grid  = batch_heads * split_v;
    const int block = head_k_dim;
    // shared memory: reduce_buf[head_k_dim] + shared_vec[max(v_chunk, 2)]
    const int shared_vec_size = v_chunk > 2 ? v_chunk : 2;
    const int smem  = (head_k_dim + shared_vec_size) * sizeof(float);
    if (recurrent_state_batch_stride == 0) {
        recurrent_state_batch_stride = num_heads * head_k_dim * head_v_dim;
    }

    recurrentDeltaRuleDecodeKernel<<<grid, block, smem, stream>>>(
        output, recurrent_state, q, k, v,
        A_log, dt_bias, alpha_ptr, beta_raw_ptr,
        num_heads, q_num_heads, kv_ratio,
        q_stride, k_stride, v_stride,
        head_k_dim, head_v_dim, split_v, v_chunk,
        recurrent_state_batch_stride,
        alpha_stride, beta_stride, alpha_col_offset, beta_col_offset);
}


// ---------------------------------------------------------------------------
// Prefill Kernel with inline gating — register-cached S_row fast path
//
// When v_chunk is small enough (≤ 16), each thread caches its
// S_row[vi_begin..vi_end) slice in registers instead of reading/writing
// global memory every token.  This eliminates 4 reads + 2 writes per
// token per v-element from the inner loop, writing back only once at
// the end of the sequence.
//
// For the target model (head_v_dim=128, split_v=16 → v_chunk=8),
// each thread holds 8 floats = 32 bytes in registers.
// ---------------------------------------------------------------------------

template<int VChunk>
__global__ void chunkDeltaRulePrefillRegKernel(half*        output,
                                               float*       recurrent_state,
                                               const half*  q,
                                               const half*  k,
                                               const half*  v,
                                               const half*  A_log,
                                               const half*  dt_bias,
                                               const half*  alpha_ptr,
                                               const half*  beta_raw_ptr,
                                               const int*   cu_seqlens,
                                               int          num_heads,
                                               int          q_num_heads,
                                               int          kv_ratio,
                                               int          q_stride,
                                               int          k_stride,
                                               int          v_stride,
                                               int          head_k_dim,
                                               int          head_v_dim,
                                               int          split_v,
                                               int          v_chunk_rt,
                                               int          recurrent_state_batch_stride,
                                               int          alpha_stride,
                                               int          beta_stride,
                                               int          alpha_col_offset,
                                               int          beta_col_offset)
{
    const int bhv        = blockIdx.x;
    const int bh         = bhv / split_v;
    const int v_split_id = bhv - bh * split_v;
    const int tid        = threadIdx.x;
    const int b          = bh / num_heads;
    const int h          = bh % num_heads;
    const int vi_begin   = v_split_id * v_chunk_rt;
    const int vi_end     = min(head_v_dim, vi_begin + v_chunk_rt);
    const int actual_vc  = vi_end - vi_begin;

    if (tid >= head_k_dim) return;
    if (actual_vc <= 0) return;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* shared_vec = smem + head_k_dim;

    const float scale = rsqrtf((float)head_k_dim);

    const float a_log_val   = __half2float(A_log[h]);
    const float dt_bias_val = __half2float(dt_bias[h]);

    const int seq_start = cu_seqlens[b];
    const int seq_end   = cu_seqlens[b + 1];
    const int seq_len   = seq_end - seq_start;

    const int s_offset = b * recurrent_state_batch_stride + h * head_k_dim * head_v_dim;
    float* S_row_global = recurrent_state + s_offset + tid * head_v_dim;

    // Load S_row slice into registers
    float S_reg[VChunk];
    #pragma unroll
    for (int i = 0; i < VChunk; ++i) {
        S_reg[i] = (i < actual_vc) ? S_row_global[vi_begin + i] : 0.f;
    }

    for (int t = 0; t < seq_len; ++t) {
        const int token_idx = seq_start + t;
        const int qk_head   = kv_ratio > 1 ? (h / kv_ratio) : h;
        const int q_offset  = token_idx * q_stride + qk_head * head_k_dim;
        const int k_offset  = token_idx * k_stride + qk_head * head_k_dim;
        const int v_offset  = token_idx * v_stride + h * head_v_dim;

        // Inline gating
        float g_val, beta_val;
        if (tid == 0) {
            float alpha_val = __half2float(alpha_ptr[token_idx * alpha_stride + alpha_col_offset + h]);
            float beta_raw  = __half2float(beta_raw_ptr[token_idx * beta_stride + beta_col_offset + h]);
            computeGating(g_val, beta_val, a_log_val, dt_bias_val, alpha_val, beta_raw);
            g_val    = isfinite(g_val) ? fminf(fmaxf(g_val, 0.f), 1.f) : 0.f;
            beta_val = isfinite(beta_val) ? fminf(fmaxf(beta_val, 0.f), 1.f) : 0.f;
            shared_vec[0] = g_val;
            shared_vec[1] = beta_val;
        }
        __syncthreads();
        g_val    = shared_vec[0];
        beta_val = shared_vec[1];
        __syncthreads();

        float q_val = __half2float(q[q_offset + tid]);
        float k_val = __half2float(k[k_offset + tid]);

        // L2 normalise q
        float q_sq_sum   = blockReduceSum(q_val * q_val, reduce_buf);
        float q_norm_den = q_sq_sum + 1e-6f;
        if (isfinite(q_norm_den) && q_norm_den > 0.f) {
            q_val *= rsqrtf(q_norm_den) * scale;
        } else {
            q_val = 0.f;
        }

        // L2 normalise k
        float k_sq_sum   = blockReduceSum(k_val * k_val, reduce_buf);
        float k_norm_den = k_sq_sum + 1e-6f;
        if (isfinite(k_norm_den) && k_norm_den > 0.f) {
            k_val *= rsqrtf(k_norm_den);
        } else {
            k_val = 0.f;
        }

        // Step 1: Decay (registers only)
        #pragma unroll
        for (int i = 0; i < VChunk; ++i) {
            if (i < actual_vc) {
                float s = S_reg[i] * g_val;
                S_reg[i] = isfinite(s) ? s : 0.f;
            }
        }

        // Step 2: kv_mem + delta (S_reg read → blockReduce → shared_vec)
        #pragma unroll
        for (int i = 0; i < VChunk; ++i) {
            if (i < actual_vc) {
                float partial = k_val * S_reg[i];
                float kv_mem_vi = blockReduceSum(partial, reduce_buf);
                if (tid == 0) {
                    float v_val = __half2float(v[v_offset + vi_begin + i]);
                    float delta = beta_val * (v_val - kv_mem_vi);
                    shared_vec[i] = isfinite(delta) ? delta : 0.f;
                }
            }
        }
        __syncthreads();

        // Step 3: rank-1 update (registers only)
        #pragma unroll
        for (int i = 0; i < VChunk; ++i) {
            if (i < actual_vc) {
                float s = S_reg[i] + k_val * shared_vec[i];
                S_reg[i] = isfinite(s) ? s : 0.f;
            }
        }
        __syncthreads();

        // Step 4: output (S_reg read → blockReduce → output)
        #pragma unroll
        for (int i = 0; i < VChunk; ++i) {
            if (i < actual_vc) {
                float partial = q_val * S_reg[i];
                float out_vi = blockReduceSum(partial, reduce_buf);
                if (tid == 0) {
                    const int out_off = token_idx * num_heads * head_v_dim + h * head_v_dim;
                    out_vi = isfinite(out_vi) ? out_vi : 0.f;
                    output[out_off + vi_begin + i] = __float2half(out_vi);
                }
            }
        }
        __syncthreads();
    }

    // Write S_row back to global memory once at the end
    #pragma unroll
    for (int i = 0; i < VChunk; ++i) {
        if (i < actual_vc) {
            S_row_global[vi_begin + i] = S_reg[i];
        }
    }
}

// Generic prefill kernel for large runtime v_chunk.
// This path keeps full correctness when v_chunk exceeds the register-cached
// template limit.
__global__ void chunkDeltaRulePrefillKernel(half*        output,
                                            float*       recurrent_state,
                                            const half*  q,
                                            const half*  k,
                                            const half*  v,
                                            const half*  A_log,
                                            const half*  dt_bias,
                                            const half*  alpha_ptr,
                                            const half*  beta_raw_ptr,
                                            const int*   cu_seqlens,
                                            int          num_heads,
                                            int          q_num_heads,
                                            int          kv_ratio,
                                            int          q_stride,
                                            int          k_stride,
                                            int          v_stride,
                                            int          head_k_dim,
                                            int          head_v_dim,
                                            int          split_v,
                                            int          v_chunk,
                                            int          recurrent_state_batch_stride,
                                            int          alpha_stride,
                                            int          beta_stride,
                                            int          alpha_col_offset,
                                            int          beta_col_offset)
{
    const int bhv        = blockIdx.x;
    const int bh         = bhv / split_v;
    const int v_split_id = bhv - bh * split_v;
    const int tid        = threadIdx.x;
    const int b          = bh / num_heads;
    const int h          = bh % num_heads;
    const int vi_begin   = v_split_id * v_chunk;
    const int vi_end     = min(head_v_dim, vi_begin + v_chunk);

    if (tid >= head_k_dim) return;
    if (vi_begin >= vi_end) return;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* shared_vec = smem + head_k_dim;

    const float scale = rsqrtf((float)head_k_dim);

    const float a_log_val   = __half2float(A_log[h]);
    const float dt_bias_val = __half2float(dt_bias[h]);

    const int seq_start = cu_seqlens[b];
    const int seq_end   = cu_seqlens[b + 1];
    const int seq_len   = seq_end - seq_start;

    const int s_offset = b * recurrent_state_batch_stride + h * head_k_dim * head_v_dim;
    float* S_row = recurrent_state + s_offset + tid * head_v_dim;

    for (int t = 0; t < seq_len; ++t) {
        const int token_idx = seq_start + t;
        const int qk_head   = kv_ratio > 1 ? (h / kv_ratio) : h;
        const int q_offset  = token_idx * q_stride + qk_head * head_k_dim;
        const int k_offset  = token_idx * k_stride + qk_head * head_k_dim;
        const int v_offset  = token_idx * v_stride + h * head_v_dim;

        // Inline gating
        float g_val, beta_val;
        if (tid == 0) {
            float alpha_val = __half2float(alpha_ptr[token_idx * alpha_stride + alpha_col_offset + h]);
            float beta_raw  = __half2float(beta_raw_ptr[token_idx * beta_stride + beta_col_offset + h]);
            computeGating(g_val, beta_val, a_log_val, dt_bias_val, alpha_val, beta_raw);
            g_val    = isfinite(g_val) ? fminf(fmaxf(g_val, 0.f), 1.f) : 0.f;
            beta_val = isfinite(beta_val) ? fminf(fmaxf(beta_val, 0.f), 1.f) : 0.f;
            shared_vec[0] = g_val;
            shared_vec[1] = beta_val;
        }
        __syncthreads();
        g_val    = shared_vec[0];
        beta_val = shared_vec[1];
        __syncthreads();

        float q_val = __half2float(q[q_offset + tid]);
        float k_val = __half2float(k[k_offset + tid]);

        // L2 normalise q
        float q_sq_sum   = blockReduceSum(q_val * q_val, reduce_buf);
        float q_norm_den = q_sq_sum + 1e-6f;
        if (isfinite(q_norm_den) && q_norm_den > 0.f) {
            q_val *= rsqrtf(q_norm_den) * scale;
        } else {
            q_val = 0.f;
        }

        // L2 normalise k
        float k_sq_sum   = blockReduceSum(k_val * k_val, reduce_buf);
        float k_norm_den = k_sq_sum + 1e-6f;
        if (isfinite(k_norm_den) && k_norm_den > 0.f) {
            k_val *= rsqrtf(k_norm_den);
        } else {
            k_val = 0.f;
        }

        // Step 1: Decay state
        for (int vi = vi_begin; vi < vi_end; ++vi) {
            float s = S_row[vi] * g_val;
            S_row[vi] = isfinite(s) ? s : 0.f;
        }
        __syncthreads();

        // Step 2: kv_mem + delta
        for (int vi = vi_begin; vi < vi_end; ++vi) {
            float partial = k_val * S_row[vi];
            float kv_mem_vi = blockReduceSum(partial, reduce_buf);
            if (tid == 0) {
                float v_val = __half2float(v[v_offset + vi]);
                float delta = beta_val * (v_val - kv_mem_vi);
                shared_vec[vi - vi_begin] = isfinite(delta) ? delta : 0.f;
            }
        }
        __syncthreads();

        // Step 3: rank-1 update
        for (int vi = vi_begin; vi < vi_end; ++vi) {
            float s = S_row[vi] + k_val * shared_vec[vi - vi_begin];
            S_row[vi] = isfinite(s) ? s : 0.f;
        }
        __syncthreads();

        // Step 4: output
        for (int vi = vi_begin; vi < vi_end; ++vi) {
            float partial = q_val * S_row[vi];
            float out_vi = blockReduceSum(partial, reduce_buf);
            if (tid == 0) {
                const int out_off = token_idx * num_heads * head_v_dim + h * head_v_dim;
                out_vi = isfinite(out_vi) ? out_vi : 0.f;
                output[out_off + vi] = __float2half(out_vi);
            }
        }
        __syncthreads();
    }
}

// Tiled register-cached prefill kernel for large runtime v_chunk.
// It processes V in fixed register tiles (TileVChunk) to reduce global-memory
// traffic while keeping register pressure bounded on older architectures.
template<int TileVChunk>
__global__ void chunkDeltaRulePrefillTiledRegKernel(half*        output,
                                                    float*       recurrent_state,
                                                    const half*  q,
                                                    const half*  k,
                                                    const half*  v,
                                                    const half*  A_log,
                                                    const half*  dt_bias,
                                                    const half*  alpha_ptr,
                                                    const half*  beta_raw_ptr,
                                                    const int*   cu_seqlens,
                                                    int          num_heads,
                                                    int          q_num_heads,
                                                    int          kv_ratio,
                                                    int          q_stride,
                                                    int          k_stride,
                                                    int          v_stride,
                                                    int          head_k_dim,
                                                    int          head_v_dim,
                                                    int          split_v,
                                                    int          v_chunk,
                                                    int          recurrent_state_batch_stride,
                                                    int          alpha_stride,
                                                    int          beta_stride,
                                                    int          alpha_col_offset,
                                                    int          beta_col_offset)
{
    const int bhv        = blockIdx.x;
    const int bh         = bhv / split_v;
    const int v_split_id = bhv - bh * split_v;
    const int tid        = threadIdx.x;
    const int b          = bh / num_heads;
    const int h          = bh % num_heads;
    const int vi_begin   = v_split_id * v_chunk;
    const int vi_end     = min(head_v_dim, vi_begin + v_chunk);
    const int actual_vc  = vi_end - vi_begin;

    if (tid >= head_k_dim) return;
    if (actual_vc <= 0) return;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* shared_vec = smem + head_k_dim;

    const float scale = rsqrtf((float)head_k_dim);

    const float a_log_val   = __half2float(A_log[h]);
    const float dt_bias_val = __half2float(dt_bias[h]);

    const int seq_start = cu_seqlens[b];
    const int seq_end   = cu_seqlens[b + 1];
    const int seq_len   = seq_end - seq_start;

    const int s_offset = b * recurrent_state_batch_stride + h * head_k_dim * head_v_dim;
    float* S_row_global = recurrent_state + s_offset + tid * head_v_dim;

    for (int t = 0; t < seq_len; ++t) {
        const int token_idx = seq_start + t;
        const int qk_head   = kv_ratio > 1 ? (h / kv_ratio) : h;
        const int q_offset  = token_idx * q_stride + qk_head * head_k_dim;
        const int k_offset  = token_idx * k_stride + qk_head * head_k_dim;
        const int v_offset  = token_idx * v_stride + h * head_v_dim;
        const int out_off   = token_idx * num_heads * head_v_dim + h * head_v_dim;

        // Inline gating
        float g_val, beta_val;
        if (tid == 0) {
            float alpha_val = __half2float(alpha_ptr[token_idx * alpha_stride + alpha_col_offset + h]);
            float beta_raw  = __half2float(beta_raw_ptr[token_idx * beta_stride + beta_col_offset + h]);
            computeGating(g_val, beta_val, a_log_val, dt_bias_val, alpha_val, beta_raw);
            g_val    = isfinite(g_val) ? fminf(fmaxf(g_val, 0.f), 1.f) : 0.f;
            beta_val = isfinite(beta_val) ? fminf(fmaxf(beta_val, 0.f), 1.f) : 0.f;
            shared_vec[0] = g_val;
            shared_vec[1] = beta_val;
        }
        __syncthreads();
        g_val    = shared_vec[0];
        beta_val = shared_vec[1];
        __syncthreads();

        float q_val = __half2float(q[q_offset + tid]);
        float k_val = __half2float(k[k_offset + tid]);

        // L2 normalise q
        float q_sq_sum   = blockReduceSum(q_val * q_val, reduce_buf);
        float q_norm_den = q_sq_sum + 1e-6f;
        if (isfinite(q_norm_den) && q_norm_den > 0.f) {
            q_val *= rsqrtf(q_norm_den) * scale;
        } else {
            q_val = 0.f;
        }

        // L2 normalise k
        float k_sq_sum   = blockReduceSum(k_val * k_val, reduce_buf);
        float k_norm_den = k_sq_sum + 1e-6f;
        if (isfinite(k_norm_den) && k_norm_den > 0.f) {
            k_val *= rsqrtf(k_norm_den);
        } else {
            k_val = 0.f;
        }

        // Process large v_chunk in fixed-size register tiles.
        for (int tile_begin = vi_begin; tile_begin < vi_end; tile_begin += TileVChunk) {
            const int tile_size = min(TileVChunk, vi_end - tile_begin);
            float     S_reg[TileVChunk];

            #pragma unroll
            for (int i = 0; i < TileVChunk; ++i) {
                if (i < tile_size) {
                    S_reg[i] = S_row_global[tile_begin + i];
                }
            }

            // Step 1: decay in registers
            #pragma unroll
            for (int i = 0; i < TileVChunk; ++i) {
                if (i < tile_size) {
                    float s = S_reg[i] * g_val;
                    S_reg[i] = isfinite(s) ? s : 0.f;
                }
            }

            // Step 2: kv_mem + delta
            #pragma unroll
            for (int i = 0; i < TileVChunk; ++i) {
                if (i < tile_size) {
                    float partial = k_val * S_reg[i];
                    float kv_mem_vi = blockReduceSum(partial, reduce_buf);
                    if (tid == 0) {
                        float v_val = __half2float(v[v_offset + tile_begin + i]);
                        float delta = beta_val * (v_val - kv_mem_vi);
                        shared_vec[i] = isfinite(delta) ? delta : 0.f;
                    }
                }
            }
            __syncthreads();

            // Step 3: rank-1 update
            #pragma unroll
            for (int i = 0; i < TileVChunk; ++i) {
                if (i < tile_size) {
                    float s = S_reg[i] + k_val * shared_vec[i];
                    S_reg[i] = isfinite(s) ? s : 0.f;
                }
            }

            // Step 4: output
            #pragma unroll
            for (int i = 0; i < TileVChunk; ++i) {
                if (i < tile_size) {
                    float partial = q_val * S_reg[i];
                    float out_vi = blockReduceSum(partial, reduce_buf);
                    if (tid == 0) {
                        out_vi = isfinite(out_vi) ? out_vi : 0.f;
                        output[out_off + tile_begin + i] = __float2half(out_vi);
                    }
                }
            }

            // Write current tile back once per token.
            #pragma unroll
            for (int i = 0; i < TileVChunk; ++i) {
                if (i < tile_size) {
                    S_row_global[tile_begin + i] = S_reg[i];
                }
            }
        }
    }
}

// Template dispatch helper — launches the register-cached prefill kernel
// with the appropriate VChunk template parameter.
template<int VChunk>
static void launchPrefillRegKernel(half*        output,
                                   float*       recurrent_state,
                                   const half*  q,
                                   const half*  k,
                                   const half*  v,
                                   const half*  A_log,
                                   const half*  dt_bias,
                                   const half*  alpha_ptr,
                                   const half*  beta_raw_ptr,
                                   const int*   cu_seqlens,
                                   int          num_heads,
                                   int          q_num_heads,
                                   int          kv_ratio,
                                   int          q_stride,
                                   int          k_stride,
                                   int          v_stride,
                                   int          head_k_dim,
                                   int          head_v_dim,
                                   int          split_v,
                                   int          v_chunk,
                                   int          grid,
                                   int          block,
                                   int          smem,
                                   cudaStream_t stream,
                                   int          recurrent_state_batch_stride,
                                   int          alpha_stride,
                                   int          beta_stride,
                                   int          alpha_col_offset,
                                   int          beta_col_offset)
{
    chunkDeltaRulePrefillRegKernel<VChunk><<<grid, block, smem, stream>>>(
        output, recurrent_state, q, k, v,
        A_log, dt_bias, alpha_ptr, beta_raw_ptr,
        cu_seqlens,
        num_heads, q_num_heads, kv_ratio,
        q_stride, k_stride, v_stride,
        head_k_dim, head_v_dim, split_v, v_chunk,
        recurrent_state_batch_stride,
        alpha_stride, beta_stride, alpha_col_offset, beta_col_offset);
}

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
                                 int          beta_col_offset)
{
    if (q_num_heads <= 0) q_num_heads = num_heads;
    kv_ratio = kv_ratio > 0 ? kv_ratio : 1;
    if (q_stride == 0) q_stride = q_num_heads * head_k_dim;
    if (k_stride == 0) k_stride = q_num_heads * head_k_dim;
    if (v_stride == 0) v_stride = num_heads * head_v_dim;

    const int batch_heads = batch * num_heads;
    int split_v = chooseSplitV(batch_heads, head_v_dim);
    if (token_num >= 1024 && batch_heads < 64 && head_v_dim >= 64) {
        while (split_v < 16) {
            const int next_split = split_v << 1;
            const int next_chunk = (head_v_dim + next_split - 1) / next_split;
            if (next_chunk < 8) break;
            split_v = next_split;
        }
    }
    const int v_chunk = (head_v_dim + split_v - 1) / split_v;
    const int grid  = batch_heads * split_v;
    const int block = head_k_dim;
    const int shared_vec_size = v_chunk > 2 ? v_chunk : 2;
    const int smem  = (head_k_dim + shared_vec_size) * sizeof(float);
    if (recurrent_state_batch_stride == 0) {
        recurrent_state_batch_stride = num_heads * head_k_dim * head_v_dim;
    }

    // Dispatch to the smallest VChunk template that covers the runtime v_chunk.
    // This keeps register usage minimal while enabling full unrolling.
    #define LAUNCH_PREFILL_REG(VC)                                              \
        launchPrefillRegKernel<VC>(output, recurrent_state, q, k, v,            \
            A_log, dt_bias, alpha_ptr, beta_raw_ptr, cu_seqlens,                \
            num_heads, q_num_heads, kv_ratio, q_stride, k_stride, v_stride,     \
            head_k_dim, head_v_dim, split_v, v_chunk, grid, block, smem,        \
            stream, recurrent_state_batch_stride,                               \
            alpha_stride, beta_stride, alpha_col_offset, beta_col_offset)

    if (v_chunk <= 4) {
        LAUNCH_PREFILL_REG(4);
    } else if (v_chunk <= 8) {
        LAUNCH_PREFILL_REG(8);
    } else if (v_chunk <= 16) {
        LAUNCH_PREFILL_REG(16);
    } else {
        // Heuristic: only use tiled-register path for large chunks in long
        // prefill, where reduced global-memory traffic outweighs extra regs.
        const bool use_tiled_reg = (token_num >= 512 && v_chunk >= 32);
        if (use_tiled_reg) {
            chunkDeltaRulePrefillTiledRegKernel<16><<<grid, block, smem, stream>>>(
                output, recurrent_state, q, k, v,
                A_log, dt_bias, alpha_ptr, beta_raw_ptr,
                cu_seqlens,
                num_heads, q_num_heads, kv_ratio,
                q_stride, k_stride, v_stride,
                head_k_dim, head_v_dim, split_v, v_chunk,
                recurrent_state_batch_stride,
                alpha_stride, beta_stride, alpha_col_offset, beta_col_offset);
        } else {
            chunkDeltaRulePrefillKernel<<<grid, block, smem, stream>>>(
                output, recurrent_state, q, k, v,
                A_log, dt_bias, alpha_ptr, beta_raw_ptr,
                cu_seqlens,
                num_heads, q_num_heads, kv_ratio,
                q_stride, k_stride, v_stride,
                head_k_dim, head_v_dim, split_v, v_chunk,
                recurrent_state_batch_stride,
                alpha_stride, beta_stride, alpha_col_offset, beta_col_offset);
        }
    }
    #undef LAUNCH_PREFILL_REG
}

}  // namespace turbomind
