// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "src/turbomind/kernels/gated_delta_net/gated_rmsnorm.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// Gated RMSNorm Kernel: y = silu(z) * rms_norm(x, weight)
//
// Grid:  (token_num * num_heads,)
// Block: (head_v_dim,)  — one thread per dimension element
//
// Each block processes one (token, head) row.
// Step 1: Compute sum(x^2) via block reduction → variance
// Step 2: For each dim: y[d] = silu(z[d]) * (x[d] * rsqrt(var + eps)) * weight[d]
// ---------------------------------------------------------------------------

__device__ float blockReduceSumGRN(float val, float* smem, int block_size)
{
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void gatedRMSNormKernel(half*       output,      // [T*H, D]
                                   const half* x,           // [T*H, D] or strided
                                   const half* z,           // [T*H, D] or strided
                                   const half* weight,      // [D]
                                   int         row_count,   // T * H
                                   int         num_heads,
                                   int         head_v_dim,  // D
                                   float       eps,
                                   int         x_stride,
                                   int         z_stride)
{
    const int row = blockIdx.x;
    const int d   = threadIdx.x;
    if (row >= row_count || d >= head_v_dim) return;

    extern __shared__ float smem[];

    // Compute token and head indices
    const int token = row / num_heads;
    const int head  = row % num_heads;

    // x offset: if x_stride > 0, x is strided per-token
    const int x_offset = x_stride > 0
        ? token * x_stride + head * head_v_dim + d
        : row * head_v_dim + d;

    // z offset: if z_stride > 0, z is strided per-token
    const int z_offset = z_stride > 0
        ? token * z_stride + head * head_v_dim + d
        : row * head_v_dim + d;

    float x_val = __half2float(x[x_offset]);
    float z_val = __half2float(z[z_offset]);
    float w_val = __half2float(weight[d]);

    // Step 1: sum of squares for RMS norm
    float sq_sum = blockReduceSumGRN(x_val * x_val, smem, head_v_dim);

    // RMS norm: x * rsqrt(mean(x^2) + eps)
    float rms_inv = rsqrtf(sq_sum / (float)head_v_dim + eps);
    float x_norm = x_val * rms_inv * w_val;

    // SiLU gating: silu(z) = z * sigmoid(z)
    float silu_z = z_val / (1.0f + expf(-z_val));

    // Output: silu(z) * rms_norm(x, weight)
    output[row * head_v_dim + d] = __float2half(silu_z * x_norm);
}

void invokeGatedRMSNorm(half*        output,
                        const half*  x,
                        const half*  z,
                        const half*  weight,
                        int          token_num,
                        int          num_heads,
                        int          head_v_dim,
                        float        eps,
                        cudaStream_t stream,
                        int          x_stride,
                        int          z_stride)
{
    const int row_count = token_num * num_heads;
    const int smem_size = head_v_dim * sizeof(float);

    gatedRMSNormKernel<<<row_count, head_v_dim, smem_size, stream>>>(
        output, x, z, weight, row_count, num_heads, head_v_dim, eps, x_stride, z_stride);
}

}  // namespace turbomind
