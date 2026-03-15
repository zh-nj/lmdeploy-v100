// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "src/turbomind/kernels/gated_delta_net/gating.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// Gating Compute Kernel
//
// Grid:  (ceil(token_num * num_heads / block_size),)
// Block: (block_size,)
//
// Each thread computes one (token, head) element of g and beta.
//
// g[t,h]    = exp( -exp(A_log[h]) * softplus(alpha[t,h] + dt_bias[h]) )
// beta[t,h] = sigmoid(b[t,h])
//
// softplus(x) = log(1 + exp(x))
// For numerical stability: softplus(x) = x + log(1 + exp(-x))  when x > 0
//                          softplus(x) = log(1 + exp(x))        when x <= 0
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

__global__ void gatingComputeKernel(float*       g,
                                    float*       beta,
                                    const half*  A_log,
                                    const half*  dt_bias,
                                    const half*  alpha,
                                    const half*  b,
                                    int          token_num,
                                    int          num_heads,
                                    int          alpha_stride,
                                    int          b_stride,
                                    int          alpha_col_offset,
                                    int          b_col_offset)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = token_num * num_heads;
    if (idx >= total) return;

    const int token = idx / num_heads;
    const int head = idx % num_heads;

    // Load per-head constants
    const float a_log_val  = __half2float(A_log[head]);
    const float dt_bias_val = __half2float(dt_bias[head]);

    // Load per-token per-head values
    const float alpha_val = __half2float(alpha[token * alpha_stride + alpha_col_offset + head]);
    const float b_val     = __half2float(b[token * b_stride + b_col_offset + head]);

    // softplus(x) = log(1 + exp(x)), numerically stable
    const float sp_input = alpha_val + dt_bias_val;
    float sp;
    if (sp_input > 20.0f) {
        sp = sp_input;  // softplus(x) ≈ x for large x
    } else if (sp_input < -20.0f) {
        sp = expf(sp_input);  // softplus(x) ≈ exp(x) for very negative x
    } else {
        sp = log1pf(expf(sp_input));
    }

    // g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    // Note: -exp(A_log) is negative, so the exponent is negative, g in (0, 1)
    const float log_decay = -expf(a_log_val) * sp;
    g[idx] = expf(log_decay);

    // beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta[idx] = 1.0f / (1.0f + expf(-b_val));
}

void invokeGatingCompute(float*       g,
                         float*       beta,
                         const half*  A_log,
                         const half*  dt_bias,
                         const half*  alpha,
                         const half*  b,
                         int          token_num,
                         int          num_heads,
                         cudaStream_t stream,
                         int          alpha_stride,
                         int          b_stride,
                         int          alpha_col_offset,
                         int          b_col_offset)
{
    if (alpha_stride == 0) {
        alpha_stride = num_heads;
    }
    if (b_stride == 0) {
        b_stride = num_heads;
    }
    const int total = token_num * num_heads;
    const int grid  = (total + kBlockSize - 1) / kBlockSize;

    gatingComputeKernel<<<grid, kBlockSize, 0, stream>>>(
        g, beta, A_log, dt_bias, alpha, b, token_num, num_heads, alpha_stride, b_stride, alpha_col_offset, b_col_offset);
}

}  // namespace turbomind
