// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// Gated RMSNorm: y = silu(z) * rms_norm(x, weight)
//
// Fuses SiLU gating and RMS normalization into a single kernel.
// Each row (token, head) is processed by one thread block.
//
// Parameters:
//   output     [token_num, num_heads, head_v_dim]  fp16
//   x          [token_num, num_heads, head_v_dim]  fp16, input to normalize
//   z          [token_num, num_heads, head_v_dim]  fp16, gating signal
//   weight     [head_v_dim]                        fp16, RMS norm weight (per-dim, shared across heads)
//   token_num  total tokens
//   num_heads  number of heads
//   head_v_dim dimension per head
//   eps        RMS norm epsilon (default 1e-6)
//   stream     CUDA stream
//   x_stride   stride between adjacent tokens in x (in half elements). 0 = compact (num_heads * head_v_dim).
//   z_stride   stride between adjacent tokens in z (in half elements). 0 = compact (num_heads * head_v_dim).
void invokeGatedRMSNorm(half*        output,
                        const half*  x,
                        const half*  z,
                        const half*  weight,
                        int          token_num,
                        int          num_heads,
                        int          head_v_dim,
                        float        eps,
                        cudaStream_t stream,
                        int          x_stride = 0,
                        int          z_stride = 0);

}  // namespace turbomind
