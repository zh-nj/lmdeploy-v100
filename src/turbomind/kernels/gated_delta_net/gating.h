// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

void invokeGatingCompute(float*       g,
                         float*       beta,
                         const half*  A_log,
                         const half*  dt_bias,
                         const half*  alpha,
                         const half*  b,
                         int          token_num,
                         int          num_heads,
                         cudaStream_t stream,
                         int          alpha_stride = 0,
                         int          b_stride = 0,
                         int          alpha_col_offset = 0,
                         int          b_col_offset = 0);

}  // namespace turbomind
