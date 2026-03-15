// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// Extract columns from a row-major matrix.
// src: [rows, src_stride], dst: [rows, width]
// Copies src[r, col_offset .. col_offset+width-1] -> dst[r, 0..width-1]
void invokeExtractColumns(half*        dst,
                           const half*  src,
                           int          rows,
                           int          width,
                           int          col_offset,
                           int          src_stride,
                           cudaStream_t stream);

// GQA repeat_interleave: expand num_k_heads to num_v_heads = num_k_heads * kv_ratio
// src: [token_num, num_k_heads * head_dim]
// dst: [token_num, num_v_heads * head_dim]
void invokeRepeatInterleave(half*        dst,
                             const half*  src,
                             int          token_num,
                             int          num_k_heads,
                             int          head_dim,
                             int          kv_ratio,
                             cudaStream_t stream);

}  // namespace turbomind
