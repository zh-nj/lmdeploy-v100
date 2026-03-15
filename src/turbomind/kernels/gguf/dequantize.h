// Copyright (c) OpenMMLab. All rights reserved.
// Unified GGUF dequantization dispatch interface.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gguf_types.h"

namespace turbomind {
namespace gguf {

// Dequantize GGUF quantized data to FP16.
// src: device pointer to raw quantized blocks
// dst: device pointer to output FP16 buffer (n elements)
// n:   total number of elements to dequantize
void dequantize_gguf(GGMLType type,
                     const void* src,
                     half*       dst,
                     int64_t     n,
                     cudaStream_t stream);

}  // namespace gguf
}  // namespace turbomind
