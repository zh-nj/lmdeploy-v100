// Copyright (c) OpenMMLab. All rights reserved.
// GGUF/GGML quantization type definitions for TurboMind.
// Reference: llama.cpp ggml-common.h

#pragma once

#include <cstddef>
#include <cstdint>

namespace turbomind {
namespace gguf {

// QK_K = super-block size for K-quants
constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;

// GGML quantization type enum (matches ggml.h ggml_type)
enum GGMLType : int {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_MXFP4 = 39,
    GGML_TYPE_COUNT,
};

// ---- Block structures ----
// Packed structs matching llama.cpp ggml-common.h layout.

// Q8_0: block_size=32, 34 bytes
// { fp16 d; int8 qs[32]; }
#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t d;       // fp16 scale (stored as raw bits)
    int8_t   qs[32];  // quantized values
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 size mismatch");

// Q4_K: super-block=256, 144 bytes
// { half2 dm; uint8 scales[12]; uint8 qs[128]; }
#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;                   // fp16 super-block scale
    uint16_t dmin;                // fp16 super-block min
    uint8_t  scales[K_SCALE_SIZE]; // 6-bit scale/min pairs
    uint8_t  qs[QK_K / 2];        // 4-bit quants
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4K) == 144, "BlockQ4K size mismatch");

// Q5_K: super-block=256, 176 bytes
// { half2 dm; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }
#pragma pack(push, 1)
struct BlockQ5K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];        // high bits
    uint8_t  qs[QK_K / 2];        // low 4 bits
};
#pragma pack(pop)
static_assert(sizeof(BlockQ5K) == 176, "BlockQ5K size mismatch");

// Q6_K: super-block=256, 210 bytes
// { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d; }
#pragma pack(push, 1)
struct BlockQ6K {
    uint8_t ql[QK_K / 2];         // lower 4 bits
    uint8_t qh[QK_K / 4];         // upper 2 bits
    int8_t  scales[QK_K / 16];    // sub-block scales
    uint16_t d;                    // fp16 super-block scale
};
#pragma pack(pop)
static_assert(sizeof(BlockQ6K) == 210, "BlockQ6K size mismatch");

// MXFP4: block_size=32, 17 bytes
// { uint8 e; uint8 qs[16]; }
constexpr int QK_MXFP4 = 32;

#pragma pack(push, 1)
struct BlockMXFP4 {
    uint8_t e;                     // E8M0 shared exponent
    uint8_t qs[QK_MXFP4 / 2];     // 4-bit e2m1 values (32 values packed)
};
#pragma pack(pop)
static_assert(sizeof(BlockMXFP4) == 17, "BlockMXFP4 size mismatch");

// ---- Helper functions ----

// Returns the byte size of one quantization block for the given type.
inline size_t ggml_type_block_size(GGMLType type)
{
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q8_0: return sizeof(BlockQ8_0);
        case GGML_TYPE_Q4_K: return sizeof(BlockQ4K);
        case GGML_TYPE_Q5_K: return sizeof(BlockQ5K);
        case GGML_TYPE_Q6_K: return sizeof(BlockQ6K);
        case GGML_TYPE_MXFP4: return sizeof(BlockMXFP4);
        default: return 0;
    }
}

// Returns the number of elements per quantization block.
inline int ggml_type_elements_per_block(GGMLType type)
{
    switch (type) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q4_K: return QK_K;
        case GGML_TYPE_Q5_K: return QK_K;
        case GGML_TYPE_Q6_K: return QK_K;
        case GGML_TYPE_MXFP4: return QK_MXFP4;
        default: return 0;
    }
}

}  // namespace gguf
}  // namespace turbomind
