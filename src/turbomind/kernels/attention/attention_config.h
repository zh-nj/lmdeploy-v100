// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "block_iterator.h"
#include "cta_map.h"
#include "impl_16816.h"
#include "impl_1688.h"
#include "impl_884.h"
#include "impl_simt.h"
#include "linear_iterator.h"
#include "mainloop_sm70.h"
#include "mainloop_sm80.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/impl.h"

namespace turbomind::attention {

enum class CacheType
{
    kLinear,
    kBlock,
};

template<class Arch, class T, int HeadDim, CacheType cache_type>
struct AttentionConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<CacheType type, class T, int CTA_S, int HeadDim>
using GetCacheIterFactory = std::conditional_t<type == CacheType::kLinear,
                                               LinearIteratorFactory<T, CTA_S, HeadDim>,
                                               GetBlockIterFactory<T, T, CTA_S, HeadDim>>;

struct Base_64x64_16x64 {
    static constexpr int CTA_Q  = 64;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 64;
};

template<class T, int HeadDim>
struct AttentionConfig<arch::Sm80, T, HeadDim, CacheType::kLinear>: Base_64x64_16x64 {
    using Attention = Impl<MMA_16816, T, T, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, HeadDim, 2>;
    using CacheIter = LinearIteratorFactory<T, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<2>, Attention>, CacheIter, AttentionCtaMap>;
};

template<class T>
struct AttentionConfig<arch::Sm80, T, 64, CacheType::kLinear> {
    static constexpr int CTA_Q  = 64;
    static constexpr int CTA_S  = 128;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 128;

    using Attention = Impl<MMA_16816, T, T, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, 64, 2>;
    using CacheIter = LinearIteratorFactory<T, CTA_S, 64>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<2>, Attention>, CacheIter, AttentionCtaMap>;
};

template<class T, int HeadDim>
struct AttentionConfig<arch::Sm80, T, HeadDim, CacheType::kBlock>: Base_64x64_16x64 {
    using Attention = Impl<MMA_16816, T, T, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, T, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, AttentionCtaMap>;
};

template<class T, int HeadDim, CacheType Ctype>
struct AttentionConfig<arch::Sm75, T, HeadDim, Ctype>: Base_64x64_16x64 {
    using Attention = Impl<MMA_1688, T, T, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, HeadDim, 2>;
    using CacheIter = GetCacheIterFactory<Ctype, T, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, AttentionCtaMap>;
};

// Toggle: set to 1 to use MMA_884 for sm70 prefill,
//         set to 0 to fall back to MMA_SIMT
#ifndef SM70_PREFILL_USE_MMA_884
#define SM70_PREFILL_USE_MMA_884 0
#endif

template<class T, int HeadDim, CacheType Ctype>
struct AttentionConfig<arch::Sm70, T, HeadDim, Ctype>: Base_64x64_16x64 {
    // Non-HD256: always MMA_884 (original git behavior).
    // The SM70_PREFILL_USE_MMA_884 toggle only affects the HD=256 specialization below.
    using Attention = Impl<MMA_884, T, T, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, HeadDim, 2>;
    using CacheIter = GetCacheIterFactory<Ctype, T, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, AttentionCtaMap>;
};

// HeadDim=256: Two-pass 128-dim MMA_884 approach.
// Split HeadDim=256 into two 128-dim halves, each processed with MMA_884 tensor core instructions.
// QK: accumulate both halves into the same FragS; PV: output to corresponding halves of FragO.
// MMA_SIMT fallback matches original git config: CTA_Q=16, CTA_S=64, WARP_Q=16, WARP_S=16.
template<class T, CacheType Ctype>
struct AttentionConfig<arch::Sm70, T, 256, Ctype>: Base_64x64_16x64 {
#if SM70_PREFILL_USE_MMA_884
    using Attention = Impl<MMA_884, T, T, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, 256, 2>;
#else
    using Attention = Impl<MMA_SIMT, T, T, 1, 16, CTA_S, 1, 16, 16, 256, 2>;
#endif
    using CacheIter = GetCacheIterFactory<Ctype, T, CTA_S, 256>;
    using Kernel    = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, AttentionCtaMap>;
};

}  // namespace turbomind::attention
