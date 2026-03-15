// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "block_iterator.h"
#include "cta_map.h"
#include "impl_81616.h"
#include "impl_884.h"
#include "impl_884_decode.h"
#include "impl_simt.h"
#include "mainloop_sm70.h"
#include "mainloop_sm80.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/mainloop.h"

#include <type_traits>

namespace turbomind::attention {

template<class Arch, class T, class Tkv, int Qh, int HeadDim, class SFINAE = void>
struct DecodingConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<class Arch, class T, class Tkv, int Qh, int HeadDim>
using Decoding = typename DecodingConfig<Arch, T, Tkv, Qh, HeadDim>::Kernel;

//////////////////////////////////////////////////////////////
template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh, HeadDim, std::enable_if_t<!(Qh > 2)>> {
    using Attention = Impl<MMA_SIMT, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh_, HeadDim, std::enable_if_t<(Qh_ > 2 && HeadDim != 256)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter         = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint8_t, Qh_, HeadDim, std::enable_if_t<(HeadDim != 192 && HeadDim != 256)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 5>;
    using CacheIter         = GetBlockIterFactory<T, uint8_t, 64, HeadDim>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint4_t, Qh_, HeadDim, std::enable_if_t<(HeadDim != 256)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, uint4_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 5>;
    using CacheIter         = GetBlockIterFactory<T, uint4_t, 64, HeadDim>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap>;
};

//////////////////////////////////////////////////////////////

template<class T, class Tkv, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm75, T, Tkv, Qh_, HeadDim, std::enable_if_t<(HeadDim != 256)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, Tkv, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, Tkv, 64, HeadDim>;
    using Kernel = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

//////////////////////////////////////////////////////////////

// Toggle: set to 1 to use MMA_884_DEC for sm70 decode (Tkv=T, HeadDim!=256),
//         set to 0 to fall back to MMA_SIMT
#ifndef SM70_DECODE_USE_MMA_884
#define SM70_DECODE_USE_MMA_884 0
#endif

// sm70 decode with fp16 KV cache (Tkv=T): MMA_884_DEC when enabled, else MMA_SIMT
template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, T, Qh, HeadDim, std::enable_if_t<(HeadDim != 256)>> {
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
#if SM70_DECODE_USE_MMA_884
    using Attention = Impl<MMA_884_DEC, T, T, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
#else
    using Attention = Impl<MMA_SIMT, T, T, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
#endif
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

// sm70 decode with quantized KV cache: always MMA_SIMT
template<class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, Tkv, Qh, HeadDim, std::enable_if_t<(!std::is_same_v<T, Tkv> && HeadDim != 256)>> {
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
    using Attention         = Impl<MMA_SIMT, T, Tkv, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, Tkv, 64, HeadDim>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

// sm70 decode HeadDim=256 with fp16 KV cache (Tkv=T): MMA_884_DEC when enabled, else MMA_SIMT
template<class T, int Qh>
struct DecodingConfig<arch::Sm70, T, T, Qh, 256> {
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
#if SM70_DECODE_USE_MMA_884
    using Attention = Impl<MMA_884_DEC, T, T, kH, 1, 64, kH, 1, 16, 256, 2>;
#else
    using Attention = Impl<MMA_SIMT, T, T, kH, 1, 64, kH, 1, 16, 256, 2>;
#endif
    using CacheIter = GetBlockIterFactory<T, T, 64, 256>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

// sm70 decode HeadDim=256 with quantized KV cache: always MMA_SIMT
template<class T, class Tkv, int Qh>
struct DecodingConfig<arch::Sm70, T, Tkv, Qh, 256, std::enable_if_t<!std::is_same_v<T, Tkv>>> {
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
    using Attention         = Impl<MMA_SIMT, T, Tkv, kH, 1, 64, kH, 1, 16, 256, 2>;
    using CacheIter         = GetBlockIterFactory<T, Tkv, 64, 256>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

// HeadDim=256 specializations: MMA_81616 doesn't support HeadDim=256 (RakedThreadMap division by zero),
// so we use MMA_SIMT for all HeadDim=256 decoding configs.

template<class T, int Qh_>
struct DecodingConfig<arch::Sm80, T, T, Qh_, 256, std::enable_if_t<(Qh_ > 2)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_SIMT, T, T, Qh, 1, 64, Qh, 1, 16, 256, 3>;
    using CacheIter         = GetBlockIterFactory<T, T, 64, 256>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh_>
struct DecodingConfig<arch::Sm80, T, uint8_t, Qh_, 256> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_SIMT, T, uint8_t, Qh, 1, 64, Qh, 1, 16, 256, 3>;
    using CacheIter         = GetBlockIterFactory<T, uint8_t, 64, 256>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh_>
struct DecodingConfig<arch::Sm80, T, uint4_t, Qh_, 256> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_SIMT, T, uint4_t, Qh, 1, 64, Qh, 1, 16, 256, 3>;
    using CacheIter         = GetBlockIterFactory<T, uint4_t, 64, 256>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, class Tkv, int Qh_>
struct DecodingConfig<arch::Sm75, T, Tkv, Qh_, 256> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_SIMT, T, Tkv, Qh, 1, 64, Qh, 1, 16, 256, 2>;
    using CacheIter         = GetBlockIterFactory<T, Tkv, 64, 256>;
    using Kernel = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

//////////////////////////////////////////////////////////////

template<class T>
struct DecodingConfig<arch::Sm80, T, uint8_t, 1, 192> {
    static constexpr int Qh      = 1;
    static constexpr int HeadDim = 192;

    using Attention = Impl<MMA_SIMT, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, uint8_t, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

}  // namespace turbomind::attention
