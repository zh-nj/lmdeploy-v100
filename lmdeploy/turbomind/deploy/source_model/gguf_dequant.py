# Copyright (c) OpenMMLab. All rights reserved.
"""GGUF dequantization functions (Python/NumPy).

Implements CPU-side dequantization for GGUF quantization types:
Q8_0, Q6_K, Q5_K, Q4_K, and passthrough for F32/F16/BF16.

Reference: llama.cpp ggml-common.h block struct definitions and
           ggml-cuda/convert.cu dequantization kernels.

Block sizes (QK_K=256):
- Q8_0:  block_size=32,  block_bytes=34  (2 d + 32 qs)
- Q6_K:  super-block=256, block_bytes=210 (128 ql + 64 qh + 16 scales + 2 d)
- Q5_K:  super-block=256, block_bytes=176 (4 dm + 12 scales + 32 qh + 128 qs)
- Q4_K:  super-block=256, block_bytes=144 (4 dm + 12 scales + 128 qs)
"""

import numpy as np
import torch

from gguf import GGMLQuantizationType

QK_K = 256
K_SCALE_SIZE = 12


def dequantize(data, ggml_type, shape):
    """Unified dequantization interface.

    Args:
        data: Raw quantized bytes as uint8 numpy array (or bytes).
        ggml_type: GGML quantization type (int or GGMLQuantizationType).
        shape: Logical tensor shape.

    Returns:
        Dequantized torch.Tensor in float32.
    """
    ggml_type = int(ggml_type)
    n = 1
    for s in shape:
        n *= s

    _DISPATCH = {
        int(GGMLQuantizationType.F32): _dequant_f32,
        int(GGMLQuantizationType.F16): _dequant_f16,
        int(GGMLQuantizationType.BF16): _dequant_bf16,
        int(GGMLQuantizationType.Q8_0): _dequant_q8_0,
        int(GGMLQuantizationType.Q6_K): _dequant_q6_k,
        int(GGMLQuantizationType.Q5_K): _dequant_q5_k,
        int(GGMLQuantizationType.Q4_K): _dequant_q4_k,
        39: _dequant_mxfp4,  # MXFP4
    }
    fn = _DISPATCH.get(ggml_type)
    if fn is None:
        raise ValueError(f'Unsupported ggml_type: {ggml_type}')

    if isinstance(data, (bytes, bytearray)):
        data = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, np.ndarray) and data.dtype != np.uint8:
        # gguf lib may return typed arrays (e.g. float32 for F32)
        data = np.frombuffer(data.tobytes(), dtype=np.uint8)

    result = fn(data, n)
    return torch.from_numpy(result.copy()).reshape(shape)


# ---- Passthrough types ----

def _dequant_f32(data, n):
    return np.frombuffer(data.tobytes(), dtype=np.float32)[:n].copy()


def _dequant_f16(data, n):
    return np.frombuffer(data.tobytes(), dtype=np.float16)[:n].astype(
        np.float32)


def _dequant_bf16(data, n):
    raw = np.frombuffer(data.tobytes(), dtype=np.uint16)[:n]
    f32_bits = raw.astype(np.uint32) << 16
    return np.frombuffer(f32_bits.tobytes(), dtype=np.float32).copy()


# ---- Q8_0 ----
# struct { fp16 d; int8 qs[32]; } = 34 bytes, block_size=32

def _dequant_q8_0(data, n):
    block_size = 32
    block_bytes = 34
    nb = n // block_size
    raw = data[:nb * block_bytes]

    # Reshape to (nb, 34): [d(2), qs(32)]
    blocks = np.frombuffer(raw.tobytes(), dtype=np.uint8).reshape(nb, block_bytes)
    d = np.frombuffer(blocks[:, :2].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)
    qs = np.frombuffer(blocks[:, 2:].tobytes(), dtype=np.int8).reshape(nb, 32).astype(np.float32)
    out = (qs * d[:, None]).reshape(-1)
    return out


# ---- Q6_K ----
# struct block_q6_K {
#     uint8 ql[128];    // lower 4 bits
#     uint8 qh[64];     // upper 2 bits
#     int8  scales[16]; // sub-block scales
#     fp16  d;          // super-block scale
# } = 210 bytes, super-block=256
#
# llama.cpp kernel (64 threads):
#   ip = tid/32 (0,1), il = tid%32 (0..31), is = 8*ip + il/16
#   ql_ptr = ql + 64*ip + il, qh_val = qh[32*ip + il]
#   y[128*ip + il +  0] = d*sc[is+0] * ((ql[0]  & 0xF) | (((qh>>0)&3)<<4) - 32)
#   y[128*ip + il + 32] = d*sc[is+2] * ((ql[32] & 0xF) | (((qh>>2)&3)<<4) - 32)
#   y[128*ip + il + 64] = d*sc[is+4] * ((ql[0]  >> 4)  | (((qh>>4)&3)<<4) - 32)
#   y[128*ip + il + 96] = d*sc[is+6] * ((ql[32] >> 4)  | (((qh>>6)&3)<<4) - 32)

def _dequant_q6_k(data, n):
    block_bytes = 210
    nb = n // QK_K
    raw = np.frombuffer(data[:nb * block_bytes].tobytes(), dtype=np.uint8).reshape(nb, block_bytes)

    # Parse block fields: ql[128], qh[64], scales[16], d[2]
    ql = raw[:, :128]                    # (nb, 128) uint8
    qh = raw[:, 128:192]                 # (nb, 64) uint8
    scales = raw[:, 192:208].view(np.int8)  # (nb, 16) int8
    d = np.frombuffer(raw[:, 208:210].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)

    # Vectorized dequant following CUDA kernel logic:
    # 64 virtual threads: ip=tid//32 (0,1), il=tid%32 (0..31)
    # For ip=0: ql_idx=il, qh_idx=il, out_base=il
    # For ip=1: ql_idx=64+il, qh_idx=32+il, out_base=128+il
    out = np.empty((nb, 256), dtype=np.float32)

    for ip in range(2):
        ql_slice = ql[:, 64 * ip:64 * ip + 32]      # (nb, 32) - ql[ql_idx]
        ql_slice2 = ql[:, 64 * ip + 32:64 * ip + 64]  # (nb, 32) - ql[ql_idx+32]
        qh_slice = qh[:, 32 * ip:32 * ip + 32]       # (nb, 32) - qh[32*ip+il]

        # is_ = 8*ip + il//16, for il=0..31: is_ = 8*ip + [0]*16 + [1]*16
        is_base = 8 * ip
        is_arr = np.array([is_base + il // 16 for il in range(32)])  # (32,)

        # Extract 6-bit values
        q0 = (ql_slice & 0xF).astype(np.int32)
        q1 = (ql_slice2 & 0xF).astype(np.int32)
        q2 = (ql_slice >> 4).astype(np.int32)
        q3 = (ql_slice2 >> 4).astype(np.int32)

        h0 = ((qh_slice >> 0) & 3).astype(np.int32)
        h1 = ((qh_slice >> 2) & 3).astype(np.int32)
        h2 = ((qh_slice >> 4) & 3).astype(np.int32)
        h3 = ((qh_slice >> 6) & 3).astype(np.int32)

        # Scales: scales[is_+0], scales[is_+2], scales[is_+4], scales[is_+6]
        sc0 = scales[:, is_arr + 0].astype(np.float32)  # (nb, 32)
        sc1 = scales[:, is_arr + 2].astype(np.float32)
        sc2 = scales[:, is_arr + 4].astype(np.float32)
        sc3 = scales[:, is_arr + 6].astype(np.float32)

        base = 128 * ip
        out[:, base + 0:base + 32] = d[:, None] * sc0 * ((q0 | (h0 << 4)) - 32)
        out[:, base + 32:base + 64] = d[:, None] * sc1 * ((q1 | (h1 << 4)) - 32)
        out[:, base + 64:base + 96] = d[:, None] * sc2 * ((q2 | (h2 << 4)) - 32)
        out[:, base + 96:base + 128] = d[:, None] * sc3 * ((q3 | (h3 << 4)) - 32)

    return out.reshape(-1)


# ---- Scale/min decoding for Q4_K and Q5_K ----
# Matches llama.cpp get_scale_min_k4(j, q, &d, &m)
# scales[12] encodes 8 pairs of (scale, min), each 6 bits.

def _get_scale_min_k4(j, scales):
    """Decode 6-bit scale and min from the 12-byte scales array."""
    if j < 4:
        sc = int(scales[j]) & 63
        m = int(scales[j + 4]) & 63
    else:
        sc = (int(scales[j + 4]) & 0xF) | ((int(scales[j - 4]) >> 6) << 4)
        m = (int(scales[j + 4]) >> 4) | ((int(scales[j]) >> 6) << 4)
    return sc, m


# ---- Q4_K ----
# struct block_q4_K {
#     half2 dm;          // d and dmin (2+2 = 4 bytes)
#     uint8 scales[12];  // 6-bit scale/min pairs
#     uint8 qs[128];     // 4-bit quants (256 values, 2 per byte)
# } = 144 bytes, super-block=256
#
# llama.cpp kernel (32 threads):
#   tid=0..31, il=tid/8, ir=tid%8, is=2*il, n=4
#   y = yy + i*256 + 64*il + n*ir
#   q = qs + 32*il + n*ir
#   sc1, m1 = get_scale_min_k4(is+0, scales)
#   sc2, m2 = get_scale_min_k4(is+1, scales)
#   y[l+0]  = dall*sc1*(q[l] & 0xF) - dmin*m1   for l in 0..3
#   y[l+32] = dall*sc2*(q[l] >> 4)  - dmin*m2    for l in 0..3

def _dequant_q4_k(data, n):
    block_bytes = 144
    nb = n // QK_K
    buf = data[:nb * block_bytes].tobytes()

    out = np.empty(n, dtype=np.float32)
    for bi in range(nb):
        base = bi * block_bytes
        dm = np.frombuffer(buf[base:base + 4], dtype=np.float16)
        dall = float(dm[0])
        dmin = float(dm[1])
        scales = np.frombuffer(buf[base + 4:base + 16], dtype=np.uint8)
        qs = np.frombuffer(buf[base + 16:base + 144], dtype=np.uint8)

        dst = np.empty(256, dtype=np.float32)
        # 32 virtual threads
        for tid in range(32):
            il = tid // 8
            ir = tid % 8
            is_ = 2 * il
            nn = 4

            y_off = 64 * il + nn * ir
            q_off = 32 * il + nn * ir

            sc1, m1 = _get_scale_min_k4(is_ + 0, scales)
            d1 = dall * sc1
            mm1 = dmin * m1
            sc2, m2 = _get_scale_min_k4(is_ + 1, scales)
            d2 = dall * sc2
            mm2 = dmin * m2

            for l in range(nn):
                dst[y_off + l] = d1 * (int(qs[q_off + l]) & 0xF) - mm1
                dst[y_off + l + 32] = d2 * (int(qs[q_off + l]) >> 4) - mm2

        out[bi * QK_K:(bi + 1) * QK_K] = dst
    return out


# ---- Q5_K ----
# struct block_q5_K {
#     half2 dm;          // d and dmin (4 bytes)
#     uint8 scales[12];  // 6-bit scale/min pairs
#     uint8 qh[32];      // high bit for each of 256 values
#     uint8 qs[128];     // low 4 bits (256 values, 2 per byte)
# } = 176 bytes, super-block=256
#
# llama.cpp kernel (64 threads):
#   tid=0..63, il=tid/16, ir=tid%16, is=2*il
#   y = yy + i*256 + 64*il + 2*ir
#   ql = qs + 32*il + 2*ir
#   qh_ptr = qh + 2*ir
#   sc1, m1 = get_scale_min_k4(is+0, scales)
#   sc2, m2 = get_scale_min_k4(is+1, scales)
#   hm = 1 << (2*il)
#   y[0]  = d1*((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1
#   y[1]  = d1*((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1
#   hm <<= 1
#   y[32] = d2*((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2
#   y[33] = d2*((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2

def _dequant_q5_k(data, n):
    block_bytes = 176
    nb = n // QK_K
    buf = data[:nb * block_bytes].tobytes()

    out = np.empty(n, dtype=np.float32)
    for bi in range(nb):
        base = bi * block_bytes
        dm = np.frombuffer(buf[base:base + 4], dtype=np.float16)
        dall = float(dm[0])
        dmin = float(dm[1])
        scales = np.frombuffer(buf[base + 4:base + 16], dtype=np.uint8)
        qh = np.frombuffer(buf[base + 16:base + 48], dtype=np.uint8)
        qs = np.frombuffer(buf[base + 48:base + 176], dtype=np.uint8)

        dst = np.empty(256, dtype=np.float32)
        # 64 virtual threads
        for tid in range(64):
            il = tid // 16
            ir = tid % 16
            is_ = 2 * il

            y_off = 64 * il + 2 * ir
            ql_off = 32 * il + 2 * ir
            qh_off = 2 * ir

            sc1, m1 = _get_scale_min_k4(is_ + 0, scales)
            d1 = dall * sc1
            mm1 = dmin * m1
            sc2, m2 = _get_scale_min_k4(is_ + 1, scales)
            d2 = dall * sc2
            mm2 = dmin * m2

            hm = 1 << (2 * il)
            h0 = 16 if (int(qh[qh_off]) & hm) else 0
            h1 = 16 if (int(qh[qh_off + 1]) & hm) else 0
            dst[y_off + 0] = d1 * ((int(qs[ql_off]) & 0xF) + h0) - mm1
            dst[y_off + 1] = d1 * ((int(qs[ql_off + 1]) & 0xF) + h1) - mm1

            hm <<= 1
            h0 = 16 if (int(qh[qh_off]) & hm) else 0
            h1 = 16 if (int(qh[qh_off + 1]) & hm) else 0
            dst[y_off + 32] = d2 * ((int(qs[ql_off]) >> 4) + h0) - mm2
            dst[y_off + 33] = d2 * ((int(qs[ql_off + 1]) >> 4) + h1) - mm2

        out[bi * QK_K:(bi + 1) * QK_K] = dst
    return out


# ---- MXFP4 ----
# block_size=32, block_bytes=17 (1 byte E8M0 + 16 bytes qs)
# E8M0 scale: d = 2^(e - 128)
# e2m1 lookup (doubled values): [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]
# Low nibble → first 16 elements, high nibble → last 16 elements

_KVALUES_MXFP4 = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int8)


def _dequant_mxfp4(data, n):
    block_size = 32
    block_bytes = 17
    nb = n // block_size
    blocks = data[:nb * block_bytes].reshape(nb, block_bytes)
    e = blocks[:, 0].astype(np.int32)
    d = np.ldexp(1.0, e - 128).astype(np.float32)
    qs = blocks[:, 1:]  # (nb, 16) uint8
    lo = _KVALUES_MXFP4[qs & 0x0F].astype(np.float32)
    hi = _KVALUES_MXFP4[qs >> 4].astype(np.float32)
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = lo * d[:, None]
    out[:, 16:] = hi * d[:, None]
    return out.reshape(-1)
