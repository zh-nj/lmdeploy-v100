# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from functools import partial

import torch

from .parameter import get_params
from .source_model.base import BaseReader
from .target_model.base import BaseOutputModel


def permute_v2(x: torch.Tensor, size_per_head: int = 128):
    """
        Contract: x.size(-1) is output dims
    """

    assert x.size(-1) > 1

    output_dims = x.size(-1)
    head_num = output_dims // size_per_head

    return x.view(-1, head_num, 2, size_per_head // 2).transpose(2, 3).reshape(x.shape)


def permute_v2_partial(x: torch.Tensor, size_per_head: int, rotary_dim: int):
    """Permute only the first `rotary_dim` dims within each head for
    TurboMind's interleaved RoPE layout. Non-RoPE dims are left as-is.

    This is needed for partial rotary models (e.g. MiniMax-M2.1 where
    rotary_dim=64 but head_dim=128).
    """
    assert x.size(-1) > 1
    output_dims = x.size(-1)
    head_num = output_dims // size_per_head
    non_rope_dim = size_per_head - rotary_dim

    # reshape to (*, head_num, size_per_head)
    orig_shape = x.shape
    x = x.view(-1, head_num, size_per_head) if x.dim() >= 2 else x.view(head_num, size_per_head)

    # split each head into rope part and non-rope part
    rope_part = x[..., :rotary_dim]      # (..., head_num, rotary_dim)
    rest_part = x[..., rotary_dim:]      # (..., head_num, non_rope_dim)

    # permute only the rope part: interleave first/second halves
    rope_part = rope_part.view(*rope_part.shape[:-1], 2, rotary_dim // 2)
    rope_part = rope_part.transpose(-2, -1).reshape(*rope_part.shape[:-2], rotary_dim)

    # recombine
    x = torch.cat([rope_part, rest_part], dim=-1)
    return x.reshape(orig_shape)


def merge_qkv_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int):
    """
        Contract: x.size(-1) is output dims
    """

    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkv = torch.cat(tuple(map(reshape, (q, k, v))), dim=-1)

    qkv = qkv.view(-1, qkv.size(-1) * tp)
    if q.dim() == 1:
        qkv.squeeze_()

    return qkv


def _is_gguf_raw(x):
    """Check if tensor is raw GGUF quantized data (uint8 with ggml_type)."""
    return x is not None and hasattr(x, 'ggml_type')


def transpose(x):
    if _is_gguf_raw(x):
        return x  # GGUF raw data: skip transpose
    return x.t() if x is not None else x


def pad_out_dims(x: torch.Tensor, dims: int):
    if _is_gguf_raw(x):
        # GGUF raw data: pad with zero bytes (zero blocks dequantize to 0).
        # dims is the padded number of GGUF columns (out_dim) divided by
        # group_size (which equals elems_per_block for GGUF).
        # But the caller passes inter_size // gs where gs = group_size.
        # For GGUF, gs = group_size = 256 (block size), so dims = padded_cols / 256.
        # We need to figure out the actual padded column count.
        # Actually, the caller computes: w1 = pad_out_dims(w1, inter_size // gs1)
        # where gs1 = group_size if 'w1' in apply_gs else 1.
        # For GGUF experts, apply_gs is empty (identity pack_fn), so gs1=1.
        # Thus dims = inter_size (the padded inter_size).
        ggml_shape = x.ggml_shape  # (out_dim, in_dim)
        out_dim, in_dim = ggml_shape
        if out_dim >= dims:
            return x  # no padding needed
        # Need to pad out_dim from current to dims.
        # GGUF row-major: gguf_rows=in_dim, gguf_cols=out_dim.
        # Each row has out_dim elements packed into blocks.
        # Padding adds (dims - out_dim) zero elements per row.
        from gguf import GGMLQuantizationType
        _block_info = {
            int(GGMLQuantizationType.Q8_0): (32, 34),
            int(GGMLQuantizationType.Q6_K): (256, 210),
            int(GGMLQuantizationType.Q5_K): (256, 176),
            int(GGMLQuantizationType.Q4_K): (256, 144),
        }
        info = _block_info.get(x.ggml_type)
        if info is None:
            return x  # unknown type, skip padding
        elems_per_block, block_bytes = info
        pad_cols = dims - out_dim
        assert pad_cols % elems_per_block == 0, \
            f'GGUF pad: {pad_cols} not aligned to block {elems_per_block}'
        pad_blocks_per_row = pad_cols // elems_per_block
        pad_bytes_per_row = pad_blocks_per_row * block_bytes
        # Current layout: flat bytes = gguf_rows * bytes_per_row
        old_blocks_per_row = out_dim // elems_per_block
        old_bytes_per_row = old_blocks_per_row * block_bytes
        new_bytes_per_row = old_bytes_per_row + pad_bytes_per_row
        gguf_rows = in_dim
        # Reshape, pad each row, flatten
        old_2d = x.view(-1).reshape(gguf_rows, old_bytes_per_row)
        pad_2d = torch.zeros(gguf_rows, pad_bytes_per_row, dtype=torch.uint8)
        new_2d = torch.cat([old_2d, pad_2d], dim=1)
        result = new_2d.reshape(-1)
        result.ggml_type = x.ggml_type
        result.ggml_shape = (dims, in_dim)
        return result
    pad = dims - x.size(-1)
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, pad), 'constant', 0)


def pad_in_dims(x: torch.Tensor, dims: int):
    if _is_gguf_raw(x):
        # GGUF raw data: pad rows (in_dim) with zero bytes.
        ggml_shape = x.ggml_shape  # (out_dim, in_dim)
        out_dim, in_dim = ggml_shape
        if in_dim >= dims:
            return x  # no padding needed
        pad_rows = dims - in_dim
        from gguf import GGMLQuantizationType
        _block_info = {
            int(GGMLQuantizationType.Q8_0): (32, 34),
            int(GGMLQuantizationType.Q6_K): (256, 210),
            int(GGMLQuantizationType.Q5_K): (256, 176),
            int(GGMLQuantizationType.Q4_K): (256, 144),
        }
        info = _block_info.get(x.ggml_type)
        if info is None:
            return x
        elems_per_block, block_bytes = info
        blocks_per_row = out_dim // elems_per_block
        bytes_per_row = blocks_per_row * block_bytes
        pad_bytes = pad_rows * bytes_per_row
        result = torch.cat([x.view(-1),
                            torch.zeros(pad_bytes, dtype=torch.uint8)], dim=0)
        result.ggml_type = x.ggml_type
        result.ggml_shape = (out_dim, dims)
        return result
    if x.dim() == 1:  # 1-dim object does not have input dim (e.g. bias)
        return x
    pad = dims - x.size(0)
    assert x.dim() == 2
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad), 'constant', 0)


# split out dims -> copy A, split-out-dims B (qkv, w1, w3)
# split  in dims -> split-in-dims A,  copy B (  o, w2)
def get_lora_flags(kind: str):
    return ('lora_a' in kind, 'lora_b' in kind)


class Module(ABC):

    def __init__(self, model: BaseOutputModel):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractmethod
    def apply(self, idx: int, r: BaseReader):
        pass


class LayerNorm(Module):

    def apply(self, i: int, r: BaseReader):
        attn_norm = r.attn_norm(i)
        ffn_norm = r.ffn_norm(i)
        self.model.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.model.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')


class Ffn(Module):
    """
    requires:
        r.ffn(i, kind)
    """

    _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.mlp_tp_size
        # inter_sizes in config are padded and may be different from what's
        # in the weights
        self.inter_size = model.model_config.inter_size
        self.group_size = max(1, model.model_config.group_size)

    def _export(self, inter_size: int, fmt: str, idx: int, w123, kind: str, pack_fn, apply_gs=[], **kwargs):
        is_lora_a, is_lora_b = get_lora_flags(kind)
        w1, w2, w3 = map(transpose, w123)

        gs1 = self.group_size if 'w1' in apply_gs else 1
        w1 = pad_out_dims(w1, inter_size // gs1)

        gs3 = self.group_size if 'w3' in apply_gs else 1
        w3 = pad_out_dims(w3, inter_size // gs3)

        gs2 = self.group_size if 'w2' in apply_gs else 1
        w2 = pad_in_dims(w2, inter_size // gs2)

        w1, w2, w3 = map(pack_fn, (w1, w2, w3))
        self.model.save_split(w1, fmt.format(idx, 'w1', kind), split_dim=-1, split_num=self.tp, copy=is_lora_a)
        self.model.save_split(w3, fmt.format(idx, 'w3', kind), split_dim=-1, split_num=self.tp, copy=is_lora_a)
        self.model.save_split(w2, fmt.format(idx, 'w2', kind), split_dim=0, split_num=self.tp, copy=is_lora_b)

    def apply(self, i: int, r: BaseReader):
        if not self.inter_size[i]:
            return
        for e in get_params(r.ffn(i, None)):
            e(partial(self._export, self.inter_size[i], self._ffn), partial(r.ffn, i), i)


class MoeFfn(Ffn):
    """
    requires:
        r.moe_ffn_expert(e, i, kind)
        r.moe_ffn_gate(i)
        r.moe_ffn_shared_gate(i)
    """

    _moe_ffn_expert = 'layers.{0}.moe_ffn.experts.E.{1}.{2}'
    _moe_ffn_gate = 'layers.{0}.moe_ffn.gate.{1}'
    _moe_ffn_shared_gate = 'layers.{0}.moe_ffn.shared_gate.weight'

    def __init__(self, model: BaseOutputModel):
        super().__init__(model)
        self.expert_num = model.model_config.expert_num
        self.inter_size = model.model_config.expert_inter_size
        self.shared_gate = model.model_config.moe_shared_gate

    def apply(self, i: int, r: BaseReader):
        if self.expert_num[i] == 0:
            return
        for p in get_params(r.moe_ffn_expert(), 1):
            for e in range(self.expert_num[i]):
                fmt = self._moe_ffn_expert.replace('E', str(e))
                p(partial(self._export, self.inter_size, fmt), partial(r.moe_ffn_expert, e, i), i)

        # router
        gate = transpose(r.moe_ffn_gate(i, 'weight'))
        self.model.save_split(gate, self._moe_ffn_gate.format(i, 'weight'))
        bias = r.moe_ffn_gate(i, 'bias')
        if bias is not None:
            self.model.save_split(bias, self._moe_ffn_gate.format(i, 'bias'))

        if self.shared_gate:
            shared_gate = transpose(r.moe_ffn_shared_gate(i))
            self.model.save_split(shared_gate, self._moe_ffn_shared_gate.format(i))


class Attn(Module):
    """
    requires:
        r.attn(i, kind)
    """

    _attn = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.attn_tp_size
        self.head_dim = model.model_config.size_per_head
        self.attn_bias = model.model_config.attn_bias
        self.qk_norm = model.model_config.qk_norm
        self.attn_sink = model.model_config.attn_sink
        self.group_size = max(1, model.model_config.group_size)
        # rotary_dim for partial rotary models (e.g. MiniMax-M2.1)
        rope_param = getattr(model.attention_config, 'rope_param', None)
        self.rotary_dim = rope_param.dim if rope_param else self.head_dim
        # Per-layer rope_dim for models with mixed rotary dimensions
        # (e.g. Step3p5 where full_attention=64, sliding_attention=128)
        rope_dim_list = getattr(model.attention_config, 'rope_dim', None)
        self.rope_dim_per_layer = rope_dim_list if rope_dim_list else []
        self._current_layer_id = None
        # Per-layer head_num for models with mixed head counts
        # (e.g. Step3p5 where full_attention=64, sliding_attention=96)
        head_num_list = getattr(model.model_config, 'head_num_per_layer', None)
        self.head_num_per_layer = head_num_list if head_num_list else []

    def _permute(self, x):
        """Permute Q/K weights for TurboMind's interleaved RoPE layout.

        Uses partial permutation when rotary_dim < head_dim to avoid
        corrupting non-RoPE dimensions.
        """
        rotary_dim = self.rotary_dim
        if (self._current_layer_id is not None
                and self.rope_dim_per_layer
                and self._current_layer_id < len(self.rope_dim_per_layer)):
            rotary_dim = self.rope_dim_per_layer[self._current_layer_id]
        if rotary_dim < self.head_dim:
            return permute_v2_partial(x, self.head_dim, rotary_dim)
        return permute_v2(x, self.head_dim)

    def _reorder_and_merge(self, qkvo, gs: int):
        q, k, v, o = qkvo
        # reorder output dim for tm's rotary embedding layout
        if self.model.permute_qk:
            if gs == 1:
                q = self._permute(q)
                k = self._permute(k)
            else:
                assert gs % self.head_dim == 0
        qkv = merge_qkv_v2(q, k, v, self.tp)
        # zero bias for `wo` when `w_qkv` has bias but `wo` doesn't
        if o is None and q.dim() == 1:
            o = torch.zeros_like(q)
        return qkv, o

    def _repeat_kv(self, qkvo, gs: int, kind: str):
        """Replicate kv."""
        q, k, v, o = qkvo
        head_dim = self.model.model_config.size_per_head // gs
        kv_head_num = self.model.model_config.kv_head_num // self.model.repeat_kv
        hidden_dim = self.model.model_config.hidden_units

        def _repeat(x):
            n = self.model.repeat_kv

            x = x.reshape(-1, kv_head_num, head_dim)
            x = x.repeat(1, 1, n)
            x = x.reshape(-1, kv_head_num * n * head_dim)

            return x

        k, v = map(_repeat, (k, v))

        if kind == 'bias':
            if o is None:
                o = torch.zeros(hidden_dim, dtype=q.dtype, device=q.device)
            q, k, v, o = map(torch.squeeze, (q, k, v, o))

        return (q, k, v, o)

    def _export(self, idx: int, qkvo, kind: str, pack_fn, apply_gs=[], **kwargs):
        if all(x is None for x in qkvo):
            return
        is_lora_a, is_lora_b = get_lora_flags(kind)
        assert not (is_lora_a or is_lora_b)

        qkvo = tuple(map(transpose, qkvo))

        gs = self.group_size if ('w1' in apply_gs) else 1

        if self.model.repeat_kv:
            qkvo = self._repeat_kv(qkvo, gs, kind)

        qkv, o = self._reorder_and_merge(qkvo, gs)

        self.model.save_split(pack_fn(qkv),
                              self._attn.format(idx, 'w_qkv', kind),
                              split_dim=-1,
                              split_num=self.tp,
                              copy=is_lora_a)
        self.model.save_split(pack_fn(o),
                              self._attn.format(idx, 'wo', kind),
                              split_dim=0,
                              split_num=self.tp,
                              copy=is_lora_b)

    def apply(self, i: int, r: BaseReader):
        self._current_layer_id = i
        for e in get_params(r.attn(i, None), bias=self.attn_bias):
            e(self._export, partial(r.attn, i), i)
        # Export attention output gate weight (gated attention)
        if hasattr(r, 'attn_gate'):
            gate = r.attn_gate(i, 'weight')
            if gate is not None:
                gate = transpose(gate)
                self.model.save_split(gate,
                                      self._attn.format(i, 'w_gate', 'weight'),
                                      split_dim=-1,
                                      split_num=self.tp)
            gate_bias = r.attn_gate(i, 'bias')
            if gate_bias is not None:
                self.model.save_split(gate_bias,
                                      self._attn.format(i, 'w_gate', 'bias'),
                                      split_dim=-1,
                                      split_num=self.tp)
        if self.qk_norm:
            q, k = r.qk_norm(i)
            if self.model.permute_qk:
                q = self._permute(q)
                k = self._permute(k)
            head_num = self.model.model_config.head_num
            # Use per-layer head_num if available
            if (self.head_num_per_layer
                    and i < len(self.head_num_per_layer)):
                head_num = self.head_num_per_layer[i]
            kv_head_num = self.model.model_config.kv_head_num
            # C++ allocates per-rank buffers: (local_head_num * head_dim)
            # Shared QK norm (head_dim,): repeat to per-rank size, no TP split
            # Per-head QK norm (num_heads * head_dim,): split across TP
            is_shared_q = (q.numel() == self.head_dim)
            is_shared_k = (k.numel() == self.head_dim)
            if is_shared_q:
                q = q.repeat(head_num // self.tp)
            if is_shared_k:
                k = k.repeat(kv_head_num // self.tp)
            # Handle repeat_kv: replicate per-head K norm weights
            if self.model.repeat_kv and k.numel() > self.head_dim:
                k = k.view(-1, self.head_dim).repeat_interleave(
                    self.model.repeat_kv, dim=0).reshape(-1)
            q_name = self._attn.format(i, 'q_norm', '')[:-1]
            k_name = self._attn.format(i, 'k_norm', '')[:-1]
            # Shared norm: already sized for one rank, export directly per rank
            # Per-head norm: full size, split across TP ranks
            if is_shared_q:
                for rank in range(self.tp):
                    self.model.export_weight(
                        q, f'layers.{i}.attention.{rank}.q_norm')
            else:
                self.model.save_split(
                    q, q_name, split_dim=-1, split_num=self.tp)
            if is_shared_k:
                for rank in range(self.tp):
                    self.model.export_weight(
                        k, f'layers.{i}.attention.{rank}.k_norm')
            else:
                self.model.save_split(
                    k, k_name, split_dim=-1, split_num=self.tp)
        if self.attn_sink:
            sinks = r.attn_sinks(i)
            self.model.save_split(sinks, self._attn.format(i, 'sinks', '')[:-1], split_dim=-1, split_num=self.tp)


class LinearAttn(Module):
    """Export GatedDeltaNet linear attention weights with TP sharding.

    All linear_attn weights are fp16 (non-quantized), so no need for
    the get_params/Parameter quantization detection pattern.

    TP sharding strategy:
    - in_proj_all: column split (output dim) by tp, merged [QKVZ, BA]
    - out_proj: row split (input dim) by tp
    - conv1d_weight: split conv_dim (dim 0) by tp
    - A_log: split by tp
    - dt_bias: split by tp
    - norm_weight: NOT split (per-head, broadcast to all ranks)

    requires:
        r.linear_attn_in_proj_qkvz(i, kind)
        r.linear_attn_in_proj_ba(i, kind)
        r.linear_attn_out_proj(i, kind)
        r.linear_attn_conv1d(i, kind)
        r.linear_attn_a_log(i)
        r.linear_attn_dt_bias(i)
        r.linear_attn_norm(i)
    """

    _prefix = 'layers.{0}.linear_attn.{1}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.attn_tp_size
        cfg = model.model_config
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.kv_ratio = self.num_v_heads // self.num_k_heads

    def _reorder_qkvz(self, w):
        """Reorder in_proj_qkvz weight from grouped to flat layout.

        HF layout (output dim): [num_k_heads, (Qk + Kk + Vr*v + Zr*v)]
        Target layout: [Q_all, K_all, V_all, Z_all]

        After reorder, a simple TP column split gives each rank:
        [Q_local, K_local, V_local, Z_local] because Q has num_k_heads
        heads, K has num_k_heads heads, V has num_v_heads heads, etc.
        and they all divide evenly by TP.

        w shape: [hidden, qkvz_dim] (already transposed)
        """
        import torch
        nk = self.num_k_heads
        hk = self.head_k_dim
        hv = self.head_v_dim
        r = self.kv_ratio
        group_size = 2 * hk + 2 * r * hv
        # [hidden, num_k_heads, group_size]
        w = w.reshape(w.shape[0], nk, group_size)
        # split each group: Q(hk), K(hk), V(r*hv), Z(r*hv)
        q = w[:, :, :hk]                          # [hidden, nk, hk]
        k = w[:, :, hk:2*hk]                      # [hidden, nk, hk]
        v = w[:, :, 2*hk:2*hk+r*hv]               # [hidden, nk, r*hv]
        z = w[:, :, 2*hk+r*hv:]                    # [hidden, nk, r*hv]
        # flatten: [hidden, nk*hk], [hidden, nk*r*hv], ...
        q = q.reshape(w.shape[0], -1)
        k = k.reshape(w.shape[0], -1)
        v = v.reshape(w.shape[0], -1)
        z = z.reshape(w.shape[0], -1)
        return torch.cat([q, k, v, z], dim=-1)

    def _reorder_ba(self, w):
        """Reorder in_proj_ba weight from grouped to flat layout.

        HF layout (output dim): [num_k_heads, (b_ratio + a_ratio)]
        Target layout: [b_all, a_all], where b is beta_raw and a is alpha

        w shape: [hidden, ba_dim] (already transposed)
        """
        import torch
        nk = self.num_k_heads
        r = self.kv_ratio
        ba_group = 2 * r  # per k-head group: b(r) + a(r)
        # [hidden, num_k_heads, 2*r]
        w = w.reshape(w.shape[0], nk, ba_group)
        b = w[:, :, :r]       # [hidden, nk, r]
        a = w[:, :, r:]       # [hidden, nk, r]
        b = b.reshape(w.shape[0], -1)  # [hidden, num_v_heads]
        a = a.reshape(w.shape[0], -1)
        return torch.cat([b, a], dim=-1)

    def _tp_split_qkvz(self, w):
        """Split reordered [Q_all, K_all, V_all, Z_all] by TP correctly.

        Each component must be split independently by TP, then concatenated
        per rank, because Q/K have num_k_heads and V/Z have num_v_heads.
        """
        import torch
        nk = self.num_k_heads
        hk = self.head_k_dim
        nv = self.num_v_heads
        hv = self.head_v_dim
        key_dim = nk * hk
        value_dim = nv * hv
        # Split into components
        q = w[:, :key_dim]
        k = w[:, key_dim:2*key_dim]
        v = w[:, 2*key_dim:2*key_dim+value_dim]
        z = w[:, 2*key_dim+value_dim:]
        # Split each by TP
        tp = self.tp
        q_chunks = q.chunk(tp, dim=-1)
        k_chunks = k.chunk(tp, dim=-1)
        v_chunks = v.chunk(tp, dim=-1)
        z_chunks = z.chunk(tp, dim=-1)
        # Per-rank: [Q_local, K_local, V_local, Z_local]
        return [torch.cat([q_chunks[r], k_chunks[r], v_chunks[r], z_chunks[r]],
                          dim=-1) for r in range(tp)]

    def _tp_split_ba(self, w):
        """Split reordered [b_all, a_all] by TP correctly."""
        import torch
        nv = self.num_v_heads
        b = w[:, :nv]
        a = w[:, nv:]
        tp = self.tp
        b_chunks = b.chunk(tp, dim=-1)
        a_chunks = a.chunk(tp, dim=-1)
        return [torch.cat([b_chunks[r], a_chunks[r]], dim=-1)
                for r in range(tp)]

    def _reorder_conv1d(self, w):
        """Reorder conv1d weight to match reordered QKV layout.

        HF conv1d weight: [conv_dim, 1, kernel_size]
        HF conv_dim layout: [Q_h0..Q_hN, K_h0..K_hN, V_h0..V_hM]
        (already flat per-head after fix_query_key_value_ordering)

        Actually, the conv1d weight channels correspond to the flat
        [Q_all, K_all, V_all] layout because the HF code concatenates
        the deinterleaved Q, K, V before conv1d. So no reorder needed.
        """
        return w

    def apply(self, i: int, r: BaseReader):
        import torch
        # Merge in_proj_qkvz + in_proj_ba into single in_proj_all
        # Layout per rank: [Q_local, K_local, V_local, Z_local, b_local, a_local]
        w_qkvz = r.linear_attn_in_proj_qkvz(i, 'weight')
        w_ba = r.linear_attn_in_proj_ba(i, 'weight')
        if w_qkvz is not None and w_ba is not None:
            w_qkvz = transpose(w_qkvz)
            w_ba = transpose(w_ba)
            w_qkvz = self._reorder_qkvz(w_qkvz)
            w_ba = self._reorder_ba(w_ba)
            qkvz_chunks = self._tp_split_qkvz(w_qkvz)
            ba_chunks = self._tp_split_ba(w_ba)
            name = self._prefix.format(i, 'in_proj_all.weight')
            for rank in range(self.tp):
                merged = torch.cat([qkvz_chunks[rank], ba_chunks[rank]],
                                   dim=-1)
                import os.path as osp
                prefix, ext = osp.splitext(name)
                self.model.export_weight(merged, f'{prefix}.{rank}{ext}')

        # out_proj: [value_dim, hidden_size] -> row split
        w = r.linear_attn_out_proj(i, 'weight')
        if w is not None:
            w = transpose(w)
            self.model.save_split(w,
                                  self._prefix.format(i, 'out_proj.weight'),
                                  split_dim=0,
                                  split_num=self.tp)

        # conv1d_weight: [conv_dim, 1, kernel_size] -> custom TP split
        # conv_dim channels are [Q_all, K_all, V_all], must split each independently
        conv1d = r.linear_attn_conv1d(i, 'weight')
        if conv1d is not None:
            import torch
            nk = self.num_k_heads
            hk = self.head_k_dim
            nv = self.num_v_heads
            hv = self.head_v_dim
            key_dim = nk * hk
            value_dim = nv * hv
            # conv1d shape: [conv_dim, 1, kernel_size]
            q_cw = conv1d[:key_dim]
            k_cw = conv1d[key_dim:2*key_dim]
            v_cw = conv1d[2*key_dim:]
            tp = self.tp
            q_chunks = q_cw.chunk(tp, dim=0)
            k_chunks = k_cw.chunk(tp, dim=0)
            v_chunks = v_cw.chunk(tp, dim=0)
            name = self._prefix.format(i, 'conv1d.weight')
            for rank in range(tp):
                chunk = torch.cat([q_chunks[rank], k_chunks[rank], v_chunks[rank]], dim=0)
                import os.path as osp
                prefix, ext = osp.splitext(name)
                self.model.export_weight(chunk, f'{prefix}.{rank}{ext}')

        # A_log: [num_v_heads] -> split
        # Note: can't use save_split because osp.splitext treats '.A_log'
        # as a file extension, producing wrong parameter names.
        a_log = r.linear_attn_a_log(i)
        if a_log is not None:
            import torch as _torch
            splits = _torch.chunk(a_log, self.tp, dim=-1)
            for rank, chunk in enumerate(splits):
                name = self._prefix.format(i, f'A_log.{rank}')
                self.model.export_weight(chunk, name)

        # dt_bias: [num_v_heads] -> split (same osp.splitext issue)
        dt_bias = r.linear_attn_dt_bias(i)
        if dt_bias is not None:
            import torch as _torch
            splits = _torch.chunk(dt_bias, self.tp, dim=-1)
            for rank, chunk in enumerate(splits):
                name = self._prefix.format(i, f'dt_bias.{rank}')
                self.model.export_weight(chunk, name)

        # norm_weight: [head_v_dim] -> NOT split (per-head)
        norm = r.linear_attn_norm(i)
        if norm is not None:
            self.model.export_weight(norm,
                                     self._prefix.format(i, 'norm.weight'))




class MLA(Module):
    """
    requires:
        r.mla(i, kind)
        r.mla_norm(i)
    """

    _mla = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model

    def _export(self, idx: int, xs, kind: str, pack_fn, **kwargs):
        if all(x is None for x in xs):
            return
        q_a, q_b, q, kv_a, kv_b, o = map(transpose, xs)

        if q is not None:
            q_b = q

        cfg = self.model.model_config

        o = o.reshape(cfg.head_num, cfg.v_head_dim, -1)
        o = torch.nn.functional.pad(o, (0, 0, 0, cfg.size_per_head - cfg.v_head_dim, 0, 0))
        o = o.view(cfg.head_num * cfg.size_per_head, cfg.hidden_units)

        tp = self.model.attn_tp_size

        if q_a is not None:
            self.model.save_split(pack_fn(q_a), self._mla.format(idx, 'q_a_proj', kind))
        q_b_name = 'q_proj' if q_a is None else 'q_b_proj'
        self.model.save_split(pack_fn(q_b), self._mla.format(idx, q_b_name, kind), split_dim=-1, split_num=tp)
        self.model.save_split(pack_fn(kv_a), self._mla.format(idx, 'kv_a_proj', kind))
        self.model.save_split(pack_fn(kv_b), self._mla.format(idx, 'kv_b_proj', kind), split_dim=-1, split_num=tp)
        self.model.save_split(pack_fn(o), self._mla.format(idx, 'wo', kind), split_dim=0, split_num=tp)

    _layernorm = 'layers.{0}.attention.{1}_a_layernorm'

    def apply(self, i: int, r: BaseReader):

        for f in get_params(r.attn(i, None), bias=False):
            f(self._export, partial(r.mla, i), i)

        q, k = r.mla_norm(i)
        if q is not None:
            self.model.save_split(q, self._layernorm.format(i, 'q'))
        self.model.save_split(k, self._layernorm.format(i, 'kv'))


class Misc(Module):
    """
    requires:
        r.tok_embeddings()
        r.norm_weight()
        r.output_weight()
    """

    def apply(self, i: int, r: BaseReader):
        """Export embedding, norm, output weight."""
        emb = r.tok_embeddings()
        norm_weight = r.norm_weight()
        output_weight = r.output_weight()

        def pad_weight(tensor: torch.Tensor, tp: int):
            pad_size = None
            vocab_size = self.model.model_config.vocab_size
            if vocab_size % tp != 0:
                pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size
            if pad_size is None:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), 'constant', 0)

        tp = self.model.attn_tp_size * self.model.attn_cp_size
        if emb is not None:
            emb = pad_weight(emb, tp=tp)
            self.model.save_split(emb, 'tok_embeddings.weight', split_dim=1, split_num=tp)
        if norm_weight is not None:
            self.model.export_weight(norm_weight, 'norm.weight')
        if output_weight is not None:
            output_weight = pad_weight(output_weight, tp=tp)
            # transpose
            self.model.save_split(output_weight.t(), 'output.weight', split_dim=1, split_num=tp)


class Transformer:

    def __init__(self, model: BaseOutputModel):
        self.model = model
        modules = [LayerNorm]
        if model.model_config.kv_lora_rank:
            modules.append(MLA)
        else:
            modules.append(Attn)
        if model.model_config.inter_size:
            modules.append(Ffn)
        if model.model_config.expert_num:
            modules.append(MoeFfn)
        self.modules = [c(model) for c in modules]
        self.misc = Misc(model)

        # Mixed attention support: layer_types dispatch
        self.layer_types = getattr(model.model_config, 'layer_types', [])
        has_linear_attn = any(t == 'linear_attention' for t in self.layer_types)
        self.linear_attn = LinearAttn(model) if has_linear_attn else None

        # Accumulate misc params across shards for deferred MTP export
        self._misc_params = {}
        self._last_misc_reader = None

    def __call__(self, i: int, r: BaseReader):
        if i >= 0:
            # Skip MTP layers (i >= num_layer) — they share the
            # model.layers.{i} namespace but are exported separately
            # via finalize_mtp_export / _try_export_mtp.
            num_layer = self.model.model_config.num_layer
            if num_layer and i >= num_layer:
                # Accumulate MTP layer params for deferred export
                if hasattr(r, 'params'):
                    self._misc_params.update(r.params)
                    self._last_misc_reader = r
                return 0
            for m in self.modules:
                # Skip Attn for linear_attention layers (LinearAttn handles them)
                if (self.linear_attn and isinstance(m, Attn)
                        and i < len(self.layer_types)
                        and self.layer_types[i] == 'linear_attention'):
                    self.linear_attn(i, r)
                    continue
                m(i, r)
            return 1
        else:
            self.misc(i, r)
            # Accumulate misc params for deferred MTP export.
            # MTP expert weights may span multiple safetensors shards,
            # so we collect all misc params and export MTP once at the end.
            if hasattr(r, 'params'):
                self._misc_params.update(r.params)
                self._last_misc_reader = r

    def finalize_mtp_export(self):
        """Export MTP weights after all misc readers have been processed.

        MTP expert weights may span multiple safetensors shards, so we
        accumulate all misc params during __call__ and export once here.
        """
        if not self._last_misc_reader or not self._misc_params:
            return
        # Inject accumulated params into the reader so all MTP keys are
        # available in a single params dict.
        r = self._last_misc_reader
        r.params = self._misc_params
        self._try_export_mtp(r)

    def _try_export_mtp(self, r):
        """Export MTP predictor weights if present in the reader."""
        num_mtp = getattr(self.model.model_config, 'num_mtp_layers', 0)
        if not num_mtp or num_mtp <= 0:
            return
        if not hasattr(r, 'mtp_pre_fc_norm_embedding'):
            return
        for mtp_idx in range(num_mtp):
            # Check if MTP weights actually exist
            w = r.mtp_pre_fc_norm_embedding(mtp_idx)
            if w is not None:
                try:
                    self.export_mtp(mtp_idx, r)
                except Exception as e:
                    from lmdeploy.utils import get_logger
                    logger = get_logger('lmdeploy')
                    logger.warning(
                        f'Failed to export MTP layer {mtp_idx}: {e}. '
                        f'MTP speculative decoding will be disabled.')
                    self.model.model_config.num_mtp_layers = 0
                    return


    class MTPReaderAdapter:
        """Adapter that wraps MTP reader methods to match the standard reader
        interface expected by Attn, MoeFfn, Ffn, and LayerNorm modules.

        This allows reusing the existing export logic for MTP decoder layer
        weights by mapping mtp_attn() -> attn(), mtp_moe_expert() ->
        moe_ffn_expert(), etc.
        """

        def __init__(self, reader, mtp_layer_idx: int):
            self.reader = reader
            self.mtp_layer_idx = mtp_layer_idx

        def attn(self, i, kind):
            """Map to mtp_attn. The layer index i is ignored (uses
            mtp_layer_idx)."""
            if not kind:
                # Return key list for parameter type detection.
                # For models where layer 0 is linear_attention (no self_attn),
                # fall back to MTP attention keys or first full_attention layer.
                keys = self.reader.attn(0, None)
                if not keys and hasattr(self.reader, 'params'):
                    # Try MTP attention keys directly
                    prefix = self.reader._mtp_layer_prefix(
                        self.mtp_layer_idx) if hasattr(
                            self.reader, '_mtp_layer_prefix') else ''
                    if prefix:
                        keys = [k for k in self.reader.params.keys()
                                if k.startswith(prefix) and 'self_attn' in k]
                    # If still empty, try first full_attention layer
                    if not keys:
                        layer_types = getattr(
                            self.reader, 'model_cfg', {}).get(
                                'layer_types', [])
                        for li, lt in enumerate(layer_types):
                            if lt == 'full_attention':
                                keys = self.reader.attn(li, None)
                                if keys:
                                    break
                # Determine whether to force Weight handler (fp16 dequant)
                # or let CompressedWeight handler process int4 natively.
                # Only force Weight handler if MTP attn projections are
                # actually fp16 but stored as weight_packed (needs dequant).
                # If MTP attn projections are genuinely quantized (no .weight
                # keys for projections), let CompressedWeight handle them.
                if keys and any('weight_packed' in k for k in keys):
                    # Check if MTP attn has native .weight (fp16) keys
                    # for projection layers (excluding norms)
                    mtp_prefix = ''
                    if hasattr(self.reader, '_mtp_layer_prefix'):
                        mtp_prefix = self.reader._mtp_layer_prefix(
                            self.mtp_layer_idx)
                    has_fp16_weight = False
                    if mtp_prefix and hasattr(self.reader, 'params'):
                        has_fp16_weight = any(
                            k.startswith(mtp_prefix) and 'self_attn' in k
                            and k.endswith('.weight')
                            and not k.endswith('.weight_packed')
                            and not k.endswith('.weight_scale')
                            and not k.endswith('.weight_shape')
                            and 'norm' not in k
                            for k in self.reader.params.keys()
                        )
                    # If MTP attn has fp16 .weight keys, the weight_packed
                    # keys are from the main model's key list (fallback).
                    # Replace to route to Weight handler for dequant.
                    if has_fp16_weight:
                        keys = [
                            k.replace('.weight_packed', '.weight')
                            .replace('.weight_scale', '.weight')
                            .replace('.weight_shape', '.weight')
                            for k in keys
                        ]
                        keys = list(set(keys))
                    # If no fp16 .weight keys, MTP attn is genuinely
                    # quantized — keep keys as-is for CompressedWeight.
                return keys
            return self.reader.mtp_attn(self.mtp_layer_idx, kind)

        def attn_gate(self, i, kind):
            return self.reader.mtp_attn_gate(self.mtp_layer_idx, kind)

        def qk_norm(self, i):
            return self.reader.mtp_qk_norm(self.mtp_layer_idx)

        def attn_norm(self, i):
            return self.reader.mtp_attn_norm(self.mtp_layer_idx)

        def ffn_norm(self, i):
            return self.reader.mtp_ffn_norm(self.mtp_layer_idx)

        def moe_ffn_expert(self, e=None, i=None, kind=None):
            if e is None and i is None and kind is None:
                # Return MTP-specific expert keys for parameter type detection.
                # If we delegate to self.reader.moe_ffn_expert(), it returns
                # main model keys → wrong handler for MTP.
                if hasattr(self.reader, '_mtp_layer_prefix'):
                    prefix = self.reader._mtp_layer_prefix(
                        self.mtp_layer_idx)
                    keys = [k for k in self.reader.params.keys()
                            if k.startswith(prefix)
                            and 'experts' in k
                            and 'shared_expert' not in k]
                    if keys:
                        # Check if MTP experts are fp16 (.weight) or
                        # quantized (.weight_packed).
                        has_fp16 = any(
                            k.endswith('.weight')
                            and not k.endswith('gate.weight')
                            for k in keys)
                        has_packed = any(
                            'weight_packed' in k for k in keys)
                        if has_packed and not has_fp16:
                            # MTP experts are quantized — keep keys as-is
                            # so CompressedWeight handler processes them.
                            pass
                        elif has_packed and has_fp16:
                            # Mixed — shouldn't happen, keep as-is
                            pass
                        # If only .weight keys (fp16), no replacement needed
                        return keys
                return self.reader.moe_ffn_expert()
            return self.reader.mtp_moe_expert(self.mtp_layer_idx, e, kind)

        def moe_ffn_gate(self, i, kind):
            return self.reader.mtp_moe_gate(self.mtp_layer_idx, kind)

        def moe_ffn_shared_gate(self, i):
            return self.reader.mtp_shared_expert_gate(self.mtp_layer_idx)

        def ffn(self, i, kind):
            """Map to mtp_shared_expert for the shared expert FFN."""
            if kind is None:
                # Return only MTP FFN keys for parameter type detection.
                # For MoE models: filter for 'shared_expert.' keys.
                # For dense models: filter for 'mlp.' keys (no shared_expert).
                # Must exclude expert keys and shared_expert_gate to avoid
                # CompressedWeight handler when weight_type is fp16.
                if hasattr(self.reader, '_mtp_layer_prefix'):
                    prefix = self.reader._mtp_layer_prefix(
                        self.mtp_layer_idx)
                    # Try shared_expert first (MoE models)
                    keys = [k for k in self.reader.params.keys()
                            if k.startswith(prefix)
                            and 'shared_expert.' in k
                            and 'shared_expert_gate' not in k]
                    if not keys:
                        # Dense model: MTP FFN uses mlp.{gate,up,down}_proj
                        keys = [k for k in self.reader.params.keys()
                                if k.startswith(prefix + '.mlp.')
                                and 'experts' not in k
                                and 'shared_expert' not in k
                                and 'gate.weight' not in k]
                    # MTP FFN weights may be GPTQ int4 in the safetensors
                    # but the Reader dequantizes them to fp16 on read.
                    # Replace GPTQ key suffixes with .weight so that
                    # get_params() selects the Weight handler (not
                    # QuantWeightOnly which expects packed int4 data).
                    if keys and any(k.endswith('.qweight') for k in keys):
                        # Check if reader dequantizes MTP FFN to fp16
                        # by testing if mtp_shared_expert returns fp16
                        # for kind='weight'.
                        test_w = self.reader.mtp_shared_expert(
                            self.mtp_layer_idx, 'weight')
                        if test_w is not None and (
                                not isinstance(test_w, tuple)
                                or any(t is not None for t in test_w)):
                            # Reader dequantizes — replace GPTQ keys
                            keys = [
                                k.replace('.qweight', '.weight')
                                .replace('.qzeros', '.weight')
                                .replace('.scales', '.weight')
                                for k in keys
                            ]
                            keys = list(set(keys))
                    elif keys and any('weight_packed' in k for k in keys):
                        has_fp16 = any(
                            k.endswith('.weight')
                            and not k.endswith('gate.weight')
                            for k in keys)
                        if has_fp16:
                            # Mixed: has both fp16 and packed — keep as-is
                            pass
                        else:
                            # All packed, no fp16 — genuinely quantized,
                            # let CompressedWeight handle
                            pass
                    return keys
                return self.reader.ffn(0, None) \
                    if hasattr(self.reader, 'ffn') else []
            return self.reader.mtp_shared_expert(self.mtp_layer_idx, kind)

        def attn_sinks(self, i):
            return None

    def export_mtp(self, mtp_layer_idx: int, reader):
        """Export MTP predictor weights for one MTP layer.

        Reuses existing Attn, MoeFfn, Ffn modules by:
        1. Wrapping the reader with MTPReaderAdapter
        2. Temporarily swapping naming prefixes from 'layers.{0}' to 'mtp.{0}'
        3. Calling the standard module apply methods
        4. Exporting MTP-specific weights (pre_fc_norm, fc, final_norm)

        Args:
            mtp_layer_idx: MTP layer index (typically 0)
            reader: The source model reader with mtp_* methods
        """
        adapter = self.MTPReaderAdapter(reader, mtp_layer_idx)
        prefix = f'mtp.{mtp_layer_idx}'

        # --- MTP-specific weights ---
        # pre_fc_norm_embedding (RMSNorm)
        w = reader.mtp_pre_fc_norm_embedding(mtp_layer_idx)
        if w is not None:
            self.model.export_weight(w, f'{prefix}.pre_fc_norm_embedding.weight')

        # pre_fc_norm_hidden (RMSNorm)
        w = reader.mtp_pre_fc_norm_hidden(mtp_layer_idx)
        if w is not None:
            self.model.export_weight(w, f'{prefix}.pre_fc_norm_hidden.weight')

        # fc linear (hidden*2 -> hidden): ColumnParallelLinear with all-gather
        # Split along output dim (dim=-1) across TP ranks
        fc_w = reader.mtp_fc(mtp_layer_idx, 'weight')
        if fc_w is not None:
            fc_w = transpose(fc_w)
            self.model.save_split(fc_w, f'{prefix}.fc.weight',
                                  split_dim=-1,
                                  split_num=self.model.attn_tp_size)
        fc_b = reader.mtp_fc(mtp_layer_idx, 'bias')
        if fc_b is not None:
            self.model.save_split(fc_b, f'{prefix}.fc.bias',
                                  split_dim=-1,
                                  split_num=self.model.attn_tp_size)

        # --- Decoder layer norms ---
        attn_norm = adapter.attn_norm(mtp_layer_idx)
        if attn_norm is not None:
            self.model.save_split(attn_norm,
                                  f'{prefix}.attention_norm.weight')
        ffn_norm = adapter.ffn_norm(mtp_layer_idx)
        if ffn_norm is not None:
            self.model.save_split(ffn_norm, f'{prefix}.ffn_norm.weight')

        # --- Decoder layer attention (reuse Attn module with swapped prefix) ---
        attn_mod = None
        for m in self.modules:
            if isinstance(m, Attn):
                attn_mod = m
                break
        if attn_mod is not None:
            saved_attn = attn_mod._attn
            attn_mod._attn = prefix + '.attention.{1}.{2}'
            # Temporarily disable qk_norm in Attn.apply — we handle it
            # separately below to avoid hardcoded 'layers.{i}' prefix
            # in the shared norm path.
            saved_qk_norm = attn_mod.qk_norm
            attn_mod.qk_norm = False
            # MTP layers use mtp_layer_idx (0,1,2) but per-layer arrays
            # are indexed by layer_id. Shift arrays so index 0 maps to
            # the first MTP layer's config.
            saved_head_num_per_layer = attn_mod.head_num_per_layer
            saved_rope_dim_per_layer = attn_mod.rope_dim_per_layer
            num_hidden = self.model.model_config.num_layer
            if (saved_head_num_per_layer
                    and len(saved_head_num_per_layer) > num_hidden):
                attn_mod.head_num_per_layer = \
                    saved_head_num_per_layer[num_hidden:]
            if (saved_rope_dim_per_layer
                    and len(saved_rope_dim_per_layer) > num_hidden):
                attn_mod.rope_dim_per_layer = \
                    saved_rope_dim_per_layer[num_hidden:]
            try:
                attn_mod.apply(mtp_layer_idx, adapter)
            finally:
                attn_mod._attn = saved_attn
                attn_mod.qk_norm = saved_qk_norm
                attn_mod.head_num_per_layer = saved_head_num_per_layer
                attn_mod.rope_dim_per_layer = saved_rope_dim_per_layer

            # Export QK norm weights with correct MTP prefix
            if saved_qk_norm:
                qk = adapter.qk_norm(mtp_layer_idx)
                if qk is not None and qk[0] is not None:
                    q, k = qk
                    if attn_mod.model.permute_qk:
                        q = attn_mod._permute(q)
                        k = attn_mod._permute(k)
                    head_num = attn_mod.model.model_config.head_num
                    kv_head_num = attn_mod.model.model_config.kv_head_num
                    tp = attn_mod.tp
                    head_dim = attn_mod.head_dim
                    is_shared_q = (q.numel() == head_dim)
                    is_shared_k = (k.numel() == head_dim)
                    if is_shared_q:
                        q = q.repeat(head_num // tp)
                    if is_shared_k:
                        k = k.repeat(kv_head_num // tp)
                    if attn_mod.model.repeat_kv and k.numel() > head_dim:
                        k = k.view(-1, head_dim).repeat_interleave(
                            attn_mod.model.repeat_kv, dim=0).reshape(-1)
                    if is_shared_q:
                        for rank in range(tp):
                            self.model.export_weight(
                                q, f'{prefix}.attention.{rank}.q_norm')
                    else:
                        q_name = f'{prefix}.attention.q_norm'
                        self.model.save_split(
                            q, q_name, split_dim=-1, split_num=tp)
                    if is_shared_k:
                        for rank in range(tp):
                            self.model.export_weight(
                                k, f'{prefix}.attention.{rank}.k_norm')
                    else:
                        k_name = f'{prefix}.attention.k_norm'
                        self.model.save_split(
                            k, k_name, split_dim=-1, split_num=tp)

        # --- Decoder layer MoE FFN (reuse MoeFfn module with swapped prefix) ---
        moe_mod = None
        for m in self.modules:
            if isinstance(m, MoeFfn):
                moe_mod = m
                break
        if moe_mod is not None:
            saved_expert = moe_mod._moe_ffn_expert
            saved_gate = moe_mod._moe_ffn_gate
            saved_shared_gate = moe_mod._moe_ffn_shared_gate
            saved_expert_num = moe_mod.expert_num
            saved_moe_inter_size = moe_mod.inter_size
            moe_mod._moe_ffn_expert = prefix + '.moe_ffn.experts.E.{1}.{2}'
            moe_mod._moe_ffn_gate = prefix + '.moe_ffn.gate.{1}'
            moe_mod._moe_ffn_shared_gate = prefix + '.moe_ffn.shared_gate.weight'
            # Shift per-layer arrays for MTP layers
            num_hidden = self.model.model_config.num_layer
            if (isinstance(saved_expert_num, (list, tuple))
                    and len(saved_expert_num) > num_hidden):
                moe_mod.expert_num = saved_expert_num[num_hidden:]
            try:
                moe_mod.apply(mtp_layer_idx, adapter)
            finally:
                moe_mod._moe_ffn_expert = saved_expert
                moe_mod._moe_ffn_gate = saved_gate
                moe_mod._moe_ffn_shared_gate = saved_shared_gate
                moe_mod.expert_num = saved_expert_num
                moe_mod.inter_size = saved_moe_inter_size

        # --- Decoder layer shared expert FFN (reuse Ffn module with swapped prefix) ---
        ffn_mod = None
        for m in self.modules:
            if isinstance(m, Ffn) and not isinstance(m, MoeFfn):
                ffn_mod = m
                break
        if ffn_mod is not None:
            saved_ffn = ffn_mod._ffn
            saved_inter_size = ffn_mod.inter_size
            ffn_mod._ffn = prefix + '.feed_forward.{1}.{2}'
            # MTP layers use mtp_layer_idx (0,1,2) but inter_size is
            # indexed by layer_id. For Step3p5, MTP inter_size is at
            # indices num_hidden_layers+i (45,46,47). Build a temporary
            # inter_size list so that inter_size[mtp_layer_idx] returns
            # the correct MTP value.
            num_hidden = self.model.model_config.num_layer
            num_mtp = getattr(self.model.model_config,
                              'num_mtp_layers', 0)
            if (num_mtp > 0
                    and len(saved_inter_size) > num_hidden):
                # Per-layer arrays include MTP layers at the end
                ffn_mod.inter_size = saved_inter_size[num_hidden:]
            try:
                ffn_mod.apply(mtp_layer_idx, adapter)
            finally:
                ffn_mod._ffn = saved_ffn
                ffn_mod.inter_size = saved_inter_size

        # --- Final norm (RMSNorm) ---
        w = reader.mtp_final_norm(mtp_layer_idx)
        if w is not None:
            self.model.export_weight(w, f'{prefix}.norm.weight')

        # --- Per-layer shared_head weights (Step3p5 MTP only) ---
        if hasattr(reader, 'mtp_shared_head_output'):
            w = reader.mtp_shared_head_norm(mtp_layer_idx)
            if w is not None:
                self.model.export_weight(
                    w, f'{prefix}.shared_head.norm.weight')

            w = reader.mtp_shared_head_output(mtp_layer_idx)
            if w is not None:
                # Pad vocab dim for TP divisibility before transpose
                tp = self.model.attn_tp_size
                vocab_size = self.model.model_config.vocab_size
                if vocab_size % tp != 0:
                    pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size
                    w = torch.nn.functional.pad(
                        w, (0, 0, 0, pad_size), 'constant', 0)
                w = transpose(w)
                # Use '.weight' suffix so osp.splitext produces correct
                # file names: mtp.{i}.shared_head.output.{rank}.weight
                # matching C++ register_module naming convention.
                self.model.save_split(
                    w, f'{prefix}.shared_head.output.weight',
                    split_dim=-1, split_num=tp)


