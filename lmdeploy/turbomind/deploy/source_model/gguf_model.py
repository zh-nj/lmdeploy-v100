# Copyright (c) OpenMMLab. All rights reserved.
"""GGUF model reader for TurboMind converter pipeline.

Provides :class:`GGUFModelReader` (a :class:`BaseReader` subclass) that
maps GGUF tensor names to the HuggingFace-style names expected by the
existing converter modules, and :class:`GGUFModel` (a
:class:`BaseInputModel` subclass) that drives the per-layer iteration.
"""

import re

import numpy as np
import torch

from gguf import GGMLQuantizationType

from .base import INPUT_MODELS, BaseInputModel, BaseReader
from .gguf_reader import GGUFSplitReader, build_model_config

# ---------------------------------------------------------------------------
# GGUF → HF name mapping
# ---------------------------------------------------------------------------

# Layer-level mappings: GGUF suffix → HF suffix (relative to
# ``model.layers.{i}.``).  ``{i}`` is the layer index.
_LAYER_MAP = {
    'attn_q.weight': 'self_attn.q_proj.weight',
    'attn_k.weight': 'self_attn.k_proj.weight',
    'attn_v.weight': 'self_attn.v_proj.weight',
    'attn_output.weight': 'self_attn.o_proj.weight',
    'attn_q_norm.weight': 'self_attn.q_norm.weight',
    'attn_k_norm.weight': 'self_attn.k_norm.weight',
    'attn_norm.weight': 'input_layernorm.weight',
    'ffn_norm.weight': 'post_attention_layernorm.weight',
    'ffn_gate_inp.weight': 'block_sparse_moe.gate.weight',
    'exp_probs_b.bias': 'block_sparse_moe.e_score_correction_bias',
}

# MoE expert 3D tensor suffixes that need per-expert splitting.
# GGUF name suffix → (HF expert weight name, dim-to-split)
_MOE_3D_MAP = {
    'ffn_gate_exps.weight': 'w1',  # gate_proj
    'ffn_down_exps.weight': 'w2',  # down_proj
    'ffn_up_exps.weight': 'w3',    # up_proj
}

# Global (non-layer) mappings.
_GLOBAL_MAP = {
    'token_embd.weight': 'model.embed_tokens.weight',
    'output.weight': 'lm_head.weight',
    'output_norm.weight': 'model.norm.weight',
}

# Unquantized GGML types that can be directly converted to torch tensors.
_UNQUANTIZED_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}

_GGML_DTYPE_TO_TORCH = {
    GGMLQuantizationType.F32: torch.float32,
    GGMLQuantizationType.F16: torch.float16,
    GGMLQuantizationType.BF16: torch.bfloat16,
}

# Regex to extract layer index from GGUF tensor name ``blk.{i}.xxx``.
_BLK_RE = re.compile(r'^blk\.(\d+)\.(.+)$')


def _numpy_to_torch(data: np.ndarray, ggml_type: GGMLQuantizationType,
                     shape: tuple) -> torch.Tensor:
    """Convert an unquantized GGUF numpy array to a torch Tensor."""
    dtype = _GGML_DTYPE_TO_TORCH.get(ggml_type)
    if dtype is None:
        raise ValueError(f'Cannot directly convert ggml_type={ggml_type}')
    t = torch.from_numpy(data).to(dtype)
    if t.shape != shape:
        t = t.reshape(shape)
    return t


# ---------------------------------------------------------------------------
# GGUFModelReader
# ---------------------------------------------------------------------------


class GGUFModelReader(BaseReader):
    """Reader that wraps a :class:`GGUFSplitReader` and exposes weights
    through the same interface as :class:`LlamaReader` /
    :class:`MiniMaxM2Reader`.

    For the current layer (*layer_idx*), weights are read on demand from
    the underlying GGUF data.  Quantized tensors are stored as raw
    ``(np.ndarray, ggml_type, shape)`` tuples — dequantization is
    handled downstream.
    """

    def __init__(self, gguf_reader: GGUFSplitReader, layer_idx: int,
                 model_cfg: dict, num_experts: int = 0):
        super().__init__()
        self._reader = gguf_reader
        self._layer = layer_idx
        self._cfg = model_cfg
        self._num_experts = num_experts
        # Build tensor info lookup: name → TensorInfo
        self._tensor_info = {
            t.name: t for t in gguf_reader.get_tensor_info()
        }
        # Cache for 3D expert tensor data to avoid re-reading from disk
        # for each expert slice.  Key: gguf_name, Value: numpy array.
        self._expert_data_cache = {}

    def clear_cache(self):
        """Release cached expert tensor data to free memory."""
        self._expert_data_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gguf_name(self, suffix: str) -> str:
        """Return the full GGUF tensor name for a layer-level suffix."""
        return f'blk.{self._layer}.{suffix}'

    def _has(self, gguf_name: str) -> bool:
        return gguf_name in self._tensor_info

    def _get_tensor(self, gguf_name: str):
        """Read a tensor from GGUF and return as torch.Tensor.

        For unquantized types, returns a proper torch tensor.
        For quantized types, dequantizes to float16 so the converter
        pipeline can process them normally (transpose, pad, split).

        GGUF stores 2D weights in (in_dim, out_dim) layout, but the
        converter expects PyTorch convention (out_dim, in_dim).  We
        transpose 2D tensors here so downstream code works unchanged.
        """
        if not self._has(gguf_name):
            return None
        info = self._tensor_info[gguf_name]
        data = self._reader.read_tensor_data(gguf_name)
        if info.ggml_type in _UNQUANTIZED_TYPES:
            t = _numpy_to_torch(data, info.ggml_type, info.shape)
        else:
            # Quantized: dequantize to float16.
            from .gguf_dequant import dequantize
            t = dequantize(data, int(info.ggml_type), info.shape).to(
                torch.float16)
        # Transpose 2D weights from GGUF (in, out) to PyTorch (out, in)
        if t.ndim == 2:
            t = t.t().contiguous()
        return t

    def _transform(self, x: torch.Tensor, kind: str):
        """No-op transform — GGUF weights don't need policy transforms."""
        return x

    # ------------------------------------------------------------------
    # Global weights (embedding, norm, output)
    # ------------------------------------------------------------------

    def tok_embeddings(self):
        return self._get_tensor('token_embd.weight')

    def norm_weight(self):
        return self._get_tensor('output_norm.weight')

    def output_weight(self):
        return self._get_tensor('output.weight')

    # ------------------------------------------------------------------
    # Attention weights
    # ------------------------------------------------------------------

    def _attn(self, i: int, kind: str):
        result = []
        for key in ['q', 'k', 'v', 'o']:
            name = self._gguf_name(f'attn_{key}.{kind}')
            # GGUF uses 'attn_output' not 'attn_o'
            if key == 'o':
                name = self._gguf_name(f'attn_output.{kind}')
            result.append(self._get_tensor(name))
        return (*result,)

    def attn(self, i: int, kind: str):
        if not kind:
            return [k for k in self._tensor_info
                    if k.startswith(f'blk.{self._layer}.attn')]
        return self._attn(i, kind)

    def attn_norm(self, i: int):
        return self._get_tensor(self._gguf_name('attn_norm.weight'))

    # ------------------------------------------------------------------
    # FFN weights (dense MLP — not used for MoE-only models)
    # ------------------------------------------------------------------

    def _ffn(self, i: int, kind: str):
        result = []
        for key in ['gate', 'down', 'up']:
            name = self._gguf_name(f'ffn_{key}.{kind}')
            result.append(self._get_tensor(name))
        return (*result,)

    def ffn(self, i: int, kind: str):
        if not kind:
            return [k for k in self._tensor_info
                    if k.startswith(f'blk.{self._layer}.ffn')]
        return self._ffn(i, kind)

    def ffn_norm(self, i: int):
        return self._get_tensor(self._gguf_name('ffn_norm.weight'))

    # ------------------------------------------------------------------
    # QK norm
    # ------------------------------------------------------------------

    def qk_norm(self, i: int):
        q = self._get_tensor(self._gguf_name('attn_q_norm.weight'))
        k = self._get_tensor(self._gguf_name('attn_k_norm.weight'))
        return (q, k)

    # ------------------------------------------------------------------
    # MoE weights
    # ------------------------------------------------------------------

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        """Return MoE expert weights.

        GGUF stores expert weights as 3D tensors
        ``blk.{i}.ffn_{gate,down,up}_exps.weight`` with shape
        ``[num_experts, out_dim, in_dim]``.  We split along dim 0 to
        get per-expert 2D tensors.

        Returns raw GGUF quantized tensors (uint8 with ggml_type and
        ggml_shape attributes).  The downstream converter pipeline
        handles TP-aware format compatibility:
        - TP=8: Q4_K (64-aligned) and Q8_0 (32-aligned) pass through;
                Q6_K/Q5_K are re-quantized to Q4_K for column alignment.
        - TP=4: Q6_K (128-aligned) also passes through.
        - TP=2: Q5_K (256-aligned) also passes through.
        - TP=1: all formats pass through.
        """
        if not kind:
            return [k for k in self._tensor_info
                    if k.startswith(f'blk.{self._layer}.ffn')
                    and 'exps' in k]
        result = []
        for gguf_suffix, hf_name in [
            ('ffn_gate_exps.weight', 'w1'),
            ('ffn_down_exps.weight', 'w2'),
            ('ffn_up_exps.weight', 'w3'),
        ]:
            tensor = self._get_3d_expert_slice(gguf_suffix, e)
            result.append(tensor)
        return (*result,)

    def _get_3d_expert_slice(self, gguf_suffix: str, expert_idx: int):
        """Extract a single expert's 2D weight from a 3D GGUF tensor.

        GGUF logical shape is ``(out_dim, in_dim, num_experts)`` but the
        ``gguf`` package returns data with experts as the first numpy
        axis: ``(num_experts, in_dim, packed_row_bytes)`` for quantized
        types, or ``(num_experts, out_dim, in_dim)`` for unquantized.

        For quantized types, returns raw uint8 tensor with ``ggml_type``
        and ``ggml_shape`` attributes so the C++ engine can store them
        natively and dequantize on-the-fly during GEMM.

        Uses a per-layer cache to avoid re-reading the full 3D tensor
        from disk for each of the 256 experts.
        """
        gguf_name = self._gguf_name(gguf_suffix)
        if not self._has(gguf_name):
            return None
        info = self._tensor_info[gguf_name]

        # Read from cache or disk
        if gguf_name not in self._expert_data_cache:
            self._expert_data_cache[gguf_name] = \
                self._reader.read_tensor_data(gguf_name)
        data = self._expert_data_cache[gguf_name]

        if info.ggml_type in _UNQUANTIZED_TYPES:
            # data shape: (num_experts, out_dim, in_dim)
            full = torch.from_numpy(np.array(data))
            return full[expert_idx]

        # Quantized: return raw bytes as uint8 tensor.
        expert_data = data[expert_idx]
        raw_bytes = expert_data.tobytes()
        raw_tensor = torch.frombuffer(bytearray(raw_bytes),
                                      dtype=torch.uint8)
        # Attach metadata for the C++ side.
        # GGUF shape is (in_dim, out_dim, num_experts).
        # ggml_shape stores (out_dim, in_dim) to match C++ emplace_gguf
        # convention and save_split's expectation.
        raw_tensor.ggml_type = int(info.ggml_type)
        raw_tensor.ggml_shape = (info.shape[1], info.shape[0])
        return raw_tensor

    def _get_3d_expert_slice_fp16(self, gguf_suffix: str, expert_idx: int):
        """Like _get_3d_expert_slice but always dequantizes to FP16.

        Returns a 2D FP16 tensor in PyTorch convention (out_dim, in_dim),
        matching what HuggingFace model readers return.  The downstream
        converter pipeline (transpose → pad → split) expects this layout.
        """
        gguf_name = self._gguf_name(gguf_suffix)
        if not self._has(gguf_name):
            return None
        info = self._tensor_info[gguf_name]

        # Read from cache or disk
        if gguf_name not in self._expert_data_cache:
            self._expert_data_cache[gguf_name] = \
                self._reader.read_tensor_data(gguf_name)
        data = self._expert_data_cache[gguf_name]

        if info.ggml_type in _UNQUANTIZED_TYPES:
            full = torch.from_numpy(np.array(data))
            expert = full[expert_idx]
            # Unquantized: gguf returns (in_dim, out_dim), transpose to
            # PyTorch (out_dim, in_dim).
            if expert.ndim == 2:
                expert = expert.t().contiguous()
            return expert

        expert_data = data[expert_idx]
        # GGUF shape: info.shape = (in_dim, out_dim, num_experts) for 3D.
        # Per-expert dequant shape = (in_dim, out_dim).
        per_expert_shape = (info.shape[0], info.shape[1])
        from .gguf_dequant import dequantize
        t = dequantize(expert_data, int(info.ggml_type),
                       per_expert_shape).to(torch.float16)
        # Transpose from GGUF (in_dim, out_dim) to PyTorch (out_dim, in_dim).
        return t.t().contiguous()

    def moe_ffn_gate(self, i, kind):
        """Return MoE router gate weight or bias."""
        if kind == 'bias':
            return self._get_tensor(
                self._gguf_name('exp_probs_b.bias'))
        return self._get_tensor(
            self._gguf_name(f'ffn_gate_inp.{kind}'))


# ---------------------------------------------------------------------------
# GGUFModel — BaseInputModel for GGUF files
# ---------------------------------------------------------------------------


@INPUT_MODELS.register_module(name='minimax-m2-gguf')
class GGUFModel(BaseInputModel):
    """Input model that reads weights directly from GGUF files.

    Uses :class:`GGUFSplitReader` to parse GGUF metadata and tensor
    data, and yields :class:`GGUFModelReader` instances for each layer.
    """

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        self._reader = GGUFSplitReader(model_path)
        self._cfg = build_model_config(self._reader)

    def model_info(self):
        """Build model info dict from GGUF metadata.

        Returns a dict compatible with what the converter pipeline
        expects — same keys as ``LlamaModel.model_info()`` plus MoE
        fields from ``MiniMaxM2Model.model_info()``.
        """
        cfg = self._cfg
        head_dim = cfg.get('head_dim') or (
            cfg['hidden_size'] // cfg['num_attention_heads'])
        rope_theta = float(cfg.get('rope_theta', 10000.0))
        max_pos = int(cfg.get('max_position_embeddings', 0))

        from ..config import RopeParam
        rope_param = RopeParam(type='default', base=rope_theta, dim=head_dim)

        info = dict(
            num_layer=cfg['num_hidden_layers'],
            hidden_units=cfg['hidden_size'],
            head_num=cfg['num_attention_heads'],
            kv_head_num=cfg.get('num_key_value_heads',
                                cfg['num_attention_heads']),
            size_per_head=head_dim,
            vocab_size=cfg.get('vocab_size', 0),
            inter_size=cfg.get('intermediate_size', 0),
            norm_eps=cfg.get('rms_norm_eps', 1e-6),
            max_position_embeddings=max_pos,
            rope_param=rope_param,
        )

        # MoE parameters.
        expert_count = cfg.get('num_local_experts')
        if expert_count and expert_count > 0:
            info.update(
                expert_num=expert_count,
                expert_inter_size=cfg.get('expert_inter_size',
                                          cfg.get('intermediate_size', 0)),
                experts_per_token=cfg.get('num_experts_per_tok', 1),
                inter_size=0,
                norm_topk_prob=True,
                scoring_func=cfg.get('scoring_func', 'softmax'),
            )
            if cfg.get('expert_router_bias'):
                info['expert_router_bias'] = True

        # QK norm — detect from tensor names.
        tensors = self._reader.get_tensor_info()
        has_qk_norm = any('attn_q_norm' in t.name for t in tensors)
        if has_qk_norm:
            info['qk_norm'] = True
            # MiniMax-M2.1 uses per-token QK norm.
            arch = cfg.get('gguf_architecture', '')
            if arch == 'minimax-m2':
                info['qk_norm_type'] = 'per_token'

        # Partial RoPE.
        rotary_dim = cfg.get('rotary_dim')
        if rotary_dim is not None:
            info['rope_param'].dim = rotary_dim

        return info

    def readers(self):
        """Yield ``(layer_idx, GGUFModelReader)`` for each layer."""
        num_layers = self._cfg['num_hidden_layers']
        num_experts = self._cfg.get('num_local_experts', 0)
        for i in range(num_layers):
            reader = GGUFModelReader(
                self._reader, i, self._cfg, num_experts=num_experts)
            yield i, reader



# ---------------------------------------------------------------------------
# Qwen35MoeGGUFModelReader — GDN + MoE + shared_expert
# ---------------------------------------------------------------------------


class Qwen35MoeGGUFModelReader(GGUFModelReader):
    """Reader for Qwen3.5 MoE GGUF (GDN + MoE mixed architecture).

    Extends GGUFModelReader with:
    - GDN linear attention tensor reading (ssm_* → linear_attn.*)
    - Shared expert reading (ffn_*_shexp → shared_expert.*)
    - Full attention gated Q splitting
    - GemmaRMSNorm +1 conversion
    """

    def __init__(self, gguf_reader, layer_idx, model_cfg,
                 num_experts=0):
        super().__init__(gguf_reader, layer_idx, model_cfg, num_experts)
        self._attn_gates = {}

    # --- GemmaRMSNorm: y = norm(x) * (1 + w) ---

    def _gemma_rms_weight(self, tensor):
        if tensor is None:
            return None
        return tensor + 1

    # --- GDN linear attention weights ---

    def linear_attn_in_proj_qkvz(self, i, kind):
        """Read fused QKVZ from GGUF attn_qkv + attn_gate tensors.

        GGUF splits the fused QKVZ into two separate tensors:
        - attn_qkv: grouped [num_k_heads, (Q_hk + K_hk + V_r*hv)]
        - attn_gate: grouped [num_k_heads, Z_r*hv]

        HF in_proj_qkvz layout (what _reorder_qkvz expects):
        - grouped [num_k_heads, (Q_hk + K_hk + V_r*hv + Z_r*hv)]

        We reshape both to grouped form and interleave Z into each
        group to reconstruct the HF layout.
        """
        qkv = self._get_tensor(self._gguf_name(f'attn_qkv.{kind}'))
        gate = self._get_tensor(self._gguf_name(f'attn_gate.{kind}'))
        if qkv is None:
            return None
        if gate is None:
            return qkv
        # Both are (out_dim, hidden) after _get_tensor transpose.
        nk = self._cfg.get('linear_num_key_heads',
                           self._cfg.get('num_key_value_heads', 16))
        # qkv: (nk * qkv_group, hidden) → (nk, qkv_group, hidden)
        qkv_group = qkv.shape[0] // nk
        qkv_g = qkv.reshape(nk, qkv_group, -1)
        # gate: (nk * z_group, hidden) → (nk, z_group, hidden)
        z_group = gate.shape[0] // nk
        gate_g = gate.reshape(nk, z_group, -1)
        # Interleave: [Q_hk, K_hk, V_r*hv, Z_r*hv] per group
        qkvz_g = torch.cat([qkv_g, gate_g], dim=1)
        return qkvz_g.reshape(-1, qkv.shape[1])

    def linear_attn_in_proj_ba(self, i, kind):
        """Read fused BA from ssm_beta + ssm_alpha.

        GGUF stores them as separate tensors, each in grouped layout:
        - ssm_beta:  [num_k_heads, b_ratio]  (b_ratio = kv_ratio)
        - ssm_alpha: [num_k_heads, a_ratio]  (a_ratio = kv_ratio)

        HF in_proj_ba layout (what _reorder_ba expects):
        - grouped [num_k_heads, (b_ratio + a_ratio)]

        We reshape both to grouped form and interleave.
        """
        beta = self._get_tensor(self._gguf_name(f'ssm_beta.{kind}'))
        alpha = self._get_tensor(self._gguf_name(f'ssm_alpha.{kind}'))
        if beta is None or alpha is None:
            return None
        nk = self._cfg.get('linear_num_key_heads',
                           self._cfg.get('num_key_value_heads', 16))
        b_per_group = beta.shape[0] // nk
        a_per_group = alpha.shape[0] // nk
        beta_g = beta.reshape(nk, b_per_group, -1)
        alpha_g = alpha.reshape(nk, a_per_group, -1)
        ba_g = torch.cat([beta_g, alpha_g], dim=1)
        return ba_g.reshape(-1, beta.shape[1])

    def linear_attn_conv1d(self, i, kind):
        return self._get_tensor(self._gguf_name(f'ssm_conv1d.{kind}'))

    def linear_attn_out_proj(self, i, kind):
        return self._get_tensor(self._gguf_name(f'ssm_out.{kind}'))

    def linear_attn_a_log(self, i):
        return self._get_tensor(self._gguf_name('ssm_a'))

    def linear_attn_dt_bias(self, i):
        return self._get_tensor(self._gguf_name('ssm_dt.bias'))

    def linear_attn_norm(self, i):
        return self._get_tensor(self._gguf_name('ssm_norm.weight'))

    # --- Attention gate ---

    def attn_gate(self, i, kind):
        """Get attention gate weight.

        For GDN layers: read from blk.{i}.attn_gate.weight
        For full attention layers: stored from gated Q split in _attn()
        """
        stored = self._attn_gates.get((i, kind))
        if stored is not None:
            return stored
        return self._get_tensor(self._gguf_name(f'attn_gate.{kind}'))

    # --- Full attention: gated Q split ---

    def _attn(self, i, kind):
        """Override to handle full attention layers with gated Q.

        GGUF attn_q contains gated Q: (num_heads * head_dim * 2, hidden).
        Split into Q and gate, store gate for attn_gate().
        For GDN layers, attn_q/k/v don't exist (they use attn_qkv).
        """
        result = []
        for key in ['q', 'k', 'v', 'o']:
            name = self._gguf_name(f'attn_{key}.{kind}')
            if key == 'o':
                name = self._gguf_name(f'attn_output.{kind}')
            result.append(self._get_tensor(name))
        q, k, v, o = result

        # Split gated Q if present (full attention layers only)
        if q is not None:
            head_dim = self._cfg.get('head_dim', 256)
            num_heads = q.shape[0] // (head_dim * 2)
            if num_heads > 0 and q.shape[0] == num_heads * head_dim * 2:
                if q.dim() == 2:
                    q_gate = q.view(num_heads, head_dim * 2, -1)
                    q_only = q_gate[:, :head_dim, :].reshape(
                        num_heads * head_dim, -1)
                    gate = q_gate[:, head_dim:, :].reshape(
                        num_heads * head_dim, -1)
                else:
                    q_gate = q.view(num_heads, head_dim * 2)
                    q_only = q_gate[:, :head_dim].reshape(-1)
                    gate = q_gate[:, head_dim:].reshape(-1)
                self._attn_gates[(i, kind)] = gate
                q = q_only

        return (q, k, v, o)

    # --- Norm weights with GemmaRMSNorm +1 ---

    def attn_norm(self, i):
        return self._gemma_rms_weight(
            self._get_tensor(self._gguf_name('attn_norm.weight')))

    def ffn_norm(self, i):
        return self._gemma_rms_weight(
            self._get_tensor(
                self._gguf_name('post_attention_norm.weight')))

    def norm_weight(self):
        return self._gemma_rms_weight(
            self._get_tensor('output_norm.weight'))

    def qk_norm(self, i):
        q = self._gemma_rms_weight(
            self._get_tensor(self._gguf_name('attn_q_norm.weight')))
        k = self._gemma_rms_weight(
            self._get_tensor(self._gguf_name('attn_k_norm.weight')))
        return (q, k)

    # --- Shared expert ---

    def _ffn(self, i, kind):
        """Read shared_expert weights."""
        result = []
        for key in ['gate', 'down', 'up']:
            name = self._gguf_name(f'ffn_{key}_shexp.{kind}')
            result.append(self._get_tensor(name))
        return (*result,)

    def moe_ffn_shared_gate(self, i):
        return self._get_tensor(
            self._gguf_name('ffn_gate_inp_shexp.weight'))

    # --- MTP (Multi-Token Prediction) weights ---

    def _mtp_gguf_name(self, layer_idx: int, suffix: str) -> str:
        """Return the full GGUF tensor name for an MTP layer-level suffix.

        MTP tensors use ``blk.mtp.{layer}.{suffix}`` naming convention.
        """
        return f'blk.mtp.{layer_idx}.{suffix}'

    def mtp_pre_fc_norm_embedding(self, layer_idx: int):
        """Read MTP pre_fc_norm for embedding branch. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx,
                                    'pre_fc_norm_embedding.weight')))

    def mtp_pre_fc_norm_hidden(self, layer_idx: int):
        """Read MTP pre_fc_norm for hidden_states branch. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx,
                                    'pre_fc_norm_hidden.weight')))

    def mtp_fc(self, layer_idx: int, kind: str):
        """Read MTP fc linear weight (hidden*2 -> hidden)."""
        return self._get_tensor(
            self._mtp_gguf_name(layer_idx, f'fc.{kind}'))

    def mtp_final_norm(self, layer_idx: int):
        """Read MTP final RMSNorm weight. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx, 'output_norm.weight')))

    def mtp_attn(self, layer_idx: int, kind: str):
        """Read MTP decoder layer attention weights (q, k, v, o).

        MTP uses full attention (not GDN), so tensors are split q/k/v.
        Handles gated Q split — same as main model full attention.
        """
        result = []
        for key in ['q', 'k', 'v', 'o']:
            name = self._mtp_gguf_name(layer_idx, f'attn_{key}.{kind}')
            if key == 'o':
                name = self._mtp_gguf_name(layer_idx,
                                           f'attn_output.{kind}')
            result.append(self._get_tensor(name))
        q, k, v, o = result

        # Split gated Q if present (same logic as _attn for full attn)
        if q is not None:
            head_dim = self._cfg.get('head_dim', 256)
            num_heads = q.shape[0] // (head_dim * 2)
            if num_heads > 0 and q.shape[0] == num_heads * head_dim * 2:
                if q.dim() == 2:
                    q_gate = q.view(num_heads, head_dim * 2, -1)
                    q_only = q_gate[:, :head_dim, :].reshape(
                        num_heads * head_dim, -1)
                    gate = q_gate[:, head_dim:, :].reshape(
                        num_heads * head_dim, -1)
                else:
                    q_gate = q.view(num_heads, head_dim * 2)
                    q_only = q_gate[:, :head_dim].reshape(-1)
                    gate = q_gate[:, head_dim:].reshape(-1)

                if not hasattr(self, '_mtp_attn_gates'):
                    self._mtp_attn_gates = {}
                self._mtp_attn_gates[(layer_idx, kind)] = gate
                q = q_only

        return (q, k, v, o)

    def mtp_attn_gate(self, layer_idx: int, kind: str):
        """Get the stored MTP attention gate weight from gated Q split.

        For MTP layers, the gate comes from splitting the gated Q in
        mtp_attn(). If mtp_attn() hasn't been called yet, try reading
        from the GGUF attn_gate tensor directly.
        """
        if hasattr(self, '_mtp_attn_gates'):
            stored = self._mtp_attn_gates.get((layer_idx, kind))
            if stored is not None:
                return stored
        return self._get_tensor(
            self._mtp_gguf_name(layer_idx, f'attn_gate.{kind}'))

    def mtp_qk_norm(self, layer_idx: int):
        """Read MTP decoder layer QK norm weights. GemmaRMSNorm."""
        q = self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx, 'attn_q_norm.weight')))
        k = self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx, 'attn_k_norm.weight')))
        return (q, k)

    def mtp_attn_norm(self, layer_idx: int):
        """Read MTP decoder layer input_layernorm. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx, 'attn_norm.weight')))

    def mtp_ffn_norm(self, layer_idx: int):
        """Read MTP decoder layer post_attention_layernorm. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self._get_tensor(
                self._mtp_gguf_name(layer_idx,
                                    'post_attention_norm.weight')))

    def _mtp_get_3d_expert_slice(self, layer_idx: int,
                                 gguf_suffix: str, expert_idx: int):
        """Extract a single MTP expert's 2D weight from a 3D GGUF tensor.

        Same logic as _get_3d_expert_slice but uses MTP tensor prefix.
        """
        gguf_name = self._mtp_gguf_name(layer_idx, gguf_suffix)
        if not self._has(gguf_name):
            return None
        info = self._tensor_info[gguf_name]

        # Read from cache or disk
        if gguf_name not in self._expert_data_cache:
            self._expert_data_cache[gguf_name] = \
                self._reader.read_tensor_data(gguf_name)
        data = self._expert_data_cache[gguf_name]

        if info.ggml_type in _UNQUANTIZED_TYPES:
            full = torch.from_numpy(np.array(data))
            return full[expert_idx]

        # Quantized: return raw bytes as uint8 tensor with metadata.
        expert_data = data[expert_idx]
        raw_bytes = expert_data.tobytes()
        raw_tensor = torch.frombuffer(bytearray(raw_bytes),
                                      dtype=torch.uint8)
        raw_tensor.ggml_type = int(info.ggml_type)
        raw_tensor.ggml_shape = (info.shape[1], info.shape[0])
        return raw_tensor

    def mtp_moe_expert(self, layer_idx: int, e: int, kind: str):
        """Read MTP decoder layer MoE expert weights (gate, down, up)."""
        result = []
        for gguf_suffix in [
            'ffn_gate_exps.weight',
            'ffn_down_exps.weight',
            'ffn_up_exps.weight',
        ]:
            tensor = self._mtp_get_3d_expert_slice(
                layer_idx, gguf_suffix, e)
            result.append(tensor)
        return (*result,)

    def mtp_moe_gate(self, layer_idx: int, kind: str):
        """Read MTP decoder layer MoE router weight."""
        return self._get_tensor(
            self._mtp_gguf_name(layer_idx, f'ffn_gate_inp.{kind}'))

    def mtp_shared_expert(self, layer_idx: int, kind: str):
        """Read MTP decoder layer shared expert weights."""
        result = []
        for key in ['gate', 'down', 'up']:
            name = self._mtp_gguf_name(
                layer_idx, f'ffn_{key}_shexp.{kind}')
            result.append(self._get_tensor(name))
        return (*result,)

    def mtp_shared_expert_gate(self, layer_idx: int):
        """Read MTP decoder layer shared expert gate weight."""
        return self._get_tensor(
            self._mtp_gguf_name(layer_idx, 'ffn_gate_inp_shexp.weight'))


# ---------------------------------------------------------------------------
# Qwen35MoeGGUFModel
# ---------------------------------------------------------------------------


@INPUT_MODELS.register_module(name='qwen3-coder-next-gguf')
class Qwen35MoeGGUFModel(GGUFModel):
    """GGUF input model for Qwen3.5 MoE (GDN + MoE mixed architecture).

    Reuses the Qwen3NextModel converter pipeline by providing compatible
    model_info() and per-layer readers.
    """

    def model_info(self):
        info = super().model_info()
        cfg = self._cfg

        # GDN parameters
        if 'layer_types' in cfg:
            info.update(
                attn_output_gate=True,
                layer_types=cfg['layer_types'],
                linear_key_head_dim=cfg['linear_key_head_dim'],
                linear_value_head_dim=cfg['linear_value_head_dim'],
                linear_num_key_heads=cfg['linear_num_key_heads'],
                linear_num_value_heads=cfg['linear_num_value_heads'],
                linear_conv_kernel_dim=cfg['linear_conv_kernel_dim'],
                qk_norm=True,
                moe_shared_gate=cfg.get('moe_shared_gate', False),
                inter_size=cfg.get('shared_expert_intermediate_size', 0),
            )

        # MTP speculative decoding
        num_mtp = cfg.get('num_mtp_layers', 0)
        if num_mtp and num_mtp > 0:
            info['num_mtp_layers'] = num_mtp

        return info

    def readers(self):
        num_layers = self._cfg['num_hidden_layers']
        num_experts = self._cfg.get('num_local_experts', 0)
        for i in range(num_layers):
            reader = Qwen35MoeGGUFModelReader(
                self._reader, i, self._cfg, num_experts=num_experts)
            yield i, reader
