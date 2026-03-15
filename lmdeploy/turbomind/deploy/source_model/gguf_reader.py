# Copyright (c) OpenMMLab. All rights reserved.
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import gguf
from gguf import ExpertGatingFuncType, GGMLQuantizationType, GGUFValueType


@dataclass
class TensorInfo:
    """Metadata for a single GGUF tensor."""
    name: str
    shape: tuple
    ggml_type: GGMLQuantizationType
    n_bytes: int
    data_offset: int


def _patch_mxfp4():
    """Monkey-patch GGMLQuantizationType to support MXFP4 (type=39)."""
    if not hasattr(GGMLQuantizationType, 'MXFP4'):
        fake = int.__new__(GGMLQuantizationType, 39)
        fake._name_ = 'MXFP4'
        fake._value_ = 39
        GGMLQuantizationType._value2member_map_[39] = fake
        GGMLQuantizationType._member_map_['MXFP4'] = fake
        type.__setattr__(GGMLQuantizationType, 'MXFP4', fake)
    # Also patch GGML_QUANT_SIZES so GGUFReader can parse MXFP4 tensors.
    # MXFP4: block_size=32, type_size=17 (1 byte E8M0 + 16 bytes qs)
    from gguf.constants import GGML_QUANT_SIZES
    if 39 not in GGML_QUANT_SIZES:
        GGML_QUANT_SIZES[39] = (32, 17)


_patch_mxfp4()


class GGUFFileReader:
    """Parse a single GGUF V3 file: header, metadata KV, tensor info.

    Uses the ``gguf`` Python package (``pip install gguf``) internally.
    """

    def __init__(self, file_path: str):
        import time as _time
        from lmdeploy.utils import get_logger
        _logger = get_logger('lmdeploy')
        self.file_path = os.path.abspath(file_path)
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f'GGUF file not found: {self.file_path}')
        _logger.info(f'GGUFFileReader: parsing {self.file_path}')
        _t0 = _time.time()
        self._reader = gguf.GGUFReader(self.file_path)
        _logger.info(f'GGUFFileReader: done in {_time.time() - _t0:.1f}s, '
                     f'{len(self._reader.tensors)} tensors, '
                     f'{len(self._reader.fields)} fields')
        self._tensor_map: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        """Return the ``general.architecture`` string (e.g. 'llama')."""
        field = self._reader.get_field('general.architecture')
        if field is None:
            raise ValueError('GGUF file missing general.architecture')
        return bytes(field.parts[-1]).decode('utf-8')

    def get_metadata(self) -> Dict[str, Any]:
        """Extract model architecture parameters from GGUF metadata.

        Returns a flat dict whose keys are the raw GGUF metadata keys
        (e.g. ``'llama.block_count'``) and values are Python scalars,
        strings, or lists.
        """
        result: Dict[str, Any] = {}
        for key, field in self._reader.fields.items():
            if key.startswith('GGUF.'):
                continue
            result[key] = self._extract_field_value(field)
        return result

    def get_tensor_info(self) -> List[TensorInfo]:
        """Return metadata for every tensor in the file."""
        return [
            TensorInfo(
                name=t.name,
                shape=tuple(t.shape.tolist()),
                ggml_type=t.tensor_type,
                n_bytes=t.n_bytes,
                data_offset=t.data_offset,
            )
            for t in self._reader.tensors
        ]

    def read_tensor_data(self, tensor_name: str) -> np.ndarray:
        """Return the raw data array for *tensor_name*.

        For unquantized types (F32/F16) the returned array has the
        tensor's native dtype.  For quantized types the returned array
        is a ``uint8`` view over the raw block data.
        """
        idx = self._get_tensor_index(tensor_name)
        tensor = self._reader.tensors[idx]
        return np.array(tensor.data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tensor_index(self, name: str) -> int:
        if self._tensor_map is None:
            self._tensor_map = {
                t.name: i for i, t in enumerate(self._reader.tensors)
            }
        if name not in self._tensor_map:
            raise KeyError(f'Tensor not found in GGUF file: {name}')
        return self._tensor_map[name]

    @staticmethod
    def _extract_field_value(field) -> Any:
        """Convert a ``ReaderField`` to a plain Python value."""
        if not field.types:
            return None
        vtype = field.types[0]
        if vtype == GGUFValueType.STRING:
            return bytes(field.parts[-1]).decode('utf-8')
        if vtype == GGUFValueType.ARRAY:
            if len(field.types) < 2:
                return []
            elem_type = field.types[1]
            if elem_type == GGUFValueType.STRING:
                return [
                    bytes(field.parts[idx]).decode('utf-8')
                    for idx in field.data
                ]
            return [field.parts[idx][0].item() for idx in field.data]
        # Scalar numeric / bool
        if field.data:
            return field.parts[field.data[0]][0].item()
        return field.parts[-1][0].item()


@dataclass
class _ShardTensorInfo:
    """Internal: tensor info with shard index for GGUFSplitReader."""
    info: TensorInfo
    shard_index: int


class GGUFSplitReader:
    """Merge tensor info across multiple split GGUF shard files.

    If *gguf_path* points to a single (non-split) file, it is wrapped as
    a single-shard reader.  If it matches the split naming pattern
    (e.g. ``model-00001-of-00004.gguf``), all shards in the same
    directory are discovered automatically.

    Metadata is read from the first shard only.  Tensor info is merged
    from all shards.
    """

    _SPLIT_RE = re.compile(r'^(.+)-(\d{5})-of-(\d{5})\.gguf$')

    def __init__(self, gguf_path: str):
        import time as _time
        from lmdeploy.utils import get_logger
        _logger = get_logger('lmdeploy')
        gguf_path = os.path.abspath(gguf_path)
        shard_paths = self._discover_shards(gguf_path)
        _logger.info(f'GGUFSplitReader: discovered {len(shard_paths)} shards')
        _t0 = _time.time()
        self._shards: List[GGUFFileReader] = [
            GGUFFileReader(p) for p in shard_paths
        ]
        _logger.info(f'GGUFSplitReader: all shards parsed in '
                     f'{_time.time() - _t0:.1f}s')
        self.metadata = self._shards[0].get_metadata()
        self._tensor_map: Dict[str, _ShardTensorInfo] = (
            self._merge_tensor_info()
        )

    # ------------------------------------------------------------------
    # Public API  (mirrors GGUFFileReader)
    # ------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        return self._shards[0].architecture

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def get_tensor_info(self) -> List[TensorInfo]:
        return [st.info for st in self._tensor_map.values()]

    def read_tensor_data(self, tensor_name: str) -> np.ndarray:
        if tensor_name not in self._tensor_map:
            raise KeyError(
                f'Tensor not found in GGUF shards: {tensor_name}'
            )
        st = self._tensor_map[tensor_name]
        return self._shards[st.shard_index].read_tensor_data(tensor_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _discover_shards(cls, gguf_path: str) -> List[str]:
        """Return sorted list of shard file paths.

        For a non-split file, returns ``[gguf_path]``.
        If gguf_path is a directory, finds the first .gguf file in it.
        """
        # Handle directory path: find first .gguf file
        if os.path.isdir(gguf_path):
            candidates = sorted(f for f in os.listdir(gguf_path)
                                if f.endswith('.gguf'))
            if not candidates:
                raise FileNotFoundError(
                    f'No .gguf files found in {gguf_path}')
            gguf_path = os.path.join(gguf_path, candidates[0])

        basename = os.path.basename(gguf_path)
        m = cls._SPLIT_RE.match(basename)
        if m is None:
            # Single (non-split) file.
            if not os.path.isfile(gguf_path):
                raise FileNotFoundError(
                    f'GGUF file not found: {gguf_path}'
                )
            return [gguf_path]

        prefix, _, total_str = m.group(1), m.group(2), m.group(3)
        total = int(total_str)
        directory = os.path.dirname(gguf_path)
        paths = []
        for i in range(1, total + 1):
            name = f'{prefix}-{i:05d}-of-{total_str}.gguf'
            p = os.path.join(directory, name)
            if not os.path.isfile(p):
                raise FileNotFoundError(f'Missing GGUF shard: {p}')
            paths.append(p)
        return paths

    def _merge_tensor_info(self) -> Dict[str, _ShardTensorInfo]:
        merged: Dict[str, _ShardTensorInfo] = {}
        for shard_idx, shard in enumerate(self._shards):
            for ti in shard.get_tensor_info():
                merged[ti.name] = _ShardTensorInfo(
                    info=ti, shard_index=shard_idx
                )
        return merged


# Map GGUF expert_gating_func enum to TurboMind scoring_func string.
_GATING_FUNC_MAP = {
    ExpertGatingFuncType.SOFTMAX: 'softmax',
    ExpertGatingFuncType.SIGMOID: 'sigmoid',
}

# Map GGUF architecture name → HuggingFace model architecture identifier.
_GGUF_ARCH_TO_HF = {
    'minimax-m2': 'MiniMaxM2ForCausalLM',
    'llama': 'LlamaForCausalLM',
    'qwen2': 'Qwen2ForCausalLM',
    'qwen2moe': 'Qwen2MoeForCausalLM',
    'deepseek2': 'DeepseekV2ForCausalLM',
    'gemma': 'GemmaForCausalLM',
    'gemma2': 'Gemma2ForCausalLM',
    'phi3': 'Phi3ForCausalLM',
    'starcoder2': 'Starcoder2ForCausalLM',
    'internlm2': 'InternLM2ForCausalLM',
    'chatglm': 'ChatGLMForCausalLM',
    'mistral': 'MistralForCausalLM',
    'mixtral': 'MixtralForCausalLM',
    'qwen35moe': 'Qwen3NextForCausalLM',
}


def _resolve_scalar(value, default=None):
    """Return a scalar from *value*, which may be a list (per-layer array).

    For per-layer arrays, return the most common non-zero value.
    """
    if value is None:
        return default
    if not isinstance(value, list):
        return value
    non_zero = [v for v in value if v]
    if not non_zero:
        return default
    # Most common non-zero value.
    from collections import Counter
    return Counter(non_zero).most_common(1)[0][0]


def build_model_config(reader):
    """Build a TurboMind-compatible model config dict from GGUF metadata.

    Args:
        reader: A :class:`GGUFFileReader` or :class:`GGUFSplitReader`.

    Returns:
        dict with keys matching what ``LlamaModel.model_info()`` and
        ``MiniMaxM2Model.model_info()`` expect from ``model_config``.
    """
    meta = reader.get_metadata()
    arch = reader.architecture

    def _get(suffix, default=None):
        return meta.get(f'{arch}.{suffix}', default)

    num_attention_heads = _get('attention.head_count')
    hidden_size = _get('embedding_length')
    head_dim = _get('attention.key_length')
    if head_dim is None and num_attention_heads and hidden_size:
        head_dim = hidden_size // num_attention_heads

    num_key_value_heads = _resolve_scalar(
        _get('attention.head_count_kv'), num_attention_heads)

    intermediate_size = _resolve_scalar(_get('feed_forward_length'), 0)

    # Expert-specific intermediate size (may differ from dense FFN).
    expert_inter_size = _get('expert_feed_forward_length', intermediate_size)

    # vocab_size: prefer metadata, fall back to tokenizer token count.
    vocab_size = _get('vocab_size')
    if vocab_size is None:
        tokens = meta.get('tokenizer.ggml.tokens')
        if isinstance(tokens, list):
            vocab_size = len(tokens)

    config = dict(
        num_hidden_layers=_get('block_count'),
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=_get('context_length', 0),
        rms_norm_eps=_get('attention.layer_norm_rms_epsilon', 1e-6),
        rope_theta=float(_get('rope.freq_base', 10000.0)),
        head_dim=head_dim,
    )

    # RoPE partial dimension (rotary_dim).
    rope_dim_count = _get('rope.dimension_count')
    if rope_dim_count is not None and head_dim and rope_dim_count != head_dim:
        config['rotary_dim'] = rope_dim_count

    # MoE parameters.
    expert_count = _get('expert_count')
    if expert_count is not None and expert_count > 0:
        config['num_local_experts'] = expert_count
        config['num_experts_per_tok'] = _get('expert_used_count', 1)
        config['expert_inter_size'] = expert_inter_size

        # Gating / scoring function.
        gating_func = _get('expert_gating_func')
        if gating_func is not None:
            try:
                gating_enum = ExpertGatingFuncType(gating_func)
                config['scoring_func'] = _GATING_FUNC_MAP.get(
                    gating_enum, 'softmax')
            except ValueError:
                config['scoring_func'] = 'softmax'

        # Router bias — inferred from tensor names.
        tensors = reader.get_tensor_info()
        has_router_bias = any('exp_probs_b.bias' in t.name
                              for t in tensors)
        if has_router_bias:
            config['expert_router_bias'] = True

        # Detect per-layer expert GGML quantization type from tensor info.
        # UD (Unsloth Dynamic) models use different quant types per layer.
        num_layers = config.get('num_hidden_layers', 0)
        layer_expert_types = {}
        layer_expert_types_w2 = {}
        for t in tensors:
            if 'ffn_gate_exps' in t.name:
                parts = t.name.split('.')
                layer_idx = int(parts[1])  # blk.{i}.ffn_gate_exps.weight
                layer_expert_types[layer_idx] = int(t.ggml_type)
            if 'ffn_down_exps' in t.name:
                parts = t.name.split('.')
                layer_idx = int(parts[1])
                layer_expert_types_w2[layer_idx] = int(t.ggml_type)
        if layer_expert_types:
            # Build per-layer list; default to GGML_TYPE_COUNT (31) for
            # layers without expert tensors (shouldn't happen for MoE).
            expert_ggml_type_list = [
                layer_expert_types.get(i, 31)
                for i in range(num_layers)
            ]
            config['expert_ggml_type'] = expert_ggml_type_list
            # Per-layer w2 types — only emit if any layer differs from w1/w3.
            has_w2_diff = any(
                layer_expert_types_w2.get(i) != layer_expert_types.get(i)
                for i in range(num_layers)
                if i in layer_expert_types_w2
            )
            if has_w2_diff:
                config['expert_ggml_type_w2'] = [
                    layer_expert_types_w2.get(i, 31)
                    for i in range(num_layers)
                ]

        # Shared expert.
        shared_count = _get('expert_shared_count')
        if shared_count is not None and shared_count > 0:
            config['shared_expert_count'] = shared_count
            shared_ffn = _get('expert_shared_feed_forward_length')
            if shared_ffn is not None:
                config['shared_expert_intermediate_size'] = shared_ffn

    # Store architecture string for downstream consumers.
    config['gguf_architecture'] = arch

    # Map GGUF architecture name to HF model_arch identifier.
    config['model_arch'] = _GGUF_ARCH_TO_HF.get(arch)

    # MTP speculative decoding: detect from metadata or tensor names.
    num_mtp = _get('num_nextn_predict_layers')
    if num_mtp is None:
        # Fall back to detecting MTP tensors by name (blk.mtp.{layer}.*)
        tensors = reader.get_tensor_info()
        mtp_layers = set()
        for t in tensors:
            if '.mtp.' in t.name:
                # e.g. blk.mtp.0.pre_fc_norm_embedding.weight
                parts = t.name.split('.')
                try:
                    idx = parts.index('mtp')
                    mtp_layers.add(int(parts[idx + 1]))
                except (ValueError, IndexError):
                    pass
        if mtp_layers:
            num_mtp = len(mtp_layers)
    if num_mtp is not None and num_mtp > 0:
        config['num_mtp_layers'] = num_mtp

    # GDN / SSM parameters (Qwen3.5 MoE / Qwen3-Coder-Next)
    conv_kernel = _get('ssm.conv_kernel')
    if conv_kernel is not None:
        full_attn_interval = _get('full_attention_interval', 4)
        num_layers = config['num_hidden_layers']
        layer_types = []
        for i in range(num_layers):
            if (i + 1) % full_attn_interval == 0:
                layer_types.append('full_attention')
            else:
                layer_types.append('linear_attention')
        config['layer_types'] = layer_types

        ssm_inner = _get('ssm.inner_size', 8192)
        ssm_state = _get('ssm.state_size', 128)
        ssm_groups = _get('ssm.group_count', 16)

        config['linear_key_head_dim'] = ssm_state
        config['linear_value_head_dim'] = ssm_state
        config['linear_num_key_heads'] = ssm_groups
        config['linear_num_value_heads'] = ssm_inner // ssm_state
        config['linear_conv_kernel_dim'] = conv_kernel

        # Shared expert gate flag
        shared_ffn = _get('expert_shared_feed_forward_length')
        if shared_ffn is not None:
            config['shared_expert_intermediate_size'] = shared_ffn
            config['moe_shared_gate'] = True

    return config
