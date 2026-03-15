# Copyright (c) OpenMMLab. All rights reserved.

import os
import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class Step3p5Reader(LlamaReader):
    """Reader for Step-3.5-Flash model (Step3p5ForCausalLM).

    Handles mixed attention (full + sliding), MoE with sigmoid router,
    head-wise attention gate (g_proj), QK norm, and zero-centered RMSNorm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Step3p5 config.json omits tie_word_embeddings, but HuggingFace
        # PretrainedConfig defaults it to True. The model has a separate
        # lm_head.weight (max_diff ~0.29 vs embed_tokens), so force untied.
        self.output_weight_key = 'lm_head.weight'
        # Parse extra_config for per-weight bit-width detection.
        # Compiled regex patterns sorted longest-first for specificity.
        self._extra_config_patterns = []
        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        quant_cfg = cfg.get('quantization_config', {})
        self._global_bits = quant_cfg.get('bits', 4)
        self._group_size = quant_cfg.get('group_size', 128)
        ec = quant_cfg.get('extra_config', {})
        import re as _re
        # Sort by pattern length descending: longer (more specific) patterns
        # match first (e.g. "model.layers.0.mlp.gate_proj" before ".*mlp.*")
        for pat, val in sorted(ec.items(), key=lambda kv: -len(kv[0])):
            bits = val.get('bits', self._global_bits)
            try:
                self._extra_config_patterns.append((_re.compile(pat), bits))
            except _re.error:
                pass

    def _transform(self, x: torch.Tensor, kind: str):
        """Transform tensor, converting bf16→fp16 for V100 compatibility."""
        # Convert bf16→fp16 on CPU before processor moves to GPU,
        # avoiding GPU OOM on large tensors like embeddings.
        if x is not None and x.dtype == torch.bfloat16:
            x = x.half()
        x = self.processor(x, kind)
        return x

    def _get_bits_for_key(self, full_key: str) -> int:
        """Resolve bit-width for a weight key using extra_config patterns.

        Returns 4, 8, or 16.
        """
        for pattern, bits in self._extra_config_patterns:
            if pattern.search(full_key):
                return bits
        return self._global_bits

    def _dequant_gptq_8bit(self, prefix: str):
        """Dequantize 8-bit GPTQ weight to fp16.

        8-bit GPTQ: qweight [K//4, N] int32 (4 values per int32),
        qzeros [K//group_size, N//4] int32, scales [K//group_size, N] fp16.
        """
        import torch
        qweight = self.params.get(f'{prefix}.qweight')
        scales = self.params.get(f'{prefix}.scales')
        qzeros = self.params.get(f'{prefix}.qzeros')
        if qweight is None or scales is None or qzeros is None:
            return None

        bits = 8
        group_size = self._group_size

        # Unpack qweight: each int32 holds 4 x 8-bit values
        in_dim_packed, out_dim = qweight.shape
        in_dim = in_dim_packed * (32 // bits)  # K//4 * 4 = K

        weight = torch.zeros(in_dim, out_dim, dtype=torch.float16)
        for j in range(32 // bits):
            weight[j::4, :] = ((qweight >> (bits * j)) & 0xFF).to(
                torch.float16)

        # Unpack qzeros: each int32 holds 4 x 8-bit values, +1 offset
        zp_rows, zp_cols_packed = qzeros.shape
        zp_cols = zp_cols_packed * (32 // bits)
        zeros = torch.zeros(zp_rows, zp_cols, dtype=torch.float16)
        for j in range(32 // bits):
            zeros[:, j::4] = (((qzeros >> (bits * j)) & 0xFF) + 1).to(
                torch.float16)
        zeros = zeros[:, :out_dim]

        # Dequantize: w = (qw - zero) * scale
        scales_cpu = scales.to(torch.float16)
        for g in range(in_dim // group_size):
            start = g * group_size
            end = start + group_size
            weight[start:end] = (weight[start:end] - zeros[g:g + 1]) * \
                scales_cpu[g:g + 1]

        # Return [out_dim, in_dim] so module.py's transpose() → [in_dim, out_dim]
        return weight.t()

    def _dequant_gptq_4bit(self, prefix: str):
        """Dequantize 4-bit GPTQ weight to fp16.

        4-bit GPTQ: qweight [K//8, N] int32 (8 values per int32),
        qzeros [K//group_size, N//8] int32, scales [K//group_size, N] fp16.
        """
        import torch
        qweight = self.params.get(f'{prefix}.qweight')
        scales = self.params.get(f'{prefix}.scales')
        qzeros = self.params.get(f'{prefix}.qzeros')
        if qweight is None or scales is None or qzeros is None:
            return None

        bits = 4
        group_size = self._group_size

        # Unpack qweight: each int32 holds 8 x 4-bit values
        in_dim_packed, out_dim = qweight.shape
        in_dim = in_dim_packed * (32 // bits)  # K//8 * 8 = K

        weight = torch.zeros(in_dim, out_dim, dtype=torch.float16)
        for j in range(32 // bits):
            weight[j::8, :] = ((qweight >> (bits * j)) & 0xF).to(
                torch.float16)

        # Unpack qzeros: each int32 holds 8 x 4-bit values, +1 offset
        zp_rows, zp_cols_packed = qzeros.shape
        zp_cols = zp_cols_packed * (32 // bits)
        zeros = torch.zeros(zp_rows, zp_cols, dtype=torch.float16)
        for j in range(32 // bits):
            zeros[:, j::8] = (((qzeros >> (bits * j)) & 0xF) + 1).to(
                torch.float16)
        zeros = zeros[:, :out_dim]

        # Dequantize: w = (qw - zero) * scale
        scales_cpu = scales.to(torch.float16)
        for g in range(in_dim // group_size):
            start = g * group_size
            end = start + group_size
            weight[start:end] = (weight[start:end] - zeros[g:g + 1]) * \
                scales_cpu[g:g + 1]

        # Return [out_dim, in_dim] so module.py's transpose() → [in_dim, out_dim]
        return weight.t()


    # Match both dense MLP (layers 0-2) and shared expert (layers 3-44)
    # but NOT MoE expert weights (moe.experts.*)
    ffn_pattern = r'(?:\.mlp\.|share_expert\.)'

    # Zero-centered RMSNorm: y = norm(x) * (1 + w)
    # TurboMind uses y = norm(x) * w, so export gamma = 1 + w
    def _gemma_rms_weight(self, tensor):
        tensor = self.transform(tensor, 'weight')
        if tensor is None:
            return None
        disable_zero_centered = os.getenv('TM_STEP3P5_DISABLE_ZERO_CENTERED',
                                          '0') not in ('', '0')
        return tensor if disable_zero_centered else (tensor + 1)

    def norm_weight(self):
        return self._gemma_rms_weight(
            self.params.get(self.norm_weight_key, None))

    def attn_norm(self, i: int):
        return self._gemma_rms_weight(
            self.params.get(
                f'{self.attn_layer_prefix}.{i}.input_layernorm.weight'))

    def ffn_norm(self, i: int):
        return self._gemma_rms_weight(
            self.params.get(
                f'{self.attn_layer_prefix}.{i}.'
                'post_attention_layernorm.weight'))

    # --- attention: read g_proj separately ---

    def _dequant_compressed_tensor(self, key_prefix):
        """Dequantize a compressed-tensors weight to fp16.

        Used for attn weights that are int4 (weight_packed) but must be
        exported as fp16 because weight_type is overridden to fp16 for
        MoE models (only experts are quantized).
        """
        import torch
        packed = self.params.get(f'{key_prefix}.weight_packed')
        scale = self.params.get(f'{key_prefix}.weight_scale')
        if packed is None or scale is None:
            return None

        packed = packed.cuda()
        scale = scale.cuda().half()

        out_dim = packed.shape[0]
        in_dim = packed.shape[1] * 8

        vals = torch.zeros(out_dim, in_dim, dtype=torch.uint8,
                           device=packed.device)
        for j in range(8):
            vals[:, j::8] = ((packed >> (4 * j)) & 0xF).to(torch.uint8)

        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        quant_cfg = cfg.get('quantization_config', {})
        group_size = 32
        for gc in quant_cfg.get('config_groups', {}).values():
            w = gc.get('weights', {})
            if w.get('group_size'):
                group_size = w['group_size']
                break

        vals_f = vals.to(torch.float16) - 8.0
        scale_exp = scale.repeat_interleave(group_size, dim=1)
        if scale_exp.shape[1] > in_dim:
            scale_exp = scale_exp[:, :in_dim]
        weight = vals_f * scale_exp
        return weight.cpu()

    def attn(self, i: int, kind: str):
        """Override to fix key routing for mixed-bit quantization.

        For compressed-tensors (AWQ): replace weight_packed keys with .weight
        so get_params() routes to Weight handler.

        For mixed-bit GPTQ (AutoRound): 8-bit attn weights are dequantized
        to fp16 in _attn(), so replace their .qweight/.scales/.qzeros keys
        with .weight so get_params() routes to Weight handler.
        16-bit (fp16) weights already have .weight keys.
        Only 4-bit weights keep .qweight keys for QuantWeightOnly handler.
        """
        if not kind:
            keys = self.filter(self.attn_pattern)
            # compressed-tensors path (AWQ)
            if any(k.endswith('.weight_packed') for k in keys):
                new_keys = []
                seen = set()
                for k in keys:
                    for suffix in ('.weight_packed', '.weight_scale',
                                   '.weight_shape'):
                        if k.endswith(suffix):
                            base = k[:-len(suffix)]
                            fp16_key = base + '.weight'
                            if fp16_key not in seen:
                                new_keys.append(fp16_key)
                                seen.add(fp16_key)
                            break
                    else:
                        new_keys.append(k)
                return new_keys
            # Mixed-bit GPTQ path: reroute 8-bit/16-bit GPTQ keys to .weight
            if self._extra_config_patterns:
                new_keys = []
                seen = set()
                for k in keys:
                    for suffix in ('.qweight', '.scales', '.qzeros'):
                        if k.endswith(suffix):
                            base = k[:-len(suffix)]
                            bits = self._get_bits_for_key(base)
                            if bits >= 8:
                                # 8-bit or 16-bit: will be dequantized to fp16
                                fp16_key = base + '.weight'
                                if fp16_key not in seen:
                                    new_keys.append(fp16_key)
                                    seen.add(fp16_key)
                            else:
                                # 4-bit: keep original GPTQ keys
                                new_keys.append(k)
                            break
                    else:
                        new_keys.append(k)
                return new_keys
            return keys
        return self._attn(i, kind)

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o for layer i.

        g_proj (head-wise attention gate) is read separately via attn_gate().

        Handles three quantization scenarios:
        - compressed-tensors (AWQ): dequantize weight_packed → fp16
        - 8-bit GPTQ (AutoRound): dequantize 8-bit qweight → fp16
        - 4-bit GPTQ: pass through to process_gptq (standard path)
        - fp16: direct weight read
        """
        result = []
        for key in ['q', 'k', 'v', 'o']:
            prefix = (f'{self.attn_layer_prefix}.{i}.self_attn.{key}_proj')
            tensor = self.params.get(f'{prefix}.{kind}')
            if tensor is None and kind == 'weight':
                # compressed-tensors: dequantize weight_packed → fp16
                tensor = self._dequant_compressed_tensor(prefix)
                if tensor is None:
                    # 8-bit GPTQ: dequantize qweight → fp16
                    bits = self._get_bits_for_key(prefix)
                    if bits == 8:
                        tensor = self._dequant_gptq_8bit(prefix)
            else:
                tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def attn_gate(self, i: int, kind: str):
        """Read head-wise attention gate (g_proj).

        For GPTQ/AutoRound models, g_proj is quantized (qweight/qzeros/scales)
        but TurboMind expects fp16 gate weights. Dequantize on the fly since
        the gate is small (hidden_dim × num_heads).
        """
        prefix = f'{self.attn_layer_prefix}.{i}.self_attn.g_proj'
        if kind == 'weight':
            # Try direct weight first (fp16 models)
            tensor = self.params.get(f'{prefix}.weight')
            if tensor is not None:
                return self.transform(tensor, kind)
            # GPTQ: dequantize qweight + qzeros + scales → fp16
            qweight = self.params.get(f'{prefix}.qweight')
            if qweight is not None:
                return self._dequant_gptq_gate(prefix)
            return None
        elif kind == 'bias':
            tensor = self.params.get(f'{prefix}.bias')
            return self.transform(tensor, kind) if tensor is not None else None
        return None

    def _dequant_gptq_gate(self, prefix: str):
        """Dequantize GPTQ int4 gate weight to fp16.

        GPTQ symmetric format: qweight [in_dim/8, out_dim] int32,
        qzeros [in_dim/group_size, out_dim/8] int32,
        scales [in_dim/group_size, out_dim] fp16.
        """
        import torch
        qweight = self.params[f'{prefix}.qweight']
        scales = self.params[f'{prefix}.scales']
        qzeros = self.params[f'{prefix}.qzeros']
        bits = 4
        group_size = 128  # AutoRound default

        # Unpack qweight: each int32 holds 8 x 4-bit values
        in_dim_packed, out_dim = qweight.shape
        in_dim = in_dim_packed * (32 // bits)

        # Unpack qweight (sequential rows)
        # Equivalent to process_gptq's torch.stack(xs, dim=1).view(-1, N)
        weight = torch.zeros(in_dim, out_dim, dtype=torch.float16)
        for j in range(32 // bits):
            weight.reshape(-1, 8, out_dim)[:, j, :] = ((qweight >> (bits * j)) & 0xF).to(torch.float16)

        # Unpack qzeros (sequential cols, +1 offset)
        zp_packed_rows, zp_cols = qzeros.shape
        zeros = torch.zeros(zp_packed_rows, zp_cols * (32 // bits), dtype=torch.float16)
        for j in range(32 // bits):
            zeros.reshape(zp_packed_rows, -1, 8)[:, :, j] = (((qzeros >> (bits * j)) & 0xF) + 1).to(torch.float16)
        zeros = zeros[:, :out_dim]

        # Dequantize: w = (qw - zero) * scale
        for g in range(in_dim // group_size):
            start = g * group_size
            end = start + group_size
            weight[start:end] = (weight[start:end] - zeros[g:g + 1]) * \
                scales[g:g + 1]

        # Return in PyTorch nn.Linear format [output_dim, input_dim]
        # so that module.py's transpose() produces [input_dim, output_dim]
        return weight.t()

    def qk_norm(self, i: int):
        """Read per-head QK norm weights (q_norm, k_norm)."""
        result = []
        for x in ['q', 'k']:
            name = f'{self.attn_layer_prefix}.{i}.self_attn.{x}_norm.weight'
            result.append(self._gemma_rms_weight(self.params.get(name)))
        return (*result, )

    # --- MoE weights ---

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'moe\.experts')
        result = []
        for key in ['gate', 'down', 'up']:
            name = (f'{self.attn_layer_prefix}.{i}'
                    f'.moe.experts.{e}.{key}_proj.{kind}')
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i, kind):
        if kind == 'bias':
            # Step3p5 stores router bias as model.layers.{i}.moe.router_bias
            # not under moe.gate.bias
            return self.params.get(
                f'{self.attn_layer_prefix}.{i}.moe.router_bias')
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.moe.gate.{kind}')

    def moe_ffn_router_bias(self, i):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.moe.router_bias')

    def _ffn(self, i: int, kind: str):
        """Get FFN weights for layer i.

        Layers 0-2: dense MLP from model.layers.{i}.mlp.*
        Layers 3-44: shared expert from model.layers.{i}.moe.share_expert.*
                     or model.layers.{i}.share_expert.* (compressed-tensors)

        For 8-bit GPTQ (AutoRound), dequantize to fp16 when kind='weight'.
        """
        # Determine prefix based on whether this layer has MoE
        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        moe_layers = cfg.get('moe_layers_enum', '')
        if isinstance(moe_layers, str):
            moe_set = set(int(x) for x in moe_layers.split(',') if x.strip())
        else:
            moe_set = set(moe_layers) if moe_layers else set()

        if i in moe_set:
            # Try moe.share_expert first (GPTQ), fallback to share_expert
            # (compressed-tensors)
            prefix = f'{self.attn_layer_prefix}.{i}.moe.share_expert'
            test_key = f'{prefix}.gate_proj.{kind}'
            if self.params.get(test_key) is None:
                prefix = f'{self.attn_layer_prefix}.{i}.share_expert'
        else:
            prefix = f'{self.attn_layer_prefix}.{i}.mlp'

        result = []
        for key in ['gate', 'down', 'up']:
            full_prefix = f'{prefix}.{key}_proj'
            tensor = self.params.get(f'{full_prefix}.{kind}')
            if tensor is None and kind == 'weight':
                # Try dequantizing GPTQ weights to fp16.
                # 8-bit GPTQ (attn/dense MLP) and 4-bit GPTQ (shared expert)
                # are both dequantized here so the converter exports .weight
                # files matching the C++ engine's expectation (weight_type=fp16).
                bits = self._get_bits_for_key(full_prefix)
                if bits == 8:
                    tensor = self._dequant_gptq_8bit(full_prefix)
                elif bits == 4:
                    tensor = self._dequant_gptq_4bit(full_prefix)
            else:
                tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int, kind: str):
        """Override to handle mixed-bit GPTQ key routing.

        When kind=None, reroute ALL GPTQ keys to .weight so get_params()
        selects Weight handler. _ffn() handles dequantization for both
        4-bit (shared expert) and 8-bit (dense MLP) GPTQ weights to fp16,
        matching the C++ engine's expectation (weight_type=float16).
        """
        if not kind:
            keys = self.filter(self.ffn_pattern)
            # Mixed-bit GPTQ: reroute all GPTQ keys to .weight
            if self._extra_config_patterns:
                new_keys = []
                seen = set()
                for k in keys:
                    for suffix in ('.qweight', '.scales', '.qzeros'):
                        if k.endswith(suffix):
                            base = k[:-len(suffix)]
                            fp16_key = base + '.weight'
                            if fp16_key not in seen:
                                new_keys.append(fp16_key)
                                seen.add(fp16_key)
                            break
                    else:
                        new_keys.append(k)
                return new_keys
            return keys
        return self._ffn(i, kind)

    # --- MTP predictor weights ---

    def _mtp_prefix(self, layer_idx: int) -> str:
        """Return the HF weight prefix for MTP top-level weights.

        Step3p5 MTP layers share the main model's model.layers.{i} namespace,
        starting at index num_hidden_layers (45).
        """
        num_hidden = self.model_cfg.get('num_hidden_layers', 45)
        return f'{self.attn_layer_prefix}.{num_hidden + layer_idx}'

    def _mtp_layer_prefix(self, layer_idx: int) -> str:
        """Return the HF weight prefix for MTP decoder layer weights.

        For Step3p5, this is the same as _mtp_prefix since MTP decoder layer
        weights live directly under model.layers.{45+i} (no nested namespace).
        """
        return self._mtp_prefix(layer_idx)

    def _mtp_dequant_if_needed(self, prefix: str, kind: str):
        """Read MTP weight, dequantizing GPTQ to fp16 if needed.

        Layer 45 weights are fp16 (.weight suffix) — read directly.
        Layer 46-47 weights are GPTQ 8-bit (.qweight/.qzeros/.scales) —
        dequantize to fp16 via _dequant_gptq_8bit() or _dequant_gptq_4bit()
        based on the bit-width from extra_config.
        """
        # Try fp16 path first
        tensor = self.params.get(f'{prefix}.{kind}')
        if tensor is not None:
            return self._transform(tensor, kind)
        # Try GPTQ dequantization (Layer 46-47)
        if kind == 'weight':
            # Detect bit-width from extra_config patterns
            bits = self._get_bits_for_key(f'{prefix}.qweight')
            if bits == 8:
                dequant = self._dequant_gptq_8bit(prefix)
                if dequant is not None:
                    return dequant
            dequant = self._dequant_gptq_4bit(prefix)
            if dequant is not None:
                return dequant
        return None

    def mtp_pre_fc_norm_embedding(self, layer_idx: int):
        """Read MTP enorm weight. GemmaRMSNorm: gamma = 1 + weight."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}.enorm.weight'))

    def mtp_pre_fc_norm_hidden(self, layer_idx: int):
        """Read MTP hnorm weight. GemmaRMSNorm: gamma = 1 + weight."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}.hnorm.weight'))

    def mtp_fc(self, layer_idx: int, kind: str):
        """Read MTP eh_proj weight. Layer 45 fp16, Layer 46-47 int4→fp16."""
        prefix = f'{self._mtp_prefix(layer_idx)}.eh_proj'
        return self._mtp_dequant_if_needed(prefix, kind)

    def mtp_final_norm(self, layer_idx: int):
        """Read MTP shared_head norm. GemmaRMSNorm: gamma = 1 + weight.

        In Step3p5, the 'final norm' before lm_head IS the per-layer
        shared_head.norm (not a separate post-FFN norm).
        """
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}'
                '.transformer.shared_head.norm.weight'))

    def mtp_attn(self, layer_idx: int, kind: str):
        """Read MTP decoder layer attention weights (q, k, v, o).

        Does NOT split gated Q — the Attn module handles it during export.
        Layer 45: fp16. Layer 46-47: GPTQ int4 -> fp16.
        """
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.self_attn'
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self._mtp_dequant_if_needed(
                f'{prefix}.{key}_proj', kind)
            result.append(tensor)
        return (*result, )

    def mtp_attn_gate(self, layer_idx: int, kind: str):
        """Read MTP head-wise attention gate (g_proj).

        Layer 45: fp16. Layer 46-47: GPTQ int4 -> fp16.
        """
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.self_attn.g_proj'
        return self._mtp_dequant_if_needed(prefix, kind)

    def mtp_qk_norm(self, layer_idx: int):
        """Read MTP QK norm weights. GemmaRMSNorm. Always fp16."""
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.self_attn'
        result = []
        for x in ['q', 'k']:
            result.append(self._gemma_rms_weight(
                self.params.get(f'{prefix}.{x}_norm.weight')))
        return (*result, )

    def mtp_attn_norm(self, layer_idx: int):
        """Read MTP input_layernorm. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_layer_prefix(layer_idx)}'
                '.input_layernorm.weight'))

    def mtp_ffn_norm(self, layer_idx: int):
        """Read MTP post_attention_layernorm. GemmaRMSNorm."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_layer_prefix(layer_idx)}'
                '.post_attention_layernorm.weight'))

    def mtp_shared_expert(self, layer_idx: int, kind: str):
        """Read MTP dense MLP weights (gate, down, up).

        Layer 45: fp16. Layer 46-47: GPTQ int4 -> fp16.
        """
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.mlp'
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self._mtp_dequant_if_needed(
                f'{prefix}.{key}_proj', kind)
            result.append(tensor)
        return (*result, )

    def mtp_moe_expert(self, layer_idx: int, e: int, kind: str):
        """MTP layers have no MoE experts."""
        return []

    def mtp_moe_gate(self, layer_idx: int, kind: str):
        """MTP layers have no MoE router."""
        return None

    def mtp_shared_expert_gate(self, layer_idx: int):
        """MTP layers have no shared expert gate."""
        return None

    def mtp_shared_head_norm(self, layer_idx: int):
        """Read per-layer shared_head norm weight. GemmaRMSNorm: gamma = 1 +
        weight."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}'
                '.transformer.shared_head.norm.weight'))

    def mtp_shared_head_output(self, layer_idx: int):
        """Read per-layer shared_head output (lm_head) weight. bf16->fp16."""
        tensor = self.params.get(
            f'{self._mtp_prefix(layer_idx)}'
            '.transformer.shared_head.output.weight')
        return self._transform(tensor, 'weight')


@INPUT_MODELS.register_module(name='step3p5')
class Step3p5Model(LlamaModel):
    """Step-3.5-Flash model with mixed attention and MoE."""

    Reader = Step3p5Reader

    def model_info(self):
        cfg = self.model_config
        num_layer = cfg['num_hidden_layers']

        # Save per-layer rope_theta before super().model_info() tries to
        # convert it to float (it's a list in Step3p5 config)
        rope_thetas_raw = cfg.get('rope_theta', [])
        if isinstance(rope_thetas_raw, list):
            # Temporarily set a scalar for the parent class
            cfg['rope_theta'] = 10000.0

        info = super().model_info()

        num_layer = cfg['num_hidden_layers']
        layer_types = cfg.get('layer_types', [])[:num_layer]
        head_dim = cfg.get('head_dim', None) or (
            cfg['hidden_size'] // cfg['num_attention_heads'])

        # Per-layer head counts: full_attention uses default num_attention_heads,
        # sliding_attention uses attention_other_setting.num_attention_heads
        default_heads = cfg.get('num_attention_heads', 64)
        other_setting = cfg.get('attention_other_setting', {})
        other_heads = other_setting.get('num_attention_heads', default_heads)
        head_num_per_layer = [
            default_heads if t == 'full_attention' else other_heads
            for t in layer_types
        ]
        # Use max head_num for buffer allocation
        max_head_num = max(head_num_per_layer) if head_num_per_layer else default_heads
        info['head_num'] = max_head_num

        # kv_head_num: Step3p5 uses num_attention_groups
        kv_head_num = cfg.get('num_attention_groups',
                              cfg.get('num_key_value_heads',
                                      cfg['num_attention_heads']))
        info['kv_head_num'] = kv_head_num

        # Per-layer window_size
        sliding_window = cfg.get('sliding_window', 512)
        window_size = [
            0 if t == 'full_attention' else sliding_window
            for t in layer_types
        ]

        # Per-layer expert_num: layers 0-2 dense, 3-44 MoE
        moe_layers_str = cfg.get('moe_layers_enum', '')
        if isinstance(moe_layers_str, str):
            moe_set = set(
                int(x) for x in moe_layers_str.split(',') if x.strip())
        else:
            moe_set = set(moe_layers_str) if moe_layers_str else set()
        num_experts = cfg.get('moe_num_experts', 0)
        expert_num = [
            num_experts if i in moe_set else 0 for i in range(num_layer)
        ]

        # Per-layer inter_size: dense MLP vs shared expert
        dense_inter = cfg.get('intermediate_size', 0)
        shared_inter = cfg.get('share_expert_dim', 0)
        inter_size = [
            shared_inter if i in moe_set else dense_inter
            for i in range(num_layer)
        ]

        # Per-layer RoPE from config arrays
        rope_thetas = rope_thetas_raw if isinstance(
            rope_thetas_raw, list) else cfg.get('rope_theta', [])
        partial_rotary_factors = cfg.get('partial_rotary_factors', [])
        if isinstance(rope_thetas, list) and len(rope_thetas) >= num_layer:
            per_layer_rope_theta = rope_thetas[:num_layer]
        else:
            # Fallback: derive from layer_types
            base_theta = float(rope_thetas) if not isinstance(
                rope_thetas, list) else 10000.0
            per_layer_rope_theta = [
                5000000.0 if t == 'full_attention' else base_theta
                for t in layer_types
            ]

        if (isinstance(partial_rotary_factors, list)
                and len(partial_rotary_factors) >= num_layer):
            per_layer_rope_dim = [
                int(head_dim * partial_rotary_factors[i])
                for i in range(num_layer)
            ]
        else:
            prf = float(partial_rotary_factors) if not isinstance(
                partial_rotary_factors, list) else 1.0
            per_layer_rope_dim = [int(head_dim * prf)] * num_layer

        # MoE parameters
        moe_top_k = cfg.get('moe_top_k', cfg.get('experts_per_token', 0))
        scoring = cfg.get('moe_router_activation', 'sigmoid')
        fp32_gate = cfg.get('need_fp32_gate', False)

        # Per-layer rope_type: only full_attention layers use llama3 scaling,
        # sliding_attention layers use default RoPE (no scaling).
        # This matches vLLM's yarn_only_types=['full_attention'] behavior.
        yarn_only_types = cfg.get('yarn_only_types', [])
        per_layer_rope_type = []
        for t in layer_types:
            if yarn_only_types and t not in yarn_only_types:
                per_layer_rope_type.append('default')
            else:
                rope_scaling = cfg.get('rope_scaling', {})
                per_layer_rope_type.append(
                    rope_scaling.get('rope_type',
                                     rope_scaling.get('type', 'default')))

        # Per-layer FFN weight type: shared expert is fp16 for both
        # compressed-tensors (AWQ, natively fp16) and GPTQ/AutoRound
        # (8-bit dequantized to fp16 in Reader).  Only MoE experts are int4.
        # Empty string means "use global weight_type".
        ffn_weight_types = [
            'float16' if i in moe_set else ''
            for i in range(num_layer)
        ]

        disable_attn_gate = os.getenv('TM_STEP3P5_DISABLE_ATTN_GATE',
                                      '0') not in ('', '0')
        use_head_wise_attn_gate = bool(cfg.get('use_head_wise_attn_gate',
                                               False)) and not disable_attn_gate

        info.update(
            # Attention features
            qk_norm=True,
            qk_norm_type='per_head',
            # Keep gate behavior configurable from HF config / overrides.
            attn_output_gate=use_head_wise_attn_gate,
            attn_output_gate_per_head=use_head_wise_attn_gate,
            # Per-layer head counts
            head_num_per_layer=head_num_per_layer,
            # Mixed attention layer types
            layer_types=layer_types,
            window_size=window_size,
            # MoE parameters
            expert_num=expert_num,
            expert_inter_size=cfg.get('moe_intermediate_size', 0),
            experts_per_token=moe_top_k,
            scoring_func=scoring,
            fp32_gate=fp32_gate,
            expert_router_bias=cfg.get('use_moe_router_bias', False),
            norm_topk_prob=cfg.get('norm_expert_weight', True),
            moe_shared_gate=False,
            routed_scale=cfg.get('moe_router_scaling_factor', 1.0),
            inter_size=inter_size,
            # Per-layer FFN weight type (shared expert fp16)
            ffn_weight_types=ffn_weight_types,
            # Activation: Step3p5 uses per-layer SwiGLU clamping via swiglu_limits/swiglu_limits_shared
            # No need for a global 'swiglustep' activation type.
            activation_type='',
            # Per-layer RoPE (new capability)
            rope_theta=per_layer_rope_theta,
            rope_dim=per_layer_rope_dim,
            rope_type=per_layer_rope_type,
            swiglu_limits=cfg.get('swiglu_limits', []),
            swiglu_limits_shared=cfg.get('swiglu_limits_shared', []),
        )

        # Set base rope_param: use sliding layer defaults (most common)
        # The per-layer arrays override this for each layer
        rope_scaling = cfg.get('rope_scaling', {})
        if rope_scaling:
            info['rope_param'].type = rope_scaling.get(
                'rope_type', rope_scaling.get('type', 'llama3'))
            info['rope_param'].factor = rope_scaling.get('factor', 1.0)
            info['rope_param'].low_freq_factor = rope_scaling.get(
                'low_freq_factor', 1.0)
            info['rope_param'].high_freq_factor = rope_scaling.get(
                'high_freq_factor', 1.0)
            info['rope_param'].original_max_position_embeddings = \
                rope_scaling.get('original_max_position_embeddings', 0)
        # Base dim = most common rope_dim (sliding layers)
        info['rope_param'].dim = head_dim

        # MTP layer detection and config
        num_nextn = cfg.get('num_nextn_predict_layers', 0)
        if not num_nextn:
            # Fallback: layer_types length - num_hidden_layers
            all_layer_types = cfg.get('layer_types', [])
            if len(all_layer_types) > num_layer:
                num_nextn = len(all_layer_types) - num_layer

        if num_nextn > 0:
            info['num_mtp_layers'] = num_nextn
            info['num_draft_tokens'] = num_nextn  # 3 layers each run once
            info['mtp_expert_weight_type'] = 'fp16'  # MTP dense MLP, Reader dequants to fp16
            info['mtp_has_shared_head'] = True  # Step3p5 MTP has per-layer shared_head

            # Extend per-layer arrays to include MTP layers (length 48 = 45 + 3)
            # MTP layers are sliding_attention + dense MLP
            all_layer_types_raw = cfg.get('layer_types', [])
            mtp_layer_types_raw = all_layer_types_raw[num_layer:num_layer + num_nextn]

            for t in mtp_layer_types_raw:
                info['layer_types'].append(t)  # raw string from config
                info['head_num_per_layer'].append(other_heads)  # 96
                info['window_size'].append(sliding_window)  # 512
                info['expert_num'].append(0)  # dense MLP, no MoE
                info['inter_size'].append(dense_inter)  # 11264
                info['rope_theta'].append(10000.0)
                info['rope_dim'].append(int(head_dim * 1.0))  # 128
                info['ffn_weight_types'].append('float16')
                info['rope_type'].append('default')

            # Extend swiglu_limits arrays if needed (MTP layers have no clamping)
            # Config may already include MTP entries (length 48), only extend if shorter
            total_layers = num_layer + num_nextn
            swiglu = info.get('swiglu_limits', [])
            swiglu_shared = info.get('swiglu_limits_shared', [])
            if isinstance(swiglu, list) and len(swiglu) < total_layers:
                swiglu.extend([0.0] * (total_layers - len(swiglu)))
            if isinstance(swiglu_shared, list) and len(swiglu_shared) < total_layers:
                swiglu_shared.extend([0.0] * (total_layers - len(swiglu_shared)))
            info['swiglu_limits'] = swiglu
            info['swiglu_limits_shared'] = swiglu_shared

        return info
