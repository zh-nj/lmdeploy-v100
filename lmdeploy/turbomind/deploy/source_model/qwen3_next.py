# Copyright (c) OpenMMLab. All rights reserved.

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class Qwen3NextReader(LlamaReader):
    """Reader for Qwen3-Coder-Next mixed attention model.

    Handles both full attention layers (self_attn.*) and
    linear attention layers (linear_attn.*), plus MoE weights.
    """

    # Override ffn_pattern to only match shared_expert (not MoE experts)
    ffn_pattern = r'shared_expert\.'

    # Qwen3-Next uses GemmaRMSNorm semantics: y = norm(x) * (1 + w).
    # TurboMind RMSNorm uses y = norm(x) * w, so export gamma = 1 + w.
    def _gemma_rms_weight(self, tensor):
        tensor = self.transform(tensor, 'weight')
        if tensor is None:
            return None
        return tensor + 1

    # --- linear attention weights ---

    def linear_attn_in_proj_qkvz(self, i, kind):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.in_proj_qkvz.{kind}')

    def linear_attn_in_proj_ba(self, i, kind):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.in_proj_ba.{kind}')

    def linear_attn_conv1d(self, i, kind):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.conv1d.{kind}')

    def linear_attn_out_proj(self, i, kind):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.out_proj.{kind}')

    def linear_attn_a_log(self, i):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.A_log')

    def linear_attn_dt_bias(self, i):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.dt_bias')

    def linear_attn_norm(self, i):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.norm.weight')

    def norm_weight(self):
        return self._gemma_rms_weight(self.params.get(self.norm_weight_key, None))

    def attn_norm(self, i: int):
        return self._gemma_rms_weight(
            self.params.get(f'{self.attn_layer_prefix}.{i}.input_layernorm.weight'))

    def ffn_norm(self, i: int):
        return self._gemma_rms_weight(
            self.params.get(f'{self.attn_layer_prefix}.{i}.post_attention_layernorm.weight'))

    # --- full attention: override _attn to split gated Q ---

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o for layer i, splitting gated Q into Q and gate.

        HF q_proj has shape (num_heads * head_dim * 2, hidden_size).
        The output is interleaved per-head: [q0, gate0, q1, gate1, ...].
        We split into Q (num_heads * head_dim, hidden) and
        gate (num_heads * head_dim, hidden).
        Only Q is returned in the qkvo tuple; gate is stored separately.
        """
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'{self.attn_layer_prefix}.{i}.self_attn.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        q, k, v, o = result

        # Split gated Q: q_proj weight is (num_heads * head_dim * 2, hidden)
        # Per-head layout: each head has head_dim*2 output dims (Q + gate)
        if q is not None:
            cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
            head_dim = cfg.get('head_dim', None) or 256  # Qwen3-Next default
            num_heads = q.shape[0] // (head_dim * 2)
            # Reshape to (num_heads, head_dim*2, ...) and split
            if q.dim() == 2:
                # weight: (num_heads * head_dim * 2, hidden)
                q_gate = q.view(num_heads, head_dim * 2, -1)
                q_only = q_gate[:, :head_dim, :].reshape(
                    num_heads * head_dim, -1)
                gate = q_gate[:, head_dim:, :].reshape(
                    num_heads * head_dim, -1)
            else:
                # bias: (num_heads * head_dim * 2,)
                q_gate = q.view(num_heads, head_dim * 2)
                q_only = q_gate[:, :head_dim].reshape(-1)
                gate = q_gate[:, head_dim:].reshape(-1)

            # Store gate for later export by Attn.apply's attn_gate(i, 'weight').
            # For quantized models (kind='weight_packed'), dequantize the gate
            # to fp16 since C++ allocates w_gate as fp16 (data_type, not weight_type).
            # IMPORTANT: Only store gate for 'weight' and 'weight_packed' kinds.
            # CompressedWeight also calls _attn with 'weight_scale' which would
            # overwrite the dequantized gate with the scale tensor.
            if not hasattr(self, '_attn_gates'):
                self._attn_gates = {}
            if kind == 'weight_packed':
                # Gate is uint8 (4-bit unpacked after process_compressed_tensor).
                # Dequantize to fp16 using the q_proj scale.
                # CRITICAL: Must dequantize the FULL Q+gate weight first, then
                # split. The scale rows are aligned to the original interleaved
                # per-head layout [Q0,gate0,Q1,gate1,...], so splitting data
                # first and using contiguous scale rows gives wrong results.
                import torch
                scale = self.params.get(
                    f'{self.attn_layer_prefix}.{i}.self_attn.q_proj.weight_scale')
                if scale is not None:
                    scale = scale.cuda().half()
                    quant_cfg = cfg.get('quantization_config', {})
                    group_size = 32
                    for gc in quant_cfg.get('config_groups', {}).values():
                        w = gc.get('weights', {})
                        if w.get('group_size'):
                            group_size = w['group_size']
                            break
                    # Reconstruct full Q+gate from q_gate (still in
                    # interleaved per-head layout before flatten)
                    full_qg = q_gate.reshape(
                        num_heads * head_dim * 2, -1)
                    # Dequantize full tensor
                    full_f = full_qg.cuda().to(torch.float16) - 8.0
                    scale_exp = scale.repeat_interleave(group_size, dim=1)
                    if scale_exp.shape[1] > full_f.shape[1]:
                        scale_exp = scale_exp[:, :full_f.shape[1]]
                    full_fp16 = full_f * scale_exp
                    # Now split from dequantized tensor
                    fp16_qg = full_fp16.view(num_heads, head_dim * 2, -1)
                    gate_fp16 = fp16_qg[:, head_dim:, :].reshape(
                        num_heads * head_dim, -1).cpu()
                    self._attn_gates[(i, 'weight')] = gate_fp16
                else:
                    self._attn_gates[(i, 'weight')] = gate
            elif kind == 'weight':
                self._attn_gates[(i, 'weight')] = gate
            q = q_only

        return (q, k, v, o)

    def attn_gate(self, i: int, kind: str):
        """Get the stored attention gate weight for layer i."""
        if not hasattr(self, '_attn_gates'):
            return None
        return self._attn_gates.get((i, kind))

    # --- full attention QK norm ---

    def qk_norm(self, i: int):
        result = []
        for x in ['q', 'k']:
            name = f'{self.attn_layer_prefix}.{i}.self_attn.{x}_norm.weight'
            result.append(self._gemma_rms_weight(self.params.get(name)))
        return (*result, )

    # --- MoE weights ---

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts')
        result = []
        for key in ['gate', 'down', 'up']:
            name = (f'{self.attn_layer_prefix}.{i}'
                    f'.mlp.experts.{e}.{key}_proj.{kind}')
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i, kind):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.mlp.gate.{kind}')

    def _ffn(self, i: int, kind: str):
        """Get shared_expert weights for layer i."""
        if not kind:
            return self.filter(r'shared_expert\.')
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params.get(
                f'{self.attn_layer_prefix}.{i}'
                f'.mlp.shared_expert.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_shared_gate(self, i):
        return self.params.get(
            f'{self.attn_layer_prefix}.{i}.mlp.shared_expert_gate.weight')

    # --- MTP predictor weights ---

    def _mtp_prefix(self, layer_idx: int) -> str:
        """Return the HF weight prefix for MTP top-level weights."""
        return 'mtp'

    def _mtp_layer_prefix(self, layer_idx: int) -> str:
        """Return the HF weight prefix for MTP decoder layer weights."""
        return f'mtp.layers.{layer_idx}'

    def mtp_pre_fc_norm_embedding(self, layer_idx: int):
        """Read pre_fc_norm for embedding branch. GemmaRMSNorm: weight + 1."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}.pre_fc_norm_embedding.weight'))

    def mtp_pre_fc_norm_hidden(self, layer_idx: int):
        """Read pre_fc_norm for hidden_states branch. GemmaRMSNorm: weight + 1."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}.pre_fc_norm_hidden.weight'))

    def mtp_fc(self, layer_idx: int, kind: str):
        """Read fc linear weight (hidden*2 -> hidden)."""
        tensor = self.params.get(
            f'{self._mtp_prefix(layer_idx)}.fc.{kind}')
        return self.transform(tensor, kind)

    def mtp_final_norm(self, layer_idx: int):
        """Read final RMSNorm weight. GemmaRMSNorm: weight + 1."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_prefix(layer_idx)}.norm.weight'))

    def mtp_attn(self, layer_idx: int, kind: str):
        """Read MTP decoder layer attention weights (q, k, v, o).

        Handles gated Q split — same as main model full attention.
        The q_proj output is (num_heads * head_dim * 2, hidden_size),
        front half is Q, back half is gate.

        When kind='weight' but the actual data is compressed-tensors
        (weight_packed/weight_scale), dequantize to fp16 to match the
        C++ weight_type allocation (which uses the global weight_type,
        typically fp16 for Qwen3-Next/Qwen3.5 where main model attention
        is in the quantization ignore list).
        """
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.self_attn'
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(f'{prefix}.{key}_proj.{kind}')
            if tensor is None and kind == 'weight':
                # Fallback: dequantize compressed-tensors to fp16
                tensor = self._dequant_mtp_attn_weight(prefix, key)
            else:
                tensor = self.transform(tensor, kind)
            result.append(tensor)
        q, k, v, o = result

        # Split gated Q (same logic as _attn for main model)
        if q is not None:
            cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
            head_dim = cfg.get('head_dim', None) or 256
            num_heads = q.shape[0] // (head_dim * 2)
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
            if kind == 'weight_packed':
                # Same fix as _attn: dequant full Q+gate first, then split.
                # Scale rows are aligned to the interleaved per-head layout.
                import torch
                scale = self.params.get(
                    f'{prefix}.q_proj.weight_scale')
                if scale is not None:
                    scale = scale.cuda().half()
                    quant_cfg = cfg.get('quantization_config', {})
                    group_size = 32
                    for gc in quant_cfg.get('config_groups', {}).values():
                        w = gc.get('weights', {})
                        if w.get('group_size'):
                            group_size = w['group_size']
                            break
                    # Use q_gate (still in interleaved per-head layout)
                    full_qg = q_gate.reshape(
                        num_heads * head_dim * 2, -1)
                    # Dequantize full tensor
                    full_f = full_qg.cuda().to(torch.float16) - 8.0
                    scale_exp = scale.repeat_interleave(group_size, dim=1)
                    if scale_exp.shape[1] > full_f.shape[1]:
                        scale_exp = scale_exp[:, :full_f.shape[1]]
                    full_fp16 = full_f * scale_exp
                    # Split from dequantized tensor
                    fp16_qg = full_fp16.view(num_heads, head_dim * 2, -1)
                    gate_fp16 = fp16_qg[:, head_dim:, :].reshape(
                        num_heads * head_dim, -1).cpu()
                    self._mtp_attn_gates[(layer_idx, 'weight')] = gate_fp16
                else:
                    self._mtp_attn_gates[(layer_idx, 'weight')] = gate
            elif kind == 'weight':
                self._mtp_attn_gates[(layer_idx, 'weight')] = gate
            q = q_only

        return (q, k, v, o)

    def _dequant_mtp_attn_weight(self, prefix, key):
        """Dequantize compressed-tensors MTP attention weight to fp16.

        compressed-tensors format: weight_packed (int32, 8×4-bit per int32),
        weight_scale (fp16/bf16, per-group), symmetric with zero_point=8.
        """
        import torch
        packed = self.params.get(f'{prefix}.{key}_proj.weight_packed')
        scale = self.params.get(f'{prefix}.{key}_proj.weight_scale')
        if packed is None or scale is None:
            return None

        packed = packed.cuda()
        scale = scale.cuda().half()

        # Unpack int32 → 8 × uint4 values per element
        # packed shape: (out_dim, in_dim // 8)
        out_dim = packed.shape[0]
        packed_cols = packed.shape[1]
        in_dim = packed_cols * 8

        # Extract 4-bit values
        vals = torch.zeros(out_dim, in_dim, dtype=torch.uint8,
                           device=packed.device)
        for j in range(8):
            vals[:, j::8] = ((packed >> (4 * j)) & 0xF).to(torch.uint8)

        # Dequantize: value = (int4_val - 8) * scale
        # scale shape: (out_dim, in_dim // group_size)
        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        quant_cfg = cfg.get('quantization_config', {})
        group_size = 32  # default for this model
        for gc in quant_cfg.get('config_groups', {}).values():
            w = gc.get('weights', {})
            if w.get('group_size'):
                group_size = w['group_size']
                break

        vals_f = vals.to(torch.float16) - 8.0
        # Expand scale to match vals shape
        # scale: (out_dim, in_dim // group_size) → repeat each column group_size times
        scale_expanded = scale.repeat_interleave(group_size, dim=1)
        if scale_expanded.shape[1] > in_dim:
            scale_expanded = scale_expanded[:, :in_dim]
        weight = vals_f * scale_expanded

        return weight.cpu()

    def mtp_attn_gate(self, layer_idx: int, kind: str):
        """Get the stored MTP attention gate weight."""
        if not hasattr(self, '_mtp_attn_gates'):
            return None
        return self._mtp_attn_gates.get((layer_idx, kind))

    def mtp_qk_norm(self, layer_idx: int):
        """Get MTP decoder layer QK norm weights."""
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.self_attn'
        result = []
        for x in ['q', 'k']:
            result.append(self._gemma_rms_weight(
                self.params.get(f'{prefix}.{x}_norm.weight')))
        return (*result, )

    def mtp_attn_norm(self, layer_idx: int):
        """Get MTP decoder layer input_layernorm. GemmaRMSNorm: weight + 1."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_layer_prefix(layer_idx)}.input_layernorm.weight'))

    def mtp_ffn_norm(self, layer_idx: int):
        """Get MTP decoder layer post_attention_layernorm. GemmaRMSNorm: weight + 1."""
        return self._gemma_rms_weight(
            self.params.get(
                f'{self._mtp_layer_prefix(layer_idx)}.post_attention_layernorm.weight'))

    def mtp_moe_expert(self, layer_idx: int, e: int, kind: str):
        """Read MTP decoder layer MoE expert weights (gate, down, up)."""
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.mlp'
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params.get(
                f'{prefix}.experts.{e}.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def mtp_moe_gate(self, layer_idx: int, kind: str):
        """Read MTP decoder layer MoE router weight."""
        return self.params.get(
            f'{self._mtp_layer_prefix(layer_idx)}.mlp.gate.{kind}')

    def mtp_shared_expert(self, layer_idx: int, kind: str):
        """Read MTP decoder layer shared expert weights (gate, down, up)."""
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.mlp'
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params.get(
                f'{prefix}.shared_expert.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def mtp_shared_expert_gate(self, layer_idx: int):
        """Read MTP decoder layer shared expert gate weight."""
        return self.params.get(
            f'{self._mtp_layer_prefix(layer_idx)}.mlp.shared_expert_gate.weight')



@INPUT_MODELS.register_module(name='qwen3-coder-next')
class Qwen3NextModel(LlamaModel):
    """Qwen3-Coder-Next model with mixed attention and MoE."""

    Reader = Qwen3NextReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info.update(
            # full attention QK norm
            qk_norm=True,
            # gated attention output
            attn_output_gate=True,
            # MoE parameters
            expert_num=cfg['num_experts'],
            expert_inter_size=cfg['moe_intermediate_size'],
            experts_per_token=cfg['num_experts_per_tok'],
            inter_size=cfg['shared_expert_intermediate_size'],
            moe_shared_gate=True,
            norm_topk_prob=cfg.get('norm_topk_prob', True),
            # Qwen3-Next uses softmax router in HF/vLLM by default.
            scoring_func=cfg.get('scoring_func', 'softmax'),
            expert_router_bias=cfg.get('expert_router_bias', False),
            # layer types for mixed attention dispatch
            layer_types=cfg['layer_types'],
            # GatedDeltaNet linear attention parameters
            linear_key_head_dim=cfg['linear_key_head_dim'],
            linear_value_head_dim=cfg['linear_value_head_dim'],
            linear_num_key_heads=cfg['linear_num_key_heads'],
            linear_num_value_heads=cfg['linear_num_value_heads'],
            linear_conv_kernel_dim=cfg['linear_conv_kernel_dim'],
        )
        # partial rotary for full attention layers
        head_dim = cfg.get('head_dim', None) or (
            cfg['hidden_size'] // cfg['num_attention_heads'])
        partial_rotary_factor = cfg.get('partial_rotary_factor', 1.0)
        rotary_dim = int(head_dim * partial_rotary_factor)
        info['rope_param'].dim = rotary_dim

        # MTP speculative decoding
        num_mtp = cfg.get('num_nextn_predict_layers', 0)
        if not num_mtp:
            # Fallback: detect MTP layers from safetensors weight index
            import re
            mtp_layer_indices = set()
            try:
                import os
                import json as _json
                idx_path = os.path.join(self.model_path,
                                        'model.safetensors.index.json')
                if os.path.exists(idx_path):
                    with open(idx_path) as f:
                        idx = _json.load(f)
                    for key in idx.get('weight_map', {}):
                        m = re.match(r'mtp\.layers\.(\d+)\.', key)
                        if m:
                            mtp_layer_indices.add(int(m.group(1)))
            except Exception:
                pass
            if mtp_layer_indices:
                num_mtp = max(mtp_layer_indices) + 1
        if num_mtp and num_mtp > 0:
            info['num_mtp_layers'] = num_mtp
            # Detect MTP expert weight type from actual key suffixes
            try:
                import os as _os
                import json as _json2
                idx_path = _os.path.join(self.model_path,
                                         'model.safetensors.index.json')
                if _os.path.exists(idx_path):
                    with open(idx_path) as f:
                        idx2 = _json2.load(f)
                    mtp_expert_keys = [
                        k for k in idx2.get('weight_map', {})
                        if 'mtp' in k and 'experts' in k
                        and 'shared_expert' not in k
                    ]
                    has_fp16 = any(
                        k.endswith('.weight') for k in mtp_expert_keys)
                    has_packed = any(
                        k.endswith('.weight_packed')
                        for k in mtp_expert_keys)
                    if has_fp16 and not has_packed:
                        info['mtp_expert_weight_type'] = 'fp16'
                    # else: leave empty = inherit from ref layer
            except Exception:
                pass

        return info


class Qwen3_5VLMoeReader(Qwen3NextReader):
    """Reader for Qwen3.5 VL MoE - remaps model.language_model.layers.{i}
    prefix for VL models.

    Key difference from Qwen3NextReader: the VL model uses separate
    in_proj_qkv + in_proj_z + in_proj_a + in_proj_b (flat layout)
    instead of combined in_proj_qkvz + in_proj_ba (grouped layout).
    This reader overrides the linear_attn methods to cat the separate
    tensors and convert from flat to grouped layout so the existing
    LinearAttn converter module works unchanged.
    """

    attn_layer_prefix = 'model.language_model.layers'
    attn_layer_patten = r'model\.language_model\.layers\.([0-9]+).'
    tok_embeddings_key = 'model.language_model.embed_tokens.weight'
    norm_weight_key = 'model.language_model.norm.weight'
    output_weight_key = 'lm_head.weight'

    def __init__(self, new_params, unused_params, last_bin, model_cfg,
                 **kwargs):
        model_cfg = model_cfg.get('text_config', model_cfg)
        super().__init__(new_params, unused_params, last_bin, model_cfg,
                         **kwargs)

    def _flat_to_grouped_qkvz(self, w_qkv, w_z):
        """Convert flat [Q,K,V] + [Z] to grouped layout for _reorder_qkvz.

        Flat layout (output dim):
          w_qkv: [Q_all, K_all, V_all] = [nk*hk + nk*hk + nv*hv]
          w_z:   [Z_all] = [nv*hv]

        Grouped layout (output dim):
          [num_k_heads, (Qk + Kk + Vr*v + Zr*v)]

        Returns tensor with shape (grouped_dim, hidden) matching the
        format that Qwen3NextReader.linear_attn_in_proj_qkvz would return.
        """
        import torch
        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        nk = cfg.get('linear_num_key_heads', 16)
        hk = cfg.get('linear_key_head_dim', 128)
        nv = cfg.get('linear_num_value_heads', 64)
        hv = cfg.get('linear_value_head_dim', 128)
        r = nv // nk  # kv_ratio

        key_dim = nk * hk
        value_dim = nv * hv

        # w_qkv shape: (key_dim*2 + value_dim, hidden) = (Q+K+V, hidden)
        # w_z shape: (value_dim, hidden) = (Z, hidden)
        q = w_qkv[:key_dim]           # (nk*hk, hidden)
        k = w_qkv[key_dim:2*key_dim]  # (nk*hk, hidden)
        v = w_qkv[2*key_dim:]         # (nv*hv, hidden)
        z = w_z                        # (nv*hv, hidden)

        hidden = q.shape[1]

        # Reshape to per-group: [nk, per_head_dim, hidden]
        q = q.reshape(nk, hk, hidden)
        k = k.reshape(nk, hk, hidden)
        v = v.reshape(nk, r * hv, hidden)
        z = z.reshape(nk, r * hv, hidden)

        # Cat per group: [nk, hk+hk+r*hv+r*hv, hidden]
        grouped = torch.cat([q, k, v, z], dim=1)
        # Flatten: [nk * group_size, hidden]
        return grouped.reshape(-1, hidden)

    def _flat_to_grouped_ba(self, w_b, w_a):
        """Convert flat [b] + [a] to grouped layout for _reorder_ba.

        Flat layout (output dim):
          w_b: [b_all] = [num_value_heads]
          w_a: [a_all] = [num_value_heads]

        Grouped layout (output dim):
          [num_k_heads, (b_ratio + a_ratio)]

        Returns tensor with shape (grouped_dim, hidden).
        """
        import torch
        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        nk = cfg.get('linear_num_key_heads', 16)
        nv = cfg.get('linear_num_value_heads', 64)
        r = nv // nk

        hidden = w_b.shape[1]

        # Reshape to per-group: [nk, r, hidden]
        b = w_b.reshape(nk, r, hidden)
        a = w_a.reshape(nk, r, hidden)

        # Cat per group: [nk, 2*r, hidden]
        grouped = torch.cat([b, a], dim=1)
        # Flatten: [nk * 2 * r, hidden]
        return grouped.reshape(-1, hidden)

    def linear_attn_in_proj_qkvz(self, i, kind):
        """Override: cat separate in_proj_qkv + in_proj_z into grouped
        layout."""
        w_qkv = self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.in_proj_qkv.{kind}')
        w_z = self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.in_proj_z.{kind}')
        if w_qkv is None or w_z is None:
            # Fall back to combined key (in case model uses grouped layout)
            return super().linear_attn_in_proj_qkvz(i, kind)
        return self._flat_to_grouped_qkvz(w_qkv, w_z)

    def linear_attn_in_proj_ba(self, i, kind):
        """Override: cat separate in_proj_b + in_proj_a into grouped
        layout."""
        w_b = self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.in_proj_b.{kind}')
        w_a = self.params.get(
            f'{self.attn_layer_prefix}.{i}.linear_attn.in_proj_a.{kind}')
        if w_b is None or w_a is None:
            # Fall back to combined key (in case model uses grouped layout)
            return super().linear_attn_in_proj_ba(i, kind)
        return self._flat_to_grouped_ba(w_b, w_a)


@INPUT_MODELS.register_module(name='qwen3-5-vl-moe')
class Qwen3_5VLMoeModel(Qwen3NextModel):
    """Qwen3.5 VL MoE model - VL wrapper around qwen3-coder-next text
    backbone."""

    Reader = Qwen3_5VLMoeReader

    def model_info(self):
        self.model_config = self.model_config.get('text_config',
                                                  self.model_config)
        cfg = self.model_config

        # Ensure intermediate_size exists — MoE models may not have it,
        # but LlamaModel.model_info() requires it. Qwen3NextModel.model_info
        # overrides inter_size with shared_expert_intermediate_size anyway.
        if cfg.get('intermediate_size') is None:
            cfg['intermediate_size'] = \
                cfg.get('shared_expert_intermediate_size', 0)

        # Extract rope config from rope_parameters if standard fields are
        # None. Qwen3.5 VL stores rope config in text_config.rope_parameters
        # instead of the standard rope_theta/rope_scaling/partial_rotary_factor
        # top-level fields.
        rp = cfg.get('rope_parameters', {})
        if rp:
            if cfg.get('rope_theta') is None and 'rope_theta' in rp:
                cfg['rope_theta'] = rp['rope_theta']
            if cfg.get('partial_rotary_factor') is None \
                    and 'partial_rotary_factor' in rp:
                cfg['partial_rotary_factor'] = rp['partial_rotary_factor']
            # Build rope_scaling from rope_parameters if not present
            if cfg.get('rope_scaling') is None \
                    and rp.get('mrope_section') is not None:
                cfg['rope_scaling'] = {
                    'type': rp.get('rope_type', 'default'),
                    'mrope_section': rp['mrope_section'],
                }

        # Fallback defaults if still None
        if cfg.get('rope_theta') is None:
            cfg['rope_theta'] = 10000000.0
        if cfg.get('partial_rotary_factor') is None:
            cfg['partial_rotary_factor'] = 1.0

        return super().model_info()


class Qwen3_5DenseReader(Qwen3_5VLMoeReader):
    """Reader for Qwen3.5 dense (non-MoE) models.

    Inherits VL prefix handling and flat→grouped GDN layout conversion
    from Qwen3_5VLMoeReader. Overrides _ffn to read standard dense MLP
    (mlp.gate_proj/up_proj/down_proj) instead of shared_expert.
    Also overrides MoE methods to return None/empty since there are no
    experts, and MTP FFN methods for dense MTP layers.

    Key difference from 122B MoE: linear_attn weights (in_proj_qkv,
    in_proj_z, out_proj) are compressed-tensors quantized, not fp16.
    LinearAttn.apply hardcodes kind='weight', so we dequantize to fp16
    in the reader when the actual data is weight_packed.
    """

    # Override: dense model uses standard mlp pattern, not shared_expert
    ffn_pattern = r'mlp'

    def _dequant_compressed_tensor(self, key_prefix):
        """Dequantize a compressed-tensors weight to fp16.

        Used for linear_attn weights that are quantized but need to be
        returned as fp16 because LinearAttn.apply only handles 'weight' kind.
        """
        import torch
        packed = self.params.get(f'{key_prefix}.weight_packed')
        scale = self.params.get(f'{key_prefix}.weight_scale')
        if packed is None or scale is None:
            return None

        packed = packed.cuda()
        scale = scale.cuda().half()

        # Unpack int32 → 8 × uint4 values per element
        out_dim = packed.shape[0]
        packed_cols = packed.shape[1]
        in_dim = packed_cols * 8

        vals = torch.zeros(out_dim, in_dim, dtype=torch.uint8,
                           device=packed.device)
        for j in range(8):
            vals[:, j::8] = ((packed >> (4 * j)) & 0xF).to(torch.uint8)

        # Dequantize: value = (int4_val - 8) * scale (symmetric, zero_point=8)
        cfg = self.model_cfg if hasattr(self, 'model_cfg') else {}
        quant_cfg = cfg.get('quantization_config', {})
        group_size = 32
        for gc in quant_cfg.get('config_groups', {}).values():
            w = gc.get('weights', {})
            if w.get('group_size'):
                group_size = w['group_size']
                break

        vals_f = vals.to(torch.float16) - 8.0
        scale_expanded = scale.repeat_interleave(group_size, dim=1)
        if scale_expanded.shape[1] > in_dim:
            scale_expanded = scale_expanded[:, :in_dim]
        weight = vals_f * scale_expanded
        return weight.cpu()

    def linear_attn_in_proj_qkvz(self, i, kind):
        """Override: handle compressed-tensors dequantization for in_proj_qkv
        and in_proj_z, then do flat→grouped conversion."""
        prefix = self.attn_layer_prefix
        w_qkv = self.params.get(
            f'{prefix}.{i}.linear_attn.in_proj_qkv.{kind}')
        w_z = self.params.get(
            f'{prefix}.{i}.linear_attn.in_proj_z.{kind}')

        # If kind='weight' but actual data is weight_packed, dequantize
        if w_qkv is None and kind == 'weight':
            w_qkv = self._dequant_compressed_tensor(
                f'{prefix}.{i}.linear_attn.in_proj_qkv')
        if w_z is None and kind == 'weight':
            w_z = self._dequant_compressed_tensor(
                f'{prefix}.{i}.linear_attn.in_proj_z')

        if w_qkv is None or w_z is None:
            return super(Qwen3_5VLMoeReader, self)\
                .linear_attn_in_proj_qkvz(i, kind)
        return self._flat_to_grouped_qkvz(w_qkv, w_z)

    def linear_attn_out_proj(self, i, kind):
        """Override: handle compressed-tensors dequantization for out_proj."""
        prefix = self.attn_layer_prefix
        w = self.params.get(
            f'{prefix}.{i}.linear_attn.out_proj.{kind}')
        if w is None and kind == 'weight':
            w = self._dequant_compressed_tensor(
                f'{prefix}.{i}.linear_attn.out_proj')
        if w is not None and kind != 'weight':
            w = self.transform(w, kind)
        return w

    def _ffn(self, i: int, kind: str):
        """Read standard dense MLP weights (not shared_expert)."""
        if not kind:
            return self.filter(r'mlp')
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params.get(
                f'{self.attn_layer_prefix}.{i}.mlp.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        """No MoE experts in dense model."""
        if not kind:
            return []
        return None, None, None

    def moe_ffn_gate(self, i, kind):
        """No MoE gate in dense model."""
        return None

    def moe_ffn_shared_gate(self, i):
        """No shared expert gate in dense model."""
        return None

    def mtp_moe_expert(self, layer_idx: int, e: int, kind: str):
        """No MoE experts in dense MTP layer."""
        return None, None, None

    def mtp_moe_gate(self, layer_idx: int, kind: str):
        """No MoE gate in dense MTP layer."""
        return None

    def mtp_shared_expert(self, layer_idx: int, kind: str):
        """Read MTP dense FFN weights (mlp.gate/down/up_proj).

        For dense models, MTP FFN may be compressed-tensors int4.
        When kind='weight' but actual data is weight_packed, dequantize
        to fp16 (same pattern as linear_attn weights in lesson 80).
        """
        prefix = f'{self._mtp_layer_prefix(layer_idx)}.mlp'
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params.get(f'{prefix}.{key}_proj.{kind}')
            if tensor is None and kind == 'weight':
                # Fallback: dequantize compressed-tensors to fp16
                tensor = self._dequant_compressed_tensor(
                    f'{prefix}.{key}_proj')
            else:
                tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def mtp_shared_expert_gate(self, layer_idx: int):
        """No shared expert gate in dense MTP layer."""
        return None


@INPUT_MODELS.register_module(name='qwen3-5-vl-dense')
class Qwen3_5DenseModel(Qwen3_5VLMoeModel):
    """Qwen3.5 dense (non-MoE) model with GDN + full attention."""

    Reader = Qwen3_5DenseReader

    def model_info(self):
        self.model_config = self.model_config.get('text_config',
                                                  self.model_config)
        cfg = self.model_config

        # Extract rope config from rope_parameters (same as VL MoE parent)
        rp = cfg.get('rope_parameters', {})
        if rp:
            if cfg.get('rope_theta') is None and 'rope_theta' in rp:
                cfg['rope_theta'] = rp['rope_theta']
            if cfg.get('partial_rotary_factor') is None \
                    and 'partial_rotary_factor' in rp:
                cfg['partial_rotary_factor'] = rp['partial_rotary_factor']
            if cfg.get('rope_scaling') is None \
                    and rp.get('mrope_section') is not None:
                cfg['rope_scaling'] = {
                    'type': rp.get('rope_type', 'default'),
                    'mrope_section': rp['mrope_section'],
                }

        if cfg.get('rope_theta') is None:
            cfg['rope_theta'] = 10000000.0
        if cfg.get('partial_rotary_factor') is None:
            cfg['partial_rotary_factor'] = 1.0

        # Call LlamaModel.model_info (skip Qwen3NextModel which sets MoE params)
        info = LlamaModel.model_info(self)

        # Add GDN + mixed attention params (same as Qwen3NextModel but no MoE)
        info.update(
            qk_norm=True,
            attn_output_gate=True,
            layer_types=cfg['layer_types'],
            linear_key_head_dim=cfg['linear_key_head_dim'],
            linear_value_head_dim=cfg['linear_value_head_dim'],
            linear_num_key_heads=cfg['linear_num_key_heads'],
            linear_num_value_heads=cfg['linear_num_value_heads'],
            linear_conv_kernel_dim=cfg['linear_conv_kernel_dim'],
        )

        # partial rotary
        head_dim = cfg.get('head_dim', None) or (
            cfg['hidden_size'] // cfg['num_attention_heads'])
        partial_rotary_factor = cfg.get('partial_rotary_factor', 1.0)
        rotary_dim = int(head_dim * partial_rotary_factor)
        info['rope_param'].dim = rotary_dim

        # MTP speculative decoding
        num_mtp = cfg.get('num_nextn_predict_layers', 0)
        if not num_mtp:
            import re
            import os
            import json as _json
            mtp_layer_indices = set()
            try:
                idx_path = os.path.join(self.model_path,
                                        'model.safetensors.index.json')
                if os.path.exists(idx_path):
                    with open(idx_path) as f:
                        idx = _json.load(f)
                    for key in idx.get('weight_map', {}):
                        m = re.match(r'mtp\.layers\.(\d+)\.', key)
                        if m:
                            mtp_layer_indices.add(int(m.group(1)))
            except Exception:
                pass
            if mtp_layer_indices:
                num_mtp = max(mtp_layer_indices) + 1
        if num_mtp and num_mtp > 0:
            info['num_mtp_layers'] = num_mtp
            # Detect MTP expert weight type (same logic as Qwen3NextModel)
            try:
                idx_path = os.path.join(self.model_path,
                                        'model.safetensors.index.json')
                if os.path.exists(idx_path):
                    with open(idx_path) as f2:
                        idx2 = _json.load(f2)
                    mtp_expert_keys = [
                        k for k in idx2.get('weight_map', {})
                        if 'mtp' in k and 'experts' in k
                        and 'shared_expert' not in k
                    ]
                    has_fp16 = any(
                        k.endswith('.weight') for k in mtp_expert_keys)
                    has_packed = any(
                        k.endswith('.weight_packed')
                        for k in mtp_expert_keys)
                    if has_fp16 and not has_packed:
                        info['mtp_expert_weight_type'] = 'fp16'
            except Exception:
                pass

        return info


