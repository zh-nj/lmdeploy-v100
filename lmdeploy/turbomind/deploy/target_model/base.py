# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from abc import ABC
from collections.abc import Sequence

import torch
import tqdm
import yaml
from mmengine import Registry

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

from ..config import AttentionConfig, LoraConfig, ModelConfig, TurbomindModelConfig, config_from_dict, config_to_dict
from ..source_model.base import BaseInputModel

OUTPUT_MODELS = Registry('target model', locations=['lmdeploy.turbomind.deploy.target_model.base'])


# ---------------------------------------------------------------------------
# GGUF sub-block column splitting
# ---------------------------------------------------------------------------

def _q4k_subblock_col_split(raw_blocks, gguf_rows, blocks_per_row,
                             cols_per_split, split_num):
    """Split Q4_K blocks along columns at 64-element il-group granularity.

    Q4_K block (256 elements, 144 bytes):
      dm[4] + scales[12] + qs[128]
    4 il-groups (il=0..3), each 64 elements:
      - qs: 32 bytes per il-group (qs[32*il : 32*il+32])
      - scales: 2 scale/min pairs per il-group (is=2*il, is+1)
      - dm: shared across all il-groups (copied to each split)

    Each il-group is independent — no cross-group bit-packing.

    When cols_per_split is not a multiple of 256, the output blocks are
    padded with zeros to form complete Q4_K blocks.  The C++ GEMM uses
    the actual cols_per_split dimension, so padding elements are never
    accessed in computation.

    Output: list of repacked Q4_K tensors, one per split rank.
    """
    from gguf import GGMLQuantizationType
    import numpy as np

    block_bytes = 144
    elems_per_block = 256
    groups_per_block = 4  # il-groups
    elems_per_group = 64
    qs_per_group = 32  # bytes of qs per il-group

    assert cols_per_split % elems_per_group == 0
    groups_per_split = cols_per_split // elems_per_group

    # Number of output blocks per row (ceil to complete blocks)
    out_blocks_per_row = (cols_per_split + elems_per_block - 1) // elems_per_block

    # Total source blocks
    total_src_blocks = gguf_rows * blocks_per_row
    block_data = raw_blocks.numpy().reshape(total_src_blocks, block_bytes)

    # Extract fields from all source blocks
    dm_all = block_data[:, :4]           # (total, 4)
    scales_all = block_data[:, 4:16]     # (total, 12)
    qs_all = block_data[:, 16:144]       # (total, 128)

    # Total il-groups per row in source
    total_groups_per_row = blocks_per_row * groups_per_block

    results = []
    for rank in range(split_num):
        group_offset = rank * groups_per_split
        rank_block_list = []

        for row in range(gguf_rows):
            row_block_offset = row * blocks_per_row

            # Collect groups for this rank from this row
            for ob in range(out_blocks_per_row):
                new_block = np.zeros(block_bytes, dtype=np.uint8)
                new_dm = new_block[:4]
                new_scales = new_block[4:16]
                new_qs = new_block[16:144]

                for new_il in range(groups_per_block):
                    src_group_idx = group_offset + ob * groups_per_block + new_il
                    if src_group_idx >= group_offset + groups_per_split:
                        break  # Padding region — leave as zeros

                    # Map to source block and il-group
                    src_block_local = src_group_idx // groups_per_block
                    src_il = src_group_idx % groups_per_block
                    src_block_idx = row_block_offset + src_block_local

                    # Copy dm from first source block in this output block
                    if new_il == 0:
                        new_dm[:] = dm_all[src_block_idx]

                    # Copy qs
                    src_qs_start = src_il * qs_per_group
                    dst_qs_start = new_il * qs_per_group
                    new_qs[dst_qs_start:dst_qs_start + qs_per_group] = \
                        qs_all[src_block_idx, src_qs_start:src_qs_start + qs_per_group]

                    # Copy scales
                    for offset in range(2):
                        src_is = 2 * src_il + offset
                        dst_is = 2 * new_il + offset
                        sc, mn = _get_scale_min_k4(src_is, scales_all[src_block_idx])
                        _set_scale_min_k4(dst_is, new_scales, sc, mn)

                rank_block_list.append(new_block)

        rank_data = np.concatenate(rank_block_list)
        rank_tensor = torch.from_numpy(rank_data.copy()).to(torch.uint8)
        rank_tensor.ggml_type = int(GGMLQuantizationType.Q4_K)
        # ggml_shape: padded to full blocks for storage, but logical dim is cols_per_split
        padded_cols = out_blocks_per_row * elems_per_block
        rank_tensor.ggml_shape = (padded_cols, gguf_rows)
        results.append(rank_tensor)

    return results


def _q6k_subblock_col_split(raw_blocks, gguf_rows, blocks_per_row,
                             cols_per_split, split_num):
    """Split Q6_K blocks along columns at 128-element ip-half granularity.

    Q6_K block (256 elements, 210 bytes):
      ql[128] + qh[64] + scales[16] + d[2]
    2 ip-halves (ip=0,1), each 128 elements:
      - ql: 64 bytes per half (ql[64*ip : 64*ip+64])
      - qh: 32 bytes per half (qh[32*ip : 32*ip+32])
      - scales: 8 per half (scales[8*ip : 8*ip+8])
      - d: shared (copied to each split)

    Each ip-half is independent in ql/qh/scales.

    When cols_per_split is not a multiple of 256, the output blocks are
    padded with zeros to form complete Q6_K blocks.

    Output: list of repacked Q6_K tensors, one per split rank.
    """
    from gguf import GGMLQuantizationType
    import numpy as np

    block_bytes = 210
    elems_per_block = 256
    halves_per_block = 2
    elems_per_half = 128

    assert cols_per_split % elems_per_half == 0
    halves_per_split = cols_per_split // elems_per_half

    out_blocks_per_row = (cols_per_split + elems_per_block - 1) // elems_per_block

    total_blocks = gguf_rows * blocks_per_row
    block_data = raw_blocks.numpy().reshape(total_blocks, block_bytes)

    # Extract fields
    ql_all = block_data[:, :128]          # (total, 128)
    qh_all = block_data[:, 128:192]       # (total, 64)
    sc_all = block_data[:, 192:208]       # (total, 16)
    d_all = block_data[:, 208:210]        # (total, 2)

    results = []
    for rank in range(split_num):
        half_offset = rank * halves_per_split
        rank_block_list = []

        for row in range(gguf_rows):
            row_block_offset = row * blocks_per_row

            for ob in range(out_blocks_per_row):
                new_block = np.zeros(block_bytes, dtype=np.uint8)
                new_ql = new_block[:128]
                new_qh = new_block[128:192]
                new_sc = new_block[192:208]
                new_d = new_block[208:210]

                for new_ip in range(halves_per_block):
                    src_half_idx = half_offset + ob * halves_per_block + new_ip
                    if src_half_idx >= half_offset + halves_per_split:
                        break  # Padding region

                    src_block_local = src_half_idx // halves_per_block
                    src_ip = src_half_idx % halves_per_block
                    src_block_idx = row_block_offset + src_block_local

                    if new_ip == 0:
                        new_d[:] = d_all[src_block_idx]

                    # ql: 64 bytes per half
                    new_ql[64 * new_ip:64 * new_ip + 64] = \
                        ql_all[src_block_idx, 64 * src_ip:64 * src_ip + 64]
                    # qh: 32 bytes per half
                    new_qh[32 * new_ip:32 * new_ip + 32] = \
                        qh_all[src_block_idx, 32 * src_ip:32 * src_ip + 32]
                    # scales: 8 per half
                    new_sc[8 * new_ip:8 * new_ip + 8] = \
                        sc_all[src_block_idx, 8 * src_ip:8 * src_ip + 8]

                rank_block_list.append(new_block)

        rank_data = np.concatenate(rank_block_list)
        rank_tensor = torch.from_numpy(rank_data.copy()).to(torch.uint8)
        rank_tensor.ggml_type = int(GGMLQuantizationType.Q6_K)
        padded_cols = out_blocks_per_row * elems_per_block
        rank_tensor.ggml_shape = (padded_cols, gguf_rows)
        results.append(rank_tensor)

    return results


def _get_scale_min_k4(j, scales_row):
    """Decode 6-bit scale and min from Q4_K/Q5_K scales[12] array."""
    if j < 4:
        sc = int(scales_row[j]) & 63
        m = int(scales_row[j + 4]) & 63
    else:
        sc = (int(scales_row[j + 4]) & 0xF) | ((int(scales_row[j - 4]) >> 6) << 4)
        m = (int(scales_row[j + 4]) >> 4) | ((int(scales_row[j]) >> 6) << 4)
    return sc, m


def _set_scale_min_k4(j, scales, sc, m):
    """Encode 6-bit scale and min into Q4_K/Q5_K scales[12] array.

    Inverse of _get_scale_min_k4. Handles the packed 6-bit encoding.
    """
    if j < 4:
        # Low 6 bits: scales[j] low6 = sc, scales[j+4] low6 = m
        scales[j] = (int(scales[j]) & 0xC0) | (sc & 63)
        scales[j + 4] = (int(scales[j + 4]) & 0xC0) | (m & 63)
    else:
        # High bits split across two bytes
        # sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        # m  = (scales[j+4] >> 4)  | ((scales[j] >> 6) << 4)
        scales[j + 4] = ((m & 0xF) << 4) | (sc & 0xF)
        scales[j - 4] = (int(scales[j - 4]) & 0x3F) | ((sc >> 4) << 6)
        scales[j] = (int(scales[j]) & 0x3F) | ((m >> 4) << 6)


def _gguf_subblock_col_split(tensor, gguf_rows, gguf_cols, cols_per_split,
                              split_num, ggml_type, elems_per_block,
                              block_bytes):
    """Dispatch sub-block column splitting by quantization type."""
    from gguf import GGMLQuantizationType

    blocks_per_row = gguf_cols // elems_per_block
    raw_blocks = tensor.view(-1)

    if ggml_type == int(GGMLQuantizationType.Q4_K):
        return _q4k_subblock_col_split(
            raw_blocks, gguf_rows, blocks_per_row,
            cols_per_split, split_num)
    elif ggml_type == int(GGMLQuantizationType.Q6_K):
        return _q6k_subblock_col_split(
            raw_blocks, gguf_rows, blocks_per_row,
            cols_per_split, split_num)
    else:
        raise ValueError(
            f'GGUF sub-block col split not implemented for type {ggml_type}. '
            f'Only Q4_K (64-element) and Q6_K (128-element) are supported.')


# ---------------------------------------------------------------------------
def tprint(*args, **kwargs):
    to_file = kwargs.pop('to_file', False)
    if not to_file:
        return
    from io import StringIO
    s = StringIO()
    print(*args, **kwargs, file=s, end='')
    tqdm.tqdm.write(s.getvalue())


def _weight_dtype_map(weight_type: str, default=None):
    """Map literal data type to torch dtype."""

    _WEIGHT_DTYPE_MAP = dict(int4=torch.float16, float16=torch.float16, float32=torch.float16, bfloat16=torch.bfloat16)

    return _WEIGHT_DTYPE_MAP.get(weight_type, default)


def _pad_inter_size(inter_size: int, group_size: int, tp: int):
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


class BaseOutputModel(ABC):
    """Base output model."""

    def __init__(self, input_model: BaseInputModel, cfg: TurbomindModelConfig, model_cls, out_dir: str = ''):
        super().__init__()
        self.input_model = input_model
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.attn_tp_size = self.model_config.attn_tp_size
        self.attn_cp_size = self.model_config.attn_cp_size
        self.mlp_tp_size = self.model_config.mlp_tp_size
        self.out_dir = out_dir
        self.to_file = True if out_dir else False
        self.tm_params = dict()

        # get `model_info` at first, which will be updated to `self.model_config` and `self.attention_config`
        self.input_model_info = self.input_model.model_info()
        self.input_model_info = self.single_to_list(self.input_model_info, keys=['inter_size', 'expert_num'])
        self.permute_qk = self.input_model_info.get('permute_qk', True)
        self.update_model_config()
        for i, v in enumerate(self.model_config.inter_size):
            self.model_config.inter_size[i] = _pad_inter_size(v, self.model_config.group_size, self.mlp_tp_size)
        if self.model_config.expert_num:
            self.model_config.expert_inter_size = _pad_inter_size(self.model_config.expert_inter_size,
                                                                  self.model_config.group_size, self.mlp_tp_size)

        # head_num is divisble by tp but kv_head_num is not
        # and tp is divisble by kv_head_num
        assert self.model_config.head_num % self.attn_tp_size == 0
        self.repeat_kv = 0
        if (self.attn_tp_size > self.model_config.kv_head_num
                and self.attn_tp_size % self.model_config.kv_head_num == 0):
            self.repeat_kv = (self.attn_tp_size // self.model_config.kv_head_num)
            self.model_config.kv_head_num = self.attn_tp_size

        self.model_config.verify()
        assert self.model_config.kv_head_num % self.attn_tp_size == 0

        # print(self.model_config)

        self.update_attention_config()
        self.update_lora_config()
        # ! Dependency on `self`
        self.model = model_cls(self)

    def single_to_list(self, config: dict, keys):
        num_layer = int(config['num_layer'])
        for k in keys:
            v = config.get(k, None)
            if v is not None and not isinstance(v, Sequence):
                config[k] = [v] * num_layer
        return config

    def update_model_config(self):
        """Update `self.model_config` according to the input_model's
        `model_info`"""
        final_cfg = config_to_dict(self.model_config)
        final_cfg.update(self.input_model_info)
        if 'embedding_size' not in self.input_model_info.keys():
            final_cfg.update(embedding_size=self.input_model_info['vocab_size'])

        self.model_config = config_from_dict(ModelConfig, final_cfg)

    def update_attention_config(self):
        """Update attention config according to input model's model info."""
        final_cfg = config_to_dict(self.attention_config)
        final_cfg.update(self.input_model_info)
        self.attention_config = config_from_dict(AttentionConfig, final_cfg)

    def update_lora_config(self):
        """Update lora config according to input model's model info."""
        final_cfg = config_to_dict(self.lora_config)
        final_cfg.update(self.input_model_info)
        self.lora_config = config_from_dict(LoraConfig, final_cfg)

    def export_config(self) -> None:
        """Export turbomind config."""
        if self.to_file:
            config_path = osp.join(self.out_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.safe_dump(self.tm_config.to_dict(), f)

    def export_weight(self, param: torch.Tensor, name: str) -> None:
        """Export turbomind weight."""

        def _tofile(tensor, path):
            """To file."""
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.view(torch.half)
            tensor.contiguous().cpu().numpy().tofile(path)

        if self.to_file:
            if torch.is_floating_point(param):
                torch_type = _weight_dtype_map(self.model_config.weight_type, torch.float16)
                param = param.to(torch_type)
            tprint(name, param.shape)
            _tofile(param, osp.join(self.out_dir, name))
        elif len(self.tm_params) > 0:
            tm_params = self.tm_params
            weight_type = self.model_config.weight_type
            data_type = self.model_config.data_type
            assert weight_type in ['float16', 'bfloat16', 'int4', 'fp8'] or \
                self.model_config.model_format == 'gguf'

            # currently, the tensor type should in
            # [torch.float, torch.half, torch.bfloat16, torch.int32]
            torch_tensor = param if param.is_contiguous() else param.contiguous()
            torch_tensor = torch_tensor.cuda()
            assert torch_tensor.dtype in [torch.int32, torch.float, torch.half, torch.bfloat16, torch.uint8]
            FLOAT_TYPES = [torch.float, torch.half, torch.bfloat16]
            if torch_tensor.dtype == torch.uint8:
                # GGUF raw quantized data — pass through without conversion.
                pass
            elif weight_type == 'fp8':
                # avoid casting float scales to half
                if torch_tensor.dtype == torch.bfloat16 and data_type == 'float16':
                    torch_tensor = torch_tensor.half()
            elif torch_tensor.dtype in FLOAT_TYPES:
                # Check if the target C++ buffer expects fp32 (e.g. fp32 MoE gate).
                # If so, keep the tensor as fp32 instead of converting to fp16.
                target_is_fp32 = False
                if name in tm_params:
                    for tm_t in tm_params[name]:
                        # Use __dlpack__ to get the actual dtype of the C++ buffer
                        try:
                            dl_tensor = torch.utils.dlpack.from_dlpack(tm_t)
                            if dl_tensor.dtype == torch.float32:
                                target_is_fp32 = True
                                break
                        except Exception:
                            pass
                if target_is_fp32:
                    torch_tensor = torch_tensor.float()
                elif weight_type in ['float16', 'int4']:
                    torch_tensor = torch_tensor.half()
                elif weight_type == 'bfloat16':
                    torch_tensor = torch_tensor.bfloat16()
                else:
                    torch_tensor = torch_tensor.half()
            if name in tm_params:
                for tm_tensor in tm_params[name]:
                    tm_tensor.copy_from(torch_tensor)
                tm_params.pop(name)
        else:
            tprint('skip export', name, param.shape)

    def save_split(self, tensor: torch.Tensor, name: str, split_dim=None, split_num=1, copy=False) -> None:
        """Save split.

        - 2D input
            shape must be (input_dims, output_dims)
        - 1D input (bias)
            shape must be (output_dims)
            split is skipped when split_dim == 0
        """
        # GGUF raw quantized data: split in quantized domain.
        # Padding ensures block alignment for all supported types.
        if hasattr(tensor, 'ggml_type'):
            if split_num <= 1:
                self.export_weight(tensor, name)
                return
            ggml_shape = tensor.ggml_shape  # (out_dim, in_dim) in PyTorch convention
            out_dim, in_dim = ggml_shape
            # GGUF row-major: gguf_rows = in_dim, gguf_cols = out_dim
            gguf_rows = in_dim
            gguf_cols = out_dim

            from gguf import GGMLQuantizationType
            _block_info = {
                int(GGMLQuantizationType.Q8_0): (32, 34),
                int(GGMLQuantizationType.Q6_K): (256, 210),
                int(GGMLQuantizationType.Q5_K): (256, 176),
                int(GGMLQuantizationType.Q4_K): (256, 144),
                39: (32, 17),  # MXFP4: block_size=32, 1 byte E8M0 + 16 bytes qs
            }
            info = _block_info.get(tensor.ggml_type)
            if info is None:
                logger.warning(f'GGUF split: unknown type {tensor.ggml_type} for {name}, '
                               f'broadcasting to all ranks')
                for i in range(split_num):
                    prefix, ext = osp.splitext(name)
                    self.export_weight(tensor, f'{prefix}.{i}{ext}')
                return
            elems_per_block, block_bytes = info

            assert gguf_cols % elems_per_block == 0, \
                f'GGUF: {gguf_cols} cols not aligned to block {elems_per_block}'
            blocks_per_row = gguf_cols // elems_per_block
            bytes_per_row = blocks_per_row * block_bytes

            if split_dim == 0:
                # Row split (w2): rows are contiguous — simple byte slice.
                assert gguf_rows % split_num == 0, \
                    f'GGUF row split: {gguf_rows} not divisible by {split_num}'
                rows_per_split = gguf_rows // split_num
                split_bytes = rows_per_split * bytes_per_row
                for i in range(split_num):
                    start = i * split_bytes
                    end = start + split_bytes
                    split_tensor = tensor[start:end].clone()
                    split_tensor.ggml_type = tensor.ggml_type
                    split_tensor.ggml_shape = (out_dim, rows_per_split)
                    prefix, ext = osp.splitext(name)
                    self.export_weight(split_tensor, f'{prefix}.{i}{ext}')
            elif split_dim == -1 or split_dim == 1:
                # Column split (w1/w3): format-aware sub-block splitting.
                assert gguf_cols % split_num == 0, \
                    f'GGUF col split: {gguf_cols} not divisible by {split_num}'
                cols_per_split = gguf_cols // split_num
                if cols_per_split % elems_per_block == 0:
                    # Standard block-aligned split.
                    blocks_per_split = cols_per_split // elems_per_block
                    split_row_bytes = blocks_per_split * block_bytes
                    raw_2d = tensor.view(-1).reshape(gguf_rows, bytes_per_row)
                    for i in range(split_num):
                        col_start = i * split_row_bytes
                        col_end = col_start + split_row_bytes
                        split_2d = raw_2d[:, col_start:col_end].contiguous()
                        split_tensor = split_2d.view(-1)
                        split_tensor.ggml_type = tensor.ggml_type
                        split_tensor.ggml_shape = (cols_per_split, in_dim)
                        prefix, ext = osp.splitext(name)
                        self.export_weight(split_tensor, f'{prefix}.{i}{ext}')
                else:
                    # Sub-block split: repack into standard blocks.
                    split_tensor_list = _gguf_subblock_col_split(
                        tensor, gguf_rows, gguf_cols, cols_per_split,
                        split_num, tensor.ggml_type,
                        elems_per_block, block_bytes)
                    for i, st in enumerate(split_tensor_list):
                        prefix, ext = osp.splitext(name)
                        self.export_weight(st, f'{prefix}.{i}{ext}')
            else:
                for i in range(split_num):
                    prefix, ext = osp.splitext(name)
                    self.export_weight(tensor, f'{prefix}.{i}{ext}')
            return

        if copy or (tensor.dim() == 1 and split_dim == 0):
            split_dim = None
            copy = True

        if split_dim is not None:
            tprint(f'*** splitting {name}, shape={tensor.shape}, '
                   f'split_dim={split_dim}, split_num={split_num}',
                   to_file=self.to_file)
            if tensor.shape[split_dim] % split_num != 0:
                raise RuntimeError(f'{name}: shape={list(tensor.shape)}, split_num={split_num}')
            split_size = tensor.shape[split_dim] // split_num
            splits = torch.split(tensor, split_size, dim=split_dim)
            for i, split in enumerate(splits):
                prefix, ext = osp.splitext(name)
                self.export_weight(split, f'{prefix}.{i}{ext}')
        elif copy:
            tprint(f'### copying {name}, shape={tensor.shape}', to_file=self.to_file)
            copies = [tensor] * split_num
            for i, copy in enumerate(copies):
                prefix, ext = osp.splitext(name)
                self.export_weight(copy, f'{prefix}.{i}{ext}')
        else:
            self.export_weight(tensor, name)

    def export(self) -> None:
        """Export to turbomind model format."""
        num_layer = self.model_config.num_layer
        from tqdm import tqdm
        pbar = tqdm(total=num_layer, desc='Convert to turbomind format', leave=self.to_file)
        self.export_config()
        last_reader = None
        had_misc = False
        for i, reader in self.input_model.readers():
            if self.model(i, reader):
                pbar.update(1)
            if i >= 0:
                last_reader = reader
            else:
                had_misc = True
        pbar.close()
        from ..module import Transformer
        if isinstance(self.model, Transformer):
            if had_misc:
                # HF models: MTP expert weights may span multiple shards.
                # finalize_mtp_export uses accumulated misc params.
                self.model.finalize_mtp_export()
            elif last_reader is not None:
                # GGUF models: readers() never yields -1 (misc), so use
                # the last layer reader which has access to MTP weights
                # via the GGUF file reader.
                self.model._try_export_mtp(last_reader)

    def export_iter(self):
        self.export_config()
        last_reader = None
        had_misc = False
        for i, reader in self.input_model.readers():
            self.model(i, reader)
            if i >= 0:
                last_reader = reader
            else:
                had_misc = True
            yield i
        from ..module import Transformer
        if isinstance(self.model, Transformer):
            if had_misc:
                self.model.finalize_mtp_export()
            elif last_reader is not None:
                self.model._try_export_mtp(last_reader)

    @property
    def tm_config(self):
        return TurbomindModelConfig(model_config=self.model_config,
                                    attention_config=self.attention_config,
                                    lora_config=self.lora_config)
