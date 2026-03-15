# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from ...utils import _get_and_verify_max_len, is_bf16_supported
from ..supported_models import SUPPORTED_ARCHS
from .config import TurbomindModelConfig
from .module import Transformer
from .policy import get_input_policy
from .source_model.base import INPUT_MODELS
from .target_model.base import OUTPUT_MODELS, BaseOutputModel

SUPPORTED_FORMATS = ['hf', 'awq', 'gptq', 'fp8', 'gguf', None]
logger = get_logger('lmdeploy')


# Minimum split unit (in elements) for each GGUF quantization type.
# Determined by internal bit-packing structure:
# - Q4_K: 64 (il-group independent, no qh field)
# - Q5_K: 256 (qh bits shared across all il-groups)
# - Q6_K: 128 (ip-half independent, qh shared within half)
# - Q8_0: 32 (block_size, no bit-packing)
_GGUF_MIN_SPLIT_UNIT = {
    12: 64,   # Q4_K
    13: 256,  # Q5_K
    14: 128,  # Q6_K
    8: 32,    # Q8_0
    39: 32,   # MXFP4 (block_size=32, no bit-packing)
}

_GGML_TYPE_NAMES = {
    8: 'Q8_0', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 39: 'MXFP4',
}


def check_gguf_tp_compatibility(expert_ggml_types, expert_inter_size, tp):
    """Check if GGUF expert quantization types are compatible with TP split.

    For each unique ggml_type in expert_ggml_types, verifies that
    expert_inter_size / tp is divisible by the minimum split unit.

    Raises ValueError with a clear message if incompatible.
    """
    if tp <= 1:
        return  # No split needed
    unique_types = set(t for t in expert_ggml_types if t != 31)
    if not unique_types:
        return  # No GGUF expert weights
    cols_per_rank = expert_inter_size // tp
    for gt in unique_types:
        min_unit = _GGUF_MIN_SPLIT_UNIT.get(gt)
        if min_unit is None:
            raise ValueError(
                f'GGUF TP split: unsupported quantization type {gt}. '
                f'Supported types: {list(_GGML_TYPE_NAMES.values())}')
        if cols_per_rank % min_unit != 0:
            name = _GGML_TYPE_NAMES.get(gt, str(gt))
            # Find compatible TP values
            compatible = []
            for t in [1, 2, 4, 8]:
                c = expert_inter_size // t
                if expert_inter_size % t == 0 and c % min_unit == 0:
                    compatible.append(t)
            raise ValueError(
                f'GGUF TP split: {name} (min_split_unit={min_unit}) is '
                f'incompatible with TP={tp} '
                f'(expert_inter_size={expert_inter_size}, '
                f'cols_per_rank={cols_per_rank}, '
                f'{cols_per_rank}%{min_unit}={cols_per_rank % min_unit}). '
                f'Compatible TP values for this model: {compatible}')


def _detect_gguf_expert_ggml_type(model_path: str, num_layers: int = 0):
    """Detect per-layer GGML quantization types for MoE expert weights.

    Returns a tuple (w1_types, w2_types) where each is a list of ggml_type
    ints (one per layer) or None if not detectable.  w2_types is None if
    all layers have the same type for w1 and w2.
    UD models may use different types per layer and per weight.
    """
    try:
        from lmdeploy.turbomind.deploy.source_model.gguf_reader import (
            GGUFSplitReader)
        import os
        if os.path.isfile(model_path):
            gguf_path = model_path
        else:
            gguf_files = sorted(f for f in os.listdir(model_path)
                                if f.endswith('.gguf'))
            if not gguf_files:
                return None, None
            gguf_path = os.path.join(model_path, gguf_files[0])
        reader = GGUFSplitReader(gguf_path)
        layer_types = {}
        layer_types_w2 = {}
        for name, info in reader._tensor_map.items():
            if 'ffn_gate_exps' in name:
                parts = name.split('.')
                layer_idx = int(parts[1])
                layer_types[layer_idx] = int(info.info.ggml_type)
            if 'ffn_down_exps' in name:
                parts = name.split('.')
                layer_idx = int(parts[1])
                layer_types_w2[layer_idx] = int(info.info.ggml_type)
        if not layer_types:
            return None, None
        if num_layers == 0:
            num_layers = max(layer_types.keys()) + 1
        w1_list = [layer_types.get(i, 31) for i in range(num_layers)]
        # Only return w2 list if any layer differs
        has_diff = any(
            layer_types_w2.get(i) != layer_types.get(i)
            for i in range(num_layers)
            if i in layer_types_w2
        )
        w2_list = None
        if has_diff:
            w2_list = [layer_types_w2.get(i, 31) for i in range(num_layers)]
        return w1_list, w2_list
    except Exception:
        return None, None


def get_input_model_registered_name(model_path: str, model_format: str):
    """Get the registered name of a model. The name will be used to access the
    INPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq']
    """
    if model_format == 'gguf':
        # GGUF models use a dedicated input model reader.
        arch = get_model_arch(model_path)[0]
        _GGUF_INPUT_MODEL_MAP = {
            'MiniMaxM2ForCausalLM': 'minimax-m2-gguf',
            'Qwen3NextForCausalLM': 'qwen3-coder-next-gguf',
        }
        name = _GGUF_INPUT_MODEL_MAP.get(arch)
        if name is None:
            raise ValueError(
                f'No GGUF input model registered for arch={arch}')
        return name
    arch = get_model_arch(model_path)[0]
    register_name = SUPPORTED_ARCHS[arch]
    return register_name


def get_output_model_registered_name_and_config(model_path: str, model_format: str, dtype: str, group_size: int):
    """Get the registered name of the turbomind model and its configuration
    according to the input model path, format and user-input config. The name
    will be used to access the OUTPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq']
        dtype (str): the data type of the model's weights and activations
        group_size (int): the size of group used by awq model
    """
    register_name = 'tm'

    has_bf16 = is_bf16_supported()

    model_arch, model_config = get_model_arch(model_path)

    # infer dtype from device and model config
    if dtype == 'auto':
        # pick dtype by device as default
        dtype = 'bfloat16' if has_bf16 else 'float16'
        # dtype from model
        torch_dtype = getattr(model_config, 'torch_dtype', None)
        if not torch_dtype:
            if model_arch in ['QWenLMHeadModel', 'GptOssForCausalLM']:
                torch_dtype = torch.bfloat16
        TORCH_DTYPE_MAP = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}
        dtype = TORCH_DTYPE_MAP.get(torch_dtype, dtype)

    if dtype == 'bfloat16' and not has_bf16:
        logger.warning('data type fallback to float16 since '
                       'torch.cuda.is_bf16_supported is False')
        dtype = 'float16'

    weight_type = dtype

    config = TurbomindModelConfig.from_dict()

    session_len = _get_and_verify_max_len(model_config, None)

    if model_format in ['awq', 'gptq', 'compressed-tensors']:
        weight_type = 'int4'
        dtype = 'float16'  # force float16 for int4 quantized weights
        group_size = 128 if group_size == 0 else group_size
        if model_format == 'compressed-tensors':
            model_format = 'awq'
    elif model_format == 'fp8':
        weight_type = 'fp8'
        group_size = 128
    elif model_format == 'mxfp4':
        weight_type = 'e2m1'
        group_size = 32
    elif model_format == 'gguf':
        # GGUF models: expert weights stay in original GGUF quantization
        # format.  Force float16 — bf16 may not be supported on V100.
        dtype = 'float16'
        weight_type = dtype
        # Detect per-layer expert ggml_type from GGUF tensor metadata.
        _num_layers = (model_config.get('num_hidden_layers', 0)
                       if isinstance(model_config, dict)
                       else getattr(model_config, 'num_hidden_layers', 0))
        expert_types = _detect_gguf_expert_ggml_type(model_path, _num_layers)
        if expert_types[0] is not None:
            config.model_config.expert_ggml_type = expert_types[0]
        if expert_types[1] is not None:
            config.model_config.expert_ggml_type_w2 = expert_types[1]
        # Set group_size to the GGUF block size so _pad_inter_size ensures
        # block alignment after TP split.  This adds minimal padding
        # (< 2 MB/GPU) — NOT the same as padding inter_size to 2048.
        # The sub-block split in save_split fills partial blocks with
        # real data + zero padding within the block structure.
        _GGUF_BLOCK_SIZE = {8: 32, 12: 256, 13: 256, 14: 256, 39: 32}  # Q8_0, Q4_K, Q5_K, Q6_K, MXFP4
        if expert_types[0]:
            all_types = set(t for t in expert_types[0] if t != 31)
            if expert_types[1]:
                all_types |= set(t for t in expert_types[1] if t != 31)
            if all_types:
                # Use the largest block size among all expert types
                group_size = max(_GGUF_BLOCK_SIZE.get(t, 1)
                                 for t in all_types)
            else:
                group_size = 0
        else:
            group_size = 0

    expert_weight_type = weight_type

    # ONLY experts are quantized, attention weights remain in native dtype
    # NOTE: Qwen3_5ForConditionalGeneration (dense, no MoE) is NOT in this
    # list because ALL its weights (attention + MLP) are quantized int4.
    # NOTE: Step3p5ForCausalLM with mixed-bit GPTQ/AutoRound has 8-bit
    # attn/MLP weights that are dequantized to fp16 in the Reader, so it
    # should be treated as moe-only-quant (same as AWQ/compressed-tensors).
    _moe_only_quant_archs = {'GptOssForCausalLM', 'MiniMaxM2ForCausalLM',
                              'Qwen3NextForCausalLM',
                              'Qwen3_5MoeForConditionalGeneration'}
    if model_arch == 'Step3p5ForCausalLM':
        # Both AWQ and mixed-bit GPTQ: attn/MLP are fp16 (AWQ natively,
        # GPTQ via 8-bit dequant in Reader). Only MoE experts are int4.
        _moe_only_quant_archs.add('Step3p5ForCausalLM')
    if model_arch in _moe_only_quant_archs:
        weight_type = dtype

    # Detect per-layer expert weight type from modules_to_not_convert.
    # Some AWQ/GPTQ models exclude certain layers from quantization
    # (e.g. MiniMax-M2.5-AWQ excludes model.layers.0), so those layers'
    # experts are fp16 while others are int4.
    expert_weight_types = []
    if model_format in ['awq', 'gptq', 'compressed-tensors']:
        quant_cfg = model_config.to_dict().get('quantization_config', {})
        not_convert = quant_cfg.get('modules_to_not_convert', [])
        num_layers = getattr(model_config, 'num_hidden_layers', 0) or 0
        if not num_layers:
            # VL models nest num_hidden_layers inside text_config
            _tc = model_config.to_dict().get('text_config', {})
            num_layers = _tc.get('num_hidden_layers', 0)
        if not_convert and num_layers > 0:
            import re
            for i in range(num_layers):
                layer_prefix = f'model.layers.{i}.'
                excluded = any(
                    layer_prefix.startswith(pat) or pat.startswith(layer_prefix)
                    for pat in not_convert
                    if re.match(r'model\.layers\.\d+\.?$', pat)
                )
                if excluded:
                    expert_weight_types.append(dtype)  # fp16
                else:
                    expert_weight_types.append(expert_weight_type)  # int4
            # If all layers have the same type, no need for per-layer list
            if len(set(expert_weight_types)) <= 1:
                expert_weight_types = []

    config.model_config.model_arch = model_arch
    config.model_config.data_type = dtype
    config.model_config.weight_type = weight_type
    config.model_config.expert_weight_type = expert_weight_type
    config.model_config.expert_weight_types = expert_weight_types
    config.model_config.model_format = model_format
    config.model_config.group_size = group_size
    config.model_config.session_len = session_len

    return register_name, config


def get_tm_model(model_path,
                 model_name,
                 chat_template_name,
                 engine_config: TurbomindEngineConfig,
                 group_size: int = None,
                 out_dir: str = None) -> BaseOutputModel:
    """Create turbomind model.

    Args:
        model_path (str): the path of the input model, which is supposed
            to be a local path, or huggingface hub repo_id, or modelscope
            hub repo_id
        model_name (str): user customized model name
        chat_template_name (str): the name of the chat template of
            the input model
        engine_config(TurbomindEngineConfig): user input engine config
        group_size(int): refers to the group_size if the input model
            is a w4a16(awq or gptq) quantized model
        out_dir(str): the output directory where to save to turbomind model.
            If it is None, the turbomind model won't be saved
    """
    _, cfg = get_model_arch(model_path)
    logger.info('get_tm_model: get_model_arch done (1st call)')
    # Auto-detect GGUF format if not explicitly set.
    from lmdeploy.archs import is_gguf_model
    if engine_config.model_format is None and is_gguf_model(model_path):
        engine_config.model_format = 'gguf'

    # GGUF models have no HF quantization_config — skip quant detection.
    quant_config = None
    if engine_config.model_format != 'gguf':
        quant_config = search_nested_config(cfg.to_dict(),
                                            'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None \
            or engine_config.model_format == quant_method \
            or (engine_config.model_format == 'gptq' and quant_method == 'auto-round'), (
            f'mismatched quant method: user input "{engine_config.model_format}" '
            f'vs model quant_config "{quant_method}"')
        assert not group_size or group_size == _group_size, (f'mismatched quant group size: user input "{group_size}" '
                                                             f'vs model quant_config "{_group_size}"')

        if quant_method == 'awq':
            assert version == 'gemm', f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and quant_config.get(
                'sym', True), f'unsupported quant config: {quant_config}'
        elif quant_method == 'fp8':
            pass
        elif quant_method == 'mxfp4':
            _group_size = 32
        elif quant_method == 'compressed-tensors':
            _format = quant_config['config_groups']['group_0']['format']
            assert _format == 'pack-quantized', ('compressed-tennsors only supports pack-quantized format, '
                                                 f'but got {_format}')
            _weights = quant_config['config_groups']['group_0']['weights']
            _group_size = _weights['group_size']
            _num_bits = _weights['num_bits']
            _type = _weights['type']
            assert _num_bits == 4 and _type == 'int', ('pack-quantized requires 4-bit int, '
                                                       f'but got {_num_bits}-bit {_type}')
        elif quant_method == 'auto-round':
            # AutoRound uses GPTQ packing format (auto_gptq)
            packing = quant_config.get('packing_format', '')
            assert 'auto_gptq' in packing or quant_config.get('sym', True), \
                f'unsupported auto-round config: {quant_config}'
            quant_method = 'gptq'  # treat as gptq for weight processing
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    if engine_config.model_format in ['awq', 'gptq']:
        # Compatible to awq models that are quantized by lmdeploy (<=v0.3.0)
        if not group_size:
            group_size = 128
        assert group_size == 128, (f'model format is "{engine_config.model_format}" '
                                   f'but group_size is {group_size}. Currently, only 128 '
                                   'is supported')
    elif engine_config.model_format == 'compressed-tensors':
        if not group_size:
            group_size = 128
        assert group_size in [32, 128], (f'model format is "{engine_config.model_format}" '
                                         f'but group_size is {group_size}. Currently, only '
                                         '32 and 128 are supported')

    logger.info('get_tm_model: calling get_input_model_registered_name')
    input_model_name = get_input_model_registered_name(model_path, engine_config.model_format)
    logger.info(f'get_tm_model: input_model_name={input_model_name}')
    input_policy = get_input_policy(engine_config.model_format)
    logger.info('get_tm_model: creating input_model')
    input_model = INPUT_MODELS.get(input_model_name)(model_path=model_path,
                                                     tokenizer_path=model_path,
                                                     input_policy=input_policy)
    logger.info('get_tm_model: input_model created')

    logger.info('get_tm_model: calling get_output_model_registered_name_and_config')
    output_model_name, tm_cfg = get_output_model_registered_name_and_config(model_path=model_path,
                                                                            model_format=engine_config.model_format,
                                                                            dtype=engine_config.dtype,
                                                                            group_size=group_size)
    logger.info(f'get_tm_model: output_model_name={output_model_name}')
    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name

    tm_cfg.model_config.attn_tp_size = engine_config.attn_tp_size or engine_config.tp
    tm_cfg.model_config.attn_cp_size = engine_config.attn_cp_size or 1
    tm_cfg.model_config.mlp_tp_size = engine_config.mlp_tp_size or engine_config.tp

    # Validate GGUF expert TP compatibility before creating output model.
    if engine_config.model_format == 'gguf':
        _expert_types = getattr(tm_cfg.model_config, 'expert_ggml_type', None)
        # expert_inter_size is not yet in tm_cfg (set during BaseOutputModel
        # init), so read it from the model config directly.
        _expert_inter = 0
        if isinstance(cfg, dict):
            _expert_inter = cfg.get('expert_inter_size',
                                    cfg.get('intermediate_size', 0))
        else:
            _expert_inter = getattr(cfg, 'expert_inter_size',
                                    getattr(cfg, 'intermediate_size', 0))
        if _expert_types and _expert_inter:
            check_gguf_tp_compatibility(
                _expert_types, _expert_inter,
                tm_cfg.model_config.mlp_tp_size)
            _expert_types_w2 = getattr(tm_cfg.model_config,
                                       'expert_ggml_type_w2', None)
            if _expert_types_w2:
                check_gguf_tp_compatibility(
                    _expert_types_w2, _expert_inter,
                    tm_cfg.model_config.mlp_tp_size)

    output_model = OUTPUT_MODELS.get(output_model_name)(input_model=input_model,
                                                        cfg=tm_cfg,
                                                        model_cls=Transformer,
                                                        out_dir=out_dir)

    return output_model
