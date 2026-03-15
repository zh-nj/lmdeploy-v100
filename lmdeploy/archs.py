# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Literal, Tuple

from transformers import AutoConfig

from .messages import PytorchEngineConfig, TurbomindEngineConfig
from .utils import get_logger

logger = get_logger('lmdeploy')


def autoget_backend(model_path: str) -> Literal['turbomind', 'pytorch']:
    """Get backend type in auto backend mode.

    Args:
         model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.

    Returns:
        str: the backend type.
    """

    turbomind_has = False
    is_turbomind_installed = True
    try:
        from lmdeploy.turbomind.supported_models import is_supported as is_supported_turbomind
        turbomind_has = is_supported_turbomind(model_path)
    except ImportError:
        is_turbomind_installed = False

    if is_turbomind_installed:
        if not turbomind_has:
            logger.warning('Fallback to pytorch engine because '
                           f'`{model_path}` not supported by turbomind'
                           ' engine.')
    else:
        logger.warning('Fallback to pytorch engine because turbomind engine is not '
                       'installed correctly. If you insist to use turbomind engine, '
                       'you may need to reinstall lmdeploy from pypi or build from '
                       'source and try again.')

    backend = 'turbomind' if turbomind_has else 'pytorch'
    return backend


def autoget_backend_config(
    model_path: str,
    backend_config: PytorchEngineConfig | TurbomindEngineConfig | None = None
) -> Tuple[Literal['turbomind', 'pytorch'], PytorchEngineConfig | TurbomindEngineConfig]:
    """Get backend config automatically.

    Args:
        model_path (str): The input model path.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): The
            input backend config. Default to None.

    Returns:
        (PytorchEngineConfig | TurbomindEngineConfig): The auto-determined
            backend engine config.
    """
    from dataclasses import asdict

    if isinstance(backend_config, PytorchEngineConfig):
        return 'pytorch', backend_config

    backend = autoget_backend(model_path)
    config = PytorchEngineConfig() if backend == 'pytorch' else TurbomindEngineConfig()
    if backend_config is not None:
        if type(backend_config) == type(config):
            config = backend_config
        else:
            data = asdict(backend_config)
            for k, v in data.items():
                if v and hasattr(config, k):
                    setattr(config, k, v)
            # map attributes with different names
            if type(backend_config) is TurbomindEngineConfig:
                config.block_size = backend_config.cache_block_seq_len
            else:
                config.cache_block_seq_len = backend_config.block_size
    return backend, config


def check_vl_llm(config: dict) -> bool:
    """Check if the model is a vl model from model config."""
    if 'auto_map' in config:
        for _, v in config['auto_map'].items():
            if 'InternLMXComposer2ForCausalLM' in v:
                return True

    if 'language_config' in config and 'vision_config' in config and config['language_config'].get(
            'architectures', [None])[0] == 'DeepseekV2ForCausalLM':
        return True

    archs = config.get('architectures')
    if not archs:
        return False
    arch = archs[0]
    supported_archs = set([
        'LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM', 'CogVLMForCausalLM', 'InternLMXComposer2ForCausalLM',
        'InternVLChatModel', 'MiniCPMV', 'LlavaForConditionalGeneration', 'LlavaNextForConditionalGeneration',
        'Phi3VForCausalLM', 'Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration',
        'Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration',
        'Qwen3_5MoeForConditionalGeneration', 'Qwen3_5ForConditionalGeneration',
        'MllamaForConditionalGeneration',
        'MolmoForCausalLM', 'Gemma3ForConditionalGeneration', 'Llama4ForConditionalGeneration',
        'InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration', 'InternS1ProForConditionalGeneration',
        'Glm4vForConditionalGeneration'
    ])
    if arch == 'QWenLMHeadModel' and 'visual' in config:
        return True
    elif arch == 'MultiModalityCausalLM' and 'language_config' in config:
        return True
    elif arch in ['ChatGLMModel', 'ChatGLMForConditionalGeneration'] and 'vision_config' in config:
        return True
    elif arch in supported_archs:
        return True
    return False


def get_task(model_path: str):
    """Get pipeline type and pipeline class from model config."""
    from lmdeploy.serve.core import AsyncEngine

    if os.path.exists(os.path.join(model_path, 'triton_models', 'weights')):
        # workspace model
        return 'llm', AsyncEngine
    _, config = get_model_arch(model_path)
    if check_vl_llm(config.to_dict()):
        from lmdeploy.serve.core import VLAsyncEngine
        return 'vlm', VLAsyncEngine

    # default task, pipeline_class
    return 'llm', AsyncEngine


def get_model_arch(model_path: str):
    """Get a model's architecture and configuration.

    Args:
        model_path(str): the model path
    """
    # GGUF file detection — bypass HF AutoConfig.
    if is_gguf_model(model_path):
        logger.info(f'get_model_arch: GGUF model detected at {model_path}')
        return _get_gguf_model_arch(model_path)

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:  # noqa
        from transformers import PretrainedConfig
        cfg = PretrainedConfig.from_pretrained(model_path, trust_remote_code=True)

    _cfg = cfg.to_dict()
    if _cfg.get('architectures', None):
        arch = _cfg['architectures'][0]
        if _cfg.get('auto_map'):
            for _, v in _cfg['auto_map'].items():
                if 'InternLMXComposer2ForCausalLM' in v:
                    arch = 'InternLMXComposer2ForCausalLM'
    elif _cfg.get('auto_map', None) and 'AutoModelForCausalLM' in _cfg['auto_map']:
        arch = _cfg['auto_map']['AutoModelForCausalLM'].split('.')[-1]
    elif _cfg.get('language_config', None) and _cfg['language_config'].get(
            'auto_map', None) and 'AutoModelForCausalLM' in _cfg['language_config']['auto_map']:
        arch = _cfg['language_config']['auto_map']['AutoModelForCausalLM'].split('.')[-1]
    else:
        raise RuntimeError(f'Could not find model architecture from config: {_cfg}')
    return arch, cfg


def search_nested_config(config, key):
    """Recursively searches for the value associated with the given key in a
    nested configuration of a model."""
    if isinstance(config, Dict):
        for k, v in config.items():
            if k == key:
                return v
            if isinstance(v, (Dict, List)):
                result = search_nested_config(v, key)
                if result is not None:
                    return result
    elif isinstance(config, List):
        for item in config:
            result = search_nested_config(item, key)
            if result is not None:
                return result
    return None

def is_gguf_model(model_path: str) -> bool:
    """Check if *model_path* points to a GGUF file (or split shard)."""
    if not isinstance(model_path, str):
        return False
    # Direct .gguf file.
    if model_path.endswith('.gguf') and os.path.isfile(model_path):
        return True
    # Directory containing .gguf files.
    if os.path.isdir(model_path):
        return any(f.endswith('.gguf') for f in os.listdir(model_path))
    return False


_gguf_model_arch_cache: dict = {}


def _get_gguf_model_arch(model_path: str):
    """Return (arch_string, config_namespace) for a GGUF model.

    The returned *config_namespace* mimics a ``PretrainedConfig`` just
    enough for the downstream converter pipeline (``to_dict()`` method).
    Results are cached to avoid re-parsing GGUF files multiple times.
    """
    import time as _time
    cache_key = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    if cache_key in _gguf_model_arch_cache:
        logger.info(f'_get_gguf_model_arch: cache hit for {model_path}')
        return _gguf_model_arch_cache[cache_key]

    from lmdeploy.turbomind.deploy.source_model.gguf_reader import (
        GGUFSplitReader, build_model_config)

    # Find the first .gguf file.
    if os.path.isfile(model_path):
        gguf_path = model_path
    else:
        gguf_files = sorted(f for f in os.listdir(model_path)
                            if f.endswith('.gguf'))
        if not gguf_files:
            raise FileNotFoundError(
                f'No .gguf files found in {model_path}')
        gguf_path = os.path.join(model_path, gguf_files[0])

    logger.info(f'_get_gguf_model_arch: creating GGUFSplitReader '
                f'for {gguf_path}')
    _t0 = _time.time()
    reader = GGUFSplitReader(gguf_path)
    logger.info(f'_get_gguf_model_arch: GGUFSplitReader done in '
                f'{_time.time() - _t0:.1f}s')
    config_dict = build_model_config(reader)
    logger.info(f'_get_gguf_model_arch: build_model_config done, '
                f'arch={config_dict.get("model_arch")}')

    arch = config_dict.get('model_arch')
    if arch is None:
        raise ValueError(
            f'Unsupported GGUF architecture: '
            f'{config_dict.get("gguf_architecture")}')

    # Wrap config_dict in a simple namespace with to_dict().
    cfg = _GGUFConfigNamespace(config_dict)
    _gguf_model_arch_cache[cache_key] = (arch, cfg)
    return arch, cfg


class _GGUFConfigNamespace:
    """Minimal config wrapper that satisfies ``cfg.to_dict()`` calls."""

    def __init__(self, d: dict):
        self._d = d
        for k, v in d.items():
            setattr(self, k, v)

    def to_dict(self):
        return dict(self._d)

