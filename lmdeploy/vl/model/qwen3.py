# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3VLModel(VisionModel):
    """Qwen3VL model."""

    _arch = ['Qwen3VLForConditionalGeneration',
             'Qwen3VLMoeForConditionalGeneration',
             'Qwen3_5MoeForConditionalGeneration',
             'Qwen3_5ForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]
        self.video_token = self.processor.video_token
        self.video_token_id = tokenizer.encode(self.video_token)[-1]
        self.mm_processor_kwargs = None

    def get_processor_args(self, mm_processor_kwargs: Optional[Dict[str, Any]] = None):
        min_pixels = self.processor.image_processor.size['shortest_edge']
        max_pixels = self.processor.image_processor.size['longest_edge']

        if mm_processor_kwargs is None:
            return min_pixels, max_pixels

        input_min_pixels = mm_processor_kwargs.get('min_pixels', None)
        input_max_pixels = mm_processor_kwargs.get('max_pixels', None)

        # boundary check for min_pixels and max_pixels
        if input_min_pixels is None:
            if input_max_pixels is not None:
                # only max_pixels is given in the input
                if input_max_pixels < min_pixels:
                    logger.warning(
                        f'input max_pixels {input_max_pixels} < default min_pixels {min_pixels}, fall back to default.')
                    return min_pixels, max_pixels
                max_pixels = input_max_pixels
        else:
            if input_max_pixels is None:
                # only min_pixels is given in the input
                if input_min_pixels > max_pixels:
                    logger.warning(
                        f'input min_pixels {input_min_pixels} > default max_pixels {max_pixels}, fall back to default.')
                    return min_pixels, max_pixels
            else:
                if input_min_pixels > input_max_pixels:
                    logger.warning(
                        f'input min_pixels {input_min_pixels} > max_pixels {input_max_pixels}, fall back to default.')
                    return min_pixels, max_pixels
                max_pixels = input_max_pixels
            min_pixels = input_min_pixels

        return min_pixels, max_pixels
    @staticmethod
    def collect_videos(messages):
        """Collect all type='video' items from messages.

        Returns:
            List of (frames, params) tuples where frames is List[PIL.Image]
            and params is a dict of extra parameters.
        """
        videos = []
        for message in messages:
            content = message['content']
            if not isinstance(content, list):
                continue
            videos.extend([(x['frames'], {
                k: v
                for k, v in x.items() if k not in {'type', 'frames'}
            }) for x in content if x['type'] == 'video'])
        return videos



    def preprocess(self, messages: List[Dict], mm_processor_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""

        min_pixels, max_pixels = self.get_processor_args(mm_processor_kwargs)
        merge_length = self.processor.image_processor.merge_size**2

        outputs = []
        # Single pass over messages to preserve content order
        for message in messages:
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get('type') == 'image':
                    image = item['image'].convert('RGB')
                    result = self.processor.image_processor(
                        images=image,
                        videos=None,
                        size={
                            'shortest_edge': min_pixels,
                            'longest_edge': max_pixels
                        },
                        return_tensors='pt')
                    image_tokens = result['image_grid_thw'].prod(
                        dim=1) // merge_length
                    result.update(
                        dict(image_size=image.size,
                             image_tokens=image_tokens,
                             image_token_id=self.image_token_id,
                             content_type='image'))
                    outputs.append(result)
                elif item.get('type') == 'video':
                    frames = [f.convert('RGB') for f in item['frames']]
                    result = self.processor.image_processor(
                        images=None,
                        videos=frames,
                        size={
                            'shortest_edge': min_pixels,
                            'longest_edge': max_pixels
                        },
                        return_tensors='pt')
                    video_tokens = result['video_grid_thw'].prod(
                        dim=1) // merge_length
                    result.update(
                        dict(image_tokens=video_tokens,
                             image_token_id=self.video_token_id,
                             content_type='video'))
                    outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def proc_messages(self, messages, chat_template, sequence_start,
                      tools=None, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        VISION_TOKEN = '<VISION_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        if VisionModel.IMAGE_TOKEN_included(messages):
            # backward compatibility
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt = ''.join(content)
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{VISION_TOKEN}<|vision_end|>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                _content = []
                for item in content:
                    if item['type'] == 'text':
                        _content.append(item['text'])
                    elif item['type'] in ['image', 'image_url']:
                        _content.append(f'<|vision_start|>{VISION_TOKEN}<|vision_end|>')
                    elif item['type'] in ['video', 'video_url']:
                        _content.append(f'<|vision_start|>{VISION_TOKEN}<|vision_end|>')
                    else:
                        raise ValueError(f'Unsupported message type: {item["type"]}')
                message = dict(role=role, content=''.join(_content))
                prompt_messages.append(message)
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start,
                                               tools=tools,
                                               **chat_template_kwargs)
        return prompt, VISION_TOKEN

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   tools: Optional[List[object]] = None,
                   chat_template_kwargs: Optional[Dict] = None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, VISION_TOKEN = self.proc_messages(
            messages, chat_template, sequence_start,
            tools=tools, chat_template_kwargs=chat_template_kwargs)

        # Collect preprocess results
        preps = [x['content'] for x in messages
                 if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        segs = prompt.split(VISION_TOKEN)
        assert len(segs) == len(preps) + 1, \
            (f'the number of {VISION_TOKEN} is not equal '
             f'to input images/videos, {len(segs) - 1} vs {len(preps)}')

        # Build input_ids with per-item token_id
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                token_id = preps[i - 1]['image_token_id']
                input_ids.extend([token_id] * image_tokens)
            token_ids = tokenizer.encode(
                seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def build_model(self):
        check_transformers()
        from transformers import Qwen3VLForConditionalGeneration
        arch = self.hf_config.architectures[0]

        if arch == 'Qwen3VLForConditionalGeneration':
            # Standard Qwen3VL — use transformers class directly
            if self.with_llm:
                self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path, device_map='cpu')
            else:
                from accelerate import init_empty_weights
                with init_empty_weights():
                    config = self.hf_config
                    config.tie_word_embeddings = False
                    if hasattr(config, 'text_config'):
                        config.text_config.tie_word_embeddings = False
                    model = Qwen3VLForConditionalGeneration._from_config(config)
                    if hasattr(Qwen3VLForConditionalGeneration, 'visual'):
                        model.visual = model.model.visual
                    del model.model
                    del model.lm_head
                    model.half()

                from accelerate import load_checkpoint_and_dispatch
                from lmdeploy.vl.model.utils import disable_logging
                with disable_logging():
                    load_checkpoint_and_dispatch(
                        model=model,
                        checkpoint=self.model_path,
                        device_map='auto',
                        max_memory=self.max_memory,
                        no_split_module_classes=['Qwen3VLVisionBlock'],
                        dtype=torch.half)
                self.model = model.eval()
        else:
            # VL MoE variants (Qwen3_5MoeForConditionalGeneration, etc.)
            # whose arch is not in transformers — load vision model directly
            self._build_vision_model_direct()

    def _build_vision_model_direct(self):
        """Load vision encoder directly for archs not in transformers.

        Creates Qwen3VLVisionModel from vision_config and loads weights
        from safetensors, bypassing the need for the full model class.
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionConfig, Qwen3VLVisionModel)

        # Build vision config from HF config's vision_config dict
        vc_dict = self.hf_config.vision_config
        if hasattr(vc_dict, 'to_dict'):
            vc_dict = vc_dict.to_dict()
        vc_dict.pop('model_type', None)
        vision_cfg = Qwen3VLVisionConfig(**vc_dict)

        # Create and load vision model
        vision_model = Qwen3VLVisionModel(vision_cfg)

        # Load visual weights from safetensors
        import json
        import os
        index_path = os.path.join(self.model_path,
                                  'model.safetensors.index.json')
        with open(index_path) as f:
            index = json.load(f)
        # Find shards containing visual weights
        visual_shards = set()
        for k, shard in index['weight_map'].items():
            if k.startswith('model.visual.'):
                visual_shards.add(shard)

        from safetensors.torch import load_file
        visual_state = {}
        for shard in visual_shards:
            shard_path = os.path.join(self.model_path, shard)
            state_dict = load_file(shard_path)
            for k, v in state_dict.items():
                if k.startswith('model.visual.'):
                    visual_state[k[len('model.visual.'):]] = v

        missing, unexpected = vision_model.load_state_dict(
            visual_state, strict=False)
        if missing:
            logger.warning(f'Vision model missing keys: {missing}')
        if unexpected:
            logger.warning(f'Vision model unexpected keys: {unexpected}')

        self.model = vision_model.half().eval()
        # Distribute vision encoder across available GPUs using accelerate,
        # matching the pattern in build_model() for standard Qwen3VL.
        # This avoids putting the entire ~910 MB encoder on GPU0 only,
        # which wastes KV cache capacity (AllReduce(min) across GPUs).
        if self.max_memory and len(self.max_memory) > 1:
            from accelerate import dispatch_model
            from accelerate.utils import (get_balanced_memory,
                                          infer_auto_device_map)
            max_memory = get_balanced_memory(
                self.model,
                max_memory=self.max_memory,
                dtype=torch.half,
                no_split_module_classes=['Qwen3VLVisionBlock'])
            device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                no_split_module_classes=['Qwen3VLVisionBlock'],
                dtype=torch.half)
            self.model = dispatch_model(self.model, device_map=device_map)
        else:
            if self.max_memory:
                device = list(self.max_memory.keys())[0]
            else:
                device = 'cuda'
            self.model = self.model.to(device)


    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        dtype = torch.half
        from transformers.models.qwen3_vl.modeling_qwen3_vl import \
            Qwen3VLVisionModel
        if isinstance(self.model, Qwen3VLVisionModel):
            visual = self.model
        else:
            visual = getattr(self.model, 'visual',
                             None) or self.model.model.visual
        device = next(visual.parameters()).device
        merge_length = self.processor.image_processor.merge_size**2
        outputs = []
        # Process items one by one to preserve order and handle mixed types
        for item in inputs:
            ct = item.get('content_type', 'image')
            if ct == 'video':
                pv = item['pixel_values_videos'].type(dtype).to(device)
                grid = item['video_grid_thw'].to(device)
            else:
                pv = item['pixel_values'].type(dtype).to(device)
                grid = item['image_grid_thw'].to(device)
            embeds = visual(pv, grid_thw=grid)
            if isinstance(embeds, tuple):
                embeds = embeds[0]
            split_size = grid.prod(dim=1) // merge_length
            embeds = embeds.split(split_size.tolist())
            outputs.extend(embeds)
        messages.append(dict(role='forward', content=outputs))
        return messages

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     tools: Optional[List[object]] = None,
                     chat_template_kwargs: Optional[Dict] = None,
                     **kwargs):
        import numpy as np
        prompt, VISION_TOKEN = self.proc_messages(
            messages, chat_template, sequence_start,
            tools=tools, chat_template_kwargs=chat_template_kwargs)

        # Collect features and preprocess results
        features = [x['content'] for x in messages
                    if x['role'] == 'forward'][0]
        features = [x.cpu() for x in features]
        inputs = [x['content'] for x in messages
                  if x['role'] == 'preprocess'][0]

        segs = prompt.split(VISION_TOKEN)
        assert len(segs) == len(features) + 1, \
            (f'the number of {VISION_TOKEN} is not equal '
             f'to input images/videos, {len(segs) - 1} vs {len(features)}')

        # Build input_ids with per-item token_id
        input_ids = []
        begins = []
        ends = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                image_dim = features[i - 1].shape[0]
                token_id = inputs[i - 1]['image_token_id']
                begins.append(len(input_ids))
                ends.append(begins[-1] + image_dim)
                input_ids.extend([token_id] * image_dim)
            seg_ids = tokenizer.encode(
                seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        ranges = np.stack([begins, ends], axis=1).tolist()

        # Build grid_thws for MRoPE (mixed image/video)
        grid_thws = []
        for x in inputs:
            if x.get('content_type') == 'video':
                grid_thws.append(x['video_grid_thw'].tolist()[0])
            else:
                grid_thws.append(x['image_grid_thw'].tolist()[0])

        seq_len = len(input_ids)
        from lmdeploy.vl.model.qwen2 import Qwen2VLModel
        mrope_position_ids, mrope_position_delta = \
            Qwen2VLModel.get_mrope_info(seq_len, grid_thws, ranges)
        meta = dict(mrope_position_ids=mrope_position_ids,
                    mrope_position_delta=mrope_position_delta)
        return dict(prompt=prompt,
                    input_ids=input_ids,
                    input_embeddings=features,
                    input_embedding_ranges=ranges,
                    input_meta=meta)
