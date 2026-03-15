# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from vLLM's Step3p5ReasoningParser
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from .reasoning_parser import ReasoningParserManager


@ReasoningParserManager.register_module(name='step3p5')
class Step3p5ReasoningParser(DeepSeekR1ReasoningParser):
    """Reasoning parser for Step3p5 models.

    Step3p5 usually starts generation inside an implicit ``<think>`` block and
    only emits ``</think>`` when reasoning ends. It also tends to add one extra
    newline immediately before and after the closing token.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self._pending_reasoning_newline = False

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        **kwargs,
    ) -> Union[DeltaMessage, None]:
        if delta_text.startswith(self.think_end_token):
            self._pending_reasoning_newline = False
            content = delta_text[len(self.think_end_token):]
            if content.startswith('\n'):
                content = content.removeprefix('\n')
            return DeltaMessage(content=content)

        if previous_text.endswith(self.think_end_token) and delta_text:
            if delta_text == '\n':
                return None
            if delta_text.startswith('\n'):
                content = delta_text.removeprefix('\n')
                return DeltaMessage(content=content) if content else None

        ret = super().extract_reasoning_content_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            **kwargs,
        )

        if ret is None:
            return None

        if (self.think_start_token_id not in previous_token_ids
                and self.think_start_token_id not in delta_token_ids):
            if self.think_end_token_id in delta_token_ids:
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                ret = DeltaMessage(
                    reasoning_content=reasoning,
                    content=content or None,
                )
            elif self.think_end_token_id in previous_token_ids:
                ret = DeltaMessage(content=delta_text)
            else:
                ret = DeltaMessage(reasoning_content=delta_text)

        reasoning = ret.reasoning_content
        content = ret.content

        if reasoning is not None:
            if self._pending_reasoning_newline:
                reasoning = '\n' + reasoning
                self._pending_reasoning_newline = False

            if reasoning.endswith('\n'):
                reasoning = reasoning.removesuffix('\n')
                if self.think_end_token not in delta_text:
                    self._pending_reasoning_newline = True

        if content is not None:
            self._pending_reasoning_newline = False
            if self.think_end_token in delta_text and content.startswith('\n'):
                content = content.removeprefix('\n')

        reasoning = reasoning or None
        content = content or None
        if reasoning is None and content is None:
            return None

        return DeltaMessage(reasoning_content=reasoning, content=content)

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        reasoning, content = super().extract_reasoning_content(
            model_output=model_output,
            request=request,
            **kwargs,
        )
        if reasoning is not None:
            reasoning = reasoning.removesuffix('\n')
        if content is not None:
            content = content.removeprefix('\n')
        return reasoning or None, content or None
