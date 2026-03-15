# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from vllm's MiniMaxM2ReasoningParser
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name='minimax-m2')
class MiniMaxM2ReasoningParser(ReasoningParser):
    """Reasoning parser for MiniMax M2/M2.5 models.

    MiniMax M2 models don't generate <think> start token, only </think>
    end token. All content before </think> is reasoning, content after
    is the actual response.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.think_start_token = '<think>'
        self.think_end_token = '</think>'
        self.think_end_token_id = self.vocab.get(self.think_end_token)

        if self.think_end_token_id is None:
            raise RuntimeError(
                'MiniMax M2 reasoning parser could not locate '
                '</think> token in the tokenizer!')

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
        # Skip single end token
        if (len(delta_token_ids) == 1
                and delta_token_ids[0] == self.think_end_token_id):
            return None

        # Already past reasoning phase
        if self.think_end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        # End token in current delta — split reasoning and content
        if self.think_end_token_id in delta_token_ids:
            end_index = delta_text.find(self.think_end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token):]
            return DeltaMessage(
                reasoning_content=reasoning if reasoning else None,
                content=content if content else None,
            )

        # No end token yet — all content is reasoning
        return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        if self.think_end_token not in model_output:
            # No reasoning markers at all — treat everything as reasoning
            # (M2 models always think first)
            return model_output, None

        # Add start token if missing (M2 doesn't generate it)
        if self.think_start_token not in model_output:
            model_output = f'{self.think_start_token}{model_output}'

        start = model_output.find(self.think_start_token)
        end = model_output.find(self.think_end_token)
        reasoning = model_output[start + len(self.think_start_token):end]
        content = model_output[end + len(self.think_end_token):]

        if reasoning.startswith('\n'):
            reasoning = reasoning[1:]
        if reasoning.endswith('\n'):
            reasoning = reasoning[:-1]

        return (reasoning if reasoning else None,
                content if content else None)
