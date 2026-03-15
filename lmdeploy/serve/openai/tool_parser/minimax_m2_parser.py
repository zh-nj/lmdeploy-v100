# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from vllm's MinimaxM2ToolParser
import json
import re
import uuid
from typing import Dict, List, Optional, Sequence, Union

from lmdeploy.serve.openai.protocol import (ChatCompletionRequest,
                                             DeltaFunctionCall, DeltaMessage,
                                             DeltaToolCall,
                                             ExtractedToolCallInformation,
                                             FunctionCall, ToolCall)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


@ToolParserManager.register_module(name='minimax-m2')
class MinimaxM2ToolParser(ToolParser):
    """Tool call parser for MiniMax M2/M2.5 models.

    Uses <minimax:tool_call>...</minimax:tool_call> tags with
    <invoke name="...">...</invoke> and <parameter name="...">...</parameter>
    syntax.

    Used when --tool-call-parser minimax-m2 is set.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)

        self.tool_call_start_token = '<minimax:tool_call>'
        self.tool_call_end_token = '</minimax:tool_call>'
        self.invoke_start_prefix = '<invoke name='
        self.invoke_end_token = '</invoke>'
        self.parameter_prefix = '<parameter name='
        self.parameter_end_token = '</parameter>'

        self.tool_call_regex = re.compile(
            r'<minimax:tool_call>(.*?)</minimax:tool_call>', re.DOTALL)
        self.invoke_regex = re.compile(
            r'<invoke name=(.*?)</invoke>', re.DOTALL)
        self.parameter_regex = re.compile(
            r'<parameter name=(.*?)</parameter>', re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                'The model tokenizer must be passed to the ToolParser '
                'constructor during construction.')

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(
            self.tool_call_end_token)

        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                'MiniMax M2 tool parser could not locate tool call '
                'start/end tokens in the tokenizer!')

    @staticmethod
    def _extract_name(name_str: str) -> str:
        """Extract name from quoted string like "func_name" or 'func_name'."""
        name_str = name_str.strip()
        if len(name_str) >= 2 and name_str[0] in ('"', "'"):
            return name_str[1:-1]
        return name_str

    def _parse_single_invoke(
        self,
        invoke_str: str,
        tools: Optional[List] = None,
    ) -> Optional[ToolCall]:
        """Parse a single <invoke> block into a ToolCall."""
        name_match = re.search(r'^([^>]+)', invoke_str)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))

        # Build param type map from tool schema
        param_config: Dict = {}
        if tools:
            for tool in tools:
                if (hasattr(tool, 'function')
                        and tool.function.name == function_name
                        and hasattr(tool.function, 'parameters')):
                    params = tool.function.parameters
                    if isinstance(params, dict) and 'properties' in params:
                        param_config = params['properties']
                    break

        # Extract parameters
        param_dict: Dict = {}
        for match in self.parameter_regex.findall(invoke_str):
            param_match = re.search(r'^([^>]+)>(.*)', match, re.DOTALL)
            if not param_match:
                continue
            param_name = self._extract_name(param_match.group(1))
            param_value = param_match.group(2).strip()
            if param_value.startswith('\n'):
                param_value = param_value[1:]
            if param_value.endswith('\n'):
                param_value = param_value[:-1]

            # Type conversion based on schema
            param_dict[param_name] = self._convert_value(
                param_value, param_config.get(param_name, {}))

        return ToolCall(
            type='function',
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(param_dict, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _convert_value(value: str, schema: dict) -> object:
        """Convert string value to appropriate type based on JSON schema."""
        if value.lower() in ('null', 'none', 'nil'):
            return None

        # Extract types from schema (handle anyOf/oneOf/type arrays)
        types = set()
        if isinstance(schema, dict):
            if 'type' in schema:
                t = schema['type']
                types.update(t if isinstance(t, list) else [t])
            for key in ('anyOf', 'oneOf'):
                if key in schema and isinstance(schema[key], list):
                    for choice in schema[key]:
                        if isinstance(choice, dict) and 'type' in choice:
                            t = choice['type']
                            types.update(
                                t if isinstance(t, list) else [t])

        # Try conversions in priority order
        for t in ['integer', 'number', 'boolean', 'object', 'array']:
            if t not in types:
                continue
            if t == 'integer':
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            elif t == 'number':
                try:
                    v = float(value)
                    return int(v) if v == int(v) else v
                except (ValueError, TypeError):
                    continue
            elif t == 'boolean':
                low = value.lower().strip()
                if low in ('true', '1', 'yes'):
                    return True
                if low in ('false', '0', 'no'):
                    return False
                continue
            elif t in ('object', 'array'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        # Fallback: try JSON parse, then return as string
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls = []
            for tc_match in self.tool_call_regex.findall(model_output):
                for inv_match in self.invoke_regex.findall(tc_match):
                    tc = self._parse_single_invoke(
                        inv_match,
                        request.tools if request else None)
                    if tc:
                        tool_calls.append(tc)

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output)

            first_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_idx] if first_idx > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content)
        except Exception:
            logger.exception('Error extracting MiniMax M2 tool calls')
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # Detect tool call start
        if self.tool_call_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        # Content before tool call
        if self.tool_call_start_token in delta_text:
            idx = delta_text.find(self.tool_call_start_token)
            content_before = delta_text[:idx]
            if content_before:
                return DeltaMessage(content=content_before)
            return None

        # Already past tool call start — check for complete invokes
        # Count completed invokes so far
        completed = len(self.invoke_regex.findall(current_text))
        current_idx = len(self.prev_tool_call_arr)

        if completed > current_idx:
            # A new invoke just completed — parse it
            all_invokes = list(self.invoke_regex.finditer(current_text))
            for i in range(current_idx, completed):
                inv_str = all_invokes[i].group(1)
                tc = self._parse_single_invoke(
                    inv_str, request.tools if request else None)
                if tc:
                    tool_id = f'call_{uuid.uuid4().hex[:24]}'
                    self.prev_tool_call_arr.append({
                        'name': tc.function.name,
                        'arguments': tc.function.arguments,
                    })
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=i,
                            id=tool_id,
                            type='function',
                            function=DeltaFunctionCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    ])

        return None
