# Copyright (c) OpenMMLab. All rights reserved.
"""Tool call parser for Qwen3.5 models.

Qwen3.5 uses an XML-attribute format for tool calls, different from Qwen3's
JSON-in-tags format:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Beijing
    </parameter>
    </function>
    </tool_call>

This parser handles both non-streaming and streaming extraction of this format.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import shortuuid

from lmdeploy.serve.openai.protocol import (ChatCompletionRequest,
                                             DeltaFunctionCall, DeltaMessage,
                                             DeltaToolCall,
                                             ExtractedToolCallInformation,
                                             FunctionCall, ToolCall)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')

# Regex for complete tool_call blocks
_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*'
    r'<function=([^>]+)>\s*'
    r'(.*?)'
    r'</function>\s*'
    r'</tool_call>',
    re.DOTALL,
)

# Regex for individual parameters inside a function block
_PARAM_RE = re.compile(
    r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>',
    re.DOTALL,
)


def _parse_xml_tool_call(func_name: str, body: str,
                         tools: Optional[list] = None) -> ToolCall:
    """Parse a single XML tool call block into a ToolCall object.

    Args:
        func_name: Function name extracted from <function=name>.
        body: The content between <function=...> and </function>.
        tools: Optional list of tool definitions for type-aware conversion.
    """
    params = {}
    param_types = _get_param_types(func_name, tools)
    for m in _PARAM_RE.finditer(body):
        pname = m.group(1).strip()
        pvalue = m.group(2).strip()
        params[pname] = _convert_param_value(pvalue, param_types.get(pname, 'string'))
    arguments = json.dumps(params, ensure_ascii=False)
    return ToolCall(function=FunctionCall(name=func_name, arguments=arguments))


def _get_param_types(func_name: str,
                     tools: Optional[list] = None) -> Dict[str, str]:
    """Extract parameter type map from tool definitions."""
    if not tools:
        return {}
    for tool in tools:
        fn = tool.get('function', tool) if isinstance(tool, dict) else None
        if fn is None:
            continue
        if fn.get('name') != func_name:
            continue
        props = fn.get('parameters', {}).get('properties', {})
        return {k: v.get('type', 'string') for k, v in props.items()}
    return {}


def _convert_param_value(raw: str, ptype: str):
    """Convert a raw string parameter value based on its declared type."""
    if raw.lower() == 'null':
        return None
    ptype = ptype.strip().lower()
    if ptype in ('integer', 'int'):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    elif ptype in ('number', 'float', 'double'):
        try:
            v = float(raw)
            return int(v) if v == int(v) else v
        except (ValueError, TypeError):
            return raw
    elif ptype in ('boolean', 'bool'):
        return raw.strip().lower() == 'true'
    elif ptype in ('array', 'object'):
        # Try JSON first, then Python literal
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            import ast
            return ast.literal_eval(raw)
        except Exception:
            return raw
    # Default: string
    return raw


@dataclass
class _StreamState:
    """Streaming parser state stored on the request object."""
    position: int = 0
    current_index: int = -1
    id: str = ''
    # Accumulate content between <tool_call> and </tool_call>
    in_tool_call: bool = False
    tool_call_buf: str = ''

    def reset_tool_call(self):
        self.id = ''
        self.in_tool_call = False
        self.tool_call_buf = ''


@ToolParserManager.register_module(['qwen3_5', 'qwen3-5', 'step3p5'])
class Qwen3_5ToolParser(ToolParser):
    """Parser for Qwen3.5 model's XML-attribute tool call format.

    Format:
        <tool_call>
        <function=func_name>
        <parameter=param1>value1</parameter>
        <parameter=param2>value2</parameter>
        </function>
        </tool_call>
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output."""
        text = model_output
        tools_raw = None
        if request and request.tools:
            tools_raw = [
                t.function.model_dump() if hasattr(t, 'function') else t
                for t in request.tools
            ]

        buf = []
        scan_pos = 0
        tool_calls = []
        for match in _TOOL_CALL_RE.finditer(text):
            buf.append(text[scan_pos:match.start()])
            scan_pos = match.end()
            func_name = match.group(1).strip()
            body = match.group(2)
            tc = _parse_xml_tool_call(func_name, body, tools_raw)
            tool_calls.append(tc)
        if scan_pos < len(text):
            buf.append(text[scan_pos:])
        text = ''.join(buf).strip()

        return ExtractedToolCallInformation(
            content=text if text else None,
            tool_calls=tool_calls,
            tools_called=bool(tool_calls),
        )

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
        """Extract tool calls from streaming model output."""
        state: _StreamState = getattr(request, '_tool_parser_state', None)
        if state is None:
            state = _StreamState()
            setattr(request, '_tool_parser_state', state)

        new_content = current_text[state.position:]
        if not new_content:
            return None

        delta = DeltaMessage()
        text_parts = []

        while new_content:
            if not state.in_tool_call:
                # Look for <tool_call> start
                idx = new_content.find(self.tool_start_token)
                if idx == -1:
                    # No tool_call tag — check if we might be seeing a
                    # partial tag at the end
                    # e.g. new_content ends with "<tool" or "<tool_cal"
                    partial = self._partial_start_match(new_content)
                    if partial > 0:
                        # Output everything before the potential partial tag
                        safe = new_content[:len(new_content) - partial]
                        if safe:
                            text_parts.append(safe)
                        state.position += len(new_content) - partial
                    else:
                        text_parts.append(new_content)
                        state.position += len(new_content)
                    break
                else:
                    # Text before the tag
                    if idx > 0:
                        text_parts.append(new_content[:idx])
                    state.in_tool_call = True
                    state.tool_call_buf = ''
                    consumed = idx + len(self.tool_start_token)
                    state.position += consumed
                    new_content = new_content[consumed:]
            else:
                # Inside a tool_call — look for </tool_call>
                end_idx = new_content.find(self.tool_end_token)
                if end_idx == -1:
                    # Haven't seen end tag yet, buffer everything
                    state.tool_call_buf += new_content
                    state.position += len(new_content)
                    break
                else:
                    state.tool_call_buf += new_content[:end_idx]
                    consumed = end_idx + len(self.tool_end_token)
                    state.position += consumed
                    new_content = new_content[consumed:]

                    # Parse the complete tool call
                    tc = self._parse_buffered_tool_call(state, request)
                    if tc is not None:
                        if delta.tool_calls is None:
                            delta.tool_calls = []
                        delta.tool_calls.append(tc)
                    state.reset_tool_call()

        if text_parts:
            delta.content = ''.join(text_parts)

        return delta

    def _partial_start_match(self, text: str) -> int:
        """Check if text ends with a partial '<tool_call>' prefix.

        Returns the length of the partial match (0 if none).
        """
        tag = self.tool_start_token
        for length in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def _parse_buffered_tool_call(
        self, state: _StreamState,
        request: ChatCompletionRequest
    ) -> Optional[DeltaToolCall]:
        """Parse a complete buffered tool call body into a DeltaToolCall."""
        body = state.tool_call_buf.strip()

        # Extract function name
        func_match = re.match(r'<function=([^>]+)>\s*(.*?)\s*</function>',
                              body, re.DOTALL)
        if not func_match:
            logger.warning('Failed to parse tool call body: %s', body[:200])
            return None

        func_name = func_match.group(1).strip()
        func_body = func_match.group(2)

        # Extract parameters
        tools_raw = None
        if request and request.tools:
            tools_raw = [
                t.function.model_dump() if hasattr(t, 'function') else t
                for t in request.tools
            ]
        param_types = _get_param_types(func_name, tools_raw)
        params = {}
        for m in _PARAM_RE.finditer(func_body):
            pname = m.group(1).strip()
            pvalue = m.group(2).strip()
            params[pname] = _convert_param_value(
                pvalue, param_types.get(pname, 'string'))

        arguments = json.dumps(params, ensure_ascii=False)

        # Allocate ID and index
        if not state.id:
            state.id = f'chatcmpl-tool-{shortuuid.random()}'
            state.current_index += 1

        return DeltaToolCall(
            id=state.id,
            index=state.current_index,
            function=DeltaFunctionCall(
                name=func_name,
                arguments=arguments,
            ).model_dump(exclude_none=True),
        )
