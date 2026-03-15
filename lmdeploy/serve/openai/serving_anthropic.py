# Copyright (c) OpenMMLab. All rights reserved.
"""Anthropic Messages API translation logic.

Converts between Anthropic Messages API format and the internal
OpenAI-compatible format used by lmdeploy's AsyncEngine.
"""
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from fastapi.responses import JSONResponse

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.anthropic_protocol import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorEvent,
    ImageBlock,
    InputJsonDelta,
    MessageDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    MessagesRequest,
    MessagesResponse,
    PingEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
)


def _ensure_toolu_prefix(tool_id: str) -> str:
    """Add ``toolu_`` prefix only if not already present (fix #1)."""
    if tool_id.startswith('toolu_'):
        return tool_id
    return f'toolu_{tool_id}'


def translate_messages_request(
    request: MessagesRequest,
) -> Tuple[List[Dict[str, Any]], GenerationConfig, Optional[List[Dict]],
           Optional[str]]:
    """Convert an Anthropic MessagesRequest to internal format.

    Returns:
        (openai_messages, gen_config, tools, tool_choice)
    """
    openai_messages: List[Dict[str, Any]] = []

    # --- system field → system-role message at index 0 ---
    if request.system is not None:
        if isinstance(request.system, str):
            sys_text = request.system
        else:
            # list of content blocks — join text blocks
            sys_text = ''.join(
                b.text for b in request.system
                if hasattr(b, 'text') and hasattr(b, 'type')
                and b.type == 'text')
        openai_messages.append({'role': 'system', 'content': sys_text})

    # --- convert each message ---
    for msg in request.messages:
        if isinstance(msg.content, str):
            openai_messages.append({
                'role': msg.role,
                'content': msg.content,
            })
            continue

        # Content is a list of blocks
        if msg.role == 'assistant':
            _translate_assistant_message(msg, openai_messages)
        else:
            _translate_user_message(msg, openai_messages)

    # --- generation config ---
    kwargs = dict(max_new_tokens=request.max_tokens)
    if request.temperature is not None:
        kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        kwargs['top_p'] = request.top_p
    if request.top_k is not None:
        kwargs['top_k'] = request.top_k
    if request.stop_sequences is not None:
        kwargs['stop_words'] = request.stop_sequences
    gen_config = GenerationConfig(**kwargs)

    # --- tools ---
    tools = None
    if request.tools:
        tools = [
            {
                'name': t.name,
                'description': t.description or '',
                'parameters': t.input_schema,
            }
            for t in request.tools
        ]

    # --- tool_choice mapping ---
    tool_choice = None
    if request.tool_choice is not None:
        tc = request.tool_choice
        if tc.type == 'auto':
            tool_choice = 'auto'
        elif tc.type == 'any':
            tool_choice = 'required'
        elif tc.type == 'tool' and tc.name:
            tool_choice = tc.name
    elif tools:
        # Match vLLM: auto-set tool_choice when tools are present
        tool_choice = 'auto'

    return openai_messages, gen_config, tools, tool_choice


def _translate_user_message(msg, openai_messages):
    """Translate a user message with content block list.

    Fix #3: tool_result with None content → empty string.
    Fix #4: tool_result containing ImageBlock → inject follow-up user
    message with the image (matching vLLM behaviour).
    """
    parts = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append({'type': 'text', 'text': block.text})
        elif isinstance(block, ImageBlock):
            src = block.source
            if src.type == 'base64':
                url = f'data:{src.media_type};base64,{src.data}'
            else:
                url = src.url or ''
            parts.append({
                'type': 'image_url',
                'image_url': {'url': url},
            })
        elif isinstance(block, ToolResultBlock):
            # Extract text content (fix #3: None → empty string)
            if block.content is None:
                content_text = ''
            elif isinstance(block.content, str):
                content_text = block.content
            else:
                # List of TextBlock / ImageBlock
                text_parts = []
                image_parts = []
                for item in block.content:
                    if isinstance(item, TextBlock):
                        text_parts.append(item.text)
                    elif isinstance(item, ImageBlock):
                        image_parts.append(item)
                content_text = ''.join(text_parts)

                # Fix #4: images inside tool_result → follow-up user message
                if image_parts:
                    openai_messages.append({
                        'role': 'tool',
                        'tool_call_id': block.tool_use_id,
                        'content': content_text,
                    })
                    # Inject a follow-up user message with the images
                    img_content = []
                    for img in image_parts:
                        src = img.source
                        if src.type == 'base64':
                            u = f'data:{src.media_type};base64,{src.data}'
                        else:
                            u = src.url or ''
                        img_content.append({
                            'type': 'image_url',
                            'image_url': {'url': u},
                        })
                    openai_messages.append({
                        'role': 'user',
                        'content': img_content,
                    })
                    continue

            openai_messages.append({
                'role': 'tool',
                'tool_call_id': block.tool_use_id,
                'content': content_text,
            })
            continue
        elif isinstance(block, ThinkingBlock):
            parts.append({'type': 'text', 'text': block.thinking})
        else:
            # ToolUseBlock in user message — treat as text
            parts.append({'type': 'text', 'text': str(block)})

    if parts:
        if len(parts) == 1 and parts[0]['type'] == 'text':
            openai_messages.append({
                'role': msg.role,
                'content': parts[0]['text'],
            })
        else:
            openai_messages.append({
                'role': msg.role,
                'content': parts,
            })


def _translate_assistant_message(msg, openai_messages):
    """Translate an assistant message with content block list.

    Fix #2: ThinkingBlock → reasoning_content field (preserves thinking
    semantics) instead of joining into text_parts.
    """
    text_parts = []
    reasoning_parts = []
    tool_calls = []

    for block in msg.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ThinkingBlock):
            # Fix #2: preserve thinking semantics
            reasoning_parts.append(block.thinking)
        elif isinstance(block, ToolUseBlock):
            tool_calls.append({
                'id': block.id,
                'type': 'function',
                'function': {
                    'name': block.name,
                    'arguments': json.dumps(block.input),
                },
            })

    result: Dict[str, Any] = {
        'role': 'assistant',
        'content': ''.join(text_parts) if text_parts else None,
    }
    if reasoning_parts:
        result['reasoning_content'] = ''.join(reasoning_parts)
    if tool_calls:
        result['tool_calls'] = tool_calls
    openai_messages.append(result)


def map_finish_reason(finish_reason: str,
                      has_tool_calls: bool = False,
                      stop_word_matched: bool = False) -> str:
    """Map engine finish_reason to Anthropic stop_reason.

    Args:
        finish_reason: Engine finish reason ('stop', 'length', etc.)
        has_tool_calls: Whether tool calls were detected in output.
        stop_word_matched: Whether a stop word/sequence was matched.

    Returns:
        Anthropic stop_reason string.
    """
    if has_tool_calls:
        return 'tool_use'
    if finish_reason == 'length':
        return 'max_tokens'
    if finish_reason == 'stop' and stop_word_matched:
        return 'stop_sequence'
    return 'end_turn'


def build_messages_response(
    request_id: str,
    model: str,
    text: str,
    tool_calls: Optional[List] = None,
    reasoning_content: Optional[str] = None,
    finish_reason: str = 'stop',
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_word_matched: bool = False,
    matched_stop_sequence: Optional[str] = None,
) -> MessagesResponse:
    """Build a non-streaming Anthropic MessagesResponse.

    Content block ordering: ThinkingBlock → TextBlock → ToolUseBlock(s).
    """
    content = []

    # Thinking block first (if reasoning parser produced content)
    if reasoning_content:
        content.append(ThinkingBlock(
            thinking=reasoning_content,
            signature=uuid.uuid4().hex))

    # Text block
    if text:
        content.append(TextBlock(text=text))

    # Tool use blocks (fix #1: no double toolu_ prefix)
    has_tool_calls = False
    if tool_calls:
        has_tool_calls = True
        for tc in tool_calls:
            tool_id = tc.id if hasattr(tc, 'id') else tc.get('id', '')
            tool_id = _ensure_toolu_prefix(tool_id)
            name = (tc.function.name if hasattr(tc, 'function')
                    else tc.get('function', {}).get('name', ''))
            args_str = (tc.function.arguments if hasattr(tc, 'function')
                        else tc.get('function', {}).get('arguments', '{}'))
            try:
                input_dict = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                input_dict = {}
            content.append(ToolUseBlock(
                id=tool_id, name=name, input=input_dict))

    stop_reason = map_finish_reason(
        finish_reason, has_tool_calls, stop_word_matched)

    # Fix #6: stop_sequence field
    stop_sequence = None
    if stop_reason == 'stop_sequence' and matched_stop_sequence:
        stop_sequence = matched_stop_sequence

    msg_id = request_id
    if not msg_id.startswith('msg_'):
        msg_id = f'msg_{msg_id}'

    return MessagesResponse(
        id=msg_id,
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )



async def anthropic_stream_generator(
    result_generator,
    request: MessagesRequest,
    request_id: str,
    model: str,
    tool_parser=None,
    reasoning_parser=None,
    enable_thinking=None,
) -> AsyncGenerator[str, None]:
    """Yield SSE event strings in Anthropic streaming format.

    Event order:
      message_start → ping → (content_block_start → content_block_delta* →
      content_block_stop)* → message_delta → message_stop

    Fix #7: streaming errors emit an ``error`` event instead of being
    silently swallowed.
    """
    msg_id = (request_id if request_id.startswith('msg_')
              else f'msg_{request_id}')

    # Emit message_start with empty content
    start_msg = MessagesResponse(
        id=msg_id,
        model=model,
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=0, output_tokens=0),
    )
    yield _sse_event('message_start',
                     MessageStartEvent(message=start_msg).model_dump_json())

    # Emit ping (keep-alive, matches Anthropic SDK expectations)
    yield _sse_event('ping', PingEvent().model_dump_json())

    block_index = 0
    current_block_type = None  # 'thinking', 'text', 'tool_use'
    thinking_started = False
    thinking_signature = None
    text_started = False

    previous_text = ''
    current_text = ''
    previous_token_ids = []
    current_token_ids = []
    final_res = None

    try:
        async for res in result_generator:
            final_res = res
            delta_text = res.response or ''
            delta_token_ids = (res.token_ids
                               if res.token_ids is not None else [])

            current_text += delta_text
            current_token_ids = current_token_ids + delta_token_ids

            # Apply reasoning parser if configured
            reasoning_delta_text = None
            content_delta_text = delta_text
            if (reasoning_parser is not None
                    and enable_thinking is not False):
                rd = reasoning_parser \
                    .extract_reasoning_content_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids)
                if rd is not None:
                    reasoning_delta_text = rd.reasoning_content
                    content_delta_text = rd.content

            # Emit thinking deltas
            if reasoning_delta_text:
                if not thinking_started:
                    thinking_started = True
                    thinking_signature = uuid.uuid4().hex
                    current_block_type = 'thinking'
                    yield _sse_event(
                        'content_block_start',
                        ContentBlockStartEvent(
                            index=block_index,
                            content_block=ThinkingBlock(thinking=''),
                        ).model_dump_json())
                yield _sse_event(
                    'content_block_delta',
                    ContentBlockDeltaEvent(
                        index=block_index,
                        delta=ThinkingDelta(
                            thinking=reasoning_delta_text),
                    ).model_dump_json())

            # Emit text deltas (skip empty deltas)
            if content_delta_text:
                if current_block_type == 'thinking':
                    # Emit signature_delta before closing thinking block
                    yield _sse_event(
                        'content_block_delta',
                        ContentBlockDeltaEvent(
                            index=block_index,
                            delta=SignatureDelta(
                                signature=thinking_signature or ''),
                        ).model_dump_json())
                    # Close thinking block, start text block
                    yield _sse_event(
                        'content_block_stop',
                        ContentBlockStopEvent(
                            index=block_index).model_dump_json())
                    block_index += 1
                    current_block_type = None

                if not text_started:
                    text_started = True
                    current_block_type = 'text'
                    yield _sse_event(
                        'content_block_start',
                        ContentBlockStartEvent(
                            index=block_index,
                            content_block=TextBlock(text=''),
                        ).model_dump_json())
                yield _sse_event(
                    'content_block_delta',
                    ContentBlockDeltaEvent(
                        index=block_index,
                        delta=TextDelta(text=content_delta_text),
                    ).model_dump_json())

            previous_text = current_text
            previous_token_ids = current_token_ids

    except Exception as exc:
        # Fix #7: emit error event instead of silent pass
        yield _sse_event(
            'error',
            ErrorEvent(
                error=AnthropicErrorDetail(
                    type='api_error',
                    message=str(exc)),
            ).model_dump_json())
        return

    # Close last content block if open
    if current_block_type is not None:
        if current_block_type == 'thinking':
            # Emit signature before closing thinking block
            yield _sse_event(
                'content_block_delta',
                ContentBlockDeltaEvent(
                    index=block_index,
                    delta=SignatureDelta(
                        signature=thinking_signature or ''),
                ).model_dump_json())
        yield _sse_event(
            'content_block_stop',
            ContentBlockStopEvent(index=block_index).model_dump_json())
        block_index += 1

    # Handle tool calls from full text (non-streaming tool parsing)
    has_tool_calls = False
    if tool_parser is not None and request.tools:
        try:
            tool_info = tool_parser.extract_tool_calls(
                current_text, request=None)
            if tool_info.tool_calls:
                has_tool_calls = True
                for tc in tool_info.tool_calls:
                    tool_id = tc.id if hasattr(tc, 'id') else ''
                    tool_id = _ensure_toolu_prefix(tool_id)
                    name = (tc.function.name
                            if hasattr(tc, 'function') else '')
                    args_str = (tc.function.arguments
                                if hasattr(tc, 'function') else '{}')
                    yield _sse_event(
                        'content_block_start',
                        ContentBlockStartEvent(
                            index=block_index,
                            content_block=ToolUseBlock(
                                id=tool_id, name=name, input={}),
                        ).model_dump_json())
                    yield _sse_event(
                        'content_block_delta',
                        ContentBlockDeltaEvent(
                            index=block_index,
                            delta=InputJsonDelta(
                                partial_json=args_str),
                        ).model_dump_json())
                    yield _sse_event(
                        'content_block_stop',
                        ContentBlockStopEvent(
                            index=block_index).model_dump_json())
                    block_index += 1
        except Exception:
            pass

    # Determine stop reason
    finish_reason = final_res.finish_reason if final_res else 'stop'
    stop_reason = map_finish_reason(
        finish_reason or 'stop', has_tool_calls, False)

    input_tokens = final_res.input_token_len if final_res else 0
    output_tokens = final_res.generate_token_len if final_res else 0

    # message_delta with stop_reason and usage
    yield _sse_event(
        'message_delta',
        MessageDeltaEvent(
            delta=MessageDelta(stop_reason=stop_reason),
            usage=AnthropicUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens),
        ).model_dump_json())

    # message_stop
    yield _sse_event('message_stop',
                     MessageStopEvent().model_dump_json())

    # Terminal [DONE] signal (matches vLLM / Anthropic SDK expectations)
    yield 'data: [DONE]\n\n'


def _sse_event(event_type: str, data: str) -> str:
    """Format a single SSE event."""
    return f'event: {event_type}\ndata: {data}\n\n'


# Status code → Anthropic error type mapping
_ERROR_TYPE_MAP = {
    400: 'invalid_request_error',
    404: 'not_found_error',
    500: 'api_error',
    529: 'overloaded_error',
}


def create_anthropic_error(status_code: int, message: str) -> JSONResponse:
    """Return a JSONResponse with Anthropic error envelope.

    Args:
        status_code: HTTP status code (400, 404, 500, 529).
        message: Human-readable error description.
    """
    error_type = _ERROR_TYPE_MAP.get(status_code, 'api_error')
    return JSONResponse(
        status_code=status_code,
        content=AnthropicErrorResponse(
            error=AnthropicErrorDetail(type=error_type, message=message),
        ).model_dump(),
    )
