# Copyright (c) OpenMMLab. All rights reserved.
"""Pydantic data models for the Anthropic Messages API.

Defines request/response schemas and SSE streaming event types for the
``/v1/messages`` endpoint.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Content block types (Task 1.1)
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """A text content block."""
    type: Literal['text'] = 'text'
    text: str


class ImageSource(BaseModel):
    """Image source — either base64-encoded data or a URL."""
    type: Literal['base64', 'url']
    media_type: Optional[
        Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    ] = None
    data: Optional[str] = None
    url: Optional[str] = None


class ImageBlock(BaseModel):
    """An image content block."""
    type: Literal['image'] = 'image'
    source: ImageSource


class ToolUseBlock(BaseModel):
    """A tool-use content block (assistant requesting a tool call)."""
    type: Literal['tool_use'] = 'tool_use'
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultBlock(BaseModel):
    """A tool-result content block (user providing tool output).

    content can be None (treated as empty string), a string, or a list
    of TextBlock / ImageBlock items.
    """
    type: Literal['tool_result'] = 'tool_result'
    tool_use_id: str
    content: Optional[Union[str, List[Union[TextBlock, ImageBlock]]]] = None
    is_error: Optional[bool] = False


class ThinkingBlock(BaseModel):
    """A thinking/reasoning content block.

    The ``signature`` field is required by the Anthropic API to verify
    thinking content integrity.  It is opaque to the server and simply
    round-tripped between client and model.
    """
    type: Literal['thinking'] = 'thinking'
    thinking: str
    signature: Optional[str] = None


# Discriminated union of all content block types.
ContentBlock = Union[
    TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
]


class AnthropicMessage(BaseModel):
    """A single message in the Anthropic conversation."""
    role: Literal['user', 'assistant']
    content: Union[str, List[ContentBlock]]


class MessageMetadata(BaseModel):
    """Optional request metadata."""
    user_id: Optional[str] = None


class AnthropicToolDef(BaseModel):
    """An Anthropic-format tool definition."""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

    @model_validator(mode='after')
    def _ensure_input_schema_type(self):
        """Auto-add ``type: "object"`` if missing (vLLM compat)."""
        if 'type' not in self.input_schema:
            self.input_schema['type'] = 'object'
        return self


class AnthropicToolChoice(BaseModel):
    """Tool-choice specification."""
    type: Literal['auto', 'any', 'tool']
    name: Optional[str] = None

    @model_validator(mode='after')
    def _validate_tool_name(self):
        if self.type == 'tool' and not self.name:
            raise ValueError("'name' is required when type is 'tool'")
        return self


# ---------------------------------------------------------------------------
# Request and response models (Task 1.2)
# ---------------------------------------------------------------------------


class MessagesRequest(BaseModel):
    """Anthropic Messages API request body."""
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[ContentBlock]]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[MessageMetadata] = None
    tools: Optional[List[AnthropicToolDef]] = None
    tool_choice: Optional[AnthropicToolChoice] = None

    @field_validator('model')
    @classmethod
    def _validate_model(cls, v):
        if not v:
            raise ValueError('model is required')
        return v

    @field_validator('max_tokens')
    @classmethod
    def _validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError('max_tokens must be positive')
        return v


class AnthropicUsage(BaseModel):
    """Token usage information."""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class MessagesResponse(BaseModel):
    """Anthropic Messages API non-streaming response."""
    id: str  # prefixed with "msg_"
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    model: str
    content: List[Union[TextBlock, ToolUseBlock, ThinkingBlock]]
    stop_reason: Optional[
        Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use']
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


class AnthropicErrorDetail(BaseModel):
    """Inner error detail."""
    type: Literal[
        'invalid_request_error',
        'not_found_error',
        'api_error',
        'overloaded_error',
    ]
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic-format error envelope."""
    type: Literal['error'] = 'error'
    error: AnthropicErrorDetail


# ---------------------------------------------------------------------------
# Count tokens models
# ---------------------------------------------------------------------------


class CountTokensRequest(BaseModel):
    """Anthropic count_tokens request body."""
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[ContentBlock]]] = None
    tools: Optional[List[AnthropicToolDef]] = None


class AnthropicContextManagement(BaseModel):
    """Context management information for token counting."""
    original_input_tokens: int


class CountTokensResponse(BaseModel):
    """Anthropic count_tokens response body."""
    input_tokens: int
    context_management: Optional[AnthropicContextManagement] = None


# ---------------------------------------------------------------------------
# SSE streaming event models (Task 1.3)
# ---------------------------------------------------------------------------


class TextDelta(BaseModel):
    """Incremental text delta."""
    type: Literal['text_delta'] = 'text_delta'
    text: str


class InputJsonDelta(BaseModel):
    """Incremental JSON delta for tool-use input."""
    type: Literal['input_json_delta'] = 'input_json_delta'
    partial_json: str


class ThinkingDelta(BaseModel):
    """Incremental thinking/reasoning delta."""
    type: Literal['thinking_delta'] = 'thinking_delta'
    thinking: str


class SignatureDelta(BaseModel):
    """Signature delta emitted when closing a thinking block."""
    type: Literal['signature_delta'] = 'signature_delta'
    signature: str


class MessageStartEvent(BaseModel):
    """First SSE event — contains the message shell."""
    type: Literal['message_start'] = 'message_start'
    message: MessagesResponse


class ContentBlockStartEvent(BaseModel):
    """Signals the start of a new content block."""
    type: Literal['content_block_start'] = 'content_block_start'
    index: int
    content_block: Union[TextBlock, ToolUseBlock, ThinkingBlock]


class ContentBlockDeltaEvent(BaseModel):
    """Carries an incremental delta for the current content block."""
    type: Literal['content_block_delta'] = 'content_block_delta'
    index: int
    delta: Union[TextDelta, InputJsonDelta, ThinkingDelta, SignatureDelta]


class ContentBlockStopEvent(BaseModel):
    """Signals the end of the current content block."""
    type: Literal['content_block_stop'] = 'content_block_stop'
    index: int


class MessageDelta(BaseModel):
    """Final message-level delta (stop reason)."""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class MessageDeltaEvent(BaseModel):
    """Carries the final stop reason and usage."""
    type: Literal['message_delta'] = 'message_delta'
    delta: MessageDelta
    usage: AnthropicUsage


class MessageStopEvent(BaseModel):
    """Terminal SSE event."""
    type: Literal['message_stop'] = 'message_stop'


class PingEvent(BaseModel):
    """Keep-alive ping event."""
    type: Literal['ping'] = 'ping'


class ErrorEvent(BaseModel):
    """Streaming error event."""
    type: Literal['error'] = 'error'
    error: AnthropicErrorDetail
