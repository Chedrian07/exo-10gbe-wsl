"""Tests for Claude Messages API conversion functions and types."""

import pydantic
import pytest

from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    finish_reason_to_claude_stop_reason,
)
from exo.api.types.claude_api import (
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeTextBlock,
)
from exo.shared.types.common import ModelId


class TestFinishReasonToClaudeStopReason:
    """Tests for finish_reason to Claude stop_reason mapping."""

    def test_stop_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("stop") == "end_turn"

    def test_length_maps_to_max_tokens(self):
        assert finish_reason_to_claude_stop_reason("length") == "max_tokens"

    def test_tool_calls_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("tool_calls") == "tool_use"

    def test_function_call_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("function_call") == "tool_use"

    def test_content_filter_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("content_filter") == "end_turn"

    def test_none_returns_none(self):
        assert finish_reason_to_claude_stop_reason(None) is None


class TestClaudeRequestToInternal:
    """Tests for converting Claude Messages API requests to TextGenerationTaskParams."""

    async def test_basic_request_conversion(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.model == "claude-3-opus"
        assert params.max_output_tokens == 100
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"
        assert params.instructions is None

    async def test_request_with_system_string(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "You are a helpful assistant."
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"

    async def test_request_with_system_text_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system=[
                ClaudeTextBlock(text="You are helpful. "),
                ClaudeTextBlock(text="Be concise."),
            ],
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "You are helpful. Be concise."
        assert isinstance(params.input, list)
        assert len(params.input) == 1

    async def test_request_with_content_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeTextBlock(text="First part. "),
                        ClaudeTextBlock(text="Second part."),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].content == "First part. Second part."

    async def test_request_with_multi_turn_conversation(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
                ClaudeMessage(role="assistant", content="Hi there!"),
                ClaudeMessage(role="user", content="How are you?"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 3
        assert params.input[0].role == "user"
        assert params.input[1].role == "assistant"
        assert params.input[2].role == "user"

    async def test_request_with_optional_parameters(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello")],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP", "END"],
            stream=True,
        )
        params = await claude_request_to_text_generation(request)

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.stop == ["STOP", "END"]
        assert params.stream is True

    async def test_system_message_in_messages_array(self):
        # Claude Code places system blocks inside `messages` rather than the top-level
        # `system` field. They must validate and fold into instructions, not 422.
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "Be brief."},
                ],
            }
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "Be brief."
        # system is folded out of the conversational turns
        assert [m.role for m in params.input] == ["user"]
        assert params.chat_template_messages is not None
        assert params.chat_template_messages[0]["role"] == "system"

    async def test_system_in_array_merges_with_top_level_system(self):
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "system": "Top level.",
                "messages": [
                    {"role": "system", "content": "In array."},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "Top level.\n\nIn array."
        assert [m.role for m in params.input] == ["user"]


class TestClaudeMessagesRequestValidation:
    """Tests for Claude Messages API request validation."""

    def test_request_requires_model(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_max_tokens(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_messages(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "max_tokens": 100,
                }
            )
