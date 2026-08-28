"""Tests for AnthropicAdapter's prompt sending and resilience wrapping.

Mirrors `tests/openai_compatible_adapter_tests.py`'s fixture/patching style, but against
`ocht.adapters.anthropic.ChatAnthropic`.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from ocht.adapters.anthropic import AnthropicAdapter
from ocht.adapters.resilience import RetryPolicy


@pytest.fixture
def mock_chat_anthropic():
    """Patch ChatAnthropic with a mock whose ainvoke/astream return canned responses."""
    with patch("ocht.adapters.anthropic.ChatAnthropic") as mock_cls:
        client = mock_cls.return_value
        client.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
        yield client


@pytest.mark.asyncio
async def test_send_prompt_async_returns_response_content(mock_chat_anthropic):
    """Should return the mocked AIMessage's content as the completion result."""
    adapter = AnthropicAdapter(model="claude-sonnet-5", api_key="sk-ant-test")

    result = await adapter.send_prompt_async("hello")

    assert result == "mock response"


@pytest.mark.asyncio
async def test_send_prompt_stream_yields_concatenated_chunks(mock_chat_anthropic):
    """Should yield each chunk's content, in order, as the stream progresses."""

    async def gen():
        yield AIMessageChunk(content="Hello")
        yield AIMessageChunk(content=" world")

    mock_chat_anthropic.astream = MagicMock(return_value=gen())
    adapter = AnthropicAdapter(model="claude-sonnet-5", api_key="sk-ant-test")

    chunks = [chunk async for chunk in adapter.send_prompt_stream("hello")]

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_send_prompt_async_recovers_from_transient_api_connection_error(mock_chat_anthropic):
    """Should retry and succeed after one transient `anthropic.APIConnectionError`."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    connection_error = anthropic.APIConnectionError(request=request)
    mock_chat_anthropic.ainvoke = AsyncMock(side_effect=[connection_error, AIMessage(content="recovered")])

    adapter = AnthropicAdapter(
        model="claude-sonnet-5",
        api_key="sk-ant-test",
        retry_policy=RetryPolicy(base_delay_seconds=0.01, jitter_seconds=0),
    )

    result = await adapter.send_prompt_async("hello")

    assert result == "recovered"
    assert mock_chat_anthropic.ainvoke.call_count == 2


def test_init_sets_max_retries_to_zero_on_underlying_client():
    """Should pass `max_retries=0` to `ChatAnthropic` so retries are owned exclusively by resilience.py."""
    with patch("ocht.adapters.anthropic.ChatAnthropic") as mock_cls:
        AnthropicAdapter(model="claude-sonnet-5", api_key="sk-ant-test")

        assert mock_cls.call_args.kwargs["max_retries"] == 0


def test_default_params_cannot_override_max_retries():
    """Regression test: a stray max_retries key in default_params must not win over max_retries=0.

    A user-edited Model.model_params JSON blob could contain its own "max_retries" key -
    max_retries=0 has to win regardless, since retries are meant to be owned exclusively by
    adapters/resilience.py. Covers a bug where default_params was spread *after* max_retries=0 in
    the constructor's dict literal, letting it silently override the value.
    """
    with patch("ocht.adapters.anthropic.ChatAnthropic") as mock_cls:
        AnthropicAdapter(model="claude-sonnet-5", api_key="sk-ant-test", default_params={"max_retries": 5})

        assert mock_cls.call_args.kwargs["max_retries"] == 0
