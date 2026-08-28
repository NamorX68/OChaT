"""Tests for OpenAICompatibleAdapter's prompt sending and resilience wrapping.

Mirrors `tests/ollama_adapter_tests.py`'s fixture/patching style, but against
`ocht.adapters.openai_compatible.ChatOpenAI` - this adapter also backs LM Studio and OpenRouter
(see CLAUDE.md's "Provider Routing Preferences" section), all of which speak the same OpenAI
chat-completions client shape.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from ocht.adapters.openai_compatible import OpenAICompatibleAdapter
from ocht.adapters.resilience import RetryPolicy


@pytest.fixture
def mock_chat_openai():
    """Patch ChatOpenAI with a mock whose ainvoke/astream return canned responses."""
    with patch("ocht.adapters.openai_compatible.ChatOpenAI") as mock_cls:
        client = mock_cls.return_value
        client.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
        yield client


@pytest.mark.asyncio
async def test_send_prompt_async_returns_response_content(mock_chat_openai):
    """Should return the mocked AIMessage's content as the completion result."""
    adapter = OpenAICompatibleAdapter(model="mock-model")

    result = await adapter.send_prompt_async("hello")

    assert result == "mock response"


@pytest.mark.asyncio
async def test_send_prompt_stream_yields_concatenated_chunks(mock_chat_openai):
    """Should yield each chunk's content, in order, as the stream progresses."""

    async def gen():
        yield AIMessageChunk(content="Hello")
        yield AIMessageChunk(content=" world")

    mock_chat_openai.astream = MagicMock(return_value=gen())
    adapter = OpenAICompatibleAdapter(model="mock-model")

    chunks = [chunk async for chunk in adapter.send_prompt_stream("hello")]

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_send_prompt_async_recovers_from_transient_api_connection_error(mock_chat_openai):
    """Should retry and succeed after one transient `openai.APIConnectionError`."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    connection_error = openai.APIConnectionError(request=request)
    mock_chat_openai.ainvoke = AsyncMock(side_effect=[connection_error, AIMessage(content="recovered")])

    adapter = OpenAICompatibleAdapter(
        model="mock-model", retry_policy=RetryPolicy(base_delay_seconds=0.01, jitter_seconds=0)
    )

    result = await adapter.send_prompt_async("hello")

    assert result == "recovered"
    assert mock_chat_openai.ainvoke.call_count == 2
