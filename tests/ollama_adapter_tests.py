"""Tests for OllamaAdapter's conversation history wiring.

Regression coverage for a bug found while migrating off LangChain's (now-removed)
ConversationSummaryMemory: the adapter used to feed HybridMemoryStrategy a single,
already-summarized message on every call (via an outer ConversationSummaryMemory whose
`load_memory_variables()` always collapses history to one SystemMessage), so
HybridMemoryStrategy's own message-retention/summarization logic never actually received more
than one message and was effectively dead code. The adapter now keeps a plain, growing list of
raw messages instead.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from ocht.adapters.ollama import OllamaAdapter
from ocht.adapters.resilience import CircuitBreaker, CircuitBreakerOpenError, RetryPolicy


@pytest.fixture
def mock_chat_ollama():
    """Patch ChatOllama with a mock whose ainvoke/astream return canned responses."""
    with patch("ocht.adapters.ollama.ChatOllama") as mock_cls:
        client = mock_cls.return_value
        client.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
        yield client


@pytest.mark.asyncio
async def test_history_grows_with_each_turn(mock_chat_ollama):
    """Test that the adapter's raw history accumulates real messages across turns."""
    adapter = OllamaAdapter(model="mock-model")
    assert adapter._history == []

    await adapter.send_prompt_async("first prompt")
    assert len(adapter._history) == 2
    assert adapter._history[0] == HumanMessage(content="first prompt")
    assert adapter._history[1] == AIMessage(content="mock response")

    await adapter.send_prompt_async("second prompt")
    assert len(adapter._history) == 4


@pytest.mark.asyncio
async def test_growing_history_is_passed_to_memory_strategy(mock_chat_ollama):
    """Test that prepare_context() receives the adapter's actual accumulated history.

    This is the specific wiring the original bug broke: prepare_context() must see the real,
    growing message list - not a single pre-summarized placeholder message - so its own
    retention/summarization logic can ever engage.
    """
    adapter = OllamaAdapter(model="mock-model")
    # `_history` is mutated in place after each call, so a mock recording the raw args object
    # would see the final, fully-grown list on every inspection - snapshot the length instead.
    captured_lengths = []
    original_prepare_context = adapter.memory_strategy.prepare_context

    async def spy_prepare_context(history, prompt):
        captured_lengths.append(len(history))
        return await original_prepare_context(history, prompt)

    adapter.memory_strategy.prepare_context = spy_prepare_context

    await adapter.send_prompt_async("first prompt")
    await adapter.send_prompt_async("second prompt")

    # First call sees empty history; second call must see the 2 messages from the first turn.
    assert captured_lengths == [0, 2]


@pytest.mark.asyncio
async def test_send_prompt_stream_recovers_from_pre_first_chunk_transient_failure(mock_chat_ollama):
    """Should retry a pre-first-chunk astream failure transparently via the resilience wrapper.

    The first `astream()` call raises before yielding anything; the second (retried) call
    succeeds and yields a normal chunk sequence - `send_prompt_stream()` should still yield the
    full response as if nothing had failed.
    """
    call_count = 0

    def astream_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("transient network error")

        async def gen():
            yield AIMessageChunk(content="Hello")
            yield AIMessageChunk(content=" world")

        return gen()

    # A plain function side_effect (rather than a list) is used deliberately: MagicMock's
    # list-based side_effect returns list members verbatim without calling them, so a list of
    # functions would hand back function objects instead of invoking them.
    mock_chat_ollama.astream = MagicMock(side_effect=astream_side_effect)
    adapter = OllamaAdapter(model="mock-model", retry_policy=RetryPolicy(base_delay_seconds=0.01, jitter_seconds=0))

    chunks = [chunk async for chunk in adapter.send_prompt_stream("prompt")]

    assert chunks == ["Hello", " world"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_send_prompt_async_propagates_non_retryable_exception_on_first_attempt(mock_chat_ollama):
    """Should propagate a non-retryable exception unmodified, without ever retrying it."""
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(401, request=request)
    auth_error = openai.AuthenticationError("invalid api key", response=response, body=None)
    mock_chat_ollama.ainvoke = AsyncMock(side_effect=auth_error)

    adapter = OllamaAdapter(model="mock-model")

    with pytest.raises(openai.AuthenticationError):
        await adapter.send_prompt_async("prompt")

    assert mock_chat_ollama.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_consecutive_send_prompt_async_failures(mock_chat_ollama):
    """Should open the breaker after enough consecutive failures and then reject without calling."""
    mock_chat_ollama.ainvoke = AsyncMock(side_effect=httpx.ConnectError("boom"))
    adapter = OllamaAdapter(
        model="mock-model",
        # max_attempts=1 so each send_prompt_async() call is exactly one failed attempt against
        # the breaker, rather than the retry loop absorbing several failures per call.
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0.01, jitter_seconds=0),
    )
    adapter._circuit_breaker = CircuitBreaker(failure_threshold=2)

    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await adapter.send_prompt_async("prompt")

    assert mock_chat_ollama.ainvoke.call_count == 2

    with pytest.raises(CircuitBreakerOpenError):
        await adapter.send_prompt_async("prompt")

    # The breaker rejected the call before it ever reached the client.
    assert mock_chat_ollama.ainvoke.call_count == 2
