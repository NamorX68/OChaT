"""Tests for OllamaAdapter's conversation history wiring.

Regression coverage for a bug found while migrating off LangChain's (now-removed)
ConversationSummaryMemory: the adapter used to feed HybridMemoryStrategy a single,
already-summarized message on every call (via an outer ConversationSummaryMemory whose
`load_memory_variables()` always collapses history to one SystemMessage), so
HybridMemoryStrategy's own message-retention/summarization logic never actually received more
than one message and was effectively dead code. The adapter now keeps a plain, growing list of
raw messages instead.
"""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ocht.adapters.ollama import OllamaAdapter


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
