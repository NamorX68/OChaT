"""LLM adapter for local Ollama models accessed through LangChain."""
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from ocht.adapters.base import LLMAdapter
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig
from ocht.adapters.resilience import RetryPolicy


class OllamaAdapter(LLMAdapter):
    """Adapter für lokale Ollama-Modelle über LangChain."""

    def __init__(
        self,
        model: str = "qwen3:30b-a3b",
        base_url: str = "http://localhost:11434",
        default_params: dict[str, Any] | None = None,
        memory_config: MemoryConfig | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        """Initializes the Ollama client and its conversation memory.

        Args:
            model: Name of the Ollama model to use.
            base_url: URL of the Ollama server.
            default_params: Optional default parameters passed to the `ChatOllama` client (e.g. temperature).
            memory_config: Optional configuration for the hybrid memory strategy.
            retry_policy: Optional retry/backoff configuration for transient call failures. See
                `LLMAdapter.__init__()`/`adapters/resilience.py`.
        """
        super().__init__(retry_policy=retry_policy)
        self.client = ChatOllama(
            model=model,
            base_url=base_url,
            **(default_params or {})
        )
        self.memory_strategy = HybridMemoryStrategy(
            config=memory_config or MemoryConfig(),
            llm=self.client
        )
        # Raw, in-memory conversation history for this session. HybridMemoryStrategy decides how
        # much of it to send verbatim vs. summarize on each call - see prepare_context().
        self._history: list[BaseMessage] = []

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        """Sends a prompt to the Ollama model and returns the full response.

        Loads conversation history via the configured memory strategy, invokes the model, and stores the
        exchange back into memory.

        Args:
            prompt: The user prompt to send.
            **kwargs: Additional parameters forwarded to the underlying LangChain client call.

        Returns:
            The model's response text.
        """
        # Geschichte konvertieren
        messages = await self.memory_strategy.prepare_context(self._history, prompt)

        # Convert tuples to message objects for LangChain
        message_objects = self._convert_tuples_to_messages(messages)

        # LLM asynchron aufrufen (mit Retry/Circuit-Breaker gegen transiente Fehler)
        response = await self._resilient_ainvoke(lambda: self.client.ainvoke(message_objects, **kwargs))

        # Kontext speichern
        self._history.append(HumanMessage(content=prompt))
        self._history.append(AIMessage(content=response.content))

        return response.content

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Sends a prompt to the Ollama model and streams the response incrementally.

        Loads conversation history via the configured memory strategy, streams the response chunk by chunk,
        and stores the full accumulated response back into memory once streaming completes.

        Args:
            prompt: The user prompt to send.
            **kwargs: Additional parameters forwarded to the underlying LangChain client call.

        Yields:
            Successive text chunks of the model's response.
        """
        # Geschichte konvertieren
        messages = await self.memory_strategy.prepare_context(self._history, prompt)

        # Convert tuples to message objects for LangChain
        message_objects = self._convert_tuples_to_messages(messages)

        # Streaming response (mit Retry/Circuit-Breaker vor dem ersten Chunk, siehe resilience.py)
        full_response = ""
        async for chunk in self._resilient_astream(lambda: self.client.astream(message_objects, **kwargs)):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        # Nach dem Streaming den vollständigen Text speichern
        if full_response:
            self._history.append(HumanMessage(content=prompt))
            self._history.append(AIMessage(content=full_response))

    # _convert_tuples_to_messages() lives on LLMAdapter (base.py) - shared with every adapter.