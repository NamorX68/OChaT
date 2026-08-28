"""LLM adapter for OpenAI and OpenAI-compatible APIs (e.g. LM Studio, OpenRouter) via LangChain."""
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ocht.adapters.base import LLMAdapter
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig
from ocht.adapters.resilience import RetryPolicy


class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs (like LM Studio, OpenRouter) via LangChain."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_params: dict[str, Any] | None = None,
        memory_config: MemoryConfig | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        """Initialize OpenAI-compatible adapter.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', or local model name for LM Studio)
            api_key: API key for OpenAI (not needed for LM Studio)
            base_url: Custom base URL (e.g., 'http://localhost:1234/v1' for LM Studio)
            default_params: Default parameters like temperature, max_tokens
            memory_config: Configuration for hybrid memory system
            retry_policy: Optional retry/backoff configuration for transient call failures. See
                `LLMAdapter.__init__()`/`adapters/resilience.py`.
        """
        super().__init__(retry_policy=retry_policy)
        # Setup client parameters
        client_kwargs = {
            'model': model,
            **(default_params or {})
        }

        # Add API key if provided (required for OpenAI, not for LM Studio)
        if api_key:
            client_kwargs['api_key'] = api_key

        # Add custom base URL if provided (for LM Studio or other compatible APIs)
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = ChatOpenAI(**client_kwargs)
        self.model_name = model
        self.base_url = base_url
        self.memory_strategy = HybridMemoryStrategy(
            config=memory_config or MemoryConfig(),
            llm=self.client
        )
        # Raw, in-memory conversation history for this session. HybridMemoryStrategy decides how
        # much of it to send verbatim vs. summarize on each call - see prepare_context().
        self._history: list[BaseMessage] = []

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        """Send prompt asynchronously to OpenAI-compatible API."""
        messages = await self.memory_strategy.prepare_context(self._history, prompt)

        # Convert tuples to message objects for LangChain
        message_objects = self._convert_tuples_to_messages(messages)

        # Call LLM asynchronously (with retry/circuit-breaker against transient failures)
        response = await self._resilient_ainvoke(lambda: self.client.ainvoke(message_objects, **kwargs))

        # Save context
        self._history.append(HumanMessage(content=prompt))
        self._history.append(AIMessage(content=response.content))

        return response.content

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Send prompt to OpenAI-compatible API with streaming response."""
        messages = await self.memory_strategy.prepare_context(self._history, prompt)

        # Convert tuples to message objects for LangChain
        message_objects = self._convert_tuples_to_messages(messages)

        # Streaming response (with retry/circuit-breaker before the first chunk, see resilience.py)
        full_response = ""
        async for chunk in self._resilient_astream(lambda: self.client.astream(message_objects, **kwargs)):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        # Save complete text after streaming
        if full_response:
            self._history.append(HumanMessage(content=prompt))
            self._history.append(AIMessage(content=full_response))

    # _convert_tuples_to_messages() lives on LLMAdapter (base.py) - shared with every adapter.
    #
    # is_local_model()/get_provider_info() were removed as dead code (no callers anywhere in the
    # codebase) - their 'LM Studio'-vs-'OpenAI' binary was already wrong for OpenRouter, and would
    # have mislabeled MLX-LM as 'LM Studio' too, purely from checking 'localhost' in base_url.
    # `AdapterManager.get_adapter_info()` already gets the real provider name from the DB row
    # (`LLMProviderConfig.prov_name`), not by guessing from the adapter's base_url.