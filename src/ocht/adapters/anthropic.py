"""LLM adapter for Anthropic's Claude models via LangChain."""
from collections.abc import AsyncIterator
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ocht.adapters.base import LLMAdapter
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig
from ocht.adapters.resilience import RetryPolicy


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic's Claude models via LangChain."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        default_params: dict[str, Any] | None = None,
        memory_config: MemoryConfig | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        """Initializes the Anthropic client and its conversation memory.

        Args:
            model: Name of the Claude model to use (e.g. 'claude-sonnet-5').
            api_key: Anthropic API key. Always passed through explicitly - `ChatAnthropic`'s own
                `ANTHROPIC_API_KEY` env-var auto-detection is never consulted, same as every other
                adapter in this project (`LLMProviderConfig.prov_api_key` is non-nullable, so a
                real key is always available).
            base_url: Optional custom API endpoint (e.g. for a proxy). Defaults to Anthropic's own
                endpoint if not provided.
            default_params: Optional default parameters passed to the `ChatAnthropic` client (e.g.
                temperature, max_tokens, top_p, top_k).
            memory_config: Optional configuration for the hybrid memory strategy.
            retry_policy: Optional retry/backoff configuration for transient call failures. See
                `LLMAdapter.__init__()`/`adapters/resilience.py`.

        Note:
            `max_retries` is fixed to `0` on the underlying `ChatAnthropic` client rather than
            left at its own default (`2`) - retries must be owned exclusively by this project's
            `adapters/resilience.py` (`RetryPolicy`/`CircuitBreaker`), not silently duplicated
            inside the `anthropic` SDK's own retry logic with a different, invisible backoff.
            Applied *after* `default_params` is merged in, so a stray `max_retries` key inside a
            model's `model_params` JSON (a free-form blob a user can edit via the Model Manager
            TUI) can never silently re-enable client-level retries.
        """
        super().__init__(retry_policy=retry_policy)
        client_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            **(default_params or {}),
            # Applied after the default_params spread so it always wins, even if a user-edited
            # Model.model_params JSON blob happens to contain its own "max_retries" key - retries
            # must stay owned exclusively by adapters/resilience.py, never re-enabled by accident.
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = ChatAnthropic(**client_kwargs)
        self.memory_strategy = HybridMemoryStrategy(
            config=memory_config or MemoryConfig(),
            llm=self.client
        )
        # Raw, in-memory conversation history for this session. HybridMemoryStrategy decides how
        # much of it to send verbatim vs. summarize on each call - see prepare_context().
        self._history: list[BaseMessage] = []

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        """Sends a prompt to the Claude model and returns the full response.

        Loads conversation history via the configured memory strategy, invokes the model, and stores the
        exchange back into memory.

        Args:
            prompt: The user prompt to send.
            **kwargs: Additional parameters forwarded to the underlying LangChain client call.

        Returns:
            The model's response text.

        Raises:
            CircuitBreakerOpenError: If this adapter's circuit breaker is currently open. See
                `LLMAdapter._resilient_ainvoke()`.
            Exception: The last exception raised by the underlying `ChatAnthropic.ainvoke()` call,
                if all retry attempts fail or the exception is classified as non-retryable by
                `adapters/resilience.py:is_retryable()`.

        Note:
            Anthropic's API can return content as a list of content blocks (e.g. with extended
            thinking or tool use enabled) rather than a plain string. For this project's current
            text-only chat usage, `response.content` behaves like `ChatOpenAI`'s and is a plain
            string - a known simplification, not yet exercised by any tool-calling/thinking flow.
        """
        messages = await self.memory_strategy.prepare_context(self._history, prompt)
        message_objects = self._convert_tuples_to_messages(messages)

        response = await self._resilient_ainvoke(lambda: self.client.ainvoke(message_objects, **kwargs))

        self._history.append(HumanMessage(content=prompt))
        self._history.append(AIMessage(content=response.content))

        return response.content

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Sends a prompt to the Claude model and streams the response incrementally.

        Loads conversation history via the configured memory strategy, streams the response chunk by chunk,
        and stores the full accumulated response back into memory once streaming completes.

        Args:
            prompt: The user prompt to send.
            **kwargs: Additional parameters forwarded to the underlying LangChain client call.

        Yields:
            Successive text chunks of the model's response.

        Raises:
            CircuitBreakerOpenError: If this adapter's circuit breaker is currently open, raised
                on the first iteration of the stream rather than when this method is called. See
                `LLMAdapter._resilient_astream()`.
            Exception: The triggering exception from the underlying `ChatAnthropic.astream()`
                call, if it occurs after streaming has already started, or if every
                pre-first-chunk retry attempt is exhausted/non-retryable. See
                `adapters/resilience.py:is_retryable()`.
        """
        messages = await self.memory_strategy.prepare_context(self._history, prompt)
        message_objects = self._convert_tuples_to_messages(messages)

        full_response = ""
        async for chunk in self._resilient_astream(lambda: self.client.astream(message_objects, **kwargs)):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        if full_response:
            self._history.append(HumanMessage(content=prompt))
            self._history.append(AIMessage(content=full_response))
