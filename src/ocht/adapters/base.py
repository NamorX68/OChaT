"""Abstract base interface that all LLM adapters must implement."""
import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ocht.adapters.resilience import CircuitBreaker, RetryPolicy, call_with_resilience, stream_with_resilience


class LLMAdapter(ABC):
    """Einheitliches Interface für alle LLM-Adapter."""

    def __init__(self, retry_policy: RetryPolicy | None = None):
        """Initializes the shared retry/circuit-breaker resilience state for this adapter.

        Concrete adapters should call `super().__init__(retry_policy=retry_policy)` and route
        their actual LangChain client calls through `_resilient_ainvoke()`/`_resilient_astream()`
        instead of calling `self.client.ainvoke()`/`self.client.astream()` directly - see
        `adapters/resilience.py` for why this exists and how the streaming case differs.

        Args:
            retry_policy: Retry/backoff configuration. Defaults to a new `RetryPolicy()` if not
                provided.
        """
        self._retry_policy = retry_policy or RetryPolicy()
        self._circuit_breaker = CircuitBreaker()

    async def _resilient_ainvoke[T](self, call: Callable[[], Awaitable[T]]) -> T:
        """Runs one atomic client call through this adapter's retry policy and circuit breaker.

        Args:
            call: A zero-argument callable returning the awaitable to run (e.g.
                `lambda: self.client.ainvoke(messages)`).

        Returns:
            The awaited result of `call()`.

        Raises:
            CircuitBreakerOpenError: If this adapter's circuit breaker is currently open.
            Exception: The last exception raised by `call()`, if all retry attempts fail or the
                exception is classified as non-retryable. See `resilience.is_retryable()`.
        """
        return await call_with_resilience(call, retry_policy=self._retry_policy, circuit_breaker=self._circuit_breaker)

    def _resilient_astream[T](self, make_stream: Callable[[], AsyncIterator[T]]) -> AsyncIterator[T]:
        """Runs a client stream through this adapter's retry policy and circuit breaker.

        Only retries a failure that happens before the first chunk arrives - see
        `adapters/resilience.py:stream_with_resilience()`'s docstring for why.

        Args:
            make_stream: A zero-argument callable returning a fresh async iterator (e.g.
                `lambda: self.client.astream(messages)`).

        Returns:
            An async iterator yielding the stream's items. Since this returns an async generator,
            no code runs (and no exception can be raised) until the caller starts consuming it,
            e.g. via `async for chunk in self._resilient_astream(...)`.

        Raises:
            CircuitBreakerOpenError: If this adapter's circuit breaker is open, raised on the
                first iteration step rather than on this call itself (see the Returns note above).
            Exception: The triggering exception, if it occurs after streaming has already started,
                or if every pre-first-chunk retry attempt is exhausted/non-retryable.
        """
        return stream_with_resilience(
            make_stream, retry_policy=self._retry_policy, circuit_breaker=self._circuit_breaker
        )

    @abstractmethod
    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        """Sendet einen Prompt asynchron an den LLM.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter (z.B. temperature, max_tokens).

        Returns:
            Die vom LLM generierte Antwort als String.
        """
        ...

    @abstractmethod
    def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Sendet einen Prompt an den LLM und gibt Streaming-Antwort zurück.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter.

        Yields:
            Text-Chunks der LLM Antwort.
        """
        ...

    def send_prompt(self, prompt: str, **kwargs) -> str:
        """Synchroner Wrapper für send_prompt_async.

        Erkennt automatisch ob Event Loop läuft.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter.

        Returns:
            Die vom LLM generierte Antwort als String.
        """
        try:
            # Prüfe ob Event Loop bereits läuft
            asyncio.get_running_loop()
            # Wenn ja, nutze run_in_executor für thread-based execution
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.send_prompt_async(prompt, **kwargs))
                )
                return future.result()
        except RuntimeError:
            # Kein Event Loop - normal mit asyncio.run()
            return asyncio.run(self.send_prompt_async(prompt, **kwargs))

    def _convert_message_to_tuple(self, msg: Any) -> tuple[str, str]:
        """Konvertiert LangChain-Message zu (role, content) Tupel.

        Standard-Implementierung für die meisten LangChain Message-Types.
        Kann in Subklassen überschrieben werden, falls Provider-spezifische
        Anpassungen nötig sind.
        """
        message_type_map = {
            'human': 'human',
            'ai': 'ai',
            'system': 'system',
            'user': 'human',      # Fallback für andere Provider
            'assistant': 'ai'     # Fallback für andere Provider
        }

        msg_type = getattr(msg, 'type', 'system')
        role = message_type_map.get(msg_type, 'system')

        return (role, msg.content)

    def _convert_tuples_to_messages(self, message_tuples: list[tuple[str, str]]) -> list[BaseMessage]:
        """Convert list of (role, content) tuples to LangChain message objects.

        Shared by every concrete adapter (previously duplicated verbatim in `ollama.py` and
        `openai_compatible.py` - moved here when `AnthropicAdapter` was added as a third adapter
        that would otherwise need its own copy).

        Args:
            message_tuples: A list of `(role, content)` pairs, typically produced by
                `HybridMemoryStrategy.prepare_context()`. `role` is matched case-insensitively;
                `human`/`user` map to `HumanMessage`, `ai`/`assistant` map to `AIMessage`, and
                `system` (or any other unrecognized role) maps to `SystemMessage`. `content` is
                used verbatim as the resulting message's `content`.

        Returns:
            The corresponding list of `HumanMessage`/`AIMessage`/`SystemMessage` objects, in the
            same order as `message_tuples`.

        Note:
            `AnthropicAdapter`'s underlying `ChatAnthropic` client raises a `ValueError` if the
            resulting list contains more than one *non-consecutive* `SystemMessage`. Today this is
            safe because `HybridMemoryStrategy.prepare_context()` never emits more than one
            `"system"`-role tuple per call (the "Previous conversation summary" line) and no
            adapter ever stores a `SystemMessage` in its own `self._history` - but this constraint
            is specific to Anthropic (`ChatOllama`/`ChatOpenAI` have no such restriction) and isn't
            enforced here. If `prepare_context()` is ever extended to emit a second system-tagged
            tuple, verify it against a real (unmocked) `AnthropicAdapter` call, not just a mocked
            `ChatAnthropic` client - the existing adapter tests don't exercise this validation.
        """
        messages = []
        for role, content in message_tuples:
            if role.lower() in ['human', 'user']:
                messages.append(HumanMessage(content=content))
            elif role.lower() in ['ai', 'assistant']:
                messages.append(AIMessage(content=content))
            elif role.lower() == 'system':
                messages.append(SystemMessage(content=content))
            else:
                # Default to system message for unknown roles
                messages.append(SystemMessage(content=content))
        return messages