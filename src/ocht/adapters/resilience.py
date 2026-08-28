"""Retry-with-backoff and circuit-breaker resilience layer wrapping adapter client calls.

Wraps the raw LangChain chat-model calls (`ChatOllama`/`ChatOpenAI` via `.ainvoke()`/`.astream()`)
so a transient network hiccup is retried with exponential backoff before it ever becomes an
exception the TUI has to handle, and so a provider that keeps failing stops being hammered
(circuit breaker) rather than retried forever. This is deliberately separate from
`tui/app.py`'s existing stream-to-async fallback: that is a *degradation strategy* (switch which
API shape is used), this is a *resilience strategy* (retry the same shape) - the two compose
rather than compete, since a stream that fails after retries here still falls through to the
TUI's fallback exactly as it does today.
"""
import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

import httpx
import ollama
import openai

# Non-retryable OpenAI-compatible errors: config/request problems that will not succeed on retry
# (wrong API key, unknown model, malformed request, etc.) - retrying would just waste time and,
# for a circuit breaker, falsely count a permanent misconfiguration as a transient outage.
_NON_RETRYABLE_OPENAI_TYPES = (
    openai.AuthenticationError,
    openai.NotFoundError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.ConflictError,
    openai.UnprocessableEntityError,
)


@dataclass
class RetryPolicy:
    """Configuration for exponential-backoff retries.

    Attributes:
        max_attempts: Maximum number of attempts (including the first, non-retry attempt).
        base_delay_seconds: Delay before the first retry.
        max_delay_seconds: Upper bound the exponential delay is capped at.
        multiplier: Growth factor applied to the delay after each retry.
        jitter_seconds: Maximum random jitter added to each delay, to avoid retry storms when
            multiple calls fail at the same time.
        rate_limit_base_delay_seconds: Base delay used instead of `base_delay_seconds` when the
            failure was a rate-limit error (429) - these typically need longer to clear than a
            plain network hiccup.
    """
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    multiplier: float = 2.0
    jitter_seconds: float = 0.25
    rate_limit_base_delay_seconds: float = 5.0


class CircuitBreakerState(Enum):
    """Lifecycle states of a `CircuitBreaker`."""
    CLOSED = "closed"
    """Normal operation - calls are allowed through."""
    OPEN = "open"
    """Too many consecutive failures - calls are rejected without being attempted."""
    HALF_OPEN = "half_open"
    """Cooldown elapsed - the next call is allowed through as a trial to test recovery."""


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is rejected because its circuit breaker is open."""


@dataclass
class CircuitBreaker:
    """Tracks consecutive failures for one adapter instance and rejects calls while "open".

    Opens after `failure_threshold` consecutive failures, then rejects every call until
    `reset_timeout_seconds` has elapsed, at which point it moves to `HALF_OPEN` and allows exactly
    one trial call through - a success there closes the breaker again, a failure re-opens it.

    State lives on the adapter instance that owns this breaker, so it resets naturally whenever a
    new adapter is constructed (e.g. on every provider/model switch via
    `services/adapter_manager.py`'s `build_adapter()`) - this is a deliberate v1 limitation
    (no persistence across adapter instances/app restarts), not an oversight.

    Attributes:
        failure_threshold: Number of consecutive failures that opens the breaker.
        reset_timeout_seconds: How long to stay open before allowing a half-open trial call.
    """
    failure_threshold: int = 5
    reset_timeout_seconds: float = 30.0
    _state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    def before_call(self) -> None:
        """Checks whether a call may proceed, raising if the breaker is open.

        Raises:
            CircuitBreakerOpenError: If the breaker is open and the cooldown hasn't elapsed yet.
        """
        if self._state is not CircuitBreakerState.OPEN:
            return
        if self._opened_at is not None and (time.monotonic() - self._opened_at) >= self.reset_timeout_seconds:
            self._state = CircuitBreakerState.HALF_OPEN
            return
        raise CircuitBreakerOpenError(
            f"Circuit breaker open after {self._consecutive_failures} consecutive failures"
        )

    def record_success(self) -> None:
        """Resets the breaker to a fully closed, healthy state."""
        self._consecutive_failures = 0
        self._state = CircuitBreakerState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        """Records a failure, opening the breaker if the threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN
            self._opened_at = time.monotonic()


def is_retryable(exc: BaseException) -> bool:
    """Classifies whether an exception from an LLM client call is worth retrying.

    Transient/retryable: rate limits (429), connection/timeout errors, and 5xx server errors.
    Non-retryable: authentication/permission/not-found/bad-request errors - these reflect a
    configuration problem that will not resolve itself on retry. Unknown exception types are
    treated as non-retryable too, so a real bug isn't silently masked as "just try again".

    Args:
        exc: The exception raised by an adapter's underlying LangChain client call.

    Returns:
        True if the call should be retried, False if it should fail immediately.
    """
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, _NON_RETRYABLE_OPENAI_TYPES):
        return False
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, ollama.ResponseError):
        # status_code defaults to -1 when Ollama's own error response didn't carry one - treat
        # that as unknown-but-worth-retrying rather than assuming it's a permanent 4xx.
        return exc.status_code >= 500 or exc.status_code == -1
    if isinstance(exc, httpx.TransportError):
        return True
    return False


def compute_delay(attempt: int, policy: RetryPolicy, exc: BaseException) -> float:
    """Computes the exponential-backoff delay (with jitter) before the next retry.

    Args:
        attempt: The 1-indexed attempt number that just failed (1 = the first attempt failed).
        policy: The retry policy to compute the delay from.
        exc: The exception that triggered the retry, used to pick a longer base delay for
            rate-limit errors.

    Returns:
        The number of seconds to wait before the next attempt.
    """
    base = policy.rate_limit_base_delay_seconds if isinstance(exc, openai.RateLimitError) else policy.base_delay_seconds
    delay = min(base * (policy.multiplier ** (attempt - 1)), policy.max_delay_seconds)
    return delay + random.uniform(0, policy.jitter_seconds)


async def call_with_resilience[T](
    call: Callable[[], Awaitable[T]],
    *,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
) -> T:
    """Runs one atomic async call with circuit-breaker gating and exponential-backoff retries.

    Args:
        call: A zero-argument callable returning the awaitable to run (e.g.
            `lambda: self.client.ainvoke(messages)`) - a callable rather than a bare awaitable so
            it can be invoked fresh on each retry attempt.
        retry_policy: Retry/backoff configuration.
        circuit_breaker: Circuit breaker to gate and record outcomes on.

    Returns:
        The awaited result of `call()`.

    Raises:
        CircuitBreakerOpenError: If the breaker is currently open.
        Exception: The last exception raised by `call()`, if all attempts fail or the exception
            is classified as non-retryable by `is_retryable()`.
    """
    circuit_breaker.before_call()
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            result = await call()
        except Exception as exc:
            if not is_retryable(exc) or attempt == retry_policy.max_attempts:
                circuit_breaker.record_failure()
                raise
            await asyncio.sleep(compute_delay(attempt, retry_policy, exc))
            continue
        circuit_breaker.record_success()
        return result
    # Unreachable: the loop always either returns or raises on its last iteration.
    raise AssertionError("call_with_resilience exited its retry loop without returning or raising")


async def stream_with_resilience[T](
    make_stream: Callable[[], AsyncIterator[T]],
    *,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
) -> AsyncIterator[T]:
    """Runs an async stream with circuit-breaker gating and pre-first-chunk retry only.

    Retrying a partially-consumed stream transparently is not safe: once a chunk has already been
    yielded to the caller (and, in the TUI, already rendered into a chat bubble), restarting the
    whole stream would duplicate or reset visible content. So this only retries a failure that
    happens *before* the first chunk arrives; once streaming has started, a failure is recorded on
    the circuit breaker (it's still a real failure) but re-raised immediately, letting the caller's
    own fallback handle it - e.g. `tui/app.py`'s existing stream-to-async fallback.

    Args:
        make_stream: A zero-argument callable returning a fresh async iterator (e.g.
            `lambda: self.client.astream(messages)`) - called again on each retry attempt.
        retry_policy: Retry/backoff configuration.
        circuit_breaker: Circuit breaker to gate and record outcomes on.

    Yields:
        Items from the underlying stream.

    Raises:
        CircuitBreakerOpenError: If the breaker is currently open. Since this is an async
            generator function, calling it does not run any code by itself - the error is only
            raised once the caller starts consuming the result (e.g. the first iteration of an
            `async for` loop), not at the point `stream_with_resilience(...)` is invoked.
        Exception: The triggering exception, if it occurs after streaming has already started, or
            if every pre-first-chunk attempt is exhausted/non-retryable.
    """
    circuit_breaker.before_call()
    for attempt in range(1, retry_policy.max_attempts + 1):
        started = False
        try:
            async for item in make_stream():
                started = True
                yield item
        except Exception as exc:
            if started:
                circuit_breaker.record_failure()
                raise
            if not is_retryable(exc) or attempt == retry_policy.max_attempts:
                circuit_breaker.record_failure()
                raise
            await asyncio.sleep(compute_delay(attempt, retry_policy, exc))
            continue
        else:
            circuit_breaker.record_success()
            return
