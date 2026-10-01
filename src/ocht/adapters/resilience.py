"""Retry-with-backoff and circuit-breaker resilience layer wrapping adapter client calls.

Wraps the raw LangChain chat-model calls (`ChatOllama`/`ChatOpenAI`/`ChatAnthropic` via
`.ainvoke()`/`.astream()`) so a transient network hiccup is retried with exponential backoff
before it ever becomes an exception the TUI has to handle, and so a provider that keeps failing
stops being hammered (circuit breaker) rather than retried forever. This is deliberately separate from
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

import anthropic
import httpx
import ollama
import openai

_HTTP_SERVER_ERROR_THRESHOLD = 500
"""Lowest HTTP status code conventionally considered a server-side (5xx) error.

Server errors are treated as transient/retryable (the provider's fault, likely to clear on retry);
client errors (4xx) below this threshold are not, except where an SDK's own status-code semantics
say otherwise - see `SdkExceptionProfile.status_code_retryable`'s Ollama override below, which
also treats status_code == -1 (no status code at all) as retryable.
"""


def _is_server_error_status(status_code: int) -> bool:
    """Default `SdkExceptionProfile.status_code_retryable`: retryable only for a 5xx status."""
    return status_code >= _HTTP_SERVER_ERROR_THRESHOLD


def _is_ollama_retryable_status(status_code: int) -> bool:
    """Ollama's `status_code_retryable` override: a 5xx status, or -1 (no status code at all).

    `ollama.ResponseError.status_code` defaults to -1 when Ollama's own error response didn't
    carry one - treated as unknown-but-worth-retrying rather than assumed to be a permanent 4xx.
    """
    return status_code >= _HTTP_SERVER_ERROR_THRESHOLD or status_code == -1


@dataclass(frozen=True)
class SdkExceptionProfile:
    """Retry-classification rules for one LLM-provider SDK's exception hierarchy.

    `is_retryable()`/`compute_delay()` used to hardcode one near-identical four-branch
    if/isinstance chain per SDK (OpenAI, Anthropic, plus Ollama's own shape) - an Open/Closed
    violation flagged as tech debt once a third SDK (`anthropic`) duplicated the pattern: adding a
    fourth adapter's SDK meant editing both functions' bodies again. This dataclass captures one
    SDK's rules as data instead, so `_SDK_PROFILES` below is the only thing a new SDK needs to
    extend - `is_retryable()`/`compute_delay()` stay unchanged.

    Checked in the fixed order `rate_limit_types` -> `non_retryable_types` -> `status_error_type`
    -> `connection_types`, matching the order the original hand-written branches used (matters
    because, in both the `openai` and `anthropic` SDKs, `RateLimitError` and the various
    non-retryable 4xx errors are themselves subclasses of the generic status-error type, so they
    must be checked before it).

    Attributes:
        rate_limit_types: Exception types representing a 429 rate limit - always retryable.
        non_retryable_types: Exception types reflecting a permanent config/request problem
            (bad auth, unknown model, malformed request, ...) that will not succeed on retry -
            retrying would just waste time and, for a circuit breaker, falsely count a permanent
            misconfiguration as a transient outage.
        status_error_type: The SDK's generic status-error type exposing `.status_code` (e.g.
            `openai.APIStatusError`), used as a catch-all for status codes with no more specific
            named type above. None if the SDK has no such generic type.
        status_code_retryable: Given `status_error_type.status_code`, returns whether that status
            is worth retrying. Defaults to "5xx only". Ollama's `ResponseError.status_code`
            defaults to -1 when its own error response didn't carry one - that SDK supplies its
            own predicate treating -1 as unknown-but-worth-retrying rather than a permanent 4xx.
        connection_types: Exception types treated as always-retryable connection/timeout errors.
    """
    rate_limit_types: tuple[type[BaseException], ...] = ()
    non_retryable_types: tuple[type[BaseException], ...] = ()
    status_error_type: type[BaseException] | None = None
    status_code_retryable: Callable[[int], bool] = _is_server_error_status
    connection_types: tuple[type[BaseException], ...] = ()


_SDK_PROFILES: tuple[SdkExceptionProfile, ...] = (
    SdkExceptionProfile(
        rate_limit_types=(openai.RateLimitError,),
        non_retryable_types=(
            openai.AuthenticationError,
            openai.NotFoundError,
            openai.PermissionDeniedError,
            openai.BadRequestError,
            openai.ConflictError,
            openai.UnprocessableEntityError,
        ),
        status_error_type=openai.APIStatusError,
        connection_types=(openai.APIConnectionError, openai.APITimeoutError),
    ),
    SdkExceptionProfile(
        rate_limit_types=(anthropic.RateLimitError,),
        non_retryable_types=(
            anthropic.AuthenticationError,
            anthropic.NotFoundError,
            anthropic.PermissionDeniedError,
            anthropic.BadRequestError,
            anthropic.ConflictError,
            anthropic.UnprocessableEntityError,
        ),
        status_error_type=anthropic.APIStatusError,
        connection_types=(anthropic.APIConnectionError, anthropic.APITimeoutError),
    ),
    SdkExceptionProfile(
        status_error_type=ollama.ResponseError,
        status_code_retryable=_is_ollama_retryable_status,
    ),
)
"""One `SdkExceptionProfile` per LLM-provider SDK this project's adapters can raise.

Add a new adapter's SDK-specific exceptions here - not by editing `is_retryable()`/
`compute_delay()` - per the project's `CLAUDE.md`, "Adding New Providers" step 4.
"""


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
    Classifies every SDK listed in `_SDK_PROFILES` (currently `openai`, `anthropic`, `ollama`),
    plus a generic `httpx.TransportError` fallback that isn't SDK-specific.

    Args:
        exc: The exception raised by an adapter's underlying LangChain client call.

    Returns:
        True if the call should be retried, False if it should fail immediately.
    """
    for profile in _SDK_PROFILES:
        if isinstance(exc, profile.rate_limit_types):
            return True
        if isinstance(exc, profile.non_retryable_types):
            return False
        if profile.status_error_type is not None and isinstance(exc, profile.status_error_type):
            return profile.status_code_retryable(exc.status_code)
        if isinstance(exc, profile.connection_types):
            return True
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
    is_rate_limit = any(isinstance(exc, profile.rate_limit_types) for profile in _SDK_PROFILES)
    base = policy.rate_limit_base_delay_seconds if is_rate_limit else policy.base_delay_seconds
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
