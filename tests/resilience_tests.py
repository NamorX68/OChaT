"""Tests for the retry-with-backoff and circuit-breaker resilience layer.

Covers `is_retryable()`'s exception classification, `CircuitBreaker`'s open/half-open/closed
lifecycle, and the two async wrapper functions `call_with_resilience()`/`stream_with_resilience()`
that adapters route their raw LangChain client calls through (see `adapters/resilience.py`'s
module docstring for the full rationale).
"""
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import ollama
import openai
import pytest

from ocht.adapters.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    RetryPolicy,
    call_with_resilience,
    compute_delay,
    is_retryable,
    stream_with_resilience,
)

_REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def _api_status_error(
    status_code: int,
    cls: type[openai.APIStatusError] | type[anthropic.APIStatusError] = openai.APIStatusError,
) -> openai.APIStatusError | anthropic.APIStatusError:
    """Builds a real `openai`/`anthropic` status-error instance at a given HTTP status code.

    Args:
        status_code: The HTTP status code the response should carry.
        cls: Which `openai`/`anthropic` exception class to instantiate - defaults to the generic
            `openai.APIStatusError` for status codes that don't have one of the more specific
            named subclasses (`RateLimitError`, `NotFoundError`, etc.) exercised elsewhere.

    Returns:
        A constructed exception instance, with a real `httpx.Response` backing `.status_code`.
    """
    response = httpx.Response(status_code, request=_REQUEST)
    return cls("error", response=response, body=None)


class TestIsRetryable:
    """Tests for `is_retryable()`'s exception classification."""

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (_api_status_error(429, openai.RateLimitError), True),
            (_api_status_error(401, openai.AuthenticationError), False),
            (_api_status_error(404, openai.NotFoundError), False),
            (_api_status_error(400, openai.BadRequestError), False),
            (_api_status_error(500), True),  # generic APIStatusError, not a named 4xx subclass
            (_api_status_error(404), False),  # generic 4xx via APIStatusError directly
            (openai.APIConnectionError(request=_REQUEST), True),
            (openai.APITimeoutError(request=_REQUEST), True),
            (ollama.ResponseError(error="boom", status_code=503), True),
            (ollama.ResponseError(error="boom", status_code=404), False),
            (ollama.ResponseError(error="boom"), True),  # status_code defaults to -1
            (_api_status_error(429, anthropic.RateLimitError), True),
            (_api_status_error(401, anthropic.AuthenticationError), False),
            (_api_status_error(404, anthropic.NotFoundError), False),
            (_api_status_error(400, anthropic.BadRequestError), False),
            (_api_status_error(500, anthropic.APIStatusError), True),  # generic 5xx APIStatusError
            (anthropic.APIConnectionError(request=_REQUEST), True),
            (anthropic.APITimeoutError(request=_REQUEST), True),
            (httpx.ConnectError("boom"), True),
            (ValueError("boom"), False),
            # `AuthenticationError`/`NotFoundError`/etc. are themselves subclasses of the generic
            # `APIStatusError` (verified via their `__mro__`), so `SdkExceptionProfile`'s documented
            # check order (`non_retryable_types` before `status_error_type`) is load-bearing: a 5xx
            # status code paired with a non-retryable error type is the only way to prove
            # `non_retryable_types` wins, since every 4xx case above would report the same `False`
            # even if `status_error_type`'s default "5xx only" rule were checked first instead.
            (_api_status_error(500, openai.AuthenticationError), False),
            (_api_status_error(500, anthropic.AuthenticationError), False),
        ],
    )
    def test_classifies_exceptions(self, exc: BaseException, expected: bool) -> None:
        """Should classify each exception type/status boundary as retryable or not."""
        assert is_retryable(exc) is expected


class TestComputeDelay:
    """Tests for `compute_delay()`'s rate-limit-aware base delay selection."""

    def test_anthropic_rate_limit_error_uses_rate_limit_base_delay(self) -> None:
        """Should use `rate_limit_base_delay_seconds` (not `base_delay_seconds`) for a RateLimitError."""
        policy = RetryPolicy(
            base_delay_seconds=0.01, rate_limit_base_delay_seconds=5.0, multiplier=1.0, jitter_seconds=0.0
        )
        rate_limit_error = _api_status_error(429, anthropic.RateLimitError)

        delay = compute_delay(attempt=1, policy=policy, exc=rate_limit_error)

        assert delay == 5.0


class TestCircuitBreaker:
    """Tests for `CircuitBreaker`'s open/half-open/closed state transitions."""

    def test_before_call_raises_once_failure_threshold_is_reached(self) -> None:
        """Should raise CircuitBreakerOpenError after `failure_threshold` consecutive failures."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30.0)
        for _ in range(3):
            breaker.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            breaker.before_call()

    def test_before_call_does_not_raise_below_failure_threshold(self) -> None:
        """Should not raise while consecutive failures remain below the threshold."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()

        breaker.before_call()  # must not raise

    def test_record_success_resets_breaker(self) -> None:
        """Should reset consecutive failures and close the breaker on a success."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()

        breaker.record_success()

        assert breaker._consecutive_failures == 0
        assert breaker._state == CircuitBreakerState.CLOSED
        breaker.before_call()  # must not raise

    def test_before_call_allows_half_open_trial_after_reset_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should transition to HALF_OPEN and allow a call through once the cooldown elapses."""
        current_time = 1000.0
        monkeypatch.setattr("ocht.adapters.resilience.time.monotonic", lambda: current_time)

        breaker = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=30.0)
        breaker.record_failure()
        breaker.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            breaker.before_call()

        current_time += 30.0  # before_call() uses >=, so exactly the timeout already qualifies

        breaker.before_call()  # should not raise now
        assert breaker._state == CircuitBreakerState.HALF_OPEN

    def test_before_call_still_raises_before_reset_timeout_elapses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should keep raising while the cooldown hasn't fully elapsed yet."""
        current_time = 1000.0
        monkeypatch.setattr("ocht.adapters.resilience.time.monotonic", lambda: current_time)

        breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=30.0)
        breaker.record_failure()

        current_time += 29.999

        with pytest.raises(CircuitBreakerOpenError):
            breaker.before_call()


class TestCallWithResilience:
    """Tests for `call_with_resilience()`'s retry loop and circuit-breaker gating."""

    @pytest.mark.asyncio
    async def test_returns_result_after_recovering_from_transient_failures(self) -> None:
        """Should retry retryable failures and return the eventual successful result."""
        call = AsyncMock(side_effect=[httpx.ConnectError("x"), httpx.ConnectError("x"), "ok"])
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, jitter_seconds=0)
        breaker = CircuitBreaker()

        result = await call_with_resilience(call, retry_policy=policy, circuit_breaker=breaker)

        assert result == "ok"
        assert call.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retryable_exception(self) -> None:
        """Should fail on the very first attempt for a non-retryable exception, without sleeping."""
        auth_error = _api_status_error(401, openai.AuthenticationError)
        call = AsyncMock(side_effect=auth_error)
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, jitter_seconds=0)
        breaker = CircuitBreaker()

        with patch("ocht.adapters.resilience.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(openai.AuthenticationError):
                await call_with_resilience(call, retry_policy=policy, circuit_breaker=breaker)

        assert call.call_count == 1
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reraises_last_exception_once_retries_are_exhausted(self) -> None:
        """Should re-raise the last exception after exhausting all attempts on a persistent failure."""
        call = AsyncMock(side_effect=httpx.ConnectError("persistent"))
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, jitter_seconds=0)
        breaker = CircuitBreaker()

        with pytest.raises(httpx.ConnectError):
            await call_with_resilience(call, retry_policy=policy, circuit_breaker=breaker)

        assert call.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_circuit_breaker_open_error_without_calling_when_breaker_is_open(self) -> None:
        """Should reject the call outright (never invoking it) when the breaker is already open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()
        call = AsyncMock(return_value="should never be reached")
        policy = RetryPolicy(max_attempts=1, base_delay_seconds=0.01, jitter_seconds=0)

        with pytest.raises(CircuitBreakerOpenError):
            await call_with_resilience(call, retry_policy=policy, circuit_breaker=breaker)

        call.assert_not_awaited()


class TestStreamWithResilience:
    """Tests for `stream_with_resilience()`'s pre-first-chunk-only retry behavior."""

    @pytest.mark.asyncio
    async def test_retries_a_pre_first_chunk_failure_and_yields_all_items(self) -> None:
        """Should retry a failure that happens before any chunk is yielded, then yield everything."""
        call_count = 0

        def make_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async def failing_gen():
                    raise httpx.ConnectError("boom before first chunk")
                    yield  # pragma: no cover - unreachable, keeps this an async generator function

                return failing_gen()

            async def gen():
                for item in ("a", "b", "c"):
                    yield item

            return gen()

        policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.01, jitter_seconds=0)
        breaker = CircuitBreaker()

        stream = stream_with_resilience(make_stream, retry_policy=policy, circuit_breaker=breaker)
        items = [item async for item in stream]

        assert items == ["a", "b", "c"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_propagates_post_first_chunk_failure_immediately_without_retry(self) -> None:
        """Should re-raise a failure that happens after streaming already started, with no retry."""
        call_count = 0

        def make_stream():
            nonlocal call_count
            call_count += 1

            async def gen():
                yield "first"
                raise httpx.ConnectError("boom after first chunk")

            return gen()

        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, jitter_seconds=0)
        breaker = CircuitBreaker()

        items = []
        with pytest.raises(httpx.ConnectError):
            async for item in stream_with_resilience(make_stream, retry_policy=policy, circuit_breaker=breaker):
                items.append(item)

        assert items == ["first"]
        assert call_count == 1  # no retry attempted once streaming had started
        assert breaker._consecutive_failures == 1
