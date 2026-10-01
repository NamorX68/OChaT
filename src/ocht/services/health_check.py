"""Validates that a model can actually complete a real chat request (not just that it's listed).

`services/model_manager.py`'s `sync_llm_models()` only confirms a model is *listed* by its
provider's model-listing endpoint (e.g. Ollama's `/api/tags`) - it never sends an actual
completion request, so it can't catch a model that's listed but broken, and it never measures
response time or tokens/second. This module does a real completion call per model instead
(`HEALTH_CHECK_PROMPT`, long enough to give a meaningful tokens/second reading - see its own
docstring), via both the streaming and non-streaming code paths (validating streaming capability
in the process), and records the result back onto `Model`.

Deliberately bypasses `LLMAdapter.send_prompt_async()`/`send_prompt_stream()` (and therefore the
retry/circuit-breaker layer in `adapters/resilience.py`) - a health check's entire purpose is to
report the model's *current, true* status, and masking a failure behind automatic retries would
produce a falsely-rosy result. It talks to the `HealthProbeClient` returned by
`LLMAdapter.get_client()` directly instead (in practice always a raw LangChain chat-model client
today), since it needs the metadata (`response_metadata`/`usage_metadata`) the public send_prompt*
methods discard, and has no use for the conversation-history bookkeeping they do.
"""
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage

from ocht.adapters.base import AdapterClientUnavailableError, HealthProbeClient
from ocht.core.db import get_session
from ocht.repositories.llm_provider_config import get_all_llm_provider_configs, get_llm_provider_config_by_id
from ocht.repositories.model import get_model_by_name, get_models_by_provider, record_health_check_result
from ocht.services.adapter_manager import build_adapter

HEALTH_CHECK_PROMPT = "Write a short story in about 200 words."
"""Fixed prompt used for every health check.

Deliberately asks for a real chunk of generated text rather than a trivial one-token reply (e.g.
"Say OK.") - tokens/second measured from a 1-3 token response is dominated by connection/prompt-eval
overhead rather than actual sustained generation throughput, making it a noisy, not-very-meaningful
figure. ~200 words (roughly 250-300 tokens for most tokenizers) is long enough to give a stable
tokens/second reading while still being a single quick request.
"""


@dataclass
class HealthCheckResult:
    """Outcome of one model's health check.

    Attributes:
        model_name: Name of the model that was checked.
        is_available: Whether the check's completion call succeeded.
        path: Which code path produced this result - `"stream"`, `"async"` (the non-streaming
            fallback), or `"none"` (the model/provider couldn't even be resolved/built).
        latency_ms: Latency of the successful call, in milliseconds, or None if it failed.
        tokens_per_second: Output tokens/second measured for the call, or None if unavailable.
        error: Error message if the check failed, or None if it succeeded.
    """
    model_name: str
    is_available: bool
    path: str
    latency_ms: float | None
    tokens_per_second: float | None
    error: str | None


def _tokens_per_second(message: Any, elapsed_seconds: float) -> float | None:
    """Derives an output tokens/second figure from a LangChain message, provider-agnostically.

    Prefers Ollama's native `eval_count`/`eval_duration` (nanoseconds) from `response_metadata`
    when present - this is pure generation time, excluding prompt-eval and network overhead, so
    it's more precise than wall-clock timing. Every other provider (routed through `ChatOpenAI`)
    carries no timing metadata at all, so falls back to `usage_metadata`'s `output_tokens` divided
    by the caller-measured wall-clock `elapsed_seconds`.

    Args:
        message: The `AIMessage`/`AIMessageChunk` returned by the completion call.
        elapsed_seconds: Wall-clock time the caller measured around the call.

    Returns:
        Tokens/second, or None if neither source of token/timing data is available.
    """
    response_metadata = getattr(message, "response_metadata", None) or {}
    eval_count = response_metadata.get("eval_count")
    eval_duration_ns = response_metadata.get("eval_duration")
    if eval_count and eval_duration_ns:
        return eval_count / (eval_duration_ns / 1e9)

    usage_metadata = getattr(message, "usage_metadata", None) or {}
    output_tokens = usage_metadata.get("output_tokens")
    if output_tokens and elapsed_seconds > 0:
        return output_tokens / elapsed_seconds

    return None


async def _try_streaming(client: HealthProbeClient, model_name: str) -> HealthCheckResult:
    """Probes a model via its streaming API, gathering the final chunk's metadata.

    Args:
        client: The client returned by `LLMAdapter.get_client()` (a `HealthProbeClient`;
            currently always a LangChain chat model in practice).
        model_name: Name of the model being checked (carried through into the result).

    Returns:
        A `HealthCheckResult` with `path="stream"`.
    """
    start = time.monotonic()
    try:
        gathered = None
        async for chunk in client.astream([HumanMessage(content=HEALTH_CHECK_PROMPT)]):
            # LangChain message-chunk addition merges content and metadata across chunks, so the
            # final `gathered` carries the same response_metadata/usage_metadata a non-streaming
            # call would have returned in one shot.
            gathered = chunk if gathered is None else gathered + chunk
        elapsed = time.monotonic() - start
        if gathered is None:
            raise RuntimeError("Stream produced no chunks")
        return HealthCheckResult(
            model_name, True, "stream", elapsed * 1000, _tokens_per_second(gathered, elapsed), None
        )
    except Exception as exc:
        return HealthCheckResult(model_name, False, "stream", (time.monotonic() - start) * 1000, None, str(exc))


async def _try_non_streaming(client: HealthProbeClient, model_name: str) -> HealthCheckResult:
    """Probes a model via its non-streaming API - used as a fallback if streaming fails.

    Args:
        client: The client returned by `LLMAdapter.get_client()` (a `HealthProbeClient`;
            currently always a LangChain chat model in practice).
        model_name: Name of the model being checked (carried through into the result).

    Returns:
        A `HealthCheckResult` with `path="async"`.
    """
    start = time.monotonic()
    try:
        response = await client.ainvoke([HumanMessage(content=HEALTH_CHECK_PROMPT)])
        elapsed = time.monotonic() - start
        return HealthCheckResult(
            model_name, True, "async", elapsed * 1000, _tokens_per_second(response, elapsed), None
        )
    except Exception as exc:
        return HealthCheckResult(model_name, False, "async", (time.monotonic() - start) * 1000, None, str(exc))


async def check_model_health(provider_id: int, model_name: str) -> HealthCheckResult:
    """Runs (but does not persist) a health check for one model: stream first, async as fallback.

    Builds its own throwaway adapter via `build_adapter()` rather than touching
    `AdapterManager`'s singleton - checking model X must never disrupt whichever model the user is
    actively chatting with.

    Args:
        provider_id: ID of the model's provider configuration.
        model_name: Name of the model to check.

    Returns:
        A `HealthCheckResult`. If both the streaming and non-streaming attempts fail, the
        streaming attempt's error is reported (it's tried first and is what a real chat would hit).
    """
    def _resolve(db):
        provider_config = get_llm_provider_config_by_id(db, provider_id)
        model = get_model_by_name(db, model_name)
        if not provider_config or not model or model.model_provider_id != provider_id:
            return None
        return build_adapter(db, provider_config, model)

    with get_session() as db:
        adapter = _resolve(db)

    if adapter is None:
        return HealthCheckResult(
            model_name, False, "none", None, None,
            "Model or provider not found, model does not belong to that provider, or unsupported provider type"
        )

    try:
        client = adapter.get_client()
    except AdapterClientUnavailableError as exc:
        # A future adapter that doesn't wrap a bare LangChain client (e.g. a native MLX-LM adapter
        # with no LangChain client at all - see CLAUDE.md's Phase 3 roadmap) would otherwise fail
        # inside _try_streaming()/_try_non_streaming() with a generic AttributeError caught by
        # their broad `except Exception`, indistinguishable from a real model outage. Surfacing it
        # here instead makes a wiring gap look like a wiring gap, not a false "model is down" result.
        return HealthCheckResult(model_name, False, "none", None, None, str(exc))

    stream_result = await _try_streaming(client, model_name)
    if stream_result.is_available:
        return stream_result

    fallback_result = await _try_non_streaming(client, model_name)
    return fallback_result if fallback_result.is_available else stream_result


async def run_health_check(provider_id: int, model_name: str) -> HealthCheckResult:
    """Runs a health check for one model and persists the result onto its `Model` row.

    Args:
        provider_id: ID of the model's provider configuration.
        model_name: Name of the model to check.

    Returns:
        The `HealthCheckResult` (same as `check_model_health()` - persistence is a side effect).
    """
    result = await check_model_health(provider_id, model_name)

    def _persist(db):
        record_health_check_result(
            db, model_name,
            is_available=result.is_available,
            checked_at=datetime.now(),
            latency_ms=result.latency_ms,
            tokens_per_second=result.tokens_per_second,
            error=result.error,
        )

    with get_session() as db:
        _persist(db)

    return result


async def run_health_check_for_provider(provider_id: int) -> list[HealthCheckResult]:
    """Runs and persists a health check for every model belonging to one provider.

    Checks run sequentially rather than concurrently - local providers (Ollama/LM Studio)
    generally can't usefully serve concurrent completions on one GPU, and sequential execution
    avoids surprising burst-concurrency/cost against paid cloud APIs. This is a deliberate default,
    not a hard constraint - a concurrency-limited fan-out (e.g. an `asyncio.Semaphore`) would be a
    reasonable future refinement if checking many cloud models sequentially proves too slow.

    Args:
        provider_id: ID of the provider whose models should all be checked.

    Returns:
        One `HealthCheckResult` per model belonging to that provider.
    """
    with get_session() as db:
        model_names = [m.model_name for m in get_models_by_provider(db, provider_id)]

    return [await run_health_check(provider_id, name) for name in model_names]


async def run_health_check_all() -> list[HealthCheckResult]:
    """Runs and persists a health check for every model across every configured provider.

    Checks every provider identically, including paid cloud ones (OpenAI/OpenRouter) - each check
    is a single minimal request, so the cost is negligible, and treating every provider the same
    way is simpler to reason about than a hidden opt-in/opt-out rule.

    Returns:
        One `HealthCheckResult` per model across all providers.
    """
    with get_session() as db:
        provider_ids = [p.prov_id for p in get_all_llm_provider_configs(db) if p.prov_id is not None]

    results: list[HealthCheckResult] = []
    for provider_id in provider_ids:
        results.extend(await run_health_check_for_provider(provider_id))
    return results
