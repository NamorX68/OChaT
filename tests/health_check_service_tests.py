"""Tests for `services/health_check.py`'s completion-based model health checking.

`_try_streaming()`/`_try_non_streaming()` are pure enough to drive directly with a minimal fake
adapter (`SimpleNamespace(client=...)`) exposing just `.astream`/`.ainvoke` - no DB or
`build_adapter()` involved. `check_model_health()`/`run_health_check()` need a real provider+model
row to resolve (via the `temp_db` fixture, matching `tests/adapter_manager_tests.py`'s pattern),
with `ocht.services.health_check.build_adapter` patched to hand back the fake adapter instead of
building a real, network-calling one.
"""
import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from ocht.core.db import create_db_engine, get_session, init_db
from ocht.repositories.llm_provider_config import create_llm_provider_config
from ocht.repositories.model import create_model, get_model_by_name
from ocht.services.adapter_manager import AdapterManager
from ocht.services.health_check import (
    _try_non_streaming,
    _try_streaming,
    check_model_health,
    run_health_check,
)


@pytest.fixture
def temp_db(monkeypatch):
    """Point DATABASE_URL at a fresh, empty temp SQLite file for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "health_check.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        init_db(create_db_engine())
        yield db_path


def _make_fake_adapter(*, astream=None, ainvoke=None) -> SimpleNamespace:
    """Builds a minimal stand-in for an adapter, exposing only the `.client` surface health checks use."""
    return SimpleNamespace(client=SimpleNamespace(astream=astream, ainvoke=ainvoke))


class TestTryStreaming:
    """Tests for `_try_streaming()`'s probe and tokens/sec extraction."""

    @pytest.mark.asyncio
    async def test_computes_exact_tokens_per_second_from_ollama_metadata(self) -> None:
        """Should compute tokens/sec precisely from Ollama's eval_count/eval_duration metadata."""

        async def astream(*_args, **_kwargs):
            yield AIMessageChunk(
                content="OK", response_metadata={"eval_count": 50, "eval_duration": 2_000_000_000}
            )

        adapter = _make_fake_adapter(astream=astream)

        result = await _try_streaming(adapter, "model-a")

        assert result.is_available is True
        assert result.path == "stream"
        assert result.tokens_per_second == 25.0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_reports_failure_when_stream_raises_before_any_chunk(self) -> None:
        """Should return an unavailable result carrying the raised exception's message."""

        async def astream(*_args, **_kwargs):
            raise httpx.ConnectError("connection refused")
            yield  # pragma: no cover - unreachable, keeps this an async generator function

        adapter = _make_fake_adapter(astream=astream)

        result = await _try_streaming(adapter, "model-a")

        assert result.is_available is False
        assert result.path == "stream"
        assert result.tokens_per_second is None
        assert result.error == "connection refused"


class TestTryNonStreaming:
    """Tests for `_try_non_streaming()`'s probe and wall-clock tokens/sec fallback."""

    @pytest.mark.asyncio
    async def test_falls_back_to_wallclock_tokens_per_second_from_usage_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should derive tokens/sec from usage_metadata's output_tokens divided by elapsed time.

        Exercised when there is no Ollama-specific response_metadata to prefer instead.
        """
        fake_message = SimpleNamespace(response_metadata={}, usage_metadata={"output_tokens": 30})

        async def ainvoke(*_args, **_kwargs):
            return fake_message

        adapter = _make_fake_adapter(ainvoke=ainvoke)

        # `health_check.py` does `import time` and calls `time.monotonic()`, so patching this
        # attribute patches the real, global `time` module for as long as monkeypatch keeps it
        # active - including any unrelated `time.monotonic()` calls pytest's own internals make
        # during the test. A repeating tail after the two values this test actually cares about
        # keeps those incidental extra calls from raising StopIteration.
        times = itertools.chain([100.0, 103.0], itertools.repeat(103.0))
        monkeypatch.setattr("ocht.services.health_check.time.monotonic", lambda: next(times))

        result = await _try_non_streaming(adapter, "model-a")

        assert result.is_available is True
        assert result.path == "async"
        assert result.tokens_per_second == pytest.approx(10.0)


class TestCheckModelHealth:
    """Tests for `check_model_health()`'s streaming-then-async-fallback orchestration."""

    @pytest.mark.asyncio
    async def test_falls_back_to_non_streaming_when_streaming_fails(self, temp_db) -> None:
        """Should report path="async" and is_available=True when only the fallback succeeds."""
        with get_session() as db:
            provider = create_llm_provider_config(db, name="Ollama", api_key="", endpoint="http://localhost:11434")
            create_model(db, "model-a", model_provider_id=provider.prov_id)
            provider_id = provider.prov_id

        async def astream(*_args, **_kwargs):
            raise httpx.ConnectError("stream boom")
            yield  # pragma: no cover - unreachable, keeps this an async generator function

        fake_adapter = _make_fake_adapter(astream=astream, ainvoke=AsyncMock(return_value=AIMessage(content="OK")))

        with patch("ocht.services.health_check.build_adapter", return_value=fake_adapter):
            result = await check_model_health(provider_id, "model-a")

        assert result.is_available is True
        assert result.path == "async"

    @pytest.mark.asyncio
    async def test_reports_streaming_error_when_both_paths_fail(self, temp_db) -> None:
        """Should report the streaming attempt's error (not the async fallback's) when both fail."""
        with get_session() as db:
            provider = create_llm_provider_config(db, name="Ollama", api_key="", endpoint="http://localhost:11434")
            create_model(db, "model-a", model_provider_id=provider.prov_id)
            provider_id = provider.prov_id

        async def astream(*_args, **_kwargs):
            raise RuntimeError("stream failure")
            yield  # pragma: no cover - unreachable, keeps this an async generator function

        async def ainvoke(*_args, **_kwargs):
            raise RuntimeError("async failure")

        fake_adapter = _make_fake_adapter(astream=astream, ainvoke=ainvoke)

        with patch("ocht.services.health_check.build_adapter", return_value=fake_adapter):
            result = await check_model_health(provider_id, "model-a")

        assert result.is_available is False
        assert result.error == "stream failure"


class TestRunHealthCheck:
    """Tests for `run_health_check()`'s persistence and its non-disruption of the active adapter."""

    @pytest.mark.asyncio
    async def test_persists_result_onto_model_row(self, temp_db) -> None:
        """Should write the check's outcome onto the model's health-check columns."""
        with get_session() as db:
            provider = create_llm_provider_config(db, name="Ollama", api_key="", endpoint="http://localhost:11434")
            create_model(db, "model-a", model_provider_id=provider.prov_id, is_available=False)
            provider_id = provider.prov_id

        async def astream(*_args, **_kwargs):
            yield AIMessageChunk(
                content="OK", response_metadata={"eval_count": 10, "eval_duration": 1_000_000_000}
            )

        fake_adapter = _make_fake_adapter(astream=astream)

        with patch("ocht.services.health_check.build_adapter", return_value=fake_adapter):
            result = await run_health_check(provider_id, "model-a")

        with get_session() as db:
            model = get_model_by_name(db, "model-a")

        assert model.is_available is True
        assert model.last_checked is not None
        assert model.last_check_latency_ms == pytest.approx(result.latency_ms)
        assert model.last_check_tokens_per_second == pytest.approx(result.tokens_per_second)
        assert model.last_check_error is None

    @pytest.mark.asyncio
    async def test_does_not_disrupt_the_active_chat_adapter(self, temp_db) -> None:
        """Health-checking model B must never change which adapter/model the manager reports active.

        `AdapterManager` builds model A's adapter for real (construction alone makes no network
        call), while `run_health_check()`'s own `build_adapter` reference is patched separately -
        so B's fake adapter is used for the check without ever touching A's real one.
        """
        with get_session() as db:
            provider = create_llm_provider_config(db, name="Ollama", api_key="", endpoint="http://localhost:11434")
            create_model(db, "model-a", model_provider_id=provider.prov_id)
            create_model(db, "model-b", model_provider_id=provider.prov_id)
            provider_id = provider.prov_id

        manager = AdapterManager()
        assert manager.switch_adapter(provider_id, "model-a") is True
        active_adapter = manager.get_current_adapter()

        async def astream(*_args, **_kwargs):
            yield AIMessageChunk(content="OK", response_metadata={"eval_count": 1, "eval_duration": 1_000_000_000})

        fake_adapter_for_b = _make_fake_adapter(astream=astream)

        with patch("ocht.services.health_check.build_adapter", return_value=fake_adapter_for_b):
            await run_health_check(provider_id, "model-b")

        assert manager.get_current_adapter() is active_adapter
        assert manager.get_current_model_name() == "model-a"
