"""Tests for AdapterManager's provider-name-to-adapter-class wiring."""
import json
import tempfile
from pathlib import Path

import pytest

from ocht.adapters.openai_compatible import OpenAICompatibleAdapter
from ocht.core.db import create_db_engine, get_session, init_db
from ocht.core.models import LLMProviderConfig
from ocht.repositories.llm_provider_config import create_llm_provider_config
from ocht.repositories.model import create_model
from ocht.services.adapter_manager import AdapterManager, _build_openai_compatible_params


@pytest.fixture
def temp_db(monkeypatch):
    """Point DATABASE_URL at a fresh, empty temp SQLite file for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "adapter_manager.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        init_db(create_db_engine())
        yield db_path


def test_switch_adapter_recognizes_openrouter_as_openai_compatible(temp_db):
    """Test that an OpenRouter provider resolves to OpenAICompatibleAdapter.

    OpenRouter speaks the OpenAI chat-completions API, so it should be handled by the same
    branch as "openai"/"lm studio" rather than falling into the "unsupported provider" case,
    which would silently make switch_adapter() return False.
    """
    with get_session() as db:
        provider = create_llm_provider_config(
            db, name="OpenRouter", api_key="sk-or-test", endpoint="https://openrouter.ai/api/v1"
        )
        create_model(db, "openai/gpt-4o-mini", model_provider_id=provider.prov_id)
        provider_id = provider.prov_id

    manager = AdapterManager()
    assert manager.switch_adapter(provider_id, "openai/gpt-4o-mini") is True
    assert isinstance(manager.get_current_adapter(), OpenAICompatibleAdapter)


def test_build_openai_compatible_params_forwards_provider_routing():
    """Test that prov_params (e.g. OpenRouter provider-routing prefs) becomes extra_body.

    This is what lets a provider like OpenRouter be configured to only route to endpoints at a
    minimum quantization / throughput, without any adapter code changes - see
    CLAUDE.md "Dependency Upgrades" / the OpenRouter provider setup for the concrete values used.
    """
    routing = {"quantizations": ["fp8", "fp16"], "preferred_min_throughput": 40}
    provider = LLMProviderConfig(
        prov_name="OpenRouter", prov_api_key="sk-or-test", prov_params=json.dumps(routing)
    )

    params = _build_openai_compatible_params(provider)

    assert params["extra_body"] == {"provider": routing}
    assert params["temperature"] == 0.7  # default stays intact alongside the routing prefs


def test_build_openai_compatible_params_ignores_malformed_json():
    """Test that a broken prov_params value degrades gracefully instead of blocking the switch."""
    provider = LLMProviderConfig(
        prov_name="OpenRouter", prov_api_key="sk-or-test", prov_params="not valid json"
    )

    params = _build_openai_compatible_params(provider)

    assert "extra_body" not in params
    assert params["temperature"] == 0.7


def test_build_openai_compatible_params_without_prov_params():
    """Test that a provider with no prov_params set gets plain default params."""
    provider = LLMProviderConfig(prov_name="OpenAI", prov_api_key="sk-test")

    params = _build_openai_compatible_params(provider)

    assert params == {"temperature": 0.7}
