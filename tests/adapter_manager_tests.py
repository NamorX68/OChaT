"""Tests for AdapterManager's provider-name-to-adapter-class wiring."""
import json
import tempfile
from pathlib import Path

import pytest

from ocht.adapters.memory import MemoryConfig
from ocht.adapters.ollama import OllamaAdapter
from ocht.adapters.openai_compatible import OpenAICompatibleAdapter
from ocht.core.db import create_db_engine, get_session, init_db
from ocht.core.models import LLMProviderConfig
from ocht.repositories.llm_provider_config import create_llm_provider_config
from ocht.repositories.model import create_model
from ocht.repositories.setting import create_setting
from ocht.services.adapter_manager import (
    AdapterManager,
    _build_memory_config,
    _build_openai_compatible_params,
    _merge_model_params,
    _read_positive_float_setting,
    _read_positive_int_setting,
)


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


def test_merge_model_params_overrides_conflicting_key():
    """Test that a model-level value wins over a provider-level default on key conflicts."""
    default_params = {"temperature": 0.7}

    merged = _merge_model_params(default_params, json.dumps({"temperature": 0.2}))

    assert merged["temperature"] == 0.2


def test_merge_model_params_adds_new_key():
    """Test that a model-level key absent from default_params is added to the merged result."""
    default_params = {"temperature": 0.7}

    merged = _merge_model_params(default_params, json.dumps({"max_tokens": 4096}))

    assert merged == {"temperature": 0.7, "max_tokens": 4096}


def test_merge_model_params_ignores_malformed_json():
    """Test that a broken model_params value degrades gracefully to default_params alone."""
    default_params = {"temperature": 0.7}

    merged = _merge_model_params(default_params, "not valid json")

    assert merged == {"temperature": 0.7}


def test_merge_model_params_with_none_returns_new_unmutated_copy():
    """Test that model_params=None returns default_params unchanged, without mutating the input."""
    default_params = {"temperature": 0.7}

    merged = _merge_model_params(default_params, None)

    assert merged == default_params
    assert merged is not default_params  # a new dict, not the same object

    # Mutating the result must not leak back into the caller's dict.
    merged["temperature"] = 0.1
    assert default_params["temperature"] == 0.7


@pytest.mark.parametrize("non_dict_json", [json.dumps([1, 2, 3]), json.dumps(42), json.dumps("a string")])
def test_merge_model_params_ignores_non_dict_json(non_dict_json):
    """Test that syntactically valid but non-dict JSON (array/number/string) is ignored."""
    default_params = {"temperature": 0.7}

    merged = _merge_model_params(default_params, non_dict_json)

    assert merged == {"temperature": 0.7}


def test_switch_adapter_flows_model_params_into_adapter(temp_db):
    """Test that Model.model_params flows end-to-end into the adapter's underlying LLM client.

    The model-level temperature/max_tokens should win over the provider-level default
    (`_build_openai_compatible_params()`'s hardcoded `temperature: 0.7`).
    """
    with get_session() as db:
        provider = create_llm_provider_config(
            db, name="OpenRouter", api_key="sk-or-test", endpoint="https://openrouter.ai/api/v1"
        )
        create_model(
            db,
            "openai/gpt-4o-mini",
            model_provider_id=provider.prov_id,
            model_params=json.dumps({"temperature": 0.2, "max_tokens": 123}),
        )
        provider_id = provider.prov_id

    manager = AdapterManager()
    assert manager.switch_adapter(provider_id, "openai/gpt-4o-mini") is True

    adapter = manager.get_current_adapter()
    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.client.temperature == 0.2
    assert adapter.client.max_tokens == 123


class TestReadPositiveIntSetting:
    """Tests for `_read_positive_int_setting`."""

    def test_returns_default_when_setting_missing(self, temp_db) -> None:
        """Should return `default` when no Setting row matches the given key."""
        with get_session() as db:
            assert _read_positive_int_setting(db, "does_not_exist", 42) == 42

    def test_returns_parsed_value_for_valid_positive_int_string(self, temp_db) -> None:
        """Should return the parsed int for a valid positive integer string."""
        with get_session() as db:
            create_setting(db, "some_int_key", "9000")
            assert _read_positive_int_setting(db, "some_int_key", 42) == 9000

    @pytest.mark.parametrize("raw_value", ["0", "-5"])
    def test_returns_default_for_non_positive_value(self, temp_db, raw_value: str) -> None:
        """Should return `default` (not the parsed value) for zero or negative integers."""
        with get_session() as db:
            create_setting(db, "some_int_key", raw_value)
            assert _read_positive_int_setting(db, "some_int_key", 42) == 42

    def test_returns_default_for_non_numeric_string(self, temp_db) -> None:
        """Should return `default` when the setting value cannot be parsed as an int."""
        with get_session() as db:
            create_setting(db, "some_int_key", "abc")
            assert _read_positive_int_setting(db, "some_int_key", 42) == 42

    def test_returns_default_for_float_string(self, temp_db) -> None:
        """Should return `default` for a float string, since `int("3.5")` raises `ValueError`.

        Confirms actual behavior rather than assuming truncation of the float string.
        """
        with get_session() as db:
            create_setting(db, "some_int_key", "3.5")
            assert _read_positive_int_setting(db, "some_int_key", 42) == 42


class TestReadPositiveFloatSetting:
    """Tests for `_read_positive_float_setting`."""

    def test_returns_default_when_setting_missing(self, temp_db) -> None:
        """Should return `default` when no Setting row matches the given key."""
        with get_session() as db:
            assert _read_positive_float_setting(db, "does_not_exist", 1.5) == 1.5

    def test_returns_parsed_value_for_valid_positive_float_string(self, temp_db) -> None:
        """Should return the parsed float for a valid positive float string."""
        with get_session() as db:
            create_setting(db, "some_float_key", "9000.5")
            assert _read_positive_float_setting(db, "some_float_key", 1.5) == 9000.5

    def test_parses_float_string_that_int_would_reject(self, temp_db) -> None:
        """Should successfully parse a float string like '1.5', unlike `_read_positive_int_setting`."""
        with get_session() as db:
            create_setting(db, "some_float_key", "1.5")
            assert _read_positive_float_setting(db, "some_float_key", 42.0) == 1.5

    @pytest.mark.parametrize("raw_value", ["0", "-5", "-5.5"])
    def test_returns_default_for_non_positive_value(self, temp_db, raw_value: str) -> None:
        """Should return `default` (not the parsed value) for zero or negative numbers."""
        with get_session() as db:
            create_setting(db, "some_float_key", raw_value)
            assert _read_positive_float_setting(db, "some_float_key", 1.5) == 1.5

    def test_returns_default_for_non_numeric_string(self, temp_db) -> None:
        """Should return `default` when the setting value cannot be parsed as a float."""
        with get_session() as db:
            create_setting(db, "some_float_key", "abc")
            assert _read_positive_float_setting(db, "some_float_key", 1.5) == 1.5


class TestBuildMemoryConfig:
    """Tests for `_build_memory_config`."""

    def test_returns_plain_defaults_when_no_settings_configured(self, temp_db) -> None:
        """Should return a `MemoryConfig` equal to `MemoryConfig()` when nothing is overridden."""
        with get_session() as db:
            assert _build_memory_config(db) == MemoryConfig()

    def test_returns_all_overridden_values_when_all_settings_configured(self, temp_db) -> None:
        """Should reflect all four Setting overrides when all are configured with valid values."""
        with get_session() as db:
            create_setting(db, AdapterManager.MEMORY_MAX_CONTEXT_TOKENS_KEY, "8000")
            create_setting(db, AdapterManager.MEMORY_RECENT_MESSAGES_COUNT_KEY, "5")
            create_setting(db, AdapterManager.MEMORY_CODE_RETENTION_PRIORITY_KEY, "3.5")
            create_setting(db, AdapterManager.MEMORY_SUMMARIZATION_THRESHOLD_KEY, "30")

            config = _build_memory_config(db)

        assert config == MemoryConfig(
            max_context_tokens=8000,
            recent_messages_count=5,
            code_retention_priority=3.5,
            summarization_threshold=30,
        )

    def test_falls_back_per_field_on_mixed_valid_and_invalid_settings(self, temp_db) -> None:
        """Should only fall back to the default for the invalid/missing field, not all fields.

        `MEMORY_MAX_CONTEXT_TOKENS_KEY` and `MEMORY_CODE_RETENTION_PRIORITY_KEY` are configured with
        valid values; `MEMORY_SUMMARIZATION_THRESHOLD_KEY` is configured with an invalid value; and
        `MEMORY_RECENT_MESSAGES_COUNT_KEY` is left unset entirely.
        """
        with get_session() as db:
            create_setting(db, AdapterManager.MEMORY_MAX_CONTEXT_TOKENS_KEY, "8000")
            create_setting(db, AdapterManager.MEMORY_CODE_RETENTION_PRIORITY_KEY, "3.5")
            create_setting(db, AdapterManager.MEMORY_SUMMARIZATION_THRESHOLD_KEY, "not-a-number")

            config = _build_memory_config(db)

        defaults = MemoryConfig()
        assert config == MemoryConfig(
            max_context_tokens=8000,
            recent_messages_count=defaults.recent_messages_count,
            code_retention_priority=3.5,
            summarization_threshold=defaults.summarization_threshold,
        )


class TestSwitchAdapterMemoryConfigOverride:
    """End-to-end tests that a configured memory Setting reaches the adapter's memory_strategy."""

    def test_ollama_adapter_uses_plain_defaults_when_no_settings_configured(self, temp_db) -> None:
        """Should give the OllamaAdapter a plain-default MemoryConfig when nothing is overridden."""
        with get_session() as db:
            provider = create_llm_provider_config(db, name="Ollama", api_key="", endpoint="http://localhost:11434")
            create_model(db, "qwen3:30b-a3b", model_provider_id=provider.prov_id)
            provider_id = provider.prov_id

        manager = AdapterManager()
        assert manager.switch_adapter(provider_id, "qwen3:30b-a3b") is True

        adapter = manager.get_current_adapter()
        assert isinstance(adapter, OllamaAdapter)
        assert adapter.memory_strategy.config == MemoryConfig()

    def test_ollama_adapter_reflects_configured_memory_override(self, temp_db) -> None:
        """Should flow a configured memory Setting through to the OllamaAdapter's memory_strategy."""
        with get_session() as db:
            create_setting(db, AdapterManager.MEMORY_RECENT_MESSAGES_COUNT_KEY, "3")
            create_setting(db, AdapterManager.MEMORY_MAX_CONTEXT_TOKENS_KEY, "1234")
            provider = create_llm_provider_config(db, name="Ollama", api_key="", endpoint="http://localhost:11434")
            create_model(db, "qwen3:30b-a3b", model_provider_id=provider.prov_id)
            provider_id = provider.prov_id

        manager = AdapterManager()
        assert manager.switch_adapter(provider_id, "qwen3:30b-a3b") is True

        adapter = manager.get_current_adapter()
        assert isinstance(adapter, OllamaAdapter)
        config = adapter.memory_strategy.config
        assert config.recent_messages_count == 3
        assert config.max_context_tokens == 1234
        # Untouched fields still fall back to their normal defaults.
        defaults = MemoryConfig()
        assert config.code_retention_priority == defaults.code_retention_priority
        assert config.summarization_threshold == defaults.summarization_threshold

    def test_openai_compatible_adapter_reflects_configured_memory_override(self, temp_db) -> None:
        """Should flow a configured memory Setting through to the OpenAICompatibleAdapter's memory_strategy."""
        with get_session() as db:
            create_setting(db, AdapterManager.MEMORY_CODE_RETENTION_PRIORITY_KEY, "5.0")
            create_setting(db, AdapterManager.MEMORY_SUMMARIZATION_THRESHOLD_KEY, "50")
            provider = create_llm_provider_config(
                db, name="OpenRouter", api_key="sk-or-test", endpoint="https://openrouter.ai/api/v1"
            )
            create_model(db, "openai/gpt-4o-mini", model_provider_id=provider.prov_id)
            provider_id = provider.prov_id

        manager = AdapterManager()
        assert manager.switch_adapter(provider_id, "openai/gpt-4o-mini") is True

        adapter = manager.get_current_adapter()
        assert isinstance(adapter, OpenAICompatibleAdapter)
        config = adapter.memory_strategy.config
        assert config.code_retention_priority == 5.0
        assert config.summarization_threshold == 50
        # Untouched fields still fall back to their normal defaults.
        defaults = MemoryConfig()
        assert config.max_context_tokens == defaults.max_context_tokens
        assert config.recent_messages_count == defaults.recent_messages_count
