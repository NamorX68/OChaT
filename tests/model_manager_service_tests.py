"""Tests for the model_manager service layer's typed generation-config fields.

Covers `max_tokens`/`context_window`, layered on top of `Model.model_params`.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ocht.core.db import create_db_engine, init_db
from ocht.services.model_manager import (
    _max_tokens_key_for_provider,
    _validate_model_params,
    build_model_params_json,
    create_model_with_validation,
    split_model_params_for_editing,
    update_model_with_validation,
)
from ocht.services.provider_manager import create_provider_with_validation


@pytest.fixture
def temp_db(monkeypatch):
    """Point DATABASE_URL at a fresh, empty temp SQLite file for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "model_manager.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        init_db(create_db_engine())
        yield db_path


class TestMaxTokensKeyForProvider:
    """Tests for `_max_tokens_key_for_provider`."""

    @pytest.mark.parametrize("provider_name", ["ollama", "Ollama", "OLLAMA"])
    def test_returns_num_predict_for_ollama_case_insensitively(self, provider_name: str) -> None:
        """Should return `num_predict` for any casing of "ollama"."""
        assert _max_tokens_key_for_provider(provider_name) == "num_predict"

    @pytest.mark.parametrize("provider_name", ["openai", "openrouter", "lm studio", "anthropic", "some-other-name"])
    def test_returns_max_tokens_for_non_ollama_providers(self, provider_name: str) -> None:
        """Should return `max_tokens` for any provider other than Ollama."""
        assert _max_tokens_key_for_provider(provider_name) == "max_tokens"


class TestValidateModelParams:
    """Tests for `_validate_model_params`."""

    def test_none_returns_none(self) -> None:
        """Should return None when given None."""
        assert _validate_model_params(None) is None

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_blank_returns_none(self, value: str) -> None:
        """Should return None for empty or whitespace-only strings."""
        assert _validate_model_params(value) is None

    def test_valid_json_object_returned_stripped(self) -> None:
        """Should return the stripped JSON string unchanged for a valid JSON object."""
        raw = '  {"temperature": 0.7}  '
        assert _validate_model_params(raw) == '{"temperature": 0.7}'

    def test_malformed_json_raises_value_error(self) -> None:
        """Should raise ValueError for malformed JSON."""
        with pytest.raises(ValueError, match="valid JSON"):
            _validate_model_params("{not valid json")

    @pytest.mark.parametrize("value", ["[1, 2, 3]", "42", '"just a string"'])
    def test_non_object_json_raises_value_error(self, value: str) -> None:
        """Should raise ValueError for valid JSON that isn't a JSON object (list/number/string)."""
        with pytest.raises(ValueError, match="JSON object"):
            _validate_model_params(value)


class TestBuildModelParamsJson:
    """Tests for `build_model_params_json`."""

    def test_only_max_tokens_ollama(self) -> None:
        """Should store max_tokens under num_predict for Ollama."""
        result = build_model_params_json(None, max_tokens=100, context_window=None, provider_name="Ollama")

        assert json.loads(result) == {"num_predict": 100}

    def test_only_max_tokens_non_ollama(self) -> None:
        """Should store max_tokens under max_tokens for a non-Ollama provider."""
        result = build_model_params_json(None, max_tokens=100, context_window=None, provider_name="OpenAI")

        assert json.loads(result) == {"max_tokens": 100}

    def test_only_context_window_ollama(self) -> None:
        """Should store context_window under num_ctx for Ollama."""
        result = build_model_params_json(None, max_tokens=None, context_window=4096, provider_name="Ollama")

        assert json.loads(result) == {"num_ctx": 4096}

    def test_only_context_window_non_ollama_is_dropped(self) -> None:
        """Should silently drop context_window for any non-Ollama provider."""
        result = build_model_params_json(None, max_tokens=None, context_window=4096, provider_name="OpenAI")

        assert result == ""

    def test_advanced_params_merged_with_max_tokens(self) -> None:
        """Should merge advanced params and the typed max_tokens field into one object."""
        result = build_model_params_json(
            '{"temperature": 1.0}', max_tokens=256, context_window=None, provider_name="OpenAI"
        )

        assert json.loads(result) == {"temperature": 1.0, "max_tokens": 256}

    def test_stale_wrong_provider_key_is_overridden_not_stacked(self) -> None:
        """Should remove a stale advanced_params key when a typed field maps elsewhere.

        A stale `max_tokens` key in advanced_params should be removed when the typed field maps to
        `num_predict` instead, rather than leaving both keys present.
        """
        result = build_model_params_json(
            '{"max_tokens": 999}', max_tokens=50, context_window=None, provider_name="Ollama"
        )

        parsed = json.loads(result)
        assert parsed == {"num_predict": 50}
        assert "max_tokens" not in parsed

    def test_stale_num_predict_removed_for_non_ollama(self) -> None:
        """Should remove a stale `num_predict` key when the model is now on a non-Ollama provider.

        The stale key would have been left over from a prior Ollama configuration.
        """
        result = build_model_params_json(
            '{"num_predict": 999}', max_tokens=50, context_window=None, provider_name="OpenAI"
        )

        parsed = json.loads(result)
        assert parsed == {"max_tokens": 50}
        assert "num_predict" not in parsed

    def test_stale_num_ctx_removed_when_context_window_not_set(self) -> None:
        """Should remove a stale `num_ctx` key when context_window is no longer provided."""
        result = build_model_params_json(
            '{"num_ctx": 8192}', max_tokens=None, context_window=None, provider_name="Ollama"
        )

        assert result == ""

    def test_everything_none_returns_empty_string(self) -> None:
        """Should return the exact empty string (not None) when nothing is configured."""
        result = build_model_params_json(None, max_tokens=None, context_window=None, provider_name="OpenAI")

        assert result == ""
        assert result is not None

    def test_empty_advanced_params_with_nothing_else_returns_empty_string(self) -> None:
        """Should return "" when advanced_params is an empty object and no typed fields are set."""
        result = build_model_params_json("{}", max_tokens=None, context_window=None, provider_name="Ollama")

        assert result == ""


class TestSplitModelParamsForEditing:
    """Tests for `split_model_params_for_editing`."""

    @pytest.mark.parametrize("value", [None, ""])
    def test_none_or_empty_returns_all_none(self, value: str | None) -> None:
        """Should return (None, None, None) for None or an empty model_params string."""
        assert split_model_params_for_editing(value, "Ollama") == (None, None, None)

    def test_whitespace_only_is_not_falsy_and_treated_as_malformed_json(self) -> None:
        """Should not treat a whitespace-only string the same as None/"".

        Such a string is truthy (unlike ""), so it falls through to the JSON parse attempt, fails
        there, and is surfaced unchanged as the leftover advanced params.
        """
        raw = "   "

        assert split_model_params_for_editing(raw, "Ollama") == (None, None, raw)

    def test_malformed_json_returned_unchanged(self) -> None:
        """Should surface malformed JSON unchanged in the advanced field rather than crashing."""
        raw = "{not valid json"

        result = split_model_params_for_editing(raw, "Ollama")

        assert result == (None, None, raw)

    def test_valid_json_non_object_returned_unchanged(self) -> None:
        """Should surface non-object JSON (e.g. a list) unchanged in the advanced field."""
        raw = "[1, 2, 3]"

        result = split_model_params_for_editing(raw, "OpenAI")

        assert result == (None, None, raw)

    def test_non_ollama_provider_ignores_num_ctx(self) -> None:
        """Should ignore num_ctx for a non-Ollama provider even if it is present.

        `context_window` stays None, and `num_ctx` is left in the remaining advanced JSON since
        it isn't a recognized key for that provider.
        """
        raw = json.dumps({"max_tokens": 100, "num_ctx": 4096})

        max_tokens, context_window, remaining = split_model_params_for_editing(raw, "OpenAI")

        assert max_tokens == 100
        assert context_window is None
        assert remaining is not None
        assert json.loads(remaining) == {"num_ctx": 4096}

    @pytest.mark.parametrize(
        "provider_name,advanced_params",
        [
            ("Ollama", None),
            ("Ollama", '{"temperature": 0.5}'),
            ("OpenAI", None),
            ("OpenAI", '{"temperature": 0.5}'),
        ],
    )
    def test_round_trips_with_build_model_params_json(
        self, provider_name: str, advanced_params: str | None
    ) -> None:
        """Should recover the max_tokens/context_window used to build a given model_params string.

        Verified for both an Ollama and a non-Ollama provider case.
        """
        max_tokens = 128
        context_window = 8192 if provider_name.lower() == "ollama" else None

        built = build_model_params_json(advanced_params, max_tokens, context_window, provider_name)

        recovered_max_tokens, recovered_context_window, remaining = split_model_params_for_editing(
            built, provider_name
        )

        assert recovered_max_tokens == max_tokens
        assert recovered_context_window == context_window
        if advanced_params:
            assert remaining is not None
            assert json.loads(remaining) == json.loads(advanced_params)
        else:
            assert remaining is None


class TestCreateModelWithValidationGenerationConfig:
    """End-to-end tests for `create_model_with_validation`'s typed generation-config fields."""

    def test_create_with_max_tokens_under_ollama_stores_num_predict(self, temp_db) -> None:
        """Should store max_tokens under num_predict when the provider is Ollama."""
        provider = create_provider_with_validation("Ollama", api_key="test-key")

        model = create_model_with_validation("llama3", provider.prov_id, max_tokens=100)

        assert json.loads(model.model_params) == {"num_predict": 100}

    def test_create_with_max_tokens_under_openai_stores_max_tokens(self, temp_db) -> None:
        """Should store max_tokens under max_tokens for a non-Ollama provider."""
        provider = create_provider_with_validation("OpenAI", api_key="sk-test")

        model = create_model_with_validation("gpt-4", provider.prov_id, max_tokens=100)

        assert json.loads(model.model_params) == {"max_tokens": 100}

    def test_create_with_nothing_configured_stores_none(self, temp_db) -> None:
        """Should store None (not a literal empty string) on a fresh row with no config."""
        provider = create_provider_with_validation("Ollama", api_key="test-key")

        model = create_model_with_validation("llama3", provider.prov_id)

        assert model.model_params is None

    def test_create_with_malformed_params_raises_before_hitting_db(self, temp_db) -> None:
        """Should raise ValueError for malformed advanced params without creating the model."""
        provider = create_provider_with_validation("Ollama", api_key="test-key")

        with pytest.raises(ValueError, match="valid JSON"):
            create_model_with_validation("llama3", provider.prov_id, params="{not valid json")


class TestUpdateModelWithValidationGenerationConfig:
    """End-to-end tests for `update_model_with_validation`'s typed generation-config fields."""

    def test_clearing_all_params_results_in_none(self, temp_db) -> None:
        """Should end up with model_params as None after clearing all generation-config fields.

        Not a stale leftover value, and not a literal empty string either.
        """
        provider = create_provider_with_validation("Ollama", api_key="test-key")
        create_model_with_validation(
            "llama3", provider.prov_id, params='{"temperature": 0.5}', max_tokens=50, context_window=4096
        )

        updated = update_model_with_validation("llama3", params=None, max_tokens=None, context_window=None)

        assert updated is not None
        assert updated.model_params is None

    def test_switching_provider_to_non_ollama_stores_under_new_key(self, temp_db) -> None:
        """Should store max_tokens under the new provider's key, not the old one.

        Applies when switching providers and setting max_tokens in the same update call.
        """
        ollama_provider = create_provider_with_validation("Ollama", api_key="test-key")
        openai_provider = create_provider_with_validation("OpenAI", api_key="sk-test")
        create_model_with_validation("some-model", ollama_provider.prov_id, max_tokens=50)

        updated = update_model_with_validation(
            "some-model", provider_id=openai_provider.prov_id, max_tokens=200
        )

        assert updated is not None
        parsed = json.loads(updated.model_params)
        assert parsed == {"max_tokens": 200}
        assert "num_predict" not in parsed

    def test_update_with_malformed_params_raises_without_touching_existing_value(self, temp_db) -> None:
        """Should raise ValueError for malformed params and leave the existing model_params intact."""
        provider = create_provider_with_validation("Ollama", api_key="test-key")
        create_model_with_validation("llama3", provider.prov_id, max_tokens=50)

        with pytest.raises(ValueError, match="valid JSON"):
            update_model_with_validation("llama3", params="{not valid json")
