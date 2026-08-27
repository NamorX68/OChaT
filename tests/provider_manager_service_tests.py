"""Tests for the provider_manager service layer's provider-params (routing preferences) validation."""
import tempfile
from pathlib import Path

import pytest

from ocht.core.db import create_db_engine, init_db
from ocht.services.provider_manager import (
    create_provider_with_validation,
    update_provider_with_validation,
)


@pytest.fixture
def temp_db(monkeypatch):
    """Point DATABASE_URL at a fresh, empty temp SQLite file for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "provider_manager.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        init_db(create_db_engine())
        yield db_path


def test_create_provider_with_valid_params(temp_db):
    """Test that a well-formed JSON params string is stored as-is."""
    params = '{"quantizations": ["fp8", "fp16"], "preferred_min_throughput": 40}'

    provider = create_provider_with_validation("OpenRouter", api_key="sk-or-test", params=params)

    assert provider.prov_params == params


def test_create_provider_with_malformed_params_raises(temp_db):
    """Test that malformed JSON is rejected before it ever reaches the database."""
    with pytest.raises(ValueError, match="valid JSON"):
        create_provider_with_validation("OpenRouter", api_key="sk-or-test", params="{not valid json")


def test_create_provider_without_params(temp_db):
    """Test that omitting params entirely leaves prov_params unset."""
    provider = create_provider_with_validation("OpenRouter", api_key="sk-or-test")

    assert provider.prov_params is None


def test_update_provider_params_roundtrip(temp_db):
    """Test that updating a provider with new params persists them."""
    provider = create_provider_with_validation("OpenRouter", api_key="sk-or-test")
    new_params = '{"quantizations": ["fp8"]}'

    updated = update_provider_with_validation(provider.prov_id, params=new_params)

    assert updated.prov_params == new_params


def test_update_provider_params_clear(temp_db):
    """Test that passing params="" clears a previously-set value."""
    params = '{"quantizations": ["fp8"]}'
    provider = create_provider_with_validation("OpenRouter", api_key="sk-or-test", params=params)

    updated = update_provider_with_validation(provider.prov_id, params="")

    assert updated.prov_params is None


def test_update_provider_without_params_leaves_existing_value(temp_db):
    """Test that omitting params on update does not clear or change the existing value."""
    params = '{"quantizations": ["fp8"]}'
    provider = create_provider_with_validation("OpenRouter", api_key="sk-or-test", params=params)

    updated = update_provider_with_validation(provider.prov_id, name="OpenRouter")

    assert updated.prov_params == params


def test_update_provider_with_malformed_params_raises(temp_db):
    """Test that malformed JSON on update is rejected without touching the existing value."""
    provider = create_provider_with_validation("OpenRouter", api_key="sk-or-test")

    with pytest.raises(ValueError, match="valid JSON"):
        update_provider_with_validation(provider.prov_id, params="{not valid json")
