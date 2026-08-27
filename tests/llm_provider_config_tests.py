"""Tests for the LLMProviderConfig repository CRUD functions."""
from unittest import mock

import pytest
from sqlmodel import Session

from ocht.core.models import LLMProviderConfig
from ocht.repositories.llm_provider_config import (
    create_llm_provider_config,
    delete_llm_provider_config,
    get_all_llm_provider_configs,
    get_llm_provider_config_by_id,
    update_llm_provider_config,
)


@pytest.fixture
def mock_db():
    """Provides an autospecced mock of a SQLAlchemy Session."""
    return mock.create_autospec(Session)

def test_create_llm_provider_config(mock_db):
    """Test that create_llm_provider_config() adds, commits, and refreshes the new config."""
    # Arrange
    db = mock_db
    name = "test_provider"
    api_key = "test_key"

    # Act
    result = create_llm_provider_config(db, name, api_key)

    # Assert
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()
    assert result.prov_name == name
    assert result.prov_api_key == api_key

def test_get_llm_provider_config_by_id_found(mock_db):
    """Test that get_llm_provider_config_by_id() returns the matching config when found."""
    # Arrange
    db = mock_db
    config_id = 1
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config

    # Act
    result = get_llm_provider_config_by_id(db, config_id)

    # Assert
    assert result == mock_config

def test_get_all_llm_provider_configs(mock_db):
    """Test that get_all_llm_provider_configs() returns all configs from the session."""
    # Arrange
    db = mock_db
    mock_configs = [mock.create_autospec(LLMProviderConfig) for _ in range(3)]
    db.exec.return_value.all.return_value = mock_configs

    # Act
    result = get_all_llm_provider_configs(db)

    # Assert
    assert len(result) == 3

def test_update_llm_provider_config(mock_db):
    """Test that update_llm_provider_config() updates and persists the config's name."""
    # Arrange
    db = mock_db
    config_id = 1
    new_name = "updated_name"
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config

    # Act
    result = update_llm_provider_config(db, config_id, name=new_name)

    # Assert
    assert result.prov_name == new_name
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_create_llm_provider_config_with_params(mock_db):
    """Test that create_llm_provider_config() stores provider routing params when given."""
    db = mock_db
    params = '{"quantizations": ["fp8"], "preferred_min_throughput": 40}'

    result = create_llm_provider_config(db, "OpenRouter", "sk-or-test", params=params)

    assert result.prov_params == params

def test_update_llm_provider_config_sets_params(mock_db):
    """Test that update_llm_provider_config() sets prov_params when a non-empty value is given."""
    db = mock_db
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config
    params = '{"quantizations": ["fp8"]}'

    result = update_llm_provider_config(db, 1, params=params)

    assert result.prov_params == params

def test_update_llm_provider_config_clears_params_on_empty_string(mock_db):
    """Test that passing params="" explicitly clears prov_params (vs. params=None, see docstring)."""
    db = mock_db
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config

    result = update_llm_provider_config(db, 1, params="")

    assert result.prov_params is None

def test_update_llm_provider_config_leaves_params_untouched_by_default(mock_db):
    """Test that omitting params (None) does not touch the existing prov_params value."""
    db = mock_db
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config
    original_params = mock_config.prov_params

    update_llm_provider_config(db, 1, name="renamed")

    assert mock_config.prov_params is original_params

def test_delete_llm_provider_config(mock_db):
    """Test that delete_llm_provider_config() deletes the config and returns True."""
    # Arrange
    db = mock_db
    config_id = 1
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config

    # Act
    result = delete_llm_provider_config(db, config_id)

    # Assert
    assert result is True
    db.delete.assert_called_once()
    db.commit.assert_called_once()