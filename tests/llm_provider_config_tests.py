import pytest
from unittest import mock
from sqlalchemy.orm import Session
from ocht.repositories.llm_provider_config import (
    create_llm_provider_config,
    get_llm_provider_config_by_id,
    get_all_llm_provider_configs,
    update_llm_provider_config,
    delete_llm_provider_config
)
from ocht.core.models import LLMProviderConfig

@pytest.fixture
def mock_db():
    return mock.create_autospec(Session)

def test_create_llm_provider_config(mock_db):
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
    assert result.name == name
    assert result.api_key == api_key

def test_get_llm_provider_config_by_id_found(mock_db):
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
    # Arrange
    db = mock_db
    mock_configs = [mock.create_autospec(LLMProviderConfig) for _ in range(3)]
    db.exec.return_value.all.return_value = mock_configs

    # Act
    result = get_all_llm_provider_configs(db)

    # Assert
    assert len(result) == 3

def test_update_llm_provider_config(mock_db):
    # Arrange
    db = mock_db
    config_id = 1
    new_name = "updated_name"
    mock_config = mock.create_autospec(LLMProviderConfig)
    db.exec.return_value.one_or_none.return_value = mock_config

    # Act
    result = update_llm_provider_config(db, config_id, name=new_name)

    # Assert
    assert result.name == new_name
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_delete_llm_provider_config(mock_db):
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