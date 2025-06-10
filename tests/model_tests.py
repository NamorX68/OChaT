import pytest
from unittest import mock
from sqlalchemy.orm import Session
from ocht.repositories.model import (
    create_model,
    get_model_by_name,
    get_all_models,
    update_model,
    delete_model
)
from ocht.core.models import Model

@pytest.fixture
def mock_db():
    return mock.create_autospec(Session)

def test_create_model(mock_db):
    # Arrange
    db = mock_db
    provider_id = 1
    model_name = "test-model"
    description = "Test description"

    # Act
    result = create_model(db, provider_id, model_name, description)

    # Assert
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()
    assert result.model_name == model_name
    assert result.model_provider_id == provider_id
    assert result.model_description == description

def test_get_model_by_name_found(mock_db):
    # Arrange
    db = mock_db
    model_name = "test-model"
    mock_model = mock.create_autospec(Model)
    db.exec.return_value.one_or_none.return_value = mock_model

    # Act
    result = get_model_by_name(db, model_name)

    # Assert
    assert result == mock_model

def test_get_model_by_name_not_found(mock_db):
    # Arrange
    db = mock_db
    model_name = "non-existent"
    db.exec.return_value.one_or_none.return_value = None

    # Act
    result = get_model_by_name(db, model_name)

    # Assert
    assert result is None

def test_get_all_models(mock_db):
    # Arrange
    db = mock_db
    mock_models = [mock.create_autospec(Model) for _ in range(3)]
    db.exec.return_value.all.return_value = mock_models

    # Act
    result = get_all_models(db)

    # Assert
    assert len(result) == 3
    assert all(isinstance(m, Model) for m in result)

def test_update_model(mock_db):
    # Arrange
    db = mock_db
    model_name = "test-model"
    new_description = "Updated description"
    mock_model = mock.create_autospec(Model)
    db.exec.return_value.one_or_none.return_value = mock_model

    # Act
    result = update_model(db, model_name, description=new_description)

    # Assert
    assert result.model_description == new_description
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_delete_model(mock_db):
    # Arrange
    db = mock_db
    model_name = "test-model"
    mock_model = mock.create_autospec(Model)
    db.exec.return_value.one_or_none.return_value = mock_model

    # Act
    result = delete_model(db, model_name)

    # Assert
    assert result is True
    db.delete.assert_called_once()
    db.commit.assert_called_once()

def test_delete_model_not_found(mock_db):
    # Arrange
    db = mock_db
    model_name = "non-existent"
    db.exec.return_value.one_or_none.return_value = None

    # Act
    result = delete_model(db, model_name)

    # Assert
    assert result is False