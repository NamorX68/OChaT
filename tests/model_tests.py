"""Tests for the Model repository CRUD functions."""
from unittest import mock

import pytest
from sqlmodel import Session

from ocht.core.models import Model
from ocht.repositories.model import create_model, delete_model, get_all_models, get_model_by_name, update_model


@pytest.fixture
def mock_db():
    """Provides an autospecced mock of a SQLAlchemy Session."""
    return mock.create_autospec(Session)

def test_create_model(mock_db):
    """Test that create_model() adds, commits, and refreshes the new model."""
    # Arrange
    db = mock_db
    provider_id = 1
    model_name = "test-model"
    description = "Test description"

    # Act
    result = create_model(db, model_name, provider_id, description)

    # Assert
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()
    assert result.model_name == model_name
    assert result.model_provider_id == provider_id
    assert result.model_description == description

def test_get_model_by_name_found(mock_db):
    """Test that get_model_by_name() returns the matching model when found."""
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
    """Test that get_model_by_name() returns None when no model matches."""
    # Arrange
    db = mock_db
    model_name = "non-existent"
    db.exec.return_value.one_or_none.return_value = None

    # Act
    result = get_model_by_name(db, model_name)

    # Assert
    assert result is None

def test_get_all_models(mock_db):
    """Test that get_all_models() returns all models from the session."""
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
    """Test that update_model() updates and persists the model's description."""
    # Arrange
    db = mock_db
    model_name = "test-model"
    new_description = "Updated description"
    mock_model = mock.create_autospec(Model)
    db.exec.return_value.one_or_none.return_value = mock_model

    # Act
    result = update_model(db, model_name, model_description=new_description)

    # Assert
    assert result.model_description == new_description
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_delete_model(mock_db):
    """Test that delete_model() deletes the model and returns True."""
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
    """Test that delete_model() returns False when no model matches."""
    # Arrange
    db = mock_db
    model_name = "non-existent"
    db.exec.return_value.one_or_none.return_value = None

    # Act
    result = delete_model(db, model_name)

    # Assert
    assert result is False