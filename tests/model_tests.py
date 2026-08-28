"""Tests for the Model repository CRUD functions."""
from datetime import datetime
from unittest import mock

import pytest
from sqlmodel import Session

from ocht.core.models import Model
from ocht.repositories.model import (
    create_model,
    delete_model,
    get_all_models,
    get_model_by_name,
    record_health_check_result,
    update_model,
)


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

def test_update_model_with_empty_string_clears_model_params(mock_db):
    """Test that update_model(model_params="") clears an existing model_params value to None.

    Rather than storing a literal empty string.
    """
    db = mock_db
    model_name = "test-model"
    mock_model = mock.create_autospec(Model)
    mock_model.model_params = '{"temperature": 0.5}'
    db.exec.return_value.one_or_none.return_value = mock_model

    result = update_model(db, model_name, model_params="")

    assert result.model_params is None
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_update_model_with_none_leaves_model_params_untouched(mock_db):
    """Test that update_model(model_params=None) leaves an existing model_params value unchanged.

    Mirrors the "None means don't touch this field" behavior already used for other fields.
    """
    db = mock_db
    model_name = "test-model"
    existing_params = '{"temperature": 0.5}'
    mock_model = mock.create_autospec(Model)
    mock_model.model_params = existing_params
    mock_model.model_description = "Original description"
    db.exec.return_value.one_or_none.return_value = mock_model

    result = update_model(db, model_name, model_params=None)

    assert result.model_params == existing_params
    assert result.model_description == "Original description"

def test_record_health_check_result_records_a_failed_check(mock_db):
    """Test that record_health_check_result() writes a failing check's error onto the model."""
    db = mock_db
    model_name = "test-model"
    mock_model = mock.create_autospec(Model)
    db.exec.return_value.one_or_none.return_value = mock_model

    result = record_health_check_result(
        db, model_name, is_available=False, checked_at=datetime.now(),
        latency_ms=None, tokens_per_second=None, error="boom",
    )

    assert result.is_available is False
    assert result.last_check_error == "boom"
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_record_health_check_result_clears_a_previous_error_with_none():
    """Test that error=None actually clears a previously-recorded last_check_error to None.

    This is the deliberate divergence from update_model()'s "None means leave unchanged"
    convention: record_health_check_result() unconditionally overwrites every field on each call,
    since a successful check after a failing one must actually clear the stale error rather than
    silently keep it.
    """
    db = mock.create_autospec(Session)
    model_name = "test-model"
    mock_model = mock.create_autospec(Model)
    mock_model.last_check_error = "boom"  # simulate a previously-recorded failure
    db.exec.return_value.one_or_none.return_value = mock_model

    result = record_health_check_result(
        db, model_name, is_available=True, checked_at=datetime.now(),
        latency_ms=120.5, tokens_per_second=42.0, error=None,
    )

    assert result.last_check_error is None
    assert result.is_available is True
    assert result.last_check_latency_ms == 120.5
    assert result.last_check_tokens_per_second == 42.0


def test_record_health_check_result_returns_none_for_unknown_model(mock_db):
    """Test that record_health_check_result() returns None (not raising) for an unknown model."""
    db = mock_db
    db.exec.return_value.one_or_none.return_value = None

    result = record_health_check_result(
        db, "does-not-exist", is_available=True, checked_at=datetime.now(),
        latency_ms=None, tokens_per_second=None, error=None,
    )

    assert result is None
    db.commit.assert_not_called()


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