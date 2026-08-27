"""Tests for the Message repository CRUD functions."""
from unittest import mock

import pytest
from sqlmodel import Session

from ocht.core.models import Message
from ocht.repositories.message import (
    create_message,
    delete_message,
    get_message_by_id,
    get_messages_by_workspace,
    update_message,
)


@pytest.fixture
def mock_db():
    """Provides an autospecced mock of a SQLAlchemy Session."""
    return mock.create_autospec(Session)

def test_create_message(mock_db):
    """Test that create_message() adds, commits, and refreshes the new message."""
    # Arrange
    db = mock_db
    workspace_id = 1
    role = "user"
    content = "Test message"

    # Act
    result = create_message(db, workspace_id, role, content)

    # Assert
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()
    assert result.msg_workspace_id == workspace_id
    assert result.msg_role == role
    assert result.msg_content == content

def test_get_message_by_id_found(mock_db):
    """Test that get_message_by_id() returns the matching message when found."""
    # Arrange
    db = mock_db
    message_id = 1
    mock_message = mock.create_autospec(Message)
    db.exec.return_value.one_or_none.return_value = mock_message

    # Act
    result = get_message_by_id(db, message_id)

    # Assert
    assert result == mock_message

def test_get_messages_by_workspace(mock_db):
    """Test that get_messages_by_workspace() returns all messages from the session."""
    # Arrange
    db = mock_db
    workspace_id = 1
    mock_messages = [mock.create_autospec(Message) for _ in range(3)]
    db.exec.return_value.all.return_value = mock_messages

    # Act
    result = get_messages_by_workspace(db, workspace_id)

    # Assert
    assert len(result) == 3

def test_update_message(mock_db):
    """Test that update_message() updates and persists the message's content."""
    # Arrange
    db = mock_db
    message_id = 1
    new_content = "Updated content"
    mock_message = mock.create_autospec(Message)
    db.exec.return_value.one_or_none.return_value = mock_message

    # Act
    result = update_message(db, message_id, content=new_content)

    # Assert
    assert result.msg_content == new_content
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_delete_message(mock_db):
    """Test that delete_message() deletes the message and returns True."""
    # Arrange
    db = mock_db
    message_id = 1
    mock_message = mock.create_autospec(Message)
    db.exec.return_value.one_or_none.return_value = mock_message

    # Act
    result = delete_message(db, message_id)

    # Assert
    assert result is True
    db.delete.assert_called_once()
    db.commit.assert_called_once()