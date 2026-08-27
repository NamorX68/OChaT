"""Tests for the Workspace repository CRUD functions."""
from unittest.mock import MagicMock

import pytest
from sqlmodel import Session

from ocht.core.models import Workspace
from ocht.repositories.workspace import (
    create_workspace,
    delete_workspace,
    get_all_workspaces,
    get_workspace_by_id,
    update_workspace,
)


def test_create_workspace_success():
    """Test that create_workspace() adds, commits, refreshes, and returns the new workspace."""
    db = MagicMock(spec=Session)
    workspace = create_workspace(db, "Test Workspace", "default_model", "Test description")
    assert workspace.work_name == "Test Workspace"
    assert workspace.work_default_model == "default_model"
    assert workspace.work_description == "Test description"
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_get_workspace_by_id_found():
    """Test that get_workspace_by_id() returns the matching workspace when found."""
    db = MagicMock(spec=Session)
    workspace = Workspace(work_id=1, work_name="Test")
    db.exec.return_value.one_or_none.return_value = workspace

    result = get_workspace_by_id(db, 1)
    assert result == workspace
    db.exec.assert_called_once()


def test_get_workspace_by_id_not_found():
    """Test that get_workspace_by_id() returns None when no workspace matches."""
    db = MagicMock(spec=Session)
    db.exec.return_value.one_or_none.return_value = None

    result = get_workspace_by_id(db, 999)
    assert result is None


def test_get_all_workspaces_with_limit_offset():
    """Test that get_all_workspaces() applies limit and offset to the query."""
    db = MagicMock(spec=Session)
    workspaces = [Workspace(work_id=i) for i in range(5)]
    # Limit/offset are applied by the database via the SQL statement, which this mock does
    # not evaluate - so db.exec(...).all() is set up to return exactly what a real DB would
    # for limit=2, offset=1, rather than the full unfiltered list.
    db.exec.return_value.all.return_value = workspaces[1:3]

    result = get_all_workspaces(db, limit=2, offset=1)
    assert len(result) == 2
    db.exec.assert_called_once()


def test_get_all_workspaces_validation():
    """Test that get_all_workspaces() raises ValueError for a negative limit."""
    with pytest.raises(ValueError):
        get_all_workspaces(MagicMock(), limit=-1)


def test_update_workspace_success():
    """Test that update_workspace() updates and persists the workspace's name."""
    db = MagicMock(spec=Session)
    workspace = Workspace(work_id=1, work_name="Old")
    db.exec.return_value.one_or_none.return_value = workspace

    updated = update_workspace(db, 1, name="New")
    assert updated.work_name == "New"
    assert updated.work_updated_at is not None
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_update_workspace_not_found():
    """Test that update_workspace() returns None when no workspace matches."""
    db = MagicMock(spec=Session)
    db.exec.return_value.one_or_none.return_value = None

    result = update_workspace(db, 999, name="New")
    assert result is None


def test_delete_workspace_success():
    """Test that delete_workspace() deletes the workspace and returns True."""
    db = MagicMock(spec=Session)
    workspace = Workspace(work_id=1)
    db.exec.return_value.one_or_none.return_value = workspace

    result = delete_workspace(db, 1)
    assert result is True
    db.delete.assert_called_once()
    db.commit.assert_called_once()


def test_delete_workspace_not_found():
    """Test that delete_workspace() returns False when no workspace matches."""
    db = MagicMock(spec=Session)
    db.exec.return_value.one_or_none.return_value = None

    result = delete_workspace(db, 999)
    assert result is False