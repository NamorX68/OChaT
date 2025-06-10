import pytest
from unittest.mock import MagicMock, patch
from sqlmodel import Session

from ocht.core.models import Workspace
from ocht.repositories.workspace import (
    create_workspace,
    get_workspace_by_id,
    get_all_workspaces,
    update_workspace,
    delete_workspace
)


def test_create_workspace_success():
    db = MagicMock(spec=Session)
    workspace = create_workspace(db, "Test Workspace", "default_model", "Test description")
    assert workspace.work_name == "Test Workspace"
    assert workspace.work_default_model == "default_model"
    assert workspace.work_description == "Test description"
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_get_workspace_by_id_found():
    db = MagicMock(spec=Session)
    workspace = Workspace(work_id=1, work_name="Test")
    db.exec.return_value.one_or_none.return_value = workspace

    result = get_workspace_by_id(db, 1)
    assert result == workspace
    db.exec.assert_called_once()


def test_get_workspace_by_id_not_found():
    db = MagicMock(spec=Session)
    db.exec.return_value.one_or_none.return_value = None

    result = get_workspace_by_id(db, 999)
    assert result is None


def test_get_all_workspaces_with_limit_offset():
    db = MagicMock(spec=Session)
    workspaces = [Workspace(work_id=i) for i in range(5)]
    db.exec.return_value.all.return_value = workspaces

    result = get_all_workspaces(db, limit=2, offset=1)
    assert len(result) == 2
    db.exec.assert_called_once()


def test_get_all_workspaces_validation():
    with pytest.raises(ValueError):
        get_all_workspaces(MagicMock(), limit=-1)


def test_update_workspace_success():
    db = MagicMock(spec=Session)
    workspace = Workspace(work_id=1, work_name="Old")
    db.exec.return_value.one_or_none.return_value = workspace

    updated = update_workspace(db, 1, name="New")
    assert updated.work_name == "New"
    assert updated.work_updated_at is not None
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_update_workspace_not_found():
    db = MagicMock(spec=Session)
    db.exec.return_value.one_or_none.return_value = None

    result = update_workspace(db, 999, name="New")
    assert result is None


def test_delete_workspace_success():
    db = MagicMock(spec=Session)
    workspace = Workspace(work_id=1)
    db.exec.return_value.one_or_none.return_value = workspace

    result = delete_workspace(db, 1)
    assert result is True
    db.delete.assert_called_once()
    db.commit.assert_called_once()


def test_delete_workspace_not_found():
    db = MagicMock(spec=Session)
    db.exec.return_value.one_or_none.return_value = None

    result = delete_workspace(db, 999)
    assert result is False