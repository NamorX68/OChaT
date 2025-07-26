from typing import List, Optional, Dict, Any, TypeVar, Callable
from ocht.core.db import get_session
from ocht.repositories.workspace import (
    get_all_workspaces,
    create_workspace,
    update_workspace,
    delete_workspace,
    get_workspace_by_id
)
from ocht.core.models import Workspace

T = TypeVar('T')


def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    db = next(get_session())
    return func(db)


def _validate_workspace_name(name: str) -> str:
    """Validates and normalizes workspace name."""
    if not name or not name.strip():
        raise ValueError("Workspace name is required")
    return name.strip()


def _check_workspace_name_uniqueness(db, name: str, exclude_id: Optional[int] = None) -> None:
    """Checks if workspace name is unique."""
    existing_workspaces = get_all_workspaces(db)
    for workspace in existing_workspaces:
        if (workspace.work_name.lower() == name.lower() and
                workspace.work_id != exclude_id):
            raise ValueError(f"Workspace '{name}' already exists")


def _ensure_workspace_exists(db, workspace_id: int) -> Workspace:
    """Ensures workspace exists and returns it."""
    workspace = get_workspace_by_id(db, workspace_id)
    if not workspace:
        raise ValueError(f"Workspace with ID {workspace_id} not found")
    return workspace


def get_available_workspaces() -> List[Workspace]:
    """
    Gets available workspaces for selection.
    Returns:
        List[Workspace]: List of available workspaces
    """
    return _with_session(get_all_workspaces)


def get_workspaces_with_info() -> List[Dict[str, Any]]:
    """
    Gets workspaces with additional information for UI display.
    Returns:
        List[Dict]: List of dictionaries with workspace information
    """

    def _get_workspaces_info(db):
        workspaces = get_all_workspaces(db)
        return [
            {
                'workspace': workspace,
                'message_count': 0,  # Could be extended to show actual message count
                'status': 'active' if workspace.work_default_model else 'inactive'
            }
            for workspace in workspaces
        ]

    return _with_session(_get_workspaces_info)


def create_workspace_with_validation(name: str, default_model: str,
                                     description: Optional[str] = None) -> Workspace:
    """
    Creates workspace with business logic validation.
    Args:
        name: Workspace name
        default_model: Default model for the workspace
        description: Optional description
    Returns:
        Workspace: The created workspace
    Raises:
        ValueError: On validation errors
    """
    validated_name = _validate_workspace_name(name)
    
    if not default_model or not default_model.strip():
        raise ValueError("Default model is required")

    def _create_workspace(db):
        _check_workspace_name_uniqueness(db, validated_name)
        return create_workspace(
            db=db,
            name=validated_name,
            default_model=default_model.strip(),
            description=description
        )

    return _with_session(_create_workspace)


def update_workspace_with_validation(workspace_id: int, name: Optional[str] = None,
                                     default_model: Optional[str] = None,
                                     description: Optional[str] = None) -> Optional[Workspace]:
    """
    Updates workspace with business logic validation.
    Args:
        workspace_id: Workspace ID
        name: New workspace name (optional, None means don't change)
        default_model: New default model (optional)
        description: New description (optional)
    Returns:
        Optional[Workspace]: The updated workspace or None if not found
    Raises:
        ValueError: On validation errors
    """

    def _update_workspace(db):
        existing_workspace = _ensure_workspace_exists(db, workspace_id)

        validated_name = name
        if name:  # Only validate if name is provided (not None)
            validated_name = _validate_workspace_name(name)
            if validated_name.lower() != existing_workspace.work_name.lower():
                _check_workspace_name_uniqueness(db, validated_name, workspace_id)

        return update_workspace(
            db=db,
            workspace_id=workspace_id,
            name=validated_name,
            default_model=default_model,
            description=description
        )

    return _with_session(_update_workspace)


def delete_workspace_with_checks(workspace_id: int) -> bool:
    """
    Deletes workspace after business logic checks.
    Args:
        workspace_id: ID of the workspace to delete
    Returns:
        bool: True if successfully deleted, False otherwise
    Raises:
        ValueError: On validation errors
    """

    def _delete_workspace(db):
        _ensure_workspace_exists(db, workspace_id)
        return delete_workspace(db, workspace_id)

    return _with_session(_delete_workspace)