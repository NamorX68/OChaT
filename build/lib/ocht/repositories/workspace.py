# CRUD functions for workspace
from datetime import datetime
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import Workspace


def create_workspace(db: Session, name: str, default_model: str, description: str = None) -> Workspace:
    """
    Creates a new workspace.

    Args:
        db (Session): The database session.
        name (str): The name of the workspace.
        default_model (str): The default model for new chats.
        description (str, optional): An optional description of the workspace. Default is None.

    Returns:
        Workspace: The created workspace object.
    """
    workspace = Workspace(
        work_name=name,
        work_default_model=default_model,
        work_description=description,
        work_created_at=datetime.now(),
        work_updated_at=datetime.now()
    )
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    return workspace


def get_workspace_by_id(db: Session, workspace_id: int) -> Workspace:
    """
    Retrieves a workspace by its ID.

    Args:
        db (Session): The database session.
        workspace_id (int): The ID of the workspace.

    Returns:
        Optional[Workspace]: The workspace object with the specified ID or None if not found.
    """
    statement = select(Workspace).where(Workspace.work_id == workspace_id)
    result = db.exec(statement)
    return result.one_or_none()


def get_all_workspaces(db: Session, limit: Optional[int] = None, offset: Optional[int] = 0) -> Sequence[Workspace]:
    """
    Retrieves all workspaces with optional limitation and offset.

    Args:
        db (Session): The database session.
        limit (Optional[int], optional): The maximum number of workspaces to return. Default is None.
        offset (Optional[int], optional): The offset for the query. Default is 0.

    Returns:
        list[Workspace]: A list of workspace objects.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit kann nicht negativ sein.")
    if offset is not None and offset < 0:
        raise ValueError("Offset kann nicht negativ sein.")

    statement = select(Workspace).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)

    workspaces = db.exec(statement).all()
    return workspaces


def update_workspace(db: Session, workspace_id: int, name: str = None, default_model: str = None,
                     description: str = None) -> Optional[Workspace]:
    """
    Updates an existing workspace.

    Args:
        db (Session): The database session.
        workspace_id (int): The ID of the workspace.
        name (str, optional): The new name for the workspace. Default is None.
        default_model (str, optional): The new default model for the workspace. Default is None.
        description (str, optional): A new optional description for the workspace. Default is None.

    Returns:
        Optional[Workspace]: The updated workspace object or None if the workspace was not found.
    """
    workspace = get_workspace_by_id(db, workspace_id)
    if not workspace:
        return None

    if name is not None:
        workspace.work_name = name
    if default_model is not None:
        workspace.work_default_model = default_model
    if description is not None:
        workspace.work_description = description

    workspace.work_updated_at = datetime.now()

    db.add(workspace)
    db.commit()
    db.refresh(workspace)

    return workspace


def delete_workspace(db: Session, workspace_id: int) -> bool:
    """
    Deletes a workspace.

    Args:
        db (Session): The database session.
        workspace_id (int): The ID of the workspace.

    Returns:
        bool: True if the workspace was successfully deleted, False otherwise.
    """
    workspace = get_workspace_by_id(db, workspace_id)
    if not workspace:
        return False
    db.delete(workspace)
    db.commit()
    return True
# CRUD functions for Workspace
