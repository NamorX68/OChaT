# CRUD functions for Message
from datetime import datetime
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import Workspace


def create_workspace(db: Session, name: str, default_model: str, description: str = None) -> Workspace:
    """
    Erstellt einen neuen Arbeitsbereich.

    Args:
        db (Session): Die Datenbanksitzung.
        name (str): Der Name des Arbeitsbereichs.
        default_model (str): Das Standardmodell für neue Chats.
        description (str, optional): Eine optionale Beschreibung des Arbeitsbereichs. Default ist None.

    Returns:
        Workspace: Das erstellte Arbeitsbereich-Objekt.
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
    Holt einen Arbeitsbereich nach seiner ID.

    Args:
        db (Session): Die Datenbanksitzung.
        workspace_id (int): Die ID des Arbeitsbereichs.

    Returns:
        Workspace: Das Arbeitsbereich-Objekt mit der angegebenen ID.
    """
    statement = select(Workspace).where(Workspace.work_id == workspace_id)
    result = db.exec(statement)
    return result.one_or_none()


def get_all_workspaces(db: Session, limit: Optional[int] = None, offset: Optional[int] = 0) -> Sequence[Workspace]:
    """
    Holt alle Arbeitsbereiche mit optionaler Begrenzung und Verschiebung.

    Args:
        db (Session): Die Datenbanksitzung.
        limit (Optional[int], optional): Die maximale Anzahl von Arbeitsbereichen, die zurückgegeben werden sollen. Default ist None.
        offset (Optional[int], optional): Der Offset für die Abfrage. Default ist 0.

    Returns:
        list[Workspace]: Eine Liste der Arbeitsbereich-Objekte.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit kann nicht negativ sein.")
    if offset is not None and offset < 0:
        raise ValueError("Offset kann nicht negativ sein.")

    statement = select(Workspace)
    if limit is not None:
        statement = statement.limit(limit).offset(offset)

    workspaces = db.exec(statement).all()
    return workspaces


def update_workspace(db: Session, workspace_id: int, name: str = None, default_model: str = None,
                     description: str = None) -> Optional[Workspace]:
    """
    Aktualisiert einen bestehenden Arbeitsbereich.

    Args:
        db (Session): Die Datenbanksitzung.
        workspace_id (int): Die ID des Arbeitsbereichs.
        name (str, optional): Der neue Name des Arbeitsbereichs. Default ist None.
        default_model (str, optional): Das neue Standardmodell für den Arbeitsbereich. Default ist None.
        description (str, optional): Eine optionale neue Beschreibung des Arbeitsbereichs. Default ist None.

    Returns:
        Optional[Workspace]: Das aktualisierte Arbeitsbereich-Objekt oder None, wenn der Arbeitsbereich nicht gefunden wurde.
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
    Löscht einen Arbeitsbereich.

    Args:
        db (Session): Die Datenbanksitzung.
        workspace_id (int): Die ID des Arbeitsbereichs.
    """
    workspace = get_workspace_by_id(db, workspace_id)
    if not workspace:
        return False
    db.delete(workspace)
    db.commit()
    return True
# CRUD functions for Workspace
