# message.py
from datetime import datetime
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import Message


def create_message(db: Session, content: str, workspace_id: int) -> Message:
    """
    Erstellt eine neue Nachricht.

    Args:
        db (Session): Die Datenbanksitzung.
        content (str): Der Inhalt der Nachricht.
        workspace_id (int): Die ID des Arbeitsbereichs, zu dem die Nachricht gehört.

    Returns:
        Message: Das erstellte Nachrichten-Objekt.
    """
    message = Message(
        msg_content=content,
        msg_workspace_id=workspace_id,
        msg_created_at=datetime.now(),
        msg_updated_at=datetime.now()
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_message_by_id(db: Session, message_id: int) -> Message:
    """
    Holt eine Nachricht nach ihrer ID.

    Args:
        db (Session): Die Datenbanksitzung.
        message_id (int): Die ID der Nachricht.

    Returns:
        Message: Das Nachrichten-Objekt mit der angegebenen ID.
    """
    statement = select(Message).where(Message.msg_id == message_id)
    result = db.exec(statement)
    return result.one_or_none()


def get_messages_by_workspace(db: Session, workspace_id: int, limit: Optional[int] = None, offset: Optional[int] = 0) -> Sequence[Message]:
    """
    Holt alle Nachrichten eines bestimmten Arbeitsbereichs mit optionaler Begrenzung und Verschiebung.

    Args:
        db (Session): Die Datenbanksitzung.
        workspace_id (int): ID des Arbeitsbereichs, für den die Nachrichten abgerufen werden sollen.
        limit (Optional[int], optional): Die maximale Anzahl von Nachrichten, die zurückgegeben werden sollen. Default ist None.
        offset (Optional[int], optional): Der Offset für die Abfrage. Default ist 0.

    Returns:
        list[Message]: Eine Liste der Nachrichten-Objekte für den angegebenen Arbeitsbereich.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit kann nicht negativ sein.")
    if offset is not None and offset < 0:
        raise ValueError("Offset kann nicht negativ sein.")

    statement = (
        select(Message)
        .where(Message.msg_workspace_id == workspace_id)
        .order_by(Message.msg_created_at)
        .offset(offset)
    )
    if limit is not None:
        statement = statement.limit(limit)

    messages = db.exec(statement).all()
    return messages


def update_message(db: Session, message_id: int, content: str = None) -> Optional[Message]:
    """
    Aktualisiert eine bestehende Nachricht.

    Args:
        db (Session): Die Datenbanksitzung.
        message_id (int): Die ID der Nachricht.
        content (str, optional): Der neue Inhalt der Nachricht. Default ist None.

    Returns:
        Optional[Message]: Das aktualisierte Nachrichten-Objekt oder None, wenn die Nachricht nicht gefunden wurde.
    """
    message = get_message_by_id(db, message_id)
    if not message:
        return None

    if content is not None:
        message.msg_content = content
    message.msg_updated_at = datetime.now()

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def delete_message(db: Session, message_id: int) -> bool:
    """
    Löscht eine Nachricht.

    Args:
        db (Session): Die Datenbanksitzung.
        message_id (int): Die ID der Nachricht.

    Returns:
        bool: True, wenn die Nachricht gelöscht wurde, andernfalls False.
    """
    message = get_message_by_id(db, message_id)
    if not message:
        return False
    db.delete(message)
    db.commit()
    return True
