# message.py
from datetime import datetime
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import Message


def create_message(db: Session, content: str, workspace_id: int) -> Message:
    """
    Creates a new message.

    Args:
        db (Session): The database session.
        content (str): The content of the message.
        workspace_id (int): The ID of the workspace to which the message belongs.

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
    Retrieves all messages for a specific workspace with optional limitation and offset.
    
    Args:
        db (Session): The database session.
        workspace_id (int): ID of the workspace to retrieve messages for.
        limit (Optional[int], optional): The maximum number of messages to return. Default is None.
        offset (Optional[int], optional): The offset for the query. Default is 0.
    
    Returns:
        list[Message]: A list of message objects for the specified workspace.
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
    Updates an existing message.

    Args:
        db (Session): The database session.
        message_id (int): The ID of the message.
        content (str, optional): The new content for the message. Default is None.

    Returns:
        Optional[Message]: The updated message object or None if the message was not found.
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
    Deletes a message.

    Args:
        db (Session): The database session.
        message_id (int): The ID of the message.

    Returns:
        bool: True if the message was deleted, False otherwise.
    """
    message = get_message_by_id(db, message_id)
    if not message:
        return False
    db.delete(message)
    db.commit()
    return True
