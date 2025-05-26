# CRUD functions for PromptTemplate
from datetime import datetime
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import PromptTemplate


def create_prompt_template(db: Session, name: str, text: str, description: Optional[str] = None) -> PromptTemplate:
    """
    Creates a new prompt template.

    Args:
        db (Session): The database session.
        name (str): The name of the template.
        text (str): The text content of the template.
        description (Optional[str], optional): A description of the template. Defaults to None.

    Returns:
        PromptTemplate: The newly created template.
    """
    prompt_template = PromptTemplate(
        templ_name=name,
        templ_text=text,
        templ_description=description,
        templ_created_at=datetime.now(),
        templ_updated_at=datetime.now()
    )
    db.add(prompt_template)
    db.commit()
    db.refresh(prompt_template)
    return prompt_template


def get_prompt_template_by_id(db: Session, template_id: int) -> Optional[PromptTemplate]:
    """
    Retrieves a prompt template by its ID.

    Args:
        db (Session): The database session.
        template_id (int): The ID of the template.

    Returns:
        Optional[PromptTemplate]: The template object or None if not found.
    """
    statement = select(PromptTemplate).where(PromptTemplate.templ_id == template_id)
    result = db.exec(statement)
    return result.first()


def get_all_prompt_templates(db: Session, limit: Optional[int] = None, offset: int = 0) -> Sequence[PromptTemplate]:
    """
    Retrieves all prompt templates with optional limitation and offset.

    Args:
        db (Session): The database session.
        limit (Optional[int], optional): The maximum number of templates to return. Defaults to None.
        offset (int, optional): The offset for the query. Defaults to 0.

    Returns:
        Sequence[PromptTemplate]: A list of template objects.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit cannot be negative.")
    if offset < 0:
        raise ValueError("Offset cannot be negative.")

    statement = select(PromptTemplate).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)

    return db.exec(statement).all()


def update_prompt_template(db: Session, template_id: int, name: Optional[str] = None, description: Optional[str] = None,
                           text: Optional[str] = None) -> Optional[PromptTemplate]:
    """
    Updates an existing prompt template.

    Args:
        db (Session): The database session.
        template_id (int): The ID of the template to update.
        name (Optional[str], optional): New name for the template. Defaults to None.
        description (Optional[str], optional): New description. Defaults to None.
        text (Optional[str], optional): New text content. Defaults to None.

    Returns:
        Optional[PromptTemplate]: The updated template or None if not found.
    """
    db_template = get_prompt_template_by_id(db, template_id)
    if not db_template:
        return None

    if name is not None:
        db_template.templ_name = name
    if description is not None:
        db_template.templ_description = description
    if text is not None:
        db_template.templ_text = text
    db_template.templ_updated_at = datetime.now()

    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    return db_template


def delete_prompt_template(db: Session, template_id: int) -> bool:
    """
    Deletes a prompt template.

    Args:
        db (Session): The database session.
        template_id (int): The ID of the template to delete.

    Returns:
        bool: True if successful, False if template not found.
    """
    db_template = get_prompt_template_by_id(db, template_id)
    if not db_template:
        return False

    db.delete(db_template)
    db.commit()
    return True
