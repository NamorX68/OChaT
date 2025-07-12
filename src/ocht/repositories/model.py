from datetime import datetime
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import Model


def create_model(db: Session, model_name: str, model_provider_id: int,
                 model_description: Optional[str] = None, model_version: Optional[str] = None,
                 model_params: Optional[str] = None) -> Model:
    """
    Creates a new model.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model.
        model_provider_id (int): Foreign key linking to LLMProviderConfig.prov_id.
        model_description (Optional[str]): Description of the model. Default is None.
        model_version (Optional[str]): Version identifier of the model. Default is None.
        model_params (Optional[str]): JSON string with default parameters. Default is None.

    Returns:
        Model: The newly created model object.
    """
    db_model = Model(
        model_name=model_name,
        model_provider_id=model_provider_id,
        model_description=model_description,
        model_version=model_version,
        model_params=model_params
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return db_model


def get_model_by_name(db: Session, model_name: str) -> Optional[Model]:
    """
    Fetches a model by its name.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model.

    Returns:
        Optional[Model]: The model object or None if not found.
    """
    statement = select(Model).where(Model.model_name == model_name)
    result = db.exec(statement)
    return result.one_or_none()


def get_all_models(db: Session, limit: Optional[int] = None, offset: Optional[int] = 0) -> Sequence[Model]:
    """
    Retrieves all models with optional limitation and offset.

    Args:
        db (Session): The database session.
        limit (Optional[int], optional): The maximum number of models to return. Default is None.
        offset (Optional[int], optional): The offset for the query. Default is 0.

    Returns:
        list[Model]: A list of model objects.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit cannot be negative.")
    if offset is not None and offset < 0:
        raise ValueError("Offset cannot be negative.")

    statement = select(Model)
    if limit is not None:
        statement = statement.limit(limit).offset(offset)

    return db.exec(statement).all()


def update_model(db: Session, model_name: str, new_model_name: Optional[str] = None,
                 model_provider_id: Optional[int] = None, model_description: Optional[str] = None,
                 model_version: Optional[str] = None, model_params: Optional[str] = None) -> Optional[Model]:
    """
    Updates an existing model.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model to be updated.
        new_model_name (Optional[str]): New name for the model. Default is None.
        model_provider_id (Optional[int]): Updated foreign key linking to LLMProviderConfig.prov_id. Default is None.
        model_description (Optional[str]): Updated description of the model. Default is None.
        model_version (Optional[str]): Updated version identifier of the model. Default is None.
        model_params (Optional[str]): Updated JSON string with default parameters. Default is None.

    Returns:
        Optional[Model]: The updated model object or None if not found.
    """
    model = get_model_by_name(db, model_name)
    if not model:
        return None

    if new_model_name is not None:
        model.model_name = new_model_name
    if model_provider_id is not None:
        model.model_provider_id = model_provider_id
    if model_description is not None:
        model.model_description = model_description
    if model_version is not None:
        model.model_version = model_version
    if model_params is not None:
        model.model_params = model_params

    model.model_updated_at = datetime.now()

    db.add(model)
    db.commit()
    db.refresh(model)

    return model


def delete_model(db: Session, model_name: str) -> bool:
    """
    Deletes a model by its name.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model to be deleted.

    Returns:
        bool: True if the deletion was successful, False otherwise.
    """
    model = get_model_by_name(db, model_name)
    if not model:
        return False

    db.delete(model)
    db.commit()

    return True
