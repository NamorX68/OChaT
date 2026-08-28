"""CRUD operations for the Model entity."""
from collections.abc import Sequence
from datetime import datetime

from sqlmodel import Session, select

from ocht.core.models import Model


def create_model(db: Session, model_name: str, model_provider_id: int,
                 model_description: str | None = None, model_version: str | None = None,
                 model_params: str | None = None, is_available: bool = True,
                 last_checked: datetime | None = None) -> Model:
    """Creates a new model.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model.
        model_provider_id (int): Foreign key linking to LLMProviderConfig.prov_id.
        model_description (Optional[str]): Description of the model. Default is None.
        model_version (Optional[str]): Version identifier of the model. Default is None.
        model_params (Optional[str]): JSON string with default parameters. Default is None.
        is_available (bool): Whether the model is available on disk. Default is True.
        last_checked (Optional[datetime]): Last time availability was checked. Default is None.

    Returns:
        Model: The newly created model object.
    """
    db_model = Model(
        model_name=model_name,
        model_provider_id=model_provider_id,
        model_description=model_description,
        model_version=model_version,
        model_params=model_params,
        is_available=is_available,
        last_checked=last_checked
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return db_model


def get_model_by_name(db: Session, model_name: str) -> Model | None:
    """Fetches a model by its name.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model.

    Returns:
        Optional[Model]: The model object or None if not found.
    """
    statement = select(Model).where(Model.model_name == model_name)
    result = db.exec(statement)
    return result.one_or_none()


def get_all_models(db: Session, limit: int | None = None, offset: int | None = 0) -> Sequence[Model]:
    """Retrieves all models with optional limitation and offset.

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


def update_model(db: Session, model_name: str, new_model_name: str | None = None,
                 model_provider_id: int | None = None, model_description: str | None = None,
                 model_version: str | None = None, model_params: str | None = None,
                 is_available: bool | None = None, last_checked: datetime | None = None) -> Model | None:
    """Updates an existing model.

    Args:
        db (Session): The database session.
        model_name (str): The name of the model to be updated.
        new_model_name (Optional[str]): New name for the model. Default is None.
        model_provider_id (Optional[int]): Updated foreign key linking to LLMProviderConfig.prov_id. Default is None.
        model_description (Optional[str]): Updated description of the model. Default is None.
        model_version (Optional[str]): Updated version identifier of the model. Default is None.
        model_params (Optional[str]): Updated JSON string with default parameters. `None` leaves the
            existing value unchanged; pass `""` to explicitly clear it (a JSON string can never
            legitimately be empty, so this is unambiguous). Default is None.
        is_available (Optional[bool]): Updated availability status. Default is None.
        last_checked (Optional[datetime]): Updated last checked timestamp. Default is None.

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
        # "" explicitly clears model_params (stored as NULL) rather than being written back
        # verbatim as a literal empty string - same None-vs-"" convention
        # `update_llm_provider_config()` uses for `prov_params`.
        model.model_params = model_params or None
    if is_available is not None:
        model.is_available = is_available
    if last_checked is not None:
        model.last_checked = last_checked

    model.model_updated_at = datetime.now()

    db.add(model)
    db.commit()
    db.refresh(model)

    return model


def record_health_check_result(
    db: Session,
    model_name: str,
    *,
    is_available: bool,
    checked_at: datetime,
    latency_ms: float | None,
    tokens_per_second: float | None,
    error: str | None,
) -> Model | None:
    """Unconditionally overwrites a model's health-check fields with one fresh result.

    Deliberately separate from `update_model()`, whose `None`-means-leave-unchanged convention
    would make it impossible to ever clear a previously-recorded `last_check_error` back to None
    (a successful check after a failing one is a legitimate, common transition that must actually
    write `None`, not skip the field).

    Args:
        db (Session): The database session.
        model_name (str): The name of the model that was checked.
        is_available (bool): Whether the health check's completion call succeeded.
        checked_at (datetime): Timestamp the check was performed at.
        latency_ms (Optional[float]): Latency of the check's completion call, in milliseconds, or
            None if it failed before a latency could be measured.
        tokens_per_second (Optional[float]): Output tokens/second measured by the check, or None
            if unavailable (e.g. the check failed, or no token/timing metadata was returned).
        error (Optional[str]): Error message if the check failed, or None if it succeeded.

    Returns:
        Optional[Model]: The updated model object, or None if no model with that name exists.
    """
    model = get_model_by_name(db, model_name)
    if not model:
        return None

    model.is_available = is_available
    model.last_checked = checked_at
    model.last_check_latency_ms = latency_ms
    model.last_check_tokens_per_second = tokens_per_second
    model.last_check_error = error
    model.model_updated_at = datetime.now()

    db.add(model)
    db.commit()
    db.refresh(model)

    return model


def delete_model(db: Session, model_name: str) -> bool:
    """Deletes a model by its name.

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


def get_models_by_provider(db: Session, provider_id: int) -> Sequence[Model]:
    """Retrieves all models for a specific provider.

    Args:
        db (Session): The database session.
        provider_id (int): The provider ID to filter by.

    Returns:
        Sequence[Model]: A list of model objects for the provider.
    """
    statement = select(Model).where(Model.model_provider_id == provider_id)
    return db.exec(statement).all()
