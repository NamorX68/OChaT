"""CRUD operations for the LLMProviderConfig entity."""
from collections.abc import Sequence
from datetime import datetime

from sqlmodel import Session, select

from ocht.core.models import LLMProviderConfig


def create_llm_provider_config(db: Session, name: str, api_key: str, endpoint: str | None = None,
                               default_model: str | None = None, params: str | None = None) -> LLMProviderConfig:
    """Creates a new LLM provider config.

    Args:
        db (Session): The database session.
        name (str): The name of the LLM provider.
        api_key (str): The API key for the LLM provider.
        endpoint (Optional[str], optional): The endpoint URL for the LLM provider. Default is None.
        default_model (Optional[str], optional): The default model for the LLM provider. Default is None.
        params (Optional[str], optional): JSON string of provider routing preferences (e.g. OpenRouter's
            `provider` object - quantizations, preferred_min_throughput, etc.). Default is None.

    Returns:
        LLMProviderConfig: Das erstellte Konfigurations-Objekt.
    """
    llm_provider_config = LLMProviderConfig(
        prov_name=name,
        prov_api_key=api_key,
        prov_endpoint=endpoint,
        prov_default_model=default_model,
        prov_params=params,
        prov_created_at=datetime.now(),
        prov_updated_at=datetime.now()
    )
    db.add(llm_provider_config)
    db.commit()
    db.refresh(llm_provider_config)
    return llm_provider_config


def get_llm_provider_config_by_id(db: Session, config_id: int) -> LLMProviderConfig | None:
    """Holt eine LLM Provider Konfiguration nach ihrer ID.

    Args:
        db (Session): Die Datenbanksitzung.
        config_id (int): Die ID der Konfiguration.

    Returns:
        Optional[LLMProviderConfig]: Das Konfigurations-Objekt mit der angegebenen ID oder None, wenn nicht gefunden.
    """
    statement = select(LLMProviderConfig).where(LLMProviderConfig.prov_id == config_id)
    result = db.exec(statement)
    return result.one_or_none()


def get_all_llm_provider_configs(
    db: Session, limit: int | None = None, offset: int | None = 0
) -> Sequence[LLMProviderConfig]:
    """Retrieves all LLM provider configurations with optional limitation and offset.

    Args:
        db (Session): The database session.
        limit (Optional[int], optional): The maximum number of configurations to return. Default is None.
        offset (Optional[int], optional): The offset for the query. Default is 0.

    Returns:
        list[LLMProviderConfig]: A list of LLM provider configuration objects.

    Raises:
        ValueError: If limit or offset are negative.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit kann nicht negativ sein.")
    if offset is not None and offset < 0:
        raise ValueError("Offset kann nicht negativ sein.")

    statement = select(LLMProviderConfig)
    if limit is not None:
        statement = statement.limit(limit).offset(offset)

    return db.exec(statement).all()


def update_llm_provider_config(
    db: Session,
    config_id: int,
    name: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    default_model: str | None = None,
    params: str | None = None,
) -> LLMProviderConfig | None:
    """Updates an existing LLM provider configuration.

    Args:
        db (Session): The database session.
        config_id (int): The ID of the configuration.
        name (Optional[str], optional): Der neue Name für die Konfiguration. Standard ist None.
        api_key (Optional[str], optional): Der neue API-Schlüssel für die Konfiguration. Standard ist None.
        endpoint (Optional[str], optional): Der neue Endpoint für die Konfiguration. Standard ist None.
        default_model (Optional[str], optional): Das neue Standard-Modell für die Konfiguration. Standard ist None.
        params (Optional[str], optional): New JSON string of provider routing preferences. `None` leaves the
            existing value unchanged; pass `""` to explicitly clear it (a JSON string can never legitimately
            be empty, so this is unambiguous).

    Returns:
        Optional[LLMProviderConfig]: Das aktualisierte Konfigurations-Objekt oder None, wenn nicht gefunden.
    """
    config = get_llm_provider_config_by_id(db, config_id)
    if not config:
        return None

    if name is not None:
        config.prov_name = name
    if api_key is not None:
        config.prov_api_key = api_key
    if endpoint is not None:
        config.prov_endpoint = endpoint
    if default_model is not None:
        config.prov_default_model = default_model
    if params is not None:
        config.prov_params = params or None
    config.prov_updated_at = datetime.now()

    db.add(config)
    db.commit()
    db.refresh(config)

    return config


def delete_llm_provider_config(db: Session, config_id: int) -> bool:
    """Deletes an LLM provider configuration.

    Args:
        db (Session): The database session.
        config_id (int): The ID of the configuration.

    Returns:
        bool: True if the configuration was deleted, False otherwise.
    """
    config = get_llm_provider_config_by_id(db, config_id)
    if not config:
        return False
    db.delete(config)
    db.commit()
    return True
