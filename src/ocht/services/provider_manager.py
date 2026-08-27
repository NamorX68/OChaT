"""Service layer for managing LLM provider configurations with business-logic validation."""
import json
from collections.abc import Callable
from typing import Any, TypeVar

from ocht.core.db import get_session
from ocht.core.models import LLMProviderConfig
from ocht.repositories.llm_provider_config import (
    create_llm_provider_config,
    delete_llm_provider_config,
    get_all_llm_provider_configs,
    get_llm_provider_config_by_id,
    update_llm_provider_config,
)

T = TypeVar('T')


def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    with get_session() as db:
        return func(db)


def _validate_provider_name(name: str) -> str:
    """Validates and normalizes provider name."""
    if not name or not name.strip():
        raise ValueError("Provider name is required")
    return name.strip()


def _validate_provider_params(params: str | None) -> str | None:
    """Validates provider routing params as well-formed JSON.

    Args:
        params: Raw JSON string from the UI, or None/empty if not set.

    Returns:
        The stripped JSON string, or None if empty/not provided.

    Raises:
        ValueError: If `params` is non-empty but not valid JSON.
    """
    if params is None or not params.strip():
        return None
    stripped = params.strip()
    try:
        json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"Provider params must be valid JSON: {e}") from e
    return stripped


def _check_provider_name_uniqueness(db, name: str, exclude_id: int | None = None) -> None:
    """Checks if provider name is unique."""
    existing_providers = get_all_llm_provider_configs(db)
    for provider in existing_providers:
        if (provider.prov_name.lower() == name.lower() and
                provider.prov_id != exclude_id):
            raise ValueError(f"Provider '{name}' already exists")


def _ensure_provider_exists(db, provider_id: int) -> LLMProviderConfig:
    """Ensures provider exists and returns it."""
    provider = get_llm_provider_config_by_id(db, provider_id)
    if not provider:
        raise ValueError(f"Provider with ID {provider_id} not found")
    return provider


def get_available_providers() -> list[LLMProviderConfig]:
    """Gets available providers for model assignment.

    Returns:
        List[LLMProviderConfig]: List of available providers
    """
    return _with_session(get_all_llm_provider_configs)


def get_providers_with_info() -> list[dict[str, Any]]:
    """Gets providers with additional information for UI display.

    Returns:
        List[Dict]: List of dictionaries with provider information
    """

    def _get_providers_info(db):
        providers = get_all_llm_provider_configs(db)
        return [
            {
                'provider': provider,
                'model_count': 0,  # Could be extended to show actual model count
                'status': 'active' if provider.prov_api_key else 'inactive'
            }
            for provider in providers
        ]

    return _with_session(_get_providers_info)


def create_provider_with_validation(name: str, api_key: str | None = None,
                                    endpoint: str | None = None,
                                    default_model: str | None = None,
                                    params: str | None = None) -> LLMProviderConfig:
    """Creates provider with business logic validation.

    Args:
        name: Provider name
        api_key: Optional API key
        endpoint: Optional endpoint URL
        default_model: Optional default model name
        params: Optional JSON string of provider routing preferences (e.g. OpenRouter's `provider`
            object - quantizations, preferred_min_throughput, etc.)

    Returns:
        LLMProviderConfig: The created provider
    Raises:
        ValueError: On validation errors
    """
    validated_name = _validate_provider_name(name)
    validated_params = _validate_provider_params(params)

    def _create_provider(db):
        _check_provider_name_uniqueness(db, validated_name)
        return create_llm_provider_config(
            db=db,
            name=validated_name,
            api_key=api_key,
            endpoint=endpoint,
            default_model=default_model,
            params=validated_params
        )

    return _with_session(_create_provider)


def update_provider_with_validation(provider_id: int, name: str | None = None,
                                    api_key: str | None = None, endpoint: str | None = None,
                                    default_model: str | None = None,
                                    params: str | None = None) -> LLMProviderConfig | None:
    """Updates provider with business logic validation.

    Args:
        provider_id: Provider ID
        name: New provider name (optional, None means don't change)
        api_key: New API key (optional)
        endpoint: New endpoint URL (optional)
        default_model: New default model (optional)
        params: New JSON string of provider routing preferences (optional). `None` leaves the
            existing value unchanged; pass `""` to explicitly clear it.

    Returns:
        Optional[LLMProviderConfig]: The updated provider or None if not found
    Raises:
        ValueError: On validation errors
    """

    def _update_provider(db):
        existing_provider = _ensure_provider_exists(db, provider_id)

        validated_name = name
        if name:  # Only validate if name is provided (not None)
            validated_name = _validate_provider_name(name)
            if validated_name.lower() != existing_provider.prov_name.lower():
                _check_provider_name_uniqueness(db, validated_name, provider_id)

        # Unlike name/endpoint/default_model, an empty string here is meaningful (explicit clear -
        # see update_llm_provider_config's docstring), so None and "" must stay distinguishable.
        if params is None:
            validated_params = None
        elif not params.strip():
            validated_params = ""
        else:
            validated_params = _validate_provider_params(params)

        return update_llm_provider_config(
            db=db,
            config_id=provider_id,
            name=validated_name,
            api_key=api_key,
            endpoint=endpoint,
            default_model=default_model,
            params=validated_params
        )

    return _with_session(_update_provider)


def delete_provider_with_checks(provider_id: int) -> bool:
    """Deletes provider after business logic checks.

    Args:
        provider_id: ID of the provider to delete
    Returns:
        bool: True if successfully deleted, False otherwise
    Raises:
        ValueError: On validation errors
    """

    def _delete_provider(db):
        _ensure_provider_exists(db, provider_id)
        return delete_llm_provider_config(db, provider_id)

    return _with_session(_delete_provider)