from typing import List, Optional, Dict, Any, TypeVar, Callable
from ocht.core.db import get_session
from ocht.repositories.llm_provider_config import (
    get_all_llm_provider_configs,
    create_llm_provider_config,
    update_llm_provider_config,
    delete_llm_provider_config,
    get_llm_provider_config_by_id
)
from ocht.core.models import LLMProviderConfig

T = TypeVar('T')


def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    db = next(get_session())
    return func(db)


def _validate_provider_name(name: str) -> str:
    """Validates and normalizes provider name."""
    if not name or not name.strip():
        raise ValueError("Provider name is required")
    return name.strip()


def _check_provider_name_uniqueness(db, name: str, exclude_id: Optional[int] = None) -> None:
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


def get_available_providers() -> List[LLMProviderConfig]:
    """
    Gets available providers for model assignment.
    Returns:
        List[LLMProviderConfig]: List of available providers
    """
    return _with_session(get_all_llm_provider_configs)


def get_providers_with_info() -> List[Dict[str, Any]]:
    """
    Gets providers with additional information for UI display.
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


def create_provider_with_validation(name: str, api_key: Optional[str] = None,
                                    endpoint: Optional[str] = None,
                                    default_model: Optional[str] = None) -> LLMProviderConfig:
    """
    Creates provider with business logic validation.
    Args:
        name: Provider name
        api_key: Optional API key
        endpoint: Optional endpoint URL
        default_model: Optional default model name
    Returns:
        LLMProviderConfig: The created provider
    Raises:
        ValueError: On validation errors
    """
    validated_name = _validate_provider_name(name)

    def _create_provider(db):
        _check_provider_name_uniqueness(db, validated_name)
        return create_llm_provider_config(
            db=db,
            prov_name=validated_name,
            prov_api_key=api_key,
            prov_endpoint=endpoint,
            prov_default_model=default_model
        )

    return _with_session(_create_provider)


def update_provider_with_validation(provider_id: int, name: Optional[str] = None,
                                    api_key: Optional[str] = None, endpoint: Optional[str] = None,
                                    default_model: Optional[str] = None) -> Optional[LLMProviderConfig]:
    """
    Updates provider with business logic validation.
    Args:
        provider_id: Provider ID
        name: New provider name (optional, None means don't change)
        api_key: New API key (optional)
        endpoint: New endpoint URL (optional)
        default_model: New default model (optional)
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

        return update_llm_provider_config(
            db=db,
            provider_id=provider_id,
            prov_name=validated_name,
            prov_api_key=api_key,
            prov_endpoint=endpoint,
            prov_default_model=default_model
        )

    return _with_session(_update_provider)


def delete_provider_with_checks(provider_id: int) -> bool:
    """
    Deletes provider after business logic checks.
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
