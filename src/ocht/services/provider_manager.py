from typing import List, Optional, Dict, Any
from ocht.core.db import get_session
from ocht.repositories.llm_provider_config import (
    get_all_llm_provider_configs,
    create_llm_provider_config,
    update_llm_provider_config,
    delete_llm_provider_config,
    get_llm_provider_config_by_id
)
from ocht.core.models import LLMProviderConfig


def get_available_providers() -> List[LLMProviderConfig]:
    """
    Gets available providers for model assignment.

    Returns:
        List[LLMProviderConfig]: List of available providers
    """
    for db in get_session():
        return get_all_llm_provider_configs(db)


def get_providers_with_info() -> List[Dict[str, Any]]:
    """
    Gets providers with additional information for UI display.

    Returns:
        List[Dict]: List of dictionaries with provider information
    """
    for db in get_session():
        providers = get_all_llm_provider_configs(db)
        return [
            {
                'provider': provider,
                'model_count': 0,  # Could be extended to show actual model count
                'status': 'active' if provider.prov_api_key else 'inactive'
            }
            for provider in providers
        ]


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
    if not name or not name.strip():
        raise ValueError("Provider name is required")

    name = name.strip()

    for db in get_session():
        # Check if provider already exists
        existing_providers = get_all_llm_provider_configs(db)
        if any(p.prov_name.lower() == name.lower() for p in existing_providers):
            raise ValueError(f"Provider '{name}' already exists")

        return create_llm_provider_config(
            db=db,
            prov_name=name,
            prov_api_key=api_key,
            prov_endpoint=endpoint,
            prov_default_model=default_model
        )


def update_provider_with_validation(provider_id: int, name: Optional[str] = None,
                                   api_key: Optional[str] = None, endpoint: Optional[str] = None,
                                   default_model: Optional[str] = None) -> Optional[LLMProviderConfig]:
    """
    Updates provider with business logic validation.

    Args:
        provider_id: Provider ID
        name: New provider name (optional)
        api_key: New API key (optional)
        endpoint: New endpoint URL (optional)
        default_model: New default model (optional)

    Returns:
        Optional[LLMProviderConfig]: The updated provider or None if not found

    Raises:
        ValueError: On validation errors
    """
    for db in get_session():
        # Check if provider exists
        existing_provider = get_llm_provider_config_by_id(db, provider_id)
        if not existing_provider:
            raise ValueError(f"Provider with ID {provider_id} not found")

        # If new name is provided, check it's valid and not already taken
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Provider name cannot be empty")

            if name.lower() != existing_provider.prov_name.lower():
                existing_providers = get_all_llm_provider_configs(db)
                if any(p.prov_name.lower() == name.lower() for p in existing_providers):
                    raise ValueError(f"Provider '{name}' already exists")

        return update_llm_provider_config(
            db=db,
            provider_id=provider_id,
            prov_name=name,
            prov_api_key=api_key,
            prov_endpoint=endpoint,
            prov_default_model=default_model
        )


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
    for db in get_session():
        # Check if provider exists
        existing_provider = get_llm_provider_config_by_id(db, provider_id)
        if not existing_provider:
            raise ValueError(f"Provider with ID {provider_id} not found")

        # Here could be additional checks (e.g., if provider is used by models)
        # For now, we just delete it
        return delete_llm_provider_config(db, provider_id)
