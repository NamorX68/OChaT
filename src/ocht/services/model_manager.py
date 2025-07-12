import requests
from typing import List, Optional, Dict, Any
from ocht.core.db import get_session
from ocht.repositories.model import (
    get_all_models, 
    create_model, 
    get_model_by_name, 
    update_model, 
    delete_model
)
from ocht.repositories.llm_provider_config import get_all_llm_provider_configs
from ocht.core.models import Model, LLMProviderConfig


def list_llm_models() -> List[Model]:
    """Reads available models from DB/Cache and returns them."""
    for db in get_session():
        return get_all_models(db)


def sync_llm_models() -> dict:
    """Gets models from external providers and stores them."""
    results = {
        'ollama': {'added': 0, 'skipped': 0, 'errors': []},
        'total_processed': 0
    }

    for db in get_session():
        # Get all providers
        providers = get_all_llm_provider_configs(db)

        for provider in providers:
            if provider.prov_name.lower() == 'ollama':
                ollama_result = _sync_ollama_models(db, provider)
                results['ollama'] = ollama_result
                results['total_processed'] += ollama_result['added'] + ollama_result['skipped']

    return results


def _sync_ollama_models(db, provider) -> dict:
    """Synchronizes Ollama models with the database."""
    result = {'added': 0, 'skipped': 0, 'errors': []}

    try:
        # Get available models from Ollama
        base_url = provider.prov_endpoint or "http://localhost:11434"
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get('models', [])

        for model_info in models:
            model_name = model_info.get('name', '')
            if not model_name:
                continue

            # Check if model already exists
            existing_model = get_model_by_name(db, model_name)
            if existing_model:
                result['skipped'] += 1
                continue

            # Create model description with size info
            model_size = model_info.get('size', 0)
            modified_at = model_info.get('modified_at', '')

            size_gb = round(model_size / (1024**3), 2) if model_size > 0 else 0
            description = f"Ollama model, Size: {size_gb} GB"
            if modified_at:
                description += f", Modified: {modified_at}"

            try:
                # Create the model in database
                create_model(
                    db=db,
                    model_name=model_name,
                    model_provider_id=provider.prov_id,
                    model_description=description,
                    model_version=None,
                    model_params=None
                )
                result['added'] += 1

            except Exception as e:
                result['errors'].append(f"Error adding model '{model_name}': {str(e)}")

    except requests.exceptions.RequestException as e:
        result['errors'].append(f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")

    return result


# === TUI Business Logic Functions ===

def get_models_with_provider_info() -> List[Dict[str, Any]]:
    """
    Gets models with provider information for UI display.

    Returns:
        List[Dict]: List of dictionaries with model and provider information
    """
    for db in get_session():
        models = get_all_models(db)
        providers = get_all_llm_provider_configs(db)

        # Create provider lookup dictionary
        provider_lookup = {provider.prov_id: provider.prov_name for provider in providers}

        return [
            {
                'model': model,
                'provider_name': provider_lookup.get(model.model_provider_id, f"ID: {model.model_provider_id}")
            }
            for model in models
        ]


def create_model_with_validation(name: str, provider_id: int, description: Optional[str] = None,
                                version: Optional[str] = None, params: Optional[str] = None) -> Model:
    """
    Creates model with business logic validation.

    Args:
        name: Model name
        provider_id: Provider ID
        description: Optional description
        version: Optional version
        params: Optional parameters

    Returns:
        Model: The created model

    Raises:
        ValueError: On validation errors
    """
    if not name or not name.strip():
        raise ValueError("Model name is required")

    name = name.strip()

    for db in get_session():
        # Check if model already exists
        existing_model = get_model_by_name(db, name)
        if existing_model:
            raise ValueError(f"Model '{name}' already exists")

        # Validate provider exists
        providers = get_all_llm_provider_configs(db)
        if not any(p.prov_id == provider_id for p in providers):
            raise ValueError(f"Provider with ID {provider_id} does not exist")

        return create_model(
            db=db,
            model_name=name,
            model_provider_id=provider_id,
            model_description=description,
            model_version=version,
            model_params=params
        )


def update_model_with_validation(old_name: str, new_name: Optional[str] = None,
                                provider_id: Optional[int] = None, description: Optional[str] = None,
                                version: Optional[str] = None, params: Optional[str] = None) -> Optional[Model]:
    """
    Updates model with business logic validation.

    Args:
        old_name: Current model name
        new_name: New model name (optional)
        provider_id: New provider ID (optional)
        description: New description (optional)
        version: New version (optional)
        params: New parameters (optional)

    Returns:
        Optional[Model]: The updated model or None if not found

    Raises:
        ValueError: On validation errors
    """
    if not old_name or not old_name.strip():
        raise ValueError("Current model name is required")

    for db in get_session():
        # Check if model exists
        existing_model = get_model_by_name(db, old_name.strip())
        if not existing_model:
            raise ValueError(f"Model '{old_name}' not found")

        # If new name is provided, check it's valid and not already taken
        if new_name is not None:
            new_name = new_name.strip()
            if not new_name:
                raise ValueError("New model name cannot be empty")

            if new_name != old_name:
                existing_with_new_name = get_model_by_name(db, new_name)
                if existing_with_new_name:
                    raise ValueError(f"Model '{new_name}' already exists")

        # Validate provider exists if provided
        if provider_id is not None:
            providers = get_all_llm_provider_configs(db)
            if not any(p.prov_id == provider_id for p in providers):
                raise ValueError(f"Provider with ID {provider_id} does not exist")

        return update_model(
            db=db,
            model_name=old_name.strip(),
            new_model_name=new_name,
            model_provider_id=provider_id,
            model_description=description,
            model_version=version,
            model_params=params
        )


def delete_model_with_checks(model_name: str) -> bool:
    """
    Deletes model after business logic checks.

    Args:
        model_name: Name of the model to delete

    Returns:
        bool: True if successfully deleted, False otherwise

    Raises:
        ValueError: On validation errors
    """
    if not model_name or not model_name.strip():
        raise ValueError("Model name is required")

    model_name = model_name.strip()

    for db in get_session():
        # Check if model exists
        existing_model = get_model_by_name(db, model_name)
        if not existing_model:
            raise ValueError(f"Model '{model_name}' not found")

        # Here could be additional checks (e.g., if model is in use)
        # For now, we just delete it
        return delete_model(db, model_name)
