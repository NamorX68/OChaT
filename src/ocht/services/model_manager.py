"""Service layer for listing, validating, and syncing LLM models from external providers."""
import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

import requests
from sqlmodel import select

from ocht.core.db import get_session
from ocht.core.models import LLMProviderConfig, Model
from ocht.repositories.llm_provider_config import get_all_llm_provider_configs, get_llm_provider_config_by_id
from ocht.repositories.model import (
    create_model,
    delete_model,
    get_all_models,
    get_model_by_name,
    get_models_by_provider,
    update_model,
)

T = TypeVar('T')


# ============================================================================
# GENERAL HELPER FUNCTIONS
# ============================================================================

def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    with get_session() as db:
        return func(db)


def _validate_model_name(name: str) -> str:
    """Validates and normalizes model name."""
    if not name or not name.strip():
        raise ValueError("Model name is required")
    return name.strip()


def _check_model_name_uniqueness(db, name: str, exclude_name: str | None = None) -> None:
    """Checks if model name is unique."""
    existing_model = get_model_by_name(db, name)
    if existing_model and name != exclude_name:
        raise ValueError(f"Model '{name}' already exists")


def _ensure_model_exists(db, model_name: str) -> Model:
    """Ensures model exists and returns it."""
    model = get_model_by_name(db, model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    return model


def _ensure_provider_exists(db, provider_id: int) -> LLMProviderConfig:
    """Ensures provider exists and returns it."""
    provider = get_llm_provider_config_by_id(db, provider_id)
    if not provider:
        raise ValueError(f"Provider with ID {provider_id} does not exist")
    return provider


# ============================================================================
# GENERAL PUBLIC API FUNCTIONS
# ============================================================================

def list_llm_models() -> list[Model]:
    """Reads available models from DB/Cache and returns them."""
    models = _with_session(get_all_models)
    
    # Print models for CLI usage
    print("📋 Available Models:")
    print("=" * 50)
    
    if not models:
        print("No models found. Run 'sync-models' to fetch from providers.")
        return models
    
    # Group models by provider
    models_with_provider = get_models_with_provider_info()
    providers = {}
    for model_info in models_with_provider:
        provider_name = model_info['provider_name']
        if provider_name not in providers:
            providers[provider_name] = []
        providers[provider_name].append(model_info['model'])
    
    # Print grouped by provider
    for provider_name, provider_models in providers.items():
        print(f"\n🔧 {provider_name} ({len(provider_models)} models):")
        for model in provider_models:
            status = "✅" if model.is_available else "❌"
            last_check = model.last_checked.strftime("%Y-%m-%d %H:%M") if model.last_checked else "Never"
            print(f"  {status} {model.model_name} (checked: {last_check})")
    
    print(f"\n📊 Total: {len(models)} models")
    return models


def get_models_with_provider_info() -> list[dict[str, Any]]:
    """Gets models with provider information for UI display.

    Returns:
        List[Dict]: List of dictionaries with model and provider information
    """

    def _get_models_info(db):
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

    return _with_session(_get_models_info)


def get_unavailable_models() -> list[Model]:
    """Gets all models that are marked as unavailable.
    
    Returns:
        List[Model]: List of unavailable models
    """
    def _get_unavailable(db):
        statement = select(Model).where(Model.is_available.is_(False))
        return db.exec(statement).all()
    
    return _with_session(_get_unavailable)


def create_model_with_validation(name: str, provider_id: int, description: str | None = None,
                                 version: str | None = None, params: str | None = None) -> Model:
    """Creates model with business logic validation.

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
    validated_name = _validate_model_name(name)

    def _create_model(db):
        _check_model_name_uniqueness(db, validated_name)
        _ensure_provider_exists(db, provider_id)

        return create_model(
            db=db,
            model_name=validated_name,
            model_provider_id=provider_id,
            model_description=description,
            model_version=version,
            model_params=params
        )

    return _with_session(_create_model)


def update_model_with_validation(old_name: str, new_name: str | None = None,
                                 provider_id: int | None = None, description: str | None = None,
                                 version: str | None = None, params: str | None = None) -> Model | None:
    """Updates model with business logic validation.

    Args:
        old_name: Current model name
        new_name: New model name (optional, None means don't change)
        provider_id: New provider ID (optional)
        description: New description (optional)
        version: New version (optional)
        params: New parameters (optional)

    Returns:
        Optional[Model]: The updated model or None if not found
    Raises:
        ValueError: On validation errors
    """
    validated_old_name = _validate_model_name(old_name)

    def _update_model(db):
        _ensure_model_exists(db, validated_old_name)

        validated_new_name = new_name
        if new_name:  # Only validate if new name is provided (not None)
            validated_new_name = _validate_model_name(new_name)
            if validated_new_name != validated_old_name:
                _check_model_name_uniqueness(db, validated_new_name, validated_old_name)

        # Validate provider exists if provided
        if provider_id is not None:
            _ensure_provider_exists(db, provider_id)

        return update_model(
            db=db,
            model_name=validated_old_name,
            new_model_name=validated_new_name,
            model_provider_id=provider_id,
            model_description=description,
            model_version=version,
            model_params=params
        )

    return _with_session(_update_model)


def delete_model_with_checks(model_name: str) -> bool:
    """Deletes model after business logic checks.

    Args:
        model_name: Name of the model to delete
    Returns:
        bool: True if successfully deleted, False otherwise
    Raises:
        ValueError: On validation errors
    """
    validated_name = _validate_model_name(model_name)

    def _delete_model(db):
        _ensure_model_exists(db, validated_name)
        # Here could be additional checks (e.g., if model is in use)
        # For now, we just delete it
        return delete_model(db, validated_name)

    return _with_session(_delete_model)


# ============================================================================
# PROVIDER SYNC FUNCTIONS
# ============================================================================

def sync_llm_models(delete_missing: bool = False) -> dict:
    """Gets models from external providers and stores them."""
    print("🔄 Syncing Models from External Providers")
    if delete_missing:
        print("⚠️  DELETE MODE: Missing models will be permanently removed from database!")
    print("=" * 50)
    
    results = {
        'ollama': {'added': 0, 'skipped': 0, 'updated': 0, 'deleted': 0, 'errors': []},
        'lm_studio': {'added': 0, 'skipped': 0, 'updated': 0, 'deleted': 0, 'errors': []},
        'total_processed': 0
    }

    def _sync_models(db):
        providers = get_all_llm_provider_configs(db)
        found_providers = []
        
        for provider in providers:
            provider_name = provider.prov_name.lower()
            if provider_name == 'ollama':
                print("\n🐋 Syncing Ollama models...")
                try:
                    ollama_result = _sync_ollama_models(db, provider, delete_missing)
                    results['ollama'] = ollama_result
                    results['total_processed'] += ollama_result['added'] + ollama_result['skipped']
                    deleted_info = f", {ollama_result.get('deleted', 0)} deleted" if delete_missing else ""
                    print(
                        f"   ✅ {ollama_result['added']} added, {ollama_result['skipped']} skipped, "
                        f"{ollama_result.get('updated', 0)} updated{deleted_info}"
                    )
                    if ollama_result['errors']:
                        print(f"   ❌ {len(ollama_result['errors'])} errors")
                        for error in ollama_result['errors']:
                            print(f"      - {error}")
                    found_providers.append('Ollama')
                except Exception as e:
                    print(f"   ❌ Failed to sync Ollama: {e}")
                    results['ollama']['errors'].append(str(e))
                    
            elif provider_name == 'lm studio':
                print("\n🖥️  Syncing LM Studio models...")
                try:
                    lmstudio_result = _sync_lmstudio_models(db, provider, delete_missing)
                    results['lm_studio'] = lmstudio_result
                    results['total_processed'] += lmstudio_result['added'] + lmstudio_result['skipped']
                    deleted_info = f", {lmstudio_result.get('deleted', 0)} deleted" if delete_missing else ""
                    print(
                        f"   ✅ {lmstudio_result['added']} added, {lmstudio_result['skipped']} skipped, "
                        f"{lmstudio_result.get('updated', 0)} updated{deleted_info}"
                    )
                    if lmstudio_result['errors']:
                        print(f"   ❌ {len(lmstudio_result['errors'])} errors")
                        for error in lmstudio_result['errors']:
                            print(f"      - {error}")
                    found_providers.append('LM Studio')
                except Exception as e:
                    print(f"   ❌ Failed to sync LM Studio: {e}")
                    results['lm_studio']['errors'].append(str(e))
        
        # Show summary
        print("\n📊 Sync Summary:")
        print(f"   Providers found: {', '.join(found_providers) if found_providers else 'None'}")
        print(f"   Total models processed: {results['total_processed']}")
        
        if not found_providers:
            print("   💡 No supported providers found. Add Ollama or LM Studio providers first.")
            
        return results

    return _with_session(_sync_models)


def restore_model(model_name: str) -> dict[str, Any]:
    """Restores a deleted Ollama model by downloading it via ollama pull.
    
    Args:
        model_name: Name of the model to restore
        
    Returns:
        Dict: Result with success status and message
        
    Raises:
        ValueError: If model not found or not an Ollama model
        RuntimeError: If download fails
    """
    validated_name = _validate_model_name(model_name)
    
    def _restore_model(db):
        # Ensure model exists in DB
        model = _ensure_model_exists(db, validated_name)
        
        # Get provider info
        provider = _ensure_provider_exists(db, model.model_provider_id)
        
        # Only support Ollama models for now
        if provider.prov_name.lower() != 'ollama':
            raise ValueError(f"Model restoration only supported for Ollama models, not {provider.prov_name}")
        
        # Check if model is already available
        if model.is_available:
            return {
                'success': True,
                'message': f"Model '{model_name}' is already available",
                'action': 'none'
            }
        
        try:
            # Execute ollama pull command
            result = subprocess.run(
                ['ollama', 'pull', validated_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Update model as available
                update_model(
                    db=db,
                    model_name=validated_name,
                    is_available=True,
                    last_checked=datetime.now()
                )
                
                return {
                    'success': True,
                    'message': f"Model '{model_name}' successfully restored",
                    'action': 'downloaded',
                    'output': result.stdout
                }
            else:
                raise RuntimeError(f"Ollama pull failed: {result.stderr}")
                
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Model download timed out after 5 minutes") from e
        except FileNotFoundError as e:
            raise RuntimeError("Ollama command not found. Please ensure Ollama is installed and in PATH") from e
        except Exception as e:
            raise RuntimeError(f"Failed to restore model: {e}") from e
    
    return _with_session(_restore_model)


# ============================================================================
# OLLAMA-SPECIFIC FUNCTIONS
# ============================================================================

def _fetch_ollama_models(provider) -> list[dict[str, Any]]:
    """Fetches available models from Ollama API."""
    base_url = provider.prov_endpoint or "http://localhost:11434"
    response = requests.get(f"{base_url}/api/tags")
    response.raise_for_status()
    data = response.json()
    return data.get('models', [])


def _create_model_description(model_info: dict[str, Any]) -> str:
    """Creates a descriptive text for an Ollama model."""
    model_size = model_info.get('size', 0)
    modified_at = model_info.get('modified_at', '')
    
    size_gb = round(model_size / (1024 ** 3), 2) if model_size > 0 else 0
    description = f"Ollama model, Size: {size_gb} GB"
    if modified_at:
        description += f", Modified: {modified_at}"
    
    return description


def _update_model_availability(
    db, provider_id: int, available_model_names: set, delete_missing: bool = False
) -> tuple[int, int]:
    """Updates availability status of existing models or deletes them if requested."""
    updated_count = 0
    deleted_count = 0
    existing_models = get_models_by_provider(db, provider_id)
    
    for existing_model in existing_models:
        is_available = existing_model.model_name in available_model_names
        
        if not is_available and delete_missing:
            # Delete missing models if delete_missing is True
            delete_model(db, existing_model.model_name)
            deleted_count += 1
        elif existing_model.is_available != is_available:
            # Update availability status
            update_model(
                db=db,
                model_name=existing_model.model_name,
                is_available=is_available,
                last_checked=datetime.now()
            )
            updated_count += 1
        else:
            # Model still exists and availability unchanged, just update last_checked
            update_model(
                db=db,
                model_name=existing_model.model_name,
                last_checked=datetime.now()
            )
    
    return updated_count, deleted_count


def _add_new_ollama_models(db, provider, model_infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Adds new models to database that don't exist yet."""
    result = {'added': 0, 'skipped': 0, 'errors': []}
    
    for model_info in model_infos:
        model_name = model_info.get('name', '')
        if not model_name:
            continue

        # Check if model already exists
        existing_model = get_model_by_name(db, model_name)
        if existing_model:
            # Update last_checked timestamp
            update_model(
                db=db,
                model_name=model_name,
                last_checked=datetime.now()
            )
            result['skipped'] += 1
            continue

        # Create new model
        try:
            description = _create_model_description(model_info)
            create_model(
                db=db,
                model_name=model_name,
                model_provider_id=provider.prov_id,
                model_description=description,
                model_version=None,
                model_params=None,
                is_available=True,
                last_checked=datetime.now()
            )
            result['added'] += 1

        except Exception as e:
            result['errors'].append(f"Error adding model '{model_name}': {str(e)}")
    
    return result


def _sync_ollama_models(db, provider, delete_missing: bool = False) -> dict:
    """Synchronizes Ollama models with the database."""
    result = {'added': 0, 'skipped': 0, 'updated': 0, 'deleted': 0, 'errors': []}

    try:
        # Fetch available models from Ollama
        model_infos = _fetch_ollama_models(provider)
        available_model_names = {model.get('name', '') for model in model_infos if model.get('name')}

        # Update availability status of existing models (or delete if requested)
        updated_count, deleted_count = _update_model_availability(
            db, provider.prov_id, available_model_names, delete_missing
        )
        result['updated'] = updated_count
        result['deleted'] = deleted_count

        # Add new models
        add_result = _add_new_ollama_models(db, provider, model_infos)
        result['added'] = add_result['added']
        result['skipped'] = add_result['skipped']
        result['errors'].extend(add_result['errors'])

    except requests.exceptions.RequestException as e:
        result['errors'].append(f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")

    return result


# ============================================================================
# LM STUDIO-SPECIFIC FUNCTIONS
# ============================================================================

def _fetch_lmstudio_models(provider) -> list[dict[str, Any]]:
    """Fetches available models from LM Studio API."""
    base_url = provider.prov_endpoint or "http://localhost:1234/v1"
    response = requests.get(f"{base_url}/models")
    response.raise_for_status()
    data = response.json()
    return data.get('data', [])


def _create_lmstudio_model_description(model_info: dict[str, Any]) -> str:
    """Creates a descriptive text for an LM Studio model."""
    model_id = model_info.get('id', '')
    owned_by = model_info.get('owned_by', '')
    
    description = f"LM Studio model: {model_id}"
    if owned_by and owned_by != 'organization_owner':
        description += f", Owner: {owned_by}"
    
    return description


def _add_new_lmstudio_models(db, provider, model_infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Adds new LM Studio models to database that don't exist yet."""
    result = {'added': 0, 'skipped': 0, 'errors': []}
    
    for model_info in model_infos:
        model_id = model_info.get('id', '')
        if not model_id:
            continue

        # Check if model already exists
        existing_model = get_model_by_name(db, model_id)
        if existing_model:
            # Update last_checked timestamp and availability
            update_model(
                db=db,
                model_name=model_id,
                is_available=True,
                last_checked=datetime.now()
            )
            result['skipped'] += 1
            continue

        # Create new model
        try:
            description = _create_lmstudio_model_description(model_info)
            create_model(
                db=db,
                model_name=model_id,
                model_provider_id=provider.prov_id,
                model_description=description,
                model_version=None,
                model_params=None,
                is_available=True,
                last_checked=datetime.now()
            )
            result['added'] += 1

        except Exception as e:
            result['errors'].append(f"Error adding model '{model_id}': {str(e)}")
    
    return result


def _sync_lmstudio_models(db, provider, delete_missing: bool = False) -> dict:
    """Synchronizes LM Studio models with the database."""
    result = {'added': 0, 'skipped': 0, 'updated': 0, 'deleted': 0, 'errors': []}

    try:
        # Fetch available models from LM Studio
        model_infos = _fetch_lmstudio_models(provider)
        available_model_names = {model.get('id', '') for model in model_infos if model.get('id')}

        # Update availability status of existing models for this provider (or delete if requested)
        updated_count, deleted_count = _update_model_availability(
            db, provider.prov_id, available_model_names, delete_missing
        )
        result['updated'] = updated_count
        result['deleted'] = deleted_count

        # Add new models
        add_result = _add_new_lmstudio_models(db, provider, model_infos)
        result['added'] = add_result['added']
        result['skipped'] = add_result['skipped']
        result['errors'].extend(add_result['errors'])

    except requests.exceptions.RequestException as e:
        result['errors'].append(f"Error connecting to LM Studio: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")

    return result