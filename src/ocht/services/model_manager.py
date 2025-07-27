import requests
import subprocess
from datetime import datetime
from typing import List, Optional, Dict, Any, TypeVar, Callable
from sqlmodel import select
from ocht.core.db import get_session
from ocht.repositories.model import (
    get_all_models,
    create_model,
    get_model_by_name,
    update_model,
    delete_model,
    get_models_by_provider
)
from ocht.repositories.llm_provider_config import get_all_llm_provider_configs, get_llm_provider_config_by_id
from ocht.core.models import Model, LLMProviderConfig

T = TypeVar('T')


# ============================================================================
# GENERAL HELPER FUNCTIONS
# ============================================================================

def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    db = next(get_session())
    return func(db)


def _validate_model_name(name: str) -> str:
    """Validates and normalizes model name."""
    if not name or not name.strip():
        raise ValueError("Model name is required")
    return name.strip()


def _check_model_name_uniqueness(db, name: str, exclude_name: Optional[str] = None) -> None:
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

def list_llm_models() -> List[Model]:
    """Reads available models from DB/Cache and returns them."""
    return _with_session(get_all_models)


def get_models_with_provider_info() -> List[Dict[str, Any]]:
    """
    Gets models with provider information for UI display.
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


def get_unavailable_models() -> List[Model]:
    """
    Gets all models that are marked as unavailable.
    
    Returns:
        List[Model]: List of unavailable models
    """
    def _get_unavailable(db):
        statement = select(Model).where(Model.is_available == False)
        return db.exec(statement).all()
    
    return _with_session(_get_unavailable)


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


def update_model_with_validation(old_name: str, new_name: Optional[str] = None,
                                 provider_id: Optional[int] = None, description: Optional[str] = None,
                                 version: Optional[str] = None, params: Optional[str] = None) -> Optional[Model]:
    """
    Updates model with business logic validation.
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
        existing_model = _ensure_model_exists(db, validated_old_name)

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
    """
    Deletes model after business logic checks.
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

def sync_llm_models() -> dict:
    """Gets models from external providers and stores them."""
    results = {
        'ollama': {'added': 0, 'skipped': 0, 'errors': []},
        'total_processed': 0
    }

    def _sync_models(db):
        providers = get_all_llm_provider_configs(db)
        for provider in providers:
            if provider.prov_name.lower() == 'ollama':
                ollama_result = _sync_ollama_models(db, provider)
                results['ollama'] = ollama_result
                results['total_processed'] += ollama_result['added'] + ollama_result['skipped']
        return results

    return _with_session(_sync_models)


def restore_model(model_name: str) -> Dict[str, Any]:
    """
    Restores a deleted Ollama model by downloading it via ollama pull.
    
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
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Model download timed out after 5 minutes")
        except FileNotFoundError:
            raise RuntimeError("Ollama command not found. Please ensure Ollama is installed and in PATH")
        except Exception as e:
            raise RuntimeError(f"Failed to restore model: {str(e)}")
    
    return _with_session(_restore_model)


# ============================================================================
# OLLAMA-SPECIFIC FUNCTIONS
# ============================================================================

def _fetch_ollama_models(provider) -> List[Dict[str, Any]]:
    """Fetches available models from Ollama API."""
    base_url = provider.prov_endpoint or "http://localhost:11434"
    response = requests.get(f"{base_url}/api/tags")
    response.raise_for_status()
    data = response.json()
    return data.get('models', [])


def _create_model_description(model_info: Dict[str, Any]) -> str:
    """Creates a descriptive text for an Ollama model."""
    model_size = model_info.get('size', 0)
    modified_at = model_info.get('modified_at', '')
    
    size_gb = round(model_size / (1024 ** 3), 2) if model_size > 0 else 0
    description = f"Ollama model, Size: {size_gb} GB"
    if modified_at:
        description += f", Modified: {modified_at}"
    
    return description


def _update_model_availability(db, provider_id: int, available_model_names: set) -> int:
    """Updates availability status of existing models."""
    updated_count = 0
    existing_models = get_models_by_provider(db, provider_id)
    
    for existing_model in existing_models:
        new_availability = existing_model.model_name in available_model_names
        
        if existing_model.is_available != new_availability:
            update_model(
                db=db,
                model_name=existing_model.model_name,
                is_available=new_availability,
                last_checked=datetime.now()
            )
            updated_count += 1
    
    return updated_count


def _add_new_ollama_models(db, provider, model_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _sync_ollama_models(db, provider) -> dict:
    """Synchronizes Ollama models with the database."""
    result = {'added': 0, 'skipped': 0, 'updated': 0, 'errors': []}

    try:
        # Fetch available models from Ollama
        model_infos = _fetch_ollama_models(provider)
        available_model_names = {model.get('name', '') for model in model_infos if model.get('name')}

        # Update availability status of existing models
        updated_count = _update_model_availability(db, provider.prov_id, available_model_names)
        result['updated'] = updated_count

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