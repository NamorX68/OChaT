from typing import List, Optional, Dict, Any, TypeVar, Callable
from ocht.core.db import get_session
from ocht.repositories.setting import (
    get_all_settings,
    create_setting,
    update_setting,
    delete_setting,
    get_setting_by_key
)
from ocht.core.models import Setting

T = TypeVar('T')


def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    with get_session() as db:
        return func(db)


def _validate_setting_key(key: str) -> str:
    """Validates and normalizes setting key."""
    if not key or not key.strip():
        raise ValueError("Setting key is required")
    return key.strip()


def _validate_setting_value(value: str) -> str:
    """Validates and normalizes setting value."""
    if not value or not value.strip():
        raise ValueError("Setting value is required")
    return value.strip()


def _check_setting_key_uniqueness(db, key: str, exclude_key: Optional[str] = None) -> None:
    """Checks if setting key is unique."""
    existing_setting = get_setting_by_key(db, key)
    if existing_setting and key != exclude_key:
        raise ValueError(f"Setting with key '{key}' already exists")


def _ensure_setting_exists(db, key: str) -> Setting:
    """Ensures setting exists and returns it."""
    setting = get_setting_by_key(db, key)
    if not setting:
        raise ValueError(f"Setting with key '{key}' not found")
    return setting


def get_all_settings_with_info() -> List[Dict[str, Any]]:
    """
    Gets all settings with additional information for UI display.
    
    Returns:
        List[Dict]: List of dictionaries with setting information
    """
    def _get_settings_info(db):
        settings = get_all_settings(db)
        return [
            {
                'setting': setting,
                'workspace_scoped': setting.setting_workspace_id is not None,
                'key_length': len(setting.setting_key),
                'value_length': len(setting.setting_value)
            }
            for setting in settings
        ]
    
    return _with_session(_get_settings_info)


def get_setting_by_key_with_info(key: str) -> Optional[Dict[str, Any]]:
    """
    Gets a specific setting by key with additional information.
    
    Args:
        key: Setting key to retrieve
        
    Returns:
        Dict with setting information or None if not found
    """
    def _get_setting_info(db):
        setting = get_setting_by_key(db, key)
        if not setting:
            return None
        return {
            'setting': setting,
            'workspace_scoped': setting.setting_workspace_id is not None,
            'key_length': len(setting.setting_key),
            'value_length': len(setting.setting_value)
        }
    
    return _with_session(_get_setting_info)


def create_setting_with_validation(key: str, value: str, 
                                 workspace_id: Optional[int] = None) -> Setting:
    """
    Creates setting with business logic validation.
    
    Args:
        key: Setting key
        value: Setting value
        workspace_id: Optional workspace ID for workspace-specific settings
        
    Returns:
        Setting: The created setting
        
    Raises:
        ValueError: On validation errors
    """
    validated_key = _validate_setting_key(key)
    validated_value = _validate_setting_value(value)
    
    def _create_setting(db):
        _check_setting_key_uniqueness(db, validated_key)
        return create_setting(
            db=db,
            key=validated_key,
            value=validated_value
        )
    
    return _with_session(_create_setting)


def update_setting_with_validation(original_key: str, new_key: Optional[str] = None,
                                 value: Optional[str] = None) -> Optional[Setting]:
    """
    Updates setting with business logic validation.
    
    Args:
        original_key: Original setting key
        new_key: New key (optional, None means don't change)
        value: New value (optional, None means don't change)
        
    Returns:
        Optional[Setting]: The updated setting or None if not found
        
    Raises:
        ValueError: On validation errors
    """
    def _update_setting(db):
        existing_setting = _ensure_setting_exists(db, original_key)
        
        validated_new_key = new_key
        if new_key:  # Only validate if new_key is provided (not None)
            validated_new_key = _validate_setting_key(new_key)
            if validated_new_key != original_key:
                _check_setting_key_uniqueness(db, validated_new_key, original_key)
        
        validated_value = value
        if value:  # Only validate if value is provided (not None)
            validated_value = _validate_setting_value(value)
        
        return update_setting(
            db=db,
            setting_key=original_key,
            new_key=validated_new_key,
            value=validated_value
        )
    
    return _with_session(_update_setting)


def delete_setting_with_checks(key: str) -> bool:
    """
    Deletes setting after business logic checks.
    
    Args:
        key: Key of the setting to delete
        
    Returns:
        bool: True if successfully deleted, False otherwise
        
    Raises:
        ValueError: On validation errors
    """
    def _delete_setting(db):
        _ensure_setting_exists(db, key)
        return delete_setting(db, key)
    
    return _with_session(_delete_setting)


def get_workspace_settings(workspace_id: int) -> List[Setting]:
    """
    Gets all settings for a specific workspace.
    
    Args:
        workspace_id: Workspace ID
        
    Returns:
        List[Setting]: List of workspace-specific settings
    """
    def _get_workspace_settings(db):
        all_settings = get_all_settings(db)
        return [s for s in all_settings if s.setting_workspace_id == workspace_id]
    
    return _with_session(_get_workspace_settings)


def get_global_settings() -> List[Setting]:
    """
    Gets all global (non-workspace-specific) settings.
    
    Returns:
        List[Setting]: List of global settings
    """
    def _get_global_settings(db):
        all_settings = get_all_settings(db)
        return [s for s in all_settings if s.setting_workspace_id is None]
    
    return _with_session(_get_global_settings)