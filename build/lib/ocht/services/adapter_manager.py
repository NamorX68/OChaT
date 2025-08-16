from typing import Optional, Dict, Any, TypeVar, Callable
from ocht.core.db import get_session
from ocht.adapters.base import LLMAdapter
from ocht.adapters.ollama import OllamaAdapter
from ocht.repositories.setting import get_setting_by_key, create_setting, update_setting
from ocht.repositories.llm_provider_config import get_llm_provider_config_by_id
from ocht.repositories.model import get_model_by_name

T = TypeVar('T')


def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    with get_session() as db:
        return func(db)


class AdapterManager:
    """Service for managing LLM adapters and their configuration."""
    
    CURRENT_PROVIDER_KEY = "current_provider_id"
    CURRENT_MODEL_KEY = "current_model_name"
    
    def __init__(self):
        self._current_adapter: Optional[LLMAdapter] = None
        self._current_provider_id: Optional[int] = None
        self._current_model_name: Optional[str] = None
    
    def get_current_adapter(self) -> Optional[LLMAdapter]:
        """Get the currently active adapter."""
        return self._current_adapter
    
    def get_current_provider_id(self) -> Optional[int]:
        """Get the currently selected provider ID."""
        return self._current_provider_id
    
    def get_current_model_name(self) -> Optional[str]:
        """Get the currently selected model name."""
        return self._current_model_name
    
    def load_settings_on_startup(self) -> bool:
        """
        Load provider and model settings on app startup.
        
        Returns:
            bool: True if settings were loaded successfully, False if missing
        """
        def _load_settings(db):
            # Load current provider
            provider_setting = get_setting_by_key(db, self.CURRENT_PROVIDER_KEY)
            model_setting = get_setting_by_key(db, self.CURRENT_MODEL_KEY)
            
            if not provider_setting or not model_setting:
                return False
            
            try:
                provider_id = int(provider_setting.setting_value)
                model_name = model_setting.setting_value
                
                # Create adapter with loaded settings
                return self._create_adapter(provider_id, model_name)
            except (ValueError, Exception):
                return False
        
        return _with_session(_load_settings)
    
    def save_current_settings(self) -> None:
        """Save current provider and model to settings."""
        if not self._current_provider_id or not self._current_model_name:
            return
        
        def _save_settings(db):
            # Save provider setting
            provider_setting = get_setting_by_key(db, self.CURRENT_PROVIDER_KEY)
            if provider_setting:
                update_setting(db, self.CURRENT_PROVIDER_KEY, value=str(self._current_provider_id))
            else:
                create_setting(db, self.CURRENT_PROVIDER_KEY, str(self._current_provider_id))
            
            # Save model setting
            model_setting = get_setting_by_key(db, self.CURRENT_MODEL_KEY)
            if model_setting:
                update_setting(db, self.CURRENT_MODEL_KEY, value=self._current_model_name)
            else:
                create_setting(db, self.CURRENT_MODEL_KEY, self._current_model_name)
        
        _with_session(_save_settings)
    
    def switch_adapter(self, provider_id: int, model_name: str) -> bool:
        """
        Switch to a new adapter configuration.
        
        Args:
            provider_id: ID of the provider
            model_name: Name of the model
            
        Returns:
            bool: True if switch was successful
        """
        if self._create_adapter(provider_id, model_name):
            self.save_current_settings()
            return True
        return False
    
    def _create_adapter(self, provider_id: int, model_name: str) -> bool:
        """
        Create and configure adapter based on provider and model.
        
        Args:
            provider_id: ID of the provider configuration
            model_name: Name of the model
            
        Returns:
            bool: True if adapter was created successfully
        """
        def _create(db):
            # Get provider configuration
            provider_config = get_llm_provider_config_by_id(db, provider_id)
            if not provider_config:
                return False
            
            # Get model configuration
            model = get_model_by_name(db, model_name)
            if not model or model.model_provider_id != provider_id:
                return False
            
            # Create adapter based on provider type
            try:
                if provider_config.prov_name.lower() == "ollama":
                    self._current_adapter = OllamaAdapter(
                        model=model_name,
                        default_params={"temperature": 0.5}
                    )
                else:
                    # TODO: Add support for other providers (OpenAI, Claude, etc.)
                    return False
                
                self._current_provider_id = provider_id
                self._current_model_name = model_name
                return True
                
            except Exception:
                return False
        
        return _with_session(_create)
    
    def requires_provider_selection(self) -> bool:
        """Check if provider selection is required (no current settings)."""
        def _check_provider(db):
            provider_setting = get_setting_by_key(db, self.CURRENT_PROVIDER_KEY)
            return provider_setting is None
        
        return _with_session(_check_provider)
    
    def requires_model_selection(self) -> bool:
        """Check if model selection is required (no current settings)."""
        def _check_model(db):
            model_setting = get_setting_by_key(db, self.CURRENT_MODEL_KEY)
            return model_setting is None
        
        return _with_session(_check_model)
    
    def has_active_chat(self) -> bool:
        """
        Check if there's an active chat session.
        This would need to be implemented based on your chat state management.
        For now, we'll assume there's always a potential active chat.
        """
        return self._current_adapter is not None
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about the current adapter including provider name."""
        def _get_info(db):
            provider_name = None
            if self._current_provider_id:
                provider_config = get_llm_provider_config_by_id(db, self._current_provider_id)
                if provider_config:
                    provider_name = provider_config.prov_name
            
            return {
                "provider_id": self._current_provider_id,
                "provider_name": provider_name,
                "model_name": self._current_model_name,
                "adapter_type": type(self._current_adapter).__name__ if self._current_adapter else None,
                "is_active": self._current_adapter is not None
            }
        
        return _with_session(_get_info)


# Global instance
adapter_manager = AdapterManager()