"""Service for managing the active LLM adapter and its provider/model configuration."""
import json
from collections.abc import Callable
from typing import Any, TypeVar

from ocht.adapters.base import LLMAdapter
from ocht.adapters.ollama import OllamaAdapter
from ocht.adapters.openai_compatible import OpenAICompatibleAdapter
from ocht.core.db import get_session
from ocht.core.models import LLMProviderConfig
from ocht.repositories.llm_provider_config import get_llm_provider_config_by_id
from ocht.repositories.model import get_model_by_name
from ocht.repositories.setting import create_setting, get_setting_by_key, update_setting

T = TypeVar('T')


def _with_session(func: Callable) -> T:
    """Helper function to execute database operations with session."""
    with get_session() as db:
        return func(db)


def _build_openai_compatible_params(provider_config: LLMProviderConfig) -> dict[str, Any]:
    """Builds the `default_params` for OpenAICompatibleAdapter, including provider routing prefs.

    `LLMProviderConfig.prov_params` is a JSON object of OpenAI-compatible "extra body" routing
    preferences (e.g. OpenRouter's `provider` object - quantization filters, throughput/latency
    preferences, provider allow/deny lists). It's forwarded verbatim via `ChatOpenAI`'s
    `extra_body={"provider": ...}`, which the `openai` SDK merges into the raw request JSON - so
    any field OpenRouter's provider-routing API supports can be configured here without adapter
    code changes. See https://openrouter.ai/docs/guides/routing/provider-selection.

    A malformed `prov_params` value is ignored (falls back to default provider routing) rather
    than blocking the adapter switch entirely.

    Args:
        provider_config: The provider configuration to build params for.

    Returns:
        Params dict ready to pass as `default_params` to `OpenAICompatibleAdapter`.
    """
    default_params: dict[str, Any] = {"temperature": 0.7}
    if provider_config.prov_params:
        try:
            provider_routing = json.loads(provider_config.prov_params)
        except (json.JSONDecodeError, TypeError):
            provider_routing = None
        if isinstance(provider_routing, dict) and provider_routing:
            default_params["extra_body"] = {"provider": provider_routing}
    return default_params


class AdapterManager:
    """Service for managing LLM adapters and their configuration."""
    
    CURRENT_PROVIDER_KEY = "current_provider_id"
    CURRENT_MODEL_KEY = "current_model_name"
    
    def __init__(self):
        """Initializes the manager with no active adapter, provider, or model selected."""
        self._current_adapter: LLMAdapter | None = None
        self._current_provider_id: int | None = None
        self._current_model_name: str | None = None
    
    def get_current_adapter(self) -> LLMAdapter | None:
        """Get the currently active adapter."""
        return self._current_adapter
    
    def get_current_provider_id(self) -> int | None:
        """Get the currently selected provider ID."""
        return self._current_provider_id
    
    def get_current_model_name(self) -> str | None:
        """Get the currently selected model name."""
        return self._current_model_name
    
    def load_settings_on_startup(self) -> bool:
        """Load provider and model settings on app startup.
        
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
        """Switch to a new adapter configuration.
        
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
        """Create and configure adapter based on provider and model.
        
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
            actual_model_name = model_name  # Keep track of the actual model name to use
            
            # If model doesn't exist or doesn't belong to this provider, find first available model for provider
            if not model or model.model_provider_id != provider_id:
                from ocht.repositories.model import get_models_by_provider
                provider_models = get_models_by_provider(db, provider_id)
                
                # Find first available model for this provider
                model = None
                for candidate_model in provider_models:
                    if candidate_model.is_available:
                        model = candidate_model
                        actual_model_name = candidate_model.model_name  # Update to use this model
                        break
                
                # If no available models, use first model anyway
                if not model and provider_models:
                    model = provider_models[0]
                    actual_model_name = provider_models[0].model_name
                
                # If still no model found, return False
                if not model:
                    return False
            
            # Create adapter based on provider type
            try:
                provider_name = provider_config.prov_name.lower()
                
                if provider_name == "ollama":
                    self._current_adapter = OllamaAdapter(
                        model=actual_model_name,
                        default_params={"temperature": 0.5}
                    )
                elif provider_name in ["openai", "lm studio", "openrouter"]:
                    # Use OpenAI-compatible adapter for OpenAI, LM Studio, and OpenRouter -
                    # all three speak the OpenAI chat-completions API, just with different
                    # base_url/api_key configuration on the LLMProviderConfig row.
                    self._current_adapter = OpenAICompatibleAdapter(
                        model=actual_model_name,
                        api_key=provider_config.prov_api_key,
                        base_url=provider_config.prov_endpoint,
                        default_params=_build_openai_compatible_params(provider_config)
                    )
                else:
                    # Unsupported provider
                    return False
                
                self._current_provider_id = provider_id
                self._current_model_name = actual_model_name  # Use the actual model name
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
        """Check if there's an active chat session.

        This would need to be implemented based on your chat state management.
        For now, we'll assume there's always a potential active chat.
        """
        return self._current_adapter is not None
    
    def get_adapter_info(self) -> dict[str, Any]:
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