"""Service for managing the active LLM adapter and its provider/model configuration."""
import json
from collections.abc import Callable
from typing import Any, TypeVar

from ocht.adapters.anthropic import AnthropicAdapter
from ocht.adapters.base import LLMAdapter
from ocht.adapters.memory import MemoryConfig
from ocht.adapters.ollama import OllamaAdapter
from ocht.adapters.openai_compatible import OpenAICompatibleAdapter
from ocht.core.db import get_session
from ocht.core.models import LLMProviderConfig, Model
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


def _build_anthropic_params(provider_config: LLMProviderConfig) -> dict[str, Any]:
    """Builds the `default_params` for AnthropicAdapter.

    Unlike `_build_openai_compatible_params()`, there is no OpenRouter-style `extra_body` routing
    concept for Anthropic - `ChatAnthropic` has no equivalent "forward this JSON blob into the raw
    request" escape hatch, so `provider_config.prov_params` is deliberately not read here. A user
    wanting to override generation parameters for a specific Anthropic model uses
    `Model.model_params` instead (merged on top by `_merge_model_params()`, same as every other
    provider) - e.g. `{"max_tokens": 8192, "temperature": 0.5, "top_p": 0.9}`.

    No `max_tokens` default is set here (unlike an earlier draft of this function) - the installed
    `langchain-anthropic` version already defaults `max_tokens` to a large, sensible value when
    unset, so hardcoding one here would only make it harder to notice if that default ever changes
    upstream.

    Args:
        provider_config: The provider configuration to build params for (unused beyond the type
            signature - kept for symmetry with `_build_openai_compatible_params()` and in case a
            future Anthropic-specific per-provider setting is added).

    Returns:
        Params dict ready to pass as `default_params` to `AnthropicAdapter`.
    """
    return {"temperature": 0.7}


def _merge_model_params(default_params: dict[str, Any], model_params: str | None) -> dict[str, Any]:
    """Merges a model's own default generation parameters onto a provider-level params dict.

    `Model.model_params` is a JSON object of default generation parameters for one specific model
    (e.g. `{"temperature": 0.2, "max_tokens": 4096}`), forwarded verbatim into the LangChain
    chat-model constructor (`ChatOllama`/`ChatOpenAI`) alongside the provider-level params built by
    `_build_openai_compatible_params()` (or the plain Ollama default). Model-level values win over
    provider-level ones on key conflicts, since a setting scoped to one model is more specific than
    a provider-wide default such as the hardcoded temperature.

    A malformed `model_params` value is ignored (falls back to `default_params` alone) rather than
    blocking the adapter switch entirely - same behavior as `_build_openai_compatible_params()`'s
    handling of `prov_params`.

    Args:
        default_params: Provider-level default parameters to start from.
        model_params: The model's `model_params` JSON string, if any.

    Returns:
        A new dict with `model_params` merged over `default_params`.
    """
    merged = dict(default_params)
    if model_params:
        try:
            parsed = json.loads(model_params)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            merged.update(parsed)
    return merged


def _read_positive_int_setting(db, key: str, default: int) -> int:
    """Reads a positive-integer app setting, falling back to `default` if unset or invalid.

    Args:
        db: Active database session.
        key: The `Setting.setting_key` to look up.
        default: Value to use if the setting is missing, not a valid integer, or not positive.

    Returns:
        The parsed integer, or `default`.
    """
    setting = get_setting_by_key(db, key)
    if not setting:
        return default
    try:
        value = int(setting.setting_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _read_positive_float_setting(db, key: str, default: float) -> float:
    """Reads a positive-float app setting, falling back to `default` if unset or invalid.

    Args:
        db: Active database session.
        key: The `Setting.setting_key` to look up.
        default: Value to use if the setting is missing, not a valid number, or not positive.

    Returns:
        The parsed float, or `default`.
    """
    setting = get_setting_by_key(db, key)
    if not setting:
        return default
    try:
        value = float(setting.setting_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _build_memory_config(db) -> MemoryConfig:
    """Builds the conversation `MemoryConfig` from user-configurable global app Settings.

    Every `MemoryConfig` field was previously a hardcoded dataclass default - no adapter ever
    constructed one with custom values, so `HybridMemoryStrategy` always ran with
    `max_context_tokens=4000`/`recent_messages_count=10`/`code_retention_priority=2.0`/
    `summarization_threshold=20` regardless of the user's actual model/workflow. This reads each
    field from the generic `Setting` key-value store (editable via the existing Settings Manager
    screen - `tui/screens/settings_manager.py`, Ctrl+N to add a row with one of the
    `AdapterManager.MEMORY_*_KEY` names below as its key), falling back to that field's original
    hardcoded default whenever the setting is absent or holds an invalid/non-positive value -
    same "ignore malformed config rather than block adapter creation" precedent already used for
    `prov_params`/`model_params`.

    Args:
        db: Active database session.

    Returns:
        A `MemoryConfig` reflecting any configured overrides, or the original defaults where none
        are set.
    """
    defaults = MemoryConfig()
    return MemoryConfig(
        max_context_tokens=_read_positive_int_setting(
            db, AdapterManager.MEMORY_MAX_CONTEXT_TOKENS_KEY, defaults.max_context_tokens
        ),
        recent_messages_count=_read_positive_int_setting(
            db, AdapterManager.MEMORY_RECENT_MESSAGES_COUNT_KEY, defaults.recent_messages_count
        ),
        code_retention_priority=_read_positive_float_setting(
            db, AdapterManager.MEMORY_CODE_RETENTION_PRIORITY_KEY, defaults.code_retention_priority
        ),
        summarization_threshold=_read_positive_int_setting(
            db, AdapterManager.MEMORY_SUMMARIZATION_THRESHOLD_KEY, defaults.summarization_threshold
        ),
    )


def build_adapter(db, provider_config: LLMProviderConfig, model: Model) -> LLMAdapter | None:
    """Builds a throwaway adapter for one exact (provider, model) pair. Touches no shared state.

    Extracted from `AdapterManager._create_adapter()` so the real switch path (which additionally
    mutates `AdapterManager`'s `_current_*` singleton state) and `services/health_check.py`'s
    throwaway-adapter construction (which must NOT touch that singleton - checking one model's
    health must never disrupt whichever model the user is actively chatting with) share identical
    construction/param-merging logic instead of duplicating it.

    Args:
        db: Active database session.
        provider_config: The provider configuration to build an adapter for.
        model: The exact model to configure the adapter with - unlike `_create_adapter()`'s
            resolution logic, this function never substitutes a different model.

    Returns:
        A configured `OllamaAdapter`/`AnthropicAdapter`/`OpenAICompatibleAdapter`, or None if
        `provider_config.prov_name` isn't one of the supported provider types.
    """
    provider_name = provider_config.prov_name.lower()
    memory_config = _build_memory_config(db)

    if provider_name == "ollama":
        return OllamaAdapter(
            model=model.model_name,
            default_params=_merge_model_params({"temperature": 0.5}, model.model_params),
            memory_config=memory_config
        )
    if provider_name == "anthropic":
        return AnthropicAdapter(
            model=model.model_name,
            api_key=provider_config.prov_api_key,
            base_url=provider_config.prov_endpoint,
            default_params=_merge_model_params(
                _build_anthropic_params(provider_config), model.model_params
            ),
            memory_config=memory_config
        )
    if provider_name in ["openai", "lm studio", "openrouter", "mlx-lm"]:
        # Use OpenAI-compatible adapter for OpenAI, LM Studio, OpenRouter, and MLX-LM - all four
        # speak the OpenAI chat-completions API. MLX-LM has no native LangChain integration worth
        # using (see CLAUDE.md's "MLX-LM: Provider Recognition, Not a Native Adapter") - users run
        # `python -m mlx_lm.server` themselves, exactly like `ollama serve`/LM Studio are already
        # separate external processes this project only ever talks HTTP to.
        return OpenAICompatibleAdapter(
            model=model.model_name,
            api_key=provider_config.prov_api_key,
            base_url=provider_config.prov_endpoint,
            default_params=_merge_model_params(
                _build_openai_compatible_params(provider_config), model.model_params
            ),
            memory_config=memory_config
        )
    return None


class AdapterManager:
    """Service for managing LLM adapters and their configuration."""

    CURRENT_PROVIDER_KEY = "current_provider_id"
    CURRENT_MODEL_KEY = "current_model_name"

    # Well-known Setting keys for globally overriding HybridMemoryStrategy's tuning - see
    # _build_memory_config()'s docstring. Any conversation, regardless of provider/model, uses
    # the same memory strategy, so these are global settings rather than per-model ones.
    MEMORY_MAX_CONTEXT_TOKENS_KEY = "memory_max_context_tokens"
    MEMORY_RECENT_MESSAGES_COUNT_KEY = "memory_recent_messages_count"
    MEMORY_CODE_RETENTION_PRIORITY_KEY = "memory_code_retention_priority"
    MEMORY_SUMMARIZATION_THRESHOLD_KEY = "memory_summarization_threshold"
    
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
            
            # Create adapter based on provider type (shared with services/health_check.py - see
            # build_adapter()'s docstring for why this is a module-level function rather than
            # inlined here)
            try:
                adapter = build_adapter(db, provider_config, model)
                if adapter is None:
                    # Unsupported provider
                    return False

                self._current_adapter = adapter
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