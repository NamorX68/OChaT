from abc import ABC, abstractmethod
from typing import Any

class LLMAdapter(ABC):
    """Einheitliches Interface für alle LLM-Adapter."""

    @abstractmethod
    def send_prompt(self, prompt: str, **kwargs) -> str:
        """
        Sendet einen Prompt an den LLM und gibt die rohe Text-Antwort zurück.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter (z.B. temperature, max_tokens).

        Returns:
            Die vom LLM generierte Antwort als String.
        """
        ...

    def _convert_message_to_tuple(self, msg: Any) -> tuple[str, str]:
        """
        Konvertiert LangChain-Message zu (role, content) Tupel.

        Standard-Implementierung für die meisten LangChain Message-Types.
        Kann in Subklassen überschrieben werden, falls Provider-spezifische
        Anpassungen nötig sind.
        """
        message_type_map = {
            'human': 'human',
            'ai': 'ai',
            'system': 'system',
            'user': 'human',      # Fallback für andere Provider
            'assistant': 'ai'     # Fallback für andere Provider
        }

        msg_type = getattr(msg, 'type', 'system')
        role = message_type_map.get(msg_type, 'system')

        return (role, msg.content)