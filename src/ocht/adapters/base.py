from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMAdapter(ABC):
    """
    Einheitliches Interface für alle LLM-Adapter.
    """

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