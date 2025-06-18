from langchain.llms import Ollama
from typing import Optional, Dict, Any

from ocht.adapters.base import LLMAdapter


class OllamaAdapter(LLMAdapter):
    """
    Adapter für lokale Ollama-Modelle über LangChain.
    """

    def __init__(
            self,
            model: str = "llama2",
            base_url: Optional[str] = None,
            default_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: Name des lokalen Ollama-Modells.
            base_url: URL zum Ollama-Server (lokal).
            default_params: Standard-Argumente für jeden Call (z.B. temperature).
        """
        self.default_params = default_params or {}
        self.client = Ollama(
            model=model,
            base_url=base_url,
            **self.default_params
        )

    def send_prompt(self, prompt: str, **kwargs) -> str:
        # Merge default_params mit call-spezifischen overrides
        call_params = {**self.default_params, **kwargs}

        # Einfache LangChain-Methode
        response = self.client(prompt, **call_params)

        # Bei Bedarf noch Post-Processing hier durchführen
        return response
