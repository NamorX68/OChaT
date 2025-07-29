import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

class LLMAdapter(ABC):
    """Einheitliches Interface für alle LLM-Adapter."""

    @abstractmethod
    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        """
        Sendet einen Prompt asynchron an den LLM.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter (z.B. temperature, max_tokens).

        Returns:
            Die vom LLM generierte Antwort als String.
        """
        ...

    @abstractmethod
    def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Sendet einen Prompt an den LLM und gibt Streaming-Antwort zurück.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter.

        Yields:
            Text-Chunks der LLM Antwort.
        """
        ...

    def send_prompt(self, prompt: str, **kwargs) -> str:
        """
        Synchroner Wrapper für send_prompt_async.
        Erkennt automatisch ob Event Loop läuft.

        Args:
            prompt: Der Eingabetext für das LLM.
            **kwargs: Provider-spezifische Parameter.

        Returns:
            Die vom LLM generierte Antwort als String.
        """
        try:
            # Prüfe ob Event Loop bereits läuft
            loop = asyncio.get_running_loop()
            # Wenn ja, nutze run_in_executor für thread-based execution
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.send_prompt_async(prompt, **kwargs))
                )
                return future.result()
        except RuntimeError:
            # Kein Event Loop - normal mit asyncio.run()
            return asyncio.run(self.send_prompt_async(prompt, **kwargs))

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