import asyncio
from typing import Optional, Dict, Any, AsyncIterator
from langchain.memory import ConversationSummaryMemory
from langchain_ollama import ChatOllama
from ocht.adapters.base import LLMAdapter

class OllamaAdapter(LLMAdapter):
    """Adapter für lokale Ollama-Modelle über LangChain."""

    def __init__(
        self,
        model: str = "qwen3:30b-a3b",
        base_url: str = "http://localhost:11434",
        default_params: Optional[Dict[str, Any]] = None,
        memory=None,
    ):
        self.client = ChatOllama(
            model=model,
            base_url=base_url,
            **(default_params or {})
        )

        self.memory = memory or ConversationSummaryMemory(
            llm=self.client,
            return_messages=True,
            output_key="output"
        )

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        # Geschichte laden und konvertieren
        messages = await self._prepare_messages(prompt)
        
        # LLM asynchron aufrufen
        response = await self.client.ainvoke(messages, **kwargs)
        
        # Kontext speichern
        await self._save_to_memory(prompt, response.content)
        
        return response.content

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # Geschichte laden und konvertieren
        messages = await self._prepare_messages(prompt)
        
        # Streaming response
        full_response = ""
        async for chunk in self.client.astream(messages, **kwargs):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        # Nach dem Streaming den vollständigen Text speichern
        if full_response:
            await self._save_to_memory(prompt, full_response)

    async def _prepare_messages(self, prompt: str) -> list[tuple[str, str]]:
        """Bereitet die Nachrichten-Historie für den LLM-Call vor."""
        # Memory operations könnten auch async sein - für jetzt sync
        history = self.memory.load_memory_variables({})["history"]
        messages = [self._convert_message_to_tuple(msg) for msg in history]
        messages.append(("human", prompt))
        return messages

    async def _save_to_memory(self, prompt: str, response: str):
        """Speichert den Kontext ins Memory."""
        # Memory operations könnten auch async sein - für jetzt sync
        await asyncio.to_thread(
            self.memory.save_context,
            {"input": prompt},
            {"output": response}
        )