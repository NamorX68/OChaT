import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List, Tuple
from langchain.memory import ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from ocht.adapters.base import LLMAdapter
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig

class OllamaAdapter(LLMAdapter):
    """Adapter für lokale Ollama-Modelle über LangChain."""

    def __init__(
        self,
        model: str = "qwen3:30b-a3b",
        base_url: str = "http://localhost:11434",
        default_params: Optional[Dict[str, Any]] = None,
        memory=None,
        use_hybrid_memory: bool = True,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.client = ChatOllama(
            model=model,
            base_url=base_url,
            **(default_params or {})
        )

        if use_hybrid_memory:
            # Use new HybridMemoryStrategy
            self.memory_strategy = HybridMemoryStrategy(
                config=memory_config or MemoryConfig(),
                llm=self.client
            )
            # Keep legacy memory for compatibility, but it won't be used
            self.memory = ConversationSummaryMemory(
                llm=self.client,
                return_messages=True,
                output_key="output"
            )
        else:
            # Legacy memory system
            self.memory_strategy = None
            self.memory = memory or ConversationSummaryMemory(
                llm=self.client,
                return_messages=True,
                output_key="output"
            )

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        # Geschichte laden und konvertieren
        if self.memory_strategy:
            # Use HybridMemoryStrategy
            memory_vars = self.memory.load_memory_variables({})
            history_messages = memory_vars.get('history', [])
            messages = await self.memory_strategy.prepare_context(history_messages, prompt)
        else:
            # Legacy method
            messages = await self._prepare_messages(prompt)
        
        # Convert tuples to message objects for LangChain
        message_objects = self._convert_tuples_to_messages(messages)
        
        # LLM asynchron aufrufen
        response = await self.client.ainvoke(message_objects, **kwargs)
        
        # Kontext speichern
        await self._save_to_memory(prompt, response.content)
        
        return response.content

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # Geschichte laden und konvertieren
        if self.memory_strategy:
            # Use HybridMemoryStrategy
            memory_vars = self.memory.load_memory_variables({})
            history_messages = memory_vars.get('history', [])
            messages = await self.memory_strategy.prepare_context(history_messages, prompt)
        else:
            # Legacy method
            messages = await self._prepare_messages(prompt)
        
        # Convert tuples to message objects for LangChain
        message_objects = self._convert_tuples_to_messages(messages)
        
        # Streaming response
        full_response = ""
        async for chunk in self.client.astream(message_objects, **kwargs):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        # Nach dem Streaming den vollständigen Text speichern
        if full_response:
            await self._save_to_memory(prompt, full_response)

    async def _prepare_messages(self, prompt: str) -> list[tuple[str, str]]:
        """Bereitet die Nachrichten-Historie für den LLM-Call vor."""
        # Memory operations könnten auch async sein - für jetzt sync
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get('history', [])
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

    def _convert_tuples_to_messages(self, message_tuples: List[Tuple[str, str]]) -> List[BaseMessage]:
        """Convert list of (role, content) tuples to LangChain message objects."""
        messages = []
        for role, content in message_tuples:
            if role.lower() in ['human', 'user']:
                messages.append(HumanMessage(content=content))
            elif role.lower() in ['ai', 'assistant']:
                messages.append(AIMessage(content=content))
            elif role.lower() == 'system':
                messages.append(SystemMessage(content=content))
            else:
                # Default to system message for unknown roles
                messages.append(SystemMessage(content=content))
        return messages