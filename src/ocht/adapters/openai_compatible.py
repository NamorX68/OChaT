import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List, Tuple
from langchain.memory import ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from ocht.adapters.base import LLMAdapter
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig

class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs (like LM Studio) via LangChain."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None,
        memory=None,
        use_hybrid_memory: bool = True,
        memory_config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize OpenAI-compatible adapter.
        
        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', or local model name for LM Studio)
            api_key: API key for OpenAI (not needed for LM Studio)
            base_url: Custom base URL (e.g., 'http://localhost:1234/v1' for LM Studio)
            default_params: Default parameters like temperature, max_tokens
            memory: Legacy memory system (optional)
            use_hybrid_memory: Whether to use the new hybrid memory strategy
            memory_config: Configuration for hybrid memory system
        """
        # Setup client parameters
        client_kwargs = {
            'model': model,
            **(default_params or {})
        }
        
        # Add API key if provided (required for OpenAI, not for LM Studio)
        if api_key:
            client_kwargs['api_key'] = api_key
            
        # Add custom base URL if provided (for LM Studio or other compatible APIs)
        if base_url:
            client_kwargs['base_url'] = base_url
            
        self.client = ChatOpenAI(**client_kwargs)
        self.model_name = model
        self.base_url = base_url

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
        """Send prompt asynchronously to OpenAI-compatible API."""
        # Load and convert history
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
        
        # Call LLM asynchronously
        response = await self.client.ainvoke(message_objects, **kwargs)
        
        # Save context
        await self._save_to_memory(prompt, response.content)
        
        return response.content

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Send prompt to OpenAI-compatible API with streaming response."""
        # Load and convert history
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
        
        # Save complete text after streaming
        if full_response:
            await self._save_to_memory(prompt, full_response)

    async def _prepare_messages(self, prompt: str) -> list[tuple[str, str]]:
        """Prepare message history for LLM call."""
        # Memory operations could be async - for now sync
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get('history', [])
        messages = [self._convert_message_to_tuple(msg) for msg in history]
        messages.append(("human", prompt))
        return messages

    async def _save_to_memory(self, prompt: str, response: str):
        """Save context to memory."""
        # Memory operations could be async - for now sync
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

    def is_local_model(self) -> bool:
        """Check if this is a local model (LM Studio) or cloud API (OpenAI)."""
        return self.base_url is not None and 'localhost' in self.base_url

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider configuration."""
        return {
            'model': self.model_name,
            'base_url': self.base_url,
            'is_local': self.is_local_model(),
            'provider': 'LM Studio' if self.is_local_model() else 'OpenAI'
        }