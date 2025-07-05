from typing import Optional, Dict, Any
from langchain.memory import ConversationSummaryMemory
from langchain_ollama import ChatOllama
from ocht.adapters.base import LLMAdapter

class OllamaAdapter(LLMAdapter):
    """Adapter für lokale Ollama-Modelle über LangChain."""

    def __init__(
        self,
        model: str = "devstral:24b-q8_0",
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

    def send_prompt(self, prompt: str, **kwargs) -> str:
        # Geschichte laden und konvertieren
        history = self.memory.load_memory_variables({})["history"]
        messages = [self._convert_message_to_tuple(msg) for msg in history]
        messages.append(("human", prompt))

        # LLM aufrufen und Kontext speichern
        response = self.client.invoke(messages, **kwargs)
        self.memory.save_context({"input": prompt}, {"output": response.content})

        return response.content