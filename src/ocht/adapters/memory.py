"""Conversation memory strategies for managing LLM context windows."""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass
class MemoryConfig:
    """Configuration for memory strategies."""
    max_context_tokens: int = 4000
    recent_messages_count: int = 10
    code_retention_priority: float = 2.0  # Higher = longer retention
    summarization_threshold: int = 20  # Start summarizing after N messages


class MemoryStrategy(ABC):
    """Abstract base class for memory management strategies."""
    
    def __init__(self, config: MemoryConfig | None = None):
        """Initializes the strategy with the given configuration.

        Args:
            config: Memory configuration to use. Defaults to a new `MemoryConfig()` if not provided.
        """
        self.config = config or MemoryConfig()
    
    @abstractmethod
    async def prepare_context(self, messages: list[BaseMessage], new_prompt: str) -> list[tuple[str, str]]:
        """Prepare conversation context for LLM call.
        
        Args:
            messages: Historical messages from memory
            new_prompt: New user prompt to be added
            
        Returns:
            List of (role, content) tuples ready for LLM
        """
        pass
    
    @abstractmethod
    async def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Determine if conversation should be summarized.
        
        Args:
            messages: Current message history
            
        Returns:
            True if summarization should occur
        """
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Improved token estimation that accounts for different text patterns.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # More sophisticated estimation based on content type
        base_tokens = len(text) // 4  # Basic char/4 estimation
        
        # Code blocks typically have more tokens per character
        if self._contains_code(text):
            # Code has more punctuation and special chars = more tokens
            code_factor = 1.3
            base_tokens = int(base_tokens * code_factor)
        
        # Account for whitespace (doesn't count as tokens)
        whitespace_chars = len(re.findall(r'\s', text))
        adjusted_tokens = base_tokens - (whitespace_chars // 8)  # Rough adjustment
        
        # Minimum of 1 token for non-empty text
        return max(1, adjusted_tokens)
    
    def _contains_code(self, text: str) -> bool:
        """Detect if message contains code blocks or code-like content.
        
        Args:
            text: Message content to analyze
            
        Returns:
            True if code is detected
        """
        code_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`\n]+`',       # Inline code
            r'\b(def|class|function|import|from|return)\b',  # Python keywords
            r'\b(async|await|const|let|var|function)\b',     # JS keywords
            r'[{}();]',         # Common code punctuation
            r'=\s*["\']',       # Assignment patterns
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


class HybridMemoryStrategy(MemoryStrategy):
    """Hybrid memory strategy that combines recent message retention with smart summarization.
    
    Features:
    - Keep last N messages completely for immediate context
    - Prioritize code-containing messages for longer retention
    - Smart summarization of older messages
    - Token-aware context management
    """
    
    def __init__(self, config: MemoryConfig | None = None, llm: BaseLanguageModel | None = None):
        """Initializes the strategy and, if an LLM is provided, enables LLM-based summarization.

        Args:
            config: Memory configuration to use. Defaults to a new `MemoryConfig()` if not provided.
            llm: Optional language model used to generate conversation summaries. If omitted, a simple
                heuristic summary is used instead.
        """
        super().__init__(config)
        self._summary_cache: str | None = None
        self._last_summarized_count: int = 0
        self._llm = llm
    
    async def prepare_context(self, messages: list[BaseMessage], new_prompt: str) -> list[tuple[str, str]]:
        """Prepare context using hybrid strategy.
        
        Strategy:
        1. Always keep recent messages (last N)
        2. For older messages: keep code-heavy ones, summarize others
        3. Ensure total context fits within token limit
        """
        if not messages:
            return [("human", new_prompt)]
        
        total_messages = len(messages)
        recent_cutoff = max(0, total_messages - self.config.recent_messages_count)
        
        # Split messages into recent and older
        older_messages = messages[:recent_cutoff]
        recent_messages = messages[recent_cutoff:]
        
        context_tuples = []
        
        # Handle older messages with summarization/selection
        if older_messages:
            summary_text = await self._get_or_create_summary(older_messages)
            if summary_text:
                context_tuples.append(("system", f"Previous conversation summary: {summary_text}"))
            
            # Keep important older messages (code-heavy ones)
            important_older = self._select_important_messages(older_messages)
            for msg in important_older:
                role, content = self._convert_message_to_tuple(msg)
                context_tuples.append((role, content))
        
        # Add recent messages (always keep these)
        for msg in recent_messages:
            role, content = self._convert_message_to_tuple(msg)
            context_tuples.append((role, content))
        
        # Add new prompt
        context_tuples.append(("human", new_prompt))
        
        # Ensure token limit compliance
        context_tuples = await self._trim_to_token_limit(context_tuples)
        
        return context_tuples
    
    async def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Check if summarization should occur based on message count and content."""
        return (
            len(messages) >= self.config.summarization_threshold and
            len(messages) > self._last_summarized_count + 5  # Re-summarize every 5 new messages
        )
    
    async def _get_or_create_summary(self, messages: list[BaseMessage]) -> str | None:
        """Get cached summary or create new one if needed."""
        if await self.should_summarize(messages):
            if self._llm:
                try:
                    self._summary_cache = await self._summarize_with_llm(messages)
                except Exception:
                    # Fallback to simple summary if LLM summarization fails
                    self._summary_cache = self._create_simple_summary(messages)
            else:
                # Fallback to simple summary
                self._summary_cache = self._create_simple_summary(messages)

            self._last_summarized_count = len(messages)

        return self._summary_cache

    async def _summarize_with_llm(self, messages: list[BaseMessage]) -> str:
        """Ask the configured LLM to summarize older conversation messages in a single call.

        Replaces the previous ConversationSummaryMemory-based approach (deprecated in LangChain
        0.3.1, removed from the `langchain` package in 1.0) with one direct call to the same LLM
        client the adapter already talks to, over the full transcript at once - rather than one
        incremental LLM call per message pair, which is what ConversationSummaryMemory did.

        Args:
            messages: Older messages to summarize.

        Returns:
            A concise text summary of the given messages.
        """
        role_labels = {HumanMessage: "Human", AIMessage: "AI", SystemMessage: "System"}
        transcript = "\n".join(
            f"{role_labels.get(type(msg), 'System')}: {msg.content}" for msg in messages
        )
        summary_prompt = [
            SystemMessage(
                content=(
                    "Summarize the following conversation concisely, preserving key facts, "
                    "decisions, and any code or technical details that were discussed. "
                    "Respond with the summary only, no preamble."
                )
            ),
            HumanMessage(content=transcript),
        ]
        response = await self._llm.ainvoke(summary_prompt)
        return response.content
    
    def _create_simple_summary(self, messages: list[BaseMessage]) -> str:
        """Create a simple summary of messages (placeholder for LangChain integration)."""
        topics = set()
        code_mentions = []
        
        for msg in messages:
            content = msg.content.lower()
            
            # Extract potential topics (very basic)
            if 'error' in content or 'bug' in content:
                topics.add('debugging')
            if 'implement' in content or 'create' in content:
                topics.add('implementation')
            if 'test' in content:
                topics.add('testing')
            
            # Note code-related discussions
            if self._contains_code(msg.content):
                # Extract function names or class names
                code_refs = re.findall(r'\b(def|class)\s+(\w+)', msg.content)
                code_mentions.extend([ref[1] for ref in code_refs])
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Discussion topics: {', '.join(topics)}")
        if code_mentions:
            summary_parts.append(f"Code references: {', '.join(set(code_mentions))}")
        
        return ". ".join(summary_parts) if summary_parts else "General conversation"
    
    def _select_important_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Select important messages from older history (prioritize code-containing ones)."""
        scored_messages = []
        
        for msg in messages:
            score = 0.0
            
            # Higher score for code content
            if self._contains_code(msg.content):
                score += self.config.code_retention_priority
            
            # Higher score for longer, detailed messages
            score += min(len(msg.content) / 1000, 1.0)
            
            # Lower score for very recent messages (they'll be in recent_messages)
            scored_messages.append((score, msg))
        
        # Sort by score and take top messages, but limit to avoid context overflow
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        max_important = min(5, max(1, len(scored_messages) // 2))  # Max 5 or 1/2 of older messages, minimum 1
        
        # Lower threshold for code messages - they should be prioritized even if short
        min_score = 0.5 if any(score >= self.config.code_retention_priority for score, _ in scored_messages) else 1.0
        
        return [msg for score, msg in scored_messages[:max_important] if score >= min_score]
    
    async def _trim_to_token_limit(self, context_tuples: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Ensure context fits within token limit by removing older messages if needed."""
        total_tokens = sum(self._estimate_tokens(content) for _, content in context_tuples)
        
        if total_tokens <= self.config.max_context_tokens:
            return context_tuples
        
        # Remove messages from the beginning (after system summary) until we fit
        # Always keep the last message (new prompt)
        trimmed = context_tuples[-1:]  # Keep new prompt
        remaining_budget = self.config.max_context_tokens - self._estimate_tokens(context_tuples[-1][1])
        
        # Add messages from end to beginning until budget exhausted
        for role, content in reversed(context_tuples[:-1]):
            token_cost = self._estimate_tokens(content)
            if remaining_budget >= token_cost:
                trimmed.insert(0, (role, content))
                remaining_budget -= token_cost
            else:
                break
        
        return trimmed
    
    def _convert_message_to_tuple(self, msg: BaseMessage) -> tuple[str, str]:
        """Convert LangChain message to (role, content) tuple."""
        if isinstance(msg, HumanMessage):
            return ("human", msg.content)
        elif isinstance(msg, AIMessage):
            return ("ai", msg.content)
        elif isinstance(msg, SystemMessage):
            return ("system", msg.content)
        else:
            # Fallback for other message types
            return ("system", msg.content)