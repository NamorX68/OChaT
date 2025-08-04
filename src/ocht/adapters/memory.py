import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationSummaryMemory
from langchain.schema.language_model import BaseLanguageModel


@dataclass
class MemoryConfig:
    """Configuration for memory strategies."""
    max_context_tokens: int = 4000
    recent_messages_count: int = 10
    code_retention_priority: float = 2.0  # Higher = longer retention
    summarization_threshold: int = 20  # Start summarizing after N messages


class MemoryStrategy(ABC):
    """Abstract base class for memory management strategies."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
    
    @abstractmethod
    async def prepare_context(self, messages: List[BaseMessage], new_prompt: str) -> List[Tuple[str, str]]:
        """
        Prepare conversation context for LLM call.
        
        Args:
            messages: Historical messages from memory
            new_prompt: New user prompt to be added
            
        Returns:
            List of (role, content) tuples ready for LLM
        """
        pass
    
    @abstractmethod
    async def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """
        Determine if conversation should be summarized.
        
        Args:
            messages: Current message history
            
        Returns:
            True if summarization should occur
        """
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Improved token estimation that accounts for different text patterns.
        
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
        """
        Detect if message contains code blocks or code-like content.
        
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
    """
    Hybrid memory strategy that combines recent message retention with smart summarization.
    
    Features:
    - Keep last N messages completely for immediate context
    - Prioritize code-containing messages for longer retention
    - Smart summarization of older messages
    - Token-aware context management
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, llm: Optional[BaseLanguageModel] = None):
        super().__init__(config)
        self._summary_cache: Optional[str] = None
        self._last_summarized_count: int = 0
        self._llm = llm
        self._summarizer: Optional[ConversationSummaryMemory] = None
        
        if llm:
            self._summarizer = ConversationSummaryMemory(
                llm=llm,
                return_messages=False,  # We want string summaries
                max_token_limit=self.config.max_context_tokens // 4  # Reserve 1/4 for summary
            )
    
    async def prepare_context(self, messages: List[BaseMessage], new_prompt: str) -> List[Tuple[str, str]]:
        """
        Prepare context using hybrid strategy.
        
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
    
    async def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Check if summarization should occur based on message count and content."""
        return (
            len(messages) >= self.config.summarization_threshold and
            len(messages) > self._last_summarized_count + 5  # Re-summarize every 5 new messages
        )
    
    async def _get_or_create_summary(self, messages: List[BaseMessage]) -> Optional[str]:
        """Get cached summary or create new one if needed."""
        if await self.should_summarize(messages):
            if self._summarizer and self._llm:
                # Use LangChain's summarization
                try:
                    # Create a temporary conversation for summarization
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            self._summarizer.save_context({"input": msg.content}, {"output": ""})
                        elif isinstance(msg, AIMessage):
                            # Find the corresponding human message
                            prev_human = next((m for m in reversed(messages[:messages.index(msg)]) 
                                             if isinstance(m, HumanMessage)), None)
                            if prev_human:
                                self._summarizer.save_context(
                                    {"input": prev_human.content}, 
                                    {"output": msg.content}
                                )
                    
                    # Get the summary
                    summary_vars = self._summarizer.load_memory_variables({})
                    self._summary_cache = summary_vars.get("history", "")
                    
                except Exception:
                    # Fallback to simple summary if LangChain summarization fails
                    self._summary_cache = self._create_simple_summary(messages)
            else:
                # Fallback to simple summary
                self._summary_cache = self._create_simple_summary(messages)
            
            self._last_summarized_count = len(messages)
        
        return self._summary_cache
    
    def _create_simple_summary(self, messages: List[BaseMessage]) -> str:
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
    
    def _select_important_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
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
    
    async def _trim_to_token_limit(self, context_tuples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
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
    
    def _convert_message_to_tuple(self, msg: BaseMessage) -> Tuple[str, str]:
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