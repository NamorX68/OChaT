import pytest
import asyncio
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig


class TestHybridMemoryStrategy:
    """Test cases for HybridMemoryStrategy."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = MemoryConfig(
            max_context_tokens=1000,
            recent_messages_count=5,
            summarization_threshold=10
        )
        self.strategy = HybridMemoryStrategy(config=self.config)
    
    def test_token_estimation(self):
        """Test token estimation for different text types."""
        # Simple text
        simple_text = "Hello world"
        tokens = self.strategy._estimate_tokens(simple_text)
        assert tokens > 0
        
        # Code text should have more tokens per character
        code_text = "def hello():\n    return 'world'"
        code_tokens = self.strategy._estimate_tokens(code_text)
        simple_tokens = self.strategy._estimate_tokens("hello world return")
        assert code_tokens > simple_tokens
    
    def test_code_detection(self):
        """Test code detection in messages."""
        # Code block
        assert self.strategy._contains_code("```python\nprint('hello')\n```")
        
        # Inline code
        assert self.strategy._contains_code("Use `print()` function")
        
        # Python keywords
        assert self.strategy._contains_code("def my_function():")
        assert self.strategy._contains_code("import os")
        
        # JavaScript keywords
        assert self.strategy._contains_code("const x = 5;")
        assert self.strategy._contains_code("function test() {}")
        
        # Regular text
        assert not self.strategy._contains_code("This is just normal text.")
    
    @pytest.mark.asyncio
    async def test_prepare_context_empty(self):
        """Test context preparation with empty message history."""
        messages = []
        prompt = "Hello, how are you?"
        
        result = await self.strategy.prepare_context(messages, prompt)
        
        assert len(result) == 1
        assert result[0] == ("human", prompt)
    
    @pytest.mark.asyncio
    async def test_prepare_context_recent_messages(self):
        """Test context preparation with only recent messages."""
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
            HumanMessage(content="Show me an example."),
            AIMessage(content="```python\nprint('Hello World')\n```"),
        ]
        prompt = "Thanks!"
        
        result = await self.strategy.prepare_context(messages, prompt)
        
        # Should have all messages + new prompt (no summarization needed)
        assert len(result) == 5
        assert result[-1] == ("human", prompt)
        assert any("python" in content.lower() for role, content in result)
    
    @pytest.mark.asyncio
    async def test_should_summarize(self):
        """Test summarization trigger logic."""
        # Few messages - no summarization
        few_messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        assert not await self.strategy.should_summarize(few_messages)
        
        # Many messages - should summarize
        many_messages = [HumanMessage(content=f"Message {i}") for i in range(15)]
        assert await self.strategy.should_summarize(many_messages)
    
    def test_select_important_messages_prioritizes_code(self):
        """Test that code-containing messages get higher priority."""
        messages = [
            HumanMessage(content="Just chatting about weather"),
            HumanMessage(content="def calculate_sum(a, b):\n    return a + b"),
            HumanMessage(content="More casual conversation"),
            HumanMessage(content="```python\nprint('Hello')\n```"),
            HumanMessage(content="Random text here"),
        ]
        
        important = self.strategy._select_important_messages(messages)
        
        # Code messages should be selected
        code_contents = [msg.content for msg in important]
        assert any("def calculate_sum" in content for content in code_contents)
        assert any("```python" in content for content in code_contents)
    
    @pytest.mark.asyncio
    async def test_token_limit_trimming(self):
        """Test that context gets trimmed to fit token limits."""
        # Create messages that exceed token limit
        long_messages = [
            ("human", "x" * 200),  # ~50 tokens each
            ("ai", "y" * 200),
            ("human", "z" * 200),
            ("ai", "w" * 200),
            ("human", "final prompt")
        ]
        
        # Set very low token limit
        self.strategy.config.max_context_tokens = 100
        
        result = await self.strategy._trim_to_token_limit(long_messages)
        
        # Should keep the final prompt and trim from beginning
        assert result[-1] == ("human", "final prompt")
        assert len(result) < len(long_messages)
        
        # Verify total tokens is within limit
        total_tokens = sum(self.strategy._estimate_tokens(content) for _, content in result)
        assert total_tokens <= self.strategy.config.max_context_tokens


if __name__ == "__main__":
    pytest.main([__file__])