# Memory System Documentation

The OChaT Memory System provides intelligent conversation context management for LLM adapters. It combines recent message retention, code-aware prioritization, smart summarization, and token-aware context management.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [MemoryConfig](#memoryconfig)
- [MemoryStrategy Interface](#memorystrategy-interface)
- [HybridMemoryStrategy](#hybridmemorystrategy)
- [Code Detection System](#code-detection-system)
- [Message Scoring Algorithm](#message-scoring-algorithm)
- [Token Management](#token-management)
- [Integration with Adapters](#integration-with-adapters)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Testing](#testing)

## Overview

The Memory System solves the challenge of maintaining relevant conversation context within LLM token limits. Traditional approaches either keep all messages (exceeding token limits) or use simple truncation (losing important context).

### Key Features

- **Hybrid Context Management**: Combines recent message retention with selective older message preservation
- **Code-Aware Prioritization**: Automatically detects and prioritizes code-containing messages
- **Smart Summarization**: Uses LangChain's ConversationSummaryMemory for intelligent context compression
- **Token-Aware Management**: Ensures context fits within configurable token limits
- **Adaptive Thresholds**: Adjusts selection criteria based on conversation type

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   MemoryConfig  │    │ MemoryStrategy   │    │ Code Detection │
│                 │    │   (Interface)    │    │    System      │
├─────────────────┤    ├──────────────────┤    ├────────────────┤
│ max_context_    │    │ prepare_context()│    │ _contains_code()│
│   tokens        │    │ should_summarize│    │ Pattern Matching│
│ recent_messages │    │ _estimate_tokens │    │ Multi-Language  │
│ code_retention  │    │ _convert_message │    │ Support         │
│ summarization   │    │                  │    │                 │
│   threshold     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │HybridMemoryStrategy│
                    ├──────────────────┤
                    │ Message Scoring  │
                    │ Selection Logic  │
                    │ Summarization    │
                    │ Token Trimming   │
                    └──────────────────┘
```

## MemoryConfig

Configuration class that controls memory system behavior.

### Class Definition

```python
@dataclass
class MemoryConfig:
    max_context_tokens: int = 4000
    recent_messages_count: int = 10
    code_retention_priority: float = 2.0
    summarization_threshold: int = 20
```

### Parameters

#### `max_context_tokens: int = 4000`
- **Purpose**: Maximum number of tokens allowed in the prepared context
- **Impact**: Prevents exceeding LLM token limits
- **Typical Values**: 
  - Small models: 2000-4000
  - Large models: 8000-16000
  - GPT-4: 32000+
- **Behavior**: Context is trimmed if this limit is exceeded

#### `recent_messages_count: int = 10`
- **Purpose**: Number of most recent messages to always keep completely
- **Impact**: Ensures immediate conversation context is preserved
- **Typical Values**: 5-15 messages
- **Behavior**: These messages are never summarized or filtered out

#### `code_retention_priority: float = 2.0`
- **Purpose**: Score bonus for messages containing code
- **Impact**: Makes code messages more likely to be retained
- **Typical Values**: 1.5-3.0
- **Behavior**: Added to message score if code is detected

#### `summarization_threshold: int = 20`
- **Purpose**: Minimum message count before summarization begins
- **Impact**: Controls when conversation history gets compressed
- **Typical Values**: 15-30 messages
- **Behavior**: No summarization occurs below this threshold

### Configuration Examples

```python
# Default configuration (balanced)
default_config = MemoryConfig()

# Code-heavy conversations
code_config = MemoryConfig(
    max_context_tokens=6000,
    recent_messages_count=15,
    code_retention_priority=3.0,
    summarization_threshold=25
)

# Chat-focused conversations  
chat_config = MemoryConfig(
    max_context_tokens=3000,
    recent_messages_count=8,
    code_retention_priority=1.5,
    summarization_threshold=15
)

# Resource-constrained environments
minimal_config = MemoryConfig(
    max_context_tokens=2000,
    recent_messages_count=5,
    code_retention_priority=2.0,
    summarization_threshold=10
)
```

## MemoryStrategy Interface

Abstract base class defining the memory management contract.

### Class Definition

```python
class MemoryStrategy(ABC):
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
    
    @abstractmethod
    async def prepare_context(self, messages: List[BaseMessage], new_prompt: str) -> List[Tuple[str, str]]:
        """Prepare conversation context for LLM call."""
        pass
    
    @abstractmethod
    async def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Determine if conversation should be summarized."""
        pass
```

### Core Methods

#### `prepare_context(messages, new_prompt) -> List[Tuple[str, str]]`
- **Purpose**: Transform message history into LLM-ready context
- **Input**: 
  - `messages`: Historical conversation messages
  - `new_prompt`: Current user prompt
- **Output**: List of (role, content) tuples
- **Implementation**: Must handle summarization, selection, and token management

#### `should_summarize(messages) -> bool`
- **Purpose**: Determine if summarization should occur
- **Input**: Current message list
- **Output**: Boolean indicating summarization need
- **Implementation**: Based on message count and content analysis

### Utility Methods

#### `_estimate_tokens(text) -> int`
- **Purpose**: Estimate token count for text
- **Algorithm**: Enhanced character-based estimation with code detection
- **Formula**: `base_tokens * code_factor - whitespace_adjustment`

#### `_contains_code(text) -> bool`
- **Purpose**: Detect code content in messages
- **Implementation**: Multi-pattern regex matching
- **Patterns**: Code blocks, inline code, keywords, punctuation

## HybridMemoryStrategy

The main implementation of the memory strategy interface.

### Core Algorithm

The HybridMemoryStrategy uses a three-tier approach:

1. **Recent Messages**: Always kept completely (last N messages)
2. **Important Older Messages**: Selectively retained based on scoring
3. **Summary**: Compressed representation of remaining older messages

### Message Processing Flow

```
Input Messages
      │
      ▼
┌─────────────┐
│   Split     │
│ Recent vs   │ ── recent_messages_count
│ Older       │
└─────────────┘
      │
      ▼
┌─────────────┐    ┌──────────────┐
│   Recent    │    │    Older     │
│ (Keep All)  │    │ (Process)    │
└─────────────┘    └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Summarization│
                   │   +          │
                   │ Selection    │
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Combine All  │
                   │ + New Prompt │
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Token Limit  │
                   │ Trimming     │
                   └──────────────┘
```

### Implementation Details

```python
async def prepare_context(self, messages: List[BaseMessage], new_prompt: str) -> List[Tuple[str, str]]:
    if not messages:
        return [("human", new_prompt)]
    
    # Split messages
    total_messages = len(messages)
    recent_cutoff = max(0, total_messages - self.config.recent_messages_count)
    older_messages = messages[:recent_cutoff]
    recent_messages = messages[recent_cutoff:]
    
    context_tuples = []
    
    # Process older messages
    if older_messages:
        # Add summary
        summary_text = await self._get_or_create_summary(older_messages)
        if summary_text:
            context_tuples.append(("system", f"Previous conversation summary: {summary_text}"))
        
        # Add important older messages
        important_older = self._select_important_messages(older_messages)
        for msg in important_older:
            role, content = self._convert_message_to_tuple(msg)
            context_tuples.append((role, content))
    
    # Add recent messages
    for msg in recent_messages:
        role, content = self._convert_message_to_tuple(msg)
        context_tuples.append((role, content))
    
    # Add new prompt
    context_tuples.append(("human", new_prompt))
    
    # Ensure token compliance
    context_tuples = await self._trim_to_token_limit(context_tuples)
    
    return context_tuples
```

## Code Detection System

The code detection system uses multi-pattern regex matching to identify code content.

### Patterns

```python
code_patterns = [
    r'```[\s\S]*?```',  # Code blocks (```python ... ```)
    r'`[^`\n]+`',       # Inline code (`variable`)
    r'\b(def|class|function|import|from|return)\b',  # Python keywords
    r'\b(async|await|const|let|var|function)\b',     # JavaScript keywords
    r'[{}();]',         # Common code punctuation
    r'=\s*["\']',       # Assignment patterns (= "value")
]
```

### Detection Logic

```python
def _contains_code(self, text: str) -> bool:
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True  # Early return for performance
    return False
```

### Examples

| Text | Detected | Reason |
|------|----------|--------|
| `print("hello")` | ✅ | Inline code |
| `def calculate():` | ✅ | Python keyword |
| `const x = 5;` | ✅ | JavaScript keyword |
| `{key: value}` | ✅ | Code punctuation |
| `name = "John"` | ✅ | Assignment pattern |
| `Just normal text` | ❌ | No patterns match |

## Message Scoring Algorithm

Messages are scored to determine retention priority.

### Scoring Formula

```
Message Score = Code Bonus + Length Bonus

Where:
- Code Bonus = code_retention_priority if code detected, else 0
- Length Bonus = min(character_count / 1000, 1.0)
```

### Scoring Examples

| Message Type | Length | Code Bonus | Length Bonus | Total Score |
|--------------|--------|------------|--------------|-------------|
| Short text | 20 chars | +0.0 | +0.02 | **0.02** |
| Long text | 1500 chars | +0.0 | +1.0 | **1.0** |
| Short code | 30 chars | +2.0 | +0.03 | **2.03** |
| Long code | 800 chars | +2.0 | +0.8 | **2.8** |

### Selection Algorithm

```python
def _select_important_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
    # Score all messages
    scored_messages = [(calculate_score(msg), msg) for msg in messages]
    
    # Sort by score (highest first)
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate limits
    max_important = min(5, max(1, len(scored_messages) // 2))
    
    # Adaptive threshold
    has_code = any(score >= self.config.code_retention_priority for score, _ in scored_messages)
    min_score = 0.5 if has_code else 1.0
    
    # Apply selection
    return [msg for score, msg in scored_messages[:max_important] if score >= min_score]
```

### Selection Limits

- **Maximum Selection**: `min(5, len(messages) // 2)` 
  - Never more than 5 messages
  - Never more than 50% of older messages
  - Always at least 1 message if any exist

- **Adaptive Threshold**:
  - **With code messages**: `min_score = 0.5` (lower threshold)
  - **Without code**: `min_score = 1.0` (higher threshold)

## Token Management

### Token Estimation

```python
def _estimate_tokens(self, text: str) -> int:
    if not text:
        return 0
    
    # Base estimation (1 token ≈ 4 characters)
    base_tokens = len(text) // 4
    
    # Code adjustment (more tokens per character)
    if self._contains_code(text):
        base_tokens = int(base_tokens * 1.3)
    
    # Whitespace adjustment (whitespace doesn't count as tokens)
    whitespace_chars = len(re.findall(r'\s', text))
    adjusted_tokens = base_tokens - (whitespace_chars // 8)
    
    return max(1, adjusted_tokens)
```

### Token Limit Trimming

```python
async def _trim_to_token_limit(self, context_tuples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    total_tokens = sum(self._estimate_tokens(content) for _, content in context_tuples)
    
    if total_tokens <= self.config.max_context_tokens:
        return context_tuples
    
    # Always keep new prompt
    trimmed = context_tuples[-1:]
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
```

### Trimming Priority

1. **New Prompt**: Never removed
2. **Recent Messages**: Highest priority
3. **Important Older Messages**: Medium priority
4. **Summary**: Lowest priority (removed first)

## Integration with Adapters

### OllamaAdapter Integration

```python
class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = "qwen3:30b-a3b",
        base_url: str = "http://localhost:11434",
        default_params: Optional[Dict[str, Any]] = None,
        use_hybrid_memory: bool = True,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.client = ChatOllama(model=model, base_url=base_url, **(default_params or {}))
        
        if use_hybrid_memory:
            self.memory_strategy = HybridMemoryStrategy(
                config=memory_config or MemoryConfig(),
                llm=self.client
            )
        else:
            self.memory_strategy = None
        
        # Legacy memory for compatibility
        self.memory = ConversationSummaryMemory(
            llm=self.client,
            return_messages=True,
            output_key="output"
        )
```

### Usage in send_prompt_async

```python
async def send_prompt_async(self, prompt: str, **kwargs) -> str:
    if self.memory_strategy:
        # Use HybridMemoryStrategy
        history_messages = self.memory.load_memory_variables({})["history"]
        messages = await self.memory_strategy.prepare_context(history_messages, prompt)
    else:
        # Legacy method
        messages = await self._prepare_messages(prompt)
    
    response = await self.client.ainvoke(messages, **kwargs)
    await self._save_to_memory(prompt, response.content)
    return response.content
```

## Usage Examples

### Basic Usage

```python
from ocht.adapters.memory import HybridMemoryStrategy, MemoryConfig
from ocht.adapters.ollama import OllamaAdapter

# Create adapter with hybrid memory
adapter = OllamaAdapter(
    model="llama2",
    use_hybrid_memory=True,
    memory_config=MemoryConfig(
        max_context_tokens=4000,
        recent_messages_count=10,
        code_retention_priority=2.0
    )
)

# Use normally - memory is handled automatically
response = await adapter.send_prompt_async("Explain Python decorators")
```

### Custom Configuration

```python
# Configuration for code-heavy conversations
code_config = MemoryConfig(
    max_context_tokens=6000,      # More tokens for code context
    recent_messages_count=15,     # Keep more recent messages
    code_retention_priority=3.0,  # Higher code priority
    summarization_threshold=25    # Summarize later
)

adapter = OllamaAdapter(
    model="codellama",
    memory_config=code_config
)

# Configuration for chat conversations
chat_config = MemoryConfig(
    max_context_tokens=3000,      # Fewer tokens needed
    recent_messages_count=8,      # Fewer recent messages
    code_retention_priority=1.5,  # Lower code priority
    summarization_threshold=15    # Summarize earlier
)

adapter = OllamaAdapter(
    model="llama2",
    memory_config=chat_config
)
```

### Direct Memory Strategy Usage

```python
from langchain.schema import HumanMessage, AIMessage

# Create strategy directly
memory_strategy = HybridMemoryStrategy(
    config=MemoryConfig(),
    llm=your_llm_instance
)

# Prepare context manually
messages = [
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language..."),
    HumanMessage(content="Show me an example"),
    AIMessage(content="```python\nprint('Hello World')\n```"),
]

context = await memory_strategy.prepare_context(messages, "Explain functions")
```

## Performance Considerations

### Computational Complexity

- **Message Scoring**: O(n × p) where n = messages, p = patterns
- **Sorting**: O(n log n) for message selection
- **Token Estimation**: O(m) where m = message length
- **Total**: O(n × (p + log n + m))

### Memory Usage

- **Message Storage**: Original messages kept in memory until processed
- **Summary Caching**: Summaries cached to avoid regeneration
- **Pattern Compilation**: Regex patterns compiled once

### Optimization Techniques

1. **Early Return**: Code detection stops at first pattern match
2. **Pattern Ordering**: Most common patterns checked first
3. **Summary Caching**: Avoid re-summarization of unchanged history
4. **Lazy Evaluation**: Token estimation only when needed

### Performance Tips

```python
# Efficient: Reuse same strategy instance
strategy = HybridMemoryStrategy(config)
for prompt in prompts:
    context = await strategy.prepare_context(messages, prompt)

# Inefficient: Create new strategy each time
for prompt in prompts:
    strategy = HybridMemoryStrategy(config)  # Don't do this!
    context = await strategy.prepare_context(messages, prompt)
```

## Testing

The memory system includes comprehensive tests covering all major functionality.

### Test Categories

1. **Unit Tests**: Individual method testing
2. **Integration Tests**: Adapter integration
3. **Scenario Tests**: Real-world conversation patterns
4. **Performance Tests**: Token limits and large conversations

### Key Test Cases

```python
# Test code detection
def test_code_detection():
    assert strategy._contains_code("```python\nprint('hello')\n```")
    assert strategy._contains_code("def function():")
    assert not strategy._contains_code("Just normal text")

# Test message scoring
def test_message_scoring():
    code_msg = HumanMessage(content="def hello(): pass")
    text_msg = HumanMessage(content="Hello world")
    
    code_score = calculate_score(code_msg)
    text_score = calculate_score(text_msg)
    
    assert code_score > text_score

# Test token limits
def test_token_limits():
    long_context = [("human", "x" * 1000) for _ in range(10)]
    trimmed = await strategy._trim_to_token_limit(long_context, max_tokens=100)
    
    total_tokens = sum(strategy._estimate_tokens(content) for _, content in trimmed)
    assert total_tokens <= 100
```

### Running Tests

```bash
# Run all memory tests
uv run pytest tests/memory_tests.py -v

# Run specific test
uv run pytest tests/memory_tests.py::TestHybridMemoryStrategy::test_code_detection -v

# Run with coverage
uv run pytest tests/memory_tests.py --cov=src/ocht/adapters/memory
```

## Troubleshooting

### Common Issues

#### Memory Not Using Hybrid Strategy

**Problem**: Adapter still using legacy memory system

**Solution**: Ensure `use_hybrid_memory=True` in adapter constructor

```python
# Wrong
adapter = OllamaAdapter(model="llama2")  # Uses legacy memory

# Correct  
adapter = OllamaAdapter(model="llama2", use_hybrid_memory=True)
```

#### Context Too Large

**Problem**: Token limit exceeded errors

**Solution**: Reduce `max_context_tokens` or increase `summarization_threshold`

```python
config = MemoryConfig(
    max_context_tokens=2000,      # Reduce limit
    summarization_threshold=10    # Summarize earlier
)
```

#### Code Not Being Prioritized

**Problem**: Code messages not retained in context

**Solution**: Increase `code_retention_priority` or check code detection patterns

```python
# Check detection
strategy = HybridMemoryStrategy()
print(strategy._contains_code("your code here"))

# Increase priority
config = MemoryConfig(code_retention_priority=3.0)
```

#### No Summarization Occurring  

**Problem**: Conversation not being summarized

**Solution**: Check `summarization_threshold` and ensure LLM is provided

```python
# Ensure LLM is provided for summarization
strategy = HybridMemoryStrategy(config, llm=your_llm_instance)

# Lower threshold
config = MemoryConfig(summarization_threshold=10)
```

### Debug Mode

```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints to understand selection
selected = strategy._select_important_messages(messages)
print(f"Selected {len(selected)} out of {len(messages)} messages")
for msg in selected:
    print(f"- {msg.content[:50]}...")
```

## Future Enhancements

### Planned Features

1. **Custom Pattern Support**: User-defined code detection patterns
2. **Language-Specific Priority**: Different priorities for different programming languages
3. **Semantic Similarity**: Use embeddings for message similarity matching
4. **Dynamic Configuration**: Runtime adjustment based on conversation analysis
5. **Metrics Collection**: Performance and effectiveness tracking

### Extension Points

```python
# Custom memory strategy
class CustomMemoryStrategy(MemoryStrategy):
    async def prepare_context(self, messages, new_prompt):
        # Your custom logic here
        pass

# Custom code detection
class AdvancedCodeDetector:
    def contains_code(self, text):
        # Advanced detection logic
        pass

# Custom scoring
def custom_score_function(message, config):
    # Custom scoring algorithm
    pass
```

---

*This documentation covers OChaT Memory System v1.0. For the latest updates and examples, see the project repository.*