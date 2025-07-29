# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OChaT is a modular Python TUI application that orchestrates Large Language Models (LLMs) via LangChain. It supports both local models (Ollama) and cloud providers (ChatGPT, Claude, Grok).

## Development Commands

### Package Management
- `uv sync` - Install dependencies and sync environment
- `uv run ocht [command]` - Run CLI commands in development mode
- `uv run pytest` - Run tests

### Building and Distribution
- `uv build` - Build the package using setuptools
- `uv install -e .` - Install package in editable mode

### Running the Application
- `uv run ocht` - Launch default chat interface
- `uv run ocht init <workspace>` - Create new workspace
- `uv run ocht chat` - Start interactive chat
- `uv run ocht list-models` - List available models
- `uv run ocht sync-models` - Sync model metadata
- `uv run ocht migrate <version>` - Run database migrations

## Architecture Overview

### Core Components

**Database Layer (`core/`)**
- `models.py` - SQLModel entities: Workspace, Message, LLMProviderConfig, Model, Setting, PromptTemplate
- `db.py` - Database engine, session management, and initialization
- `migration.py` - Alembic integration for schema migrations

**Repository Layer (`repositories/`)**
- CRUD operations for each entity
- Direct database access abstraction
- Files: `workspace.py`, `message.py`, `llm_provider_config.py`, `model.py`, `setting.py`, `prompt_template.py`

**Service Layer (`services/`)**
- Business logic and use cases
- Orchestrates repositories and external APIs
- Files: `workspace.py`, `chat.py`, `config.py`, `model_manager.py`, `provider_manager.py`, `prompt_manager.py`

**Adapter Layer (`adapters/`)**
- LangChain integration
- `base.py` - Abstract LLMAdapter interface
- `ollama.py` - Ollama-specific implementation

**TUI Layer (`tui/`)**
- Textual-based user interface
- `app.py` - Main TUI application
- `screens/` - UI screens for model/provider management
- `widgets/` - Custom UI components (chat bubbles, etc.)
- `styles/` - TCSS styling files

### Data Models

**Primary Entities:**
- `Workspace` - Chat workspace container
- `Message` - Individual chat messages with role, content, metadata
- `LLMProviderConfig` - API credentials and provider settings
- `Model` - Available LLM models per provider
- `Setting` - Key-value configuration storage
- `PromptTemplate` - Reusable prompt templates

**Key Relationships:**
- Workspace → Messages (1:many)
- LLMProviderConfig → Models (1:many)
- Workspace → default_model (foreign key to LLMProviderConfig)
- Messages support threading via parent_id self-reference

### CLI Structure

CLI commands map to service layer functions:
- `init` → `workspace.create_workspace()`
- `chat` → `chat.start_chat()`
- `config` → `config.open_conf()`
- `list-models` → `model_manager.list_llm_models()`
- `sync-models` → `model_manager.sync_llm_models()`

### Database Configuration

- Default: SQLite at `data/ocht.db`
- Configurable via `DATABASE_URL` environment variable
- Uses SQLModel/SQLAlchemy for ORM
- Alembic for schema migrations

## Development Notes

### Database Initialization
The database is automatically initialized when first accessed. Use `init_db()` to create tables manually.

### Adding New Providers
1. Create adapter in `adapters/` extending `LLMAdapter`
2. Add provider configuration to `LLMProviderConfig`
3. Update `provider_manager.py` service
4. Add TUI screens if needed

### Testing
Tests are configured via pytest. Use `uv run pytest` to run the test suite.

### Workspace Management
Workspaces are self-contained chat environments with their own configuration and message history. Each workspace references a default model configuration.

## Warnings and Precautions
- Starte nie die App mit uv run ocht da es sich um eine TUI App handelt die du nicht steuern kannst!

## Development Guidelines
- Merke Debug in der TUI Anwendung nur mit self.notify() erstellen

## Translation Guidelines
- Alle Texte in der Anwendung in englisch erstellen.

## Adapter Roadmap & Next Steps

### Phase 1: Memory System Improvements (High Priority)
- [ ] Implement HybridMemoryStrategy
  - Keep last 8-10 messages completely (for code context)
  - Smart summarization for older messages
  - Code blocks retained longer than natural text
  - Function names/references separate indexing
  - Token-aware context management

### Phase 2: Configuration & Health Monitoring (Medium Priority)
- [ ] Provider-agnostic AdapterConfig class
  - temperature, max_tokens, context_window
  - streaming_enabled, memory_strategy
- [ ] Health check system
  - Model availability testing
  - Response time monitoring
  - Streaming capability validation
- [ ] Error recovery & retry logic
  - Exponential backoff
  - Circuit breaker pattern
  - Graceful degradation (stream → async → error)

### Phase 3: New Adapters (Medium Priority)
- [ ] OpenAI-compatible adapter (OpenAI, Groq, local APIs)
- [ ] MLX-LM adapter (Apple Silicon local models)
- [ ] Anthropic Claude adapter (API)

### Phase 4: Advanced Features (Low Priority)
- [ ] Context-aware parameter adjustment
- [ ] Multi-model conversation support
- [ ] Adapter performance metrics
- [ ] Custom memory strategies per use case

### Current Status (2025-07-29)
✅ Base LLMAdapter with async/sync/stream methods
✅ Enhanced OllamaAdapter with streaming support
✅ TUI streaming implementation with live updates
✅ Mouse escape sequence filtering
🟡 Memory system needs improvement (current: basic ConversationSummaryMemory)