# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OChaT is a modular Python TUI application that orchestrates Large Language Models (LLMs) via LangChain. It supports both local models (Ollama) and cloud providers (ChatGPT, Claude, Grok).

## Development Commands

### Package Management
- `uv sync` - Install dependencies and sync environment
- `uv run ocht [command]` - Run CLI commands in development mode
- `uv run pytest` - Run tests

### Linting
- `uv run ruff check .` - Lint the codebase
- Run this after writing or modifying any Python code and fix all reported findings before
  considering the task complete. Ruleset (`[tool.ruff]` in `pyproject.toml`): pycodestyle/pyflakes
  (`E`, `F`), import sorting (`I`), bugbear (`B`), Google-style docstring checks (`D`), and
  pyupgrade (`UP` - modern syntax such as `X | None` instead of `Optional[X]`, `list[str]`
  instead of `List[str]`).
- `ruff format` is available but not enforced yet - the codebase has not been migrated to it, so
  don't run it wholesale; keep new/edited code consistent with the surrounding style by hand.
- Python target is 3.12 (`requires-python = ">=3.12"`, `.python-version`); prefer modern syntax
  (`X | None`, `list[str]`/`dict[str, str]`) over `typing.Optional`/`Union`/`List` (Ruff `UP` rules
  enforce this).

### Dependency Upgrades

- `uv audit` - Check the locked dependency set for known CVEs (uses the OSV feed; no `pip-audit`
  needed).
- The LangChain family (`langchain`, `langchain-core`, `langchain-openai`, `langchain-ollama`,
  `langchain-text-splitters`) is on the **1.x line** (`>=1.0.0,<2` in `pyproject.toml`). Migration
  history:
  - **0.3.x CVE patches (done, superseded):** an interim stage that bumped within 0.3.x to close
    every CVE with a 0.3.x backport, before the full 1.x migration below.
  - **`textual` upgrade (done):** bumped `3.2.0` -> `8.2.8` (5 majors). None of the documented
    breaking changes across those majors (`Widget.anchor` semantics, `Static`/`Label`
    `renderable` -> `content`, `HeaderTitle` reactives, Markdown component-class moves,
    `Select.BLANK` -> `Select.NULL`) touched this codebase - confirmed by grep before the bump and
    by the headless smoke tests in `tests/tui_smoke_tests.py` (uses Textual's `App.run_test()`
    Pilot API, which runs without a real terminal - safe for Claude Code to run, unlike
    `uv run ocht` itself, see Warnings section). Those smoke tests only mount the app and open/close
    each management screen against an empty DB - they are not a substitute for a human clicking
    through the real app, which still needs to happen once after any further TUI changes.
  - **LangChain 1.x migration (done):** `langchain.schema` and `langchain.memory` no longer exist
    in the `langchain` package as of 1.0. Fixes applied in `src/ocht/adapters/`:
    - Message classes (`HumanMessage`, `AIMessage`, `SystemMessage`, `BaseMessage`) and
      `BaseLanguageModel` now import from `langchain_core.messages` /
      `langchain_core.language_models` instead of the removed `langchain.schema*` paths.
    - `ConversationSummaryMemory` is gone from `langchain.memory`. Rather than pulling in
      `langchain_classic` (LangChain's official-but-deprecated compatibility package for exactly
      this class), `HybridMemoryStrategy._summarize_with_llm()` (`adapters/memory.py`) now makes
      one direct `llm.ainvoke()` call with our own summarization prompt - see that method's
      docstring for the reasoning.
    - This also fixed a real, pre-existing wiring bug found while doing the migration: each
      adapter used to keep its own *second*, outer `ConversationSummaryMemory` purely as a message
      store, but that class's `load_memory_variables()` always collapses history down to a single
      pre-summarized message (by design - it maintains one running summary string, not a message
      list). That single message was what got fed into `HybridMemoryStrategy.prepare_context()`,
      so `HybridMemoryStrategy`'s own retention/summarization logic (recent-message keeping,
      code-block priority, token trimming) never saw more than one message and was effectively
      dead code in production - only exercised directly in `tests/memory_tests.py`, never through
      the real adapter call path. The adapters (`ollama.py`, `openai_compatible.py`) now keep a
      plain `self._history: list[BaseMessage]` instead of a second memory object, so
      `HybridMemoryStrategy` receives the real, growing history. Regression coverage:
      `tests/ollama_adapter_tests.py`.
    - `use_hybrid_memory`/`memory` constructor parameters were removed from both adapters (the
      `use_hybrid_memory=False` legacy path was dead - never called anywhere in the codebase).
    - `langchain-openai`'s new "Responses API" default (which restructures `.content` into content
      blocks instead of a plain string) only activates for OpenAI's own newer endpoints - verified
      live against both Ollama and OpenRouter that `.content` stays a plain `str` with our custom
      `base_url` setups, so no `output_version="v0"` workaround was needed.
    - `langchain` 1.x now hard-depends on `langgraph` (agent orchestration framework) even though
      this project only uses the plain chat-model client - unavoidable extra weight from staying
      on `langchain`, unrelated to anything we do with it.
    - `uv audit` confirms all previously-open LangChain-family CVEs are closed at the 1.x
      versions now resolved, with no new CVEs introduced by `langgraph`/`langgraph-checkpoint`/
      `langgraph-prebuilt`/`langgraph-sdk`/`langchain-protocol`/`ormsgpack`.
  - Transitive dependencies with CVEs but no direct import in this codebase (`langsmith`, `mako`,
    `pygments`, `requests`, `urllib3`) are floored via `[tool.uv].constraint-dependencies` instead
    of being added to `dependencies`, since we don't call their APIs directly.

### Provider Routing Preferences (`LLMProviderConfig.prov_params`)

- `prov_params` is a JSON object of OpenAI-compatible "extra body" routing preferences, forwarded
  verbatim to the provider via `ChatOpenAI(extra_body={"provider": <prov_params>})`
  (`_build_openai_compatible_params()` in `services/adapter_manager.py`). Only wired for the
  `openai`/`lm studio`/`openrouter` branch (`OpenAICompatibleAdapter`) - not for Ollama, whose
  equivalent tuning knobs (`options`) have different semantics and aren't wired up.
- For OpenRouter specifically, this is their [provider routing
  object](https://openrouter.ai/docs/guides/routing/provider-selection) - e.g.
  `quantizations` (list of acceptable quantization levels) and `preferred_min_throughput`
  (tokens/sec; a soft preference that deprioritizes slower endpoints rather than excluding them -
  unlike `quantizations`, which hard-excludes and can 404 a request if no endpoint matches).
- The OpenRouter provider row in this project's DB is currently set to
  `{"quantizations": ["int8", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown"],
  "preferred_min_throughput": 40}` - i.e. "q8-or-better, or undisclosed quantization." `unknown`
  is deliberately included: 3 of the 7 configured OpenRouter models (`qwen/qwen3.8-flash`,
  `qwen/qwen3.8-max`, `deepseek/deepseek-v4-flash-vision-exp`) only have endpoints with undisclosed
  quantization today, so excluding `unknown` would 404 every request for those models. Check
  `GET /api/v1/models/{model}/endpoints` before tightening this filter.
- A malformed `prov_params` value is ignored gracefully (falls back to default provider routing)
  rather than blocking the adapter switch - see `_build_openai_compatible_params()`.
- Editable via the UI: `tui/screens/provider_manager.py`'s provider edit form has a "Routing
  Params" field. Validated as JSON before saving (`services/provider_manager.py`'s
  `_validate_provider_params()`); clearing the field explicitly clears `prov_params` (empty string
  is unambiguous - a JSON string can never legitimately be empty - see
  `update_llm_provider_config()`'s docstring for the `None` vs `""` distinction).

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