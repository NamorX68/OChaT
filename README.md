# OChaT

[![PyPI version](https://img.shields.io/pypi/v/ocht.svg)](https://pypi.org/project/ocht/) [![Build Status](https://github.com/dein-username/OChaT/actions/workflows/ci.yml/badge.svg)](https://github.com/dein-username/OChaT/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OChaT** is a sophisticated Python TUI application that orchestrates Large Language Models (LLMs) via LangChain. It provides a unified interface for both local models (Ollama, LM Studio) and cloud providers (OpenAI, ChatGPT) with advanced memory management and real-time streaming capabilities.

---

## ✨ Features

- **Multi-Provider Support**: Seamlessly switch between Ollama, LM Studio, and OpenAI
- **Real-Time Streaming**: Live chat interface with streaming responses
- **Advanced Memory Management**: Hybrid memory strategy with intelligent context management
- **Model Discovery**: Automatic model detection and synchronization from providers
- **Professional TUI**: Textual-based interface with custom widgets and styling
- **Workspace Management**: Isolated chat environments with persistent history
- **Database Architecture**: SQLModel-based persistence with migration support
- **CLI Integration**: Comprehensive command-line interface for all operations

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/dein-username/OChaT.git
   cd OChaT
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Install package in development mode**

   ```bash
   uv install -e .
   ```

> **Note:** By default, `uv sync` installs all dependencies from `pyproject.toml`, including:
>
> - `alembic>=1.15.2` - Database migrations
> - `click>=8.1.8` - CLI framework
> - `langchain>=0.3.26` - LLM orchestration
> - `langchain-ollama>=0.3.3` - Ollama integration
> - `langchain-openai>=0.3.30` - OpenAI integration
> - `ollama>=0.4.8` - Ollama client
> - `pyperclip>=1.8.2` - Clipboard operations
> - `rich>=14.0.0` - Terminal formatting
> - `sqlmodel>=0.0.24` - Database ORM
> - `textual>=3.2.0` - TUI framework

---

## ⚡ Quick Start

Launch the TUI application:

```bash
uv run ocht
```

Or use specific commands:

- **`uv run ocht init <workspace>`** - Create new workspace
- **`uv run ocht chat`** - Start interactive chat
- **`uv run ocht list-models`** - List available models
- **`uv run ocht sync-models`** - Sync model metadata
- **`uv run ocht sync-models --delete-missing`** - Sync models and remove unavailable ones
- **`uv run ocht config`** - Open configuration editor
- **`uv run ocht migrate <version>`** - Run database migrations

### Available CLI Commands

| Command                          | Description                                                      |
| -------------------------------- | ---------------------------------------------------------------- |
| `init <name>`                    | Creates a new chat workspace with configuration file and history |
| `chat`                           | Starts interactive chat session based on current workspace       |
| `config`                         | Opens configuration in default editor                            |
| `export-config <file>`           | Exports current settings as YAML or JSON file                    |
| `import-config <file>`           | Imports settings from YAML or JSON file                          |
| `list-models`                    | Lists available LLM models grouped by provider                   |
| `sync-models [--delete-missing]` | Synchronizes model metadata from external providers              |
| `migrate <version>`              | Runs Alembic migrations to specified target version              |
| `version`                        | Shows current CLI/package version                                |
| `help [command]`                 | Shows detailed help for a command                                |

<details>
<summary>Example Usage</summary>

```bash
# Create new chat workspace
uv run ocht init my-workspace

# Start chat session
uv run ocht chat

# List available models
uv run ocht list-models

# Sync model metadata with cleanup
uv run ocht sync-models --delete-missing

# View models grouped by provider
uv run ocht list-models
```

</details>

---

## 🏗️ Architecture

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

- LangChain integration with multiple providers
- `base.py` - Abstract LLMAdapter interface
- `ollama.py` - Ollama-specific implementation
- `openai_compatible.py` - Unified OpenAI/LM Studio adapter
- `memory.py` - Advanced memory management strategies

**TUI Layer (`tui/`)**

- Textual-based user interface
- `app.py` - Main TUI application with streaming support
- `screens/` - UI screens for provider/model/workspace management
- `widgets/` - Custom UI components (chat bubbles, dialogs, footer)
- `styles/` - TCSS styling files

### Supported Providers

**Ollama**

- Local model hosting via Ollama server
- Automatic model discovery and synchronization
- Support for all Ollama-compatible models
- Endpoint: `http://localhost:11434` (configurable)

**LM Studio**

- Local model hosting via LM Studio API
- OpenAI-compatible API interface
- Real-time model availability detection
- Endpoint: `http://localhost:1234/v1` (configurable)

**OpenAI**

- Cloud-based models (GPT-3.5, GPT-4, etc.)
- Full streaming and async support
- API key authentication required
- Endpoint: `https://api.openai.com/v1`

---

## 🧠 Memory Management

OChaT features an advanced HybridMemoryStrategy that intelligently manages conversation context:

### Memory Features

- **Recent Message Retention**: Always keeps the last 8-10 messages for immediate context
- **Code-Aware Prioritization**: Code blocks and technical discussions are retained longer
- **Smart Summarization**: Older messages are summarized while preserving key information
- **Token Management**: Automatic context trimming to fit model token limits
- **Function Reference Tracking**: Code functions and class names are indexed separately

### Memory Configuration

```python
# Default configuration
max_context_tokens: 4000        # Maximum tokens for context
recent_messages_count: 10       # Always keep last N messages
code_retention_priority: 2.0    # Higher = longer code retention
summarization_threshold: 20     # Start summarizing after N messages
```

---

## 🗂️ Project Structure

```text
OChaT/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── CLAUDE.md              # Claude Code project instructions
├── alembic.ini            # Alembic configuration
├── migrations/            # Database migration files
├── docs/
├── tests/                 # Test files
├── src/
│   └── ocht/
│       ├── __init__.py
│       ├── cli.py             # CLI entry point (Click commands)
│       ├── core/              # Database engine, sessions & models
│       │   ├── db.py          # Engine & session factory
│       │   ├── migration.py   # Alembic integration
│       │   ├── models.py      # SQLModel entities
│       │   └── version.py     # Version management
│       ├── repositories/      # CRUD logic per entity
│       │   ├── workspace.py
│       │   ├── message.py
│       │   ├── llm_provider_config.py
│       │   ├── model.py
│       │   ├── setting.py
│       │   └── prompt_template.py
│       ├── services/          # Business logic / use cases
│       │   ├── workspace.py
│       │   ├── chat.py
│       │   ├── config.py
│       │   ├── model_manager.py    # Enhanced with provider-specific sync
│       │   ├── provider_manager.py
│       │   ├── prompt_manager.py
│       │   ├── settings_manager.py
│       │   ├── workspace_manager.py
│       │   └── adapter_manager.py
│       ├── adapters/          # LangChain adapters
│       │   ├── base.py        # Abstract adapter interface
│       │   ├── memory.py      # HybridMemoryStrategy implementation
│       │   ├── ollama.py      # Ollama implementation
│       │   └── openai_compatible.py  # OpenAI/LM Studio unified adapter
│       ├── tui/               # Text-based UI components
│       │   ├── app.py         # Main TUI application with streaming
│       │   ├── screens/       # UI screens for management
│       │   │   ├── provider_manager.py
│       │   │   ├── provider_selector.py
│       │   │   ├── model_manager.py
│       │   │   ├── model_selector.py
│       │   │   ├── workspace_manager.py
│       │   │   ├── workspace_selector.py
│       │   │   └── settings_manager.py
│       │   ├── widgets/       # Custom widgets
│       │   │   ├── chat_bubble.py
│       │   │   ├── custom_footer.py
│       │   │   └── confirmation_dialog.py
│       │   └── styles/        # TCSS styling
│       └── data/              # SQLite database
└── uv.lock
```

## 🧭 Project layout

- **adapters/** — LangChain adapter implementations for each LLM provider (Ollama, OpenAI/LM Studio, Anthropic), plus a resilience layer with retry and circuit-breaker logic and a hybrid memory strategy.
- **core/** — Database engine and session management, SQLModel entity definitions, Alembic migration helper, and package version helper.
- **data/** — Runtime directory for the SQLite database file, created on first use under `DEFAULT_DB_PATH` (`src/ocht/data/ocht.db`) and overridable via the `DATABASE_URL` environment variable; not checked into the repository.
- **repositories/** — CRUD functions for each domain entity (workspace, message, LLM provider configuration, model, setting, prompt template).
- **services/** — Business logic and use cases: chat session management, workspace and configuration handling, model and provider synchronization, prompt and adapter management, settings, and model health checks.
- **tui/** — Textual-based user interface with the main `ChatApp` entry point, screens for provider/model/workspace/settings management, custom widgets (chat bubbles, footer, confirmation dialogs), and TCSS styling files.

---

## 🚀 Usage Examples

### Setting Up Providers

```bash
# Start the TUI to configure providers
uv run ocht

# Or sync models for existing providers
uv run ocht sync-models

# Sync and cleanup missing models
uv run ocht sync-models --delete-missing
```

### TUI Navigation

- **Ctrl+C**: Quit application
- **Ctrl+L**: Clear chat history
- **Escape**: Focus input field
- **Ctrl+Shift+C**: Copy last bot message
- **Ctrl+Shift+U**: Copy last user message

### Provider Switching

Within the TUI:

1. Use provider selector to switch between Ollama, LM Studio, and OpenAI
2. Model selector automatically filters available models for current provider
3. Changes are persisted automatically

---

## 🧪 Testing

Run tests using pytest:

```bash
uv run pytest
```

Test coverage includes:

- Database operations and migrations
- Memory management strategies
- Provider adapters and model synchronization
- CLI command functionality

See `### Running tests and the linter` under `## 🛠️ Development` for setup and linting details.

---

## 🛠️ Development

### Adding New Providers

1. Create adapter in `adapters/` extending `LLMAdapter`
2. Add provider configuration to `LLMProviderConfig`
3. Update `provider_manager.py` service
4. Add sync logic to `model_manager.py`
5. Add TUI screens if needed

### Database Changes

Use Alembic for schema modifications:

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migration
uv run ocht migrate head
```

### Running tests and the linter

Set up the development environment and run checks:

```bash
# Install dev dependencies
uv sync

# Run the test suite
uv run pytest

# Lint the codebase
uv run ruff check .
```

`ruff format` is not enforced yet in this project.

---

## ⚠️ Important Notes

- **TUI Mode**: Never run `uv run ocht` in automated scripts - it's an interactive TUI application
- **Debug Output**: In TUI mode, use `self.notify()` for debugging instead of print statements
- **Language**: All user-facing text is in English per project guidelines
- **Provider Dependencies**: Ensure Ollama or LM Studio is running locally before syncing models

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. **Create feature branch**:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Commit changes**:
   ```bash
   git commit -m "feat: description of my feature"
   ```
4. **Push to fork**:
   ```bash
   git push origin feature/my-feature
   ```
5. **Open Pull Request**

Please follow our coding guidelines:

- All imports at the beginning of modules
- Code documentation in English, including inline comments
- Docstrings in Google format
- Add tests for new features

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
