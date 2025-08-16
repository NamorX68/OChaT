# OChaT Project Guidelines

## Project Overview

**OChaT** is a modular Python TUI (Text User Interface) application for controlling Large Language Models (LLMs) via LangChain. The application provides a comprehensive interface for managing LLM providers, models, and chat interactions through both CLI and TUI interfaces.

### Key Features
- **Provider Management**: Configure and manage LLM providers (OpenAI, Ollama, etc.)
- **Model Management**: Create, update, delete, and synchronize LLM models
- **Chat Interface**: Interactive chat sessions with LLM models
- **Database Integration**: SQLite database with SQLModel ORM
- **Migration Support**: Alembic-based database migrations
- **Extensible Architecture**: Layered architecture with clear separation of concerns

## Project Structure

```
OChaT/
├── src/ocht/                    # Main application package
│   ├── adapters/               # External service adapters (Ollama, etc.)
│   ├── cli.py                  # Command-line interface
│   ├── core/                   # Core functionality
│   │   ├── db.py              # Database configuration and session management
│   │   ├── models.py          # SQLModel database models
│   │   ├── migration.py       # Migration utilities
│   │   └── version.py         # Version management
│   ├── repositories/          # Data access layer (Repository pattern)
│   │   ├── llm_provider_config.py
│   │   ├── model.py
│   │   ├── message.py
│   │   ├── prompt_template.py
│   │   ├── setting.py
│   │   └── workspace.py
│   ├── services/              # Business logic layer (Service pattern)
│   │   ├── chat.py
│   │   ├── config.py
│   │   ├── model_manager.py
│   │   ├── prompt_manager.py
│   │   ├── provider_manager.py
│   │   └── workspace.py
│   └── tui/                   # Text User Interface
│       ├── app.py             # Main TUI application
│       ├── screens/           # TUI screens
│       │   ├── model_manager.py
│       │   ├── model_selector.py
│       │   ├── provider_manager.py
│       │   └── provider_selector.py
│       ├── styles/            # CSS-like styling for TUI
│       └── widgets/           # Custom TUI widgets
├── tests/                     # Test suite
├── migrations/                # Alembic database migrations
├── data/                      # Application data (SQLite database)
├── docs/                      # Documentation
└── pyproject.toml            # Project configuration and dependencies
```

## Architecture

The project follows a **layered architecture** with clear separation of concerns:

### 1. **Repository Layer** (`src/ocht/repositories/`)
- **Purpose**: Pure data access operations (CRUD)
- **Responsibilities**: Database queries, data persistence
- **Pattern**: Repository pattern with SQLModel/SQLAlchemy
- **Dependencies**: Only database models and SQLModel

### 2. **Service Layer** (`src/ocht/services/`)
- **Purpose**: Business logic and validation
- **Responsibilities**: Complex operations, validation, external API integration
- **Pattern**: Service pattern with business logic encapsulation
- **Dependencies**: Repository layer, external adapters

### 3. **Presentation Layer** (`src/ocht/tui/` and `src/ocht/cli.py`)
- **Purpose**: User interface and interaction
- **Responsibilities**: User input handling, display logic, navigation
- **Pattern**: Screen/Command pattern
- **Dependencies**: Service layer (NOT repository layer directly)

### 4. **Core Layer** (`src/ocht/core/`)
- **Purpose**: Shared utilities and models
- **Responsibilities**: Database models, configuration, utilities
- **Dependencies**: Minimal external dependencies

## Development Workflow

### Code Style and Conventions

1. **Python Style**:
   - Follow PEP 8 conventions
   - Use type hints for all function parameters and return values
   - Prefer descriptive variable names over abbreviations
   - Use docstrings for all public functions and classes

2. **Architecture Rules**:
   - **TUI/CLI → Service Layer → Repository Layer** (strict layering)
   - Never import repository functions directly in TUI/CLI code
   - Service layer functions should include validation and error handling
   - Repository functions should be pure data access operations

3. **Naming Conventions**:
   - Service functions: `verb_noun_with_validation()` (e.g., `create_model_with_validation`)
   - Repository functions: `verb_noun()` (e.g., `create_model`)
   - TUI screens: `NounManagerScreen`, `NounEditScreen`
   - Database models: PascalCase (e.g., `Model`, `LLMProviderConfig`)

4. **Error Handling**:
   - Service layer should raise `ValueError` for validation errors
   - TUI screens should catch and display errors appropriately
   - Repository layer should handle database-specific errors

## Testing Requirements

### Test Strategy
Junie **MUST** run tests to verify correctness of proposed solutions.

### Test Types and Locations

1. **Unit Tests** (`tests/`):
   - Database model tests
   - Repository function tests
   - Service layer business logic tests
   - Run with: `python -m pytest tests/`

2. **Integration Tests**:
   - Create temporary test scripts for complex changes
   - Test TUI screen functionality
   - Test service layer integration
   - Remove test scripts after verification

### Test Execution Commands

```bash
# Run all unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/model_tests.py

# Create and run integration test (example)
python test_integration.py
rm test_integration.py  # Clean up after testing
```

### Test Requirements for Changes

1. **Repository Changes**: Must pass existing repository tests
2. **Service Layer Changes**: Create integration tests to verify business logic
3. **TUI Changes**: Create functional tests to verify screen behavior
4. **Architecture Changes**: Create comprehensive tests covering all layers

## Build and Deployment

### Package Manager
The project uses **uv** as the primary package manager and build tool. Only use pip as a fallback when uv is not available or doesn't work.

### Development Setup
```bash
# Install dependencies (preferred - using uv)
uv sync

# Alternative: Install dependencies (fallback - using pip)
pip install -e .

# Initialize database
python -c "from ocht.core.db import init_db; init_db()"

# Run migrations (if needed)
python -c "from ocht.core.migration import migrate_to; migrate_to('head')"
```

### Build Process
```bash
# Build package (preferred - using uv)
uv build

# Alternative: Build package (fallback - using pip/build)
python -m build

# Install locally (preferred - using uv)
uv pip install -e .

# Alternative: Install locally (fallback - using pip)
pip install -e .
```

### Development Dependencies
```bash
# Install development dependencies (preferred - using uv)
uv sync --group dev

# Alternative: Install development dependencies (fallback - using pip)
pip install -e ".[dev]"
```

### No Build Required for Testing
- The project doesn't require building before testing
- Tests can be run directly on source code
- Use `uv sync` for development installation (preferred) or `pip install -e .` as fallback

## Junie-Specific Instructions

### When Making Changes

1. **Always Follow Architecture**:
   - Respect the layered architecture
   - Don't bypass the service layer in TUI/CLI code
   - Keep repository functions focused on data access

2. **Testing Protocol**:
   - Run existing tests: `python -m pytest tests/`
   - Create integration tests for complex changes
   - Test TUI functionality with temporary test scripts
   - Clean up test files after verification

3. **Code Quality Checks**:
   - Ensure all functions have proper type hints
   - Add docstrings for new public functions
   - Follow naming conventions consistently
   - Validate error handling patterns

4. **Database Changes**:
   - If modifying models, consider migration needs
   - Test database operations thoroughly
   - Ensure foreign key relationships are maintained

### Common Patterns to Follow

1. **Service Layer Functions**:
```python
def create_entity_with_validation(name: str, **kwargs) -> Entity:
    """Creates entity with business logic validation."""
    # Validation logic
    # Database operations via repository
    # Return created entity
```

2. **TUI Screen Integration**:
```python
from ocht.services.entity_manager import (
    get_entities_with_info,
    create_entity_with_validation,
    update_entity_with_validation,
    delete_entity_with_checks
)
```

3. **Error Handling in TUI**:
```python
try:
    result = service_function(...)
    self.notify("Success message", severity="information")
except ValueError as e:
    self.notify(str(e), severity="error")
except Exception as e:
    self.notify(f"Unexpected error: {str(e)}", severity="error")
```

### Files to Pay Special Attention To

- **Database Models** (`src/ocht/core/models.py`): Changes affect entire system
- **Service Layer** (`src/ocht/services/`): Core business logic
- **TUI Screens** (`src/ocht/tui/screens/`): User interface consistency
- **Repository Layer** (`src/ocht/repositories/`): Data integrity

### Quality Assurance

Before submitting changes:
1. ✅ All existing tests pass
2. ✅ New functionality is tested
3. ✅ Architecture principles are followed
4. ✅ Code style is consistent
5. ✅ Error handling is appropriate
6. ✅ Documentation is updated if needed

This project emphasizes **clean architecture**, **comprehensive testing**, and **maintainable code**. Always prioritize these principles when making changes.
