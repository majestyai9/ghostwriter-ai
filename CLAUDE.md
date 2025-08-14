# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
# Main application with Python 3.11 on Windows
"C:\Program Files\Python311\python.exe" main.py

# Get help
"C:\Program Files\Python311\python.exe" main.py --help
```

### Testing
```bash
# Run all tests
"C:\Program Files\Python311\python.exe" -m pytest tests/ -v

# Run unit tests only
"C:\Program Files\Python311\python.exe" -m pytest tests/unit/ -v

# Run integration tests
"C:\Program Files\Python311\python.exe" -m pytest tests/test_integration.py -v

# Run specific test
"C:\Program Files\Python311\python.exe" -m pytest tests/test_integration.py::test_book_generation_smoke -v -s

# Run with short traceback
"C:\Program Files\Python311\python.exe" -m pytest tests/ -v --tb=short
```

### Code Quality
```bash
# Lint with ruff (auto-fix issues)
"C:\Program Files\Python311\python.exe" -m ruff check . --fix

# Format code with black
black .

# Type checking with mypy
mypy .
```

### Development Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Architecture Overview

### Core Application Flow
The application follows a modular architecture with dependency injection:

1. **Entry Point** (`main.py`): Sets up event handlers, initializes services via DI container, and orchestrates book generation
2. **Service Layer** (`services/generation_service.py`): Main business logic for book generation, manages the flow between providers and components
3. **Provider System** (`providers/`): Abstracted LLM providers with a common base class that handles retries, rate limiting, and error handling
4. **Dependency Injection** (`containers.py`, `app_config.py`): Configuration and DI container setup using dependency-injector

### Key Components

#### LLM Provider Architecture
- **Base Provider** (`providers/base.py`): Abstract base class with centralized retry logic, exponential backoff, and error handling
- **Provider Implementations**: OpenAI, Anthropic, Cohere, Gemini, OpenRouter - all inherit from base
- **Factory Pattern** (`providers/factory.py`): Creates appropriate provider based on configuration
- **Token Counting**: Each provider uses official SDK methods for accurate token counting

#### Event System
- **Event Manager** (`events.py`): Pub/sub system for monitoring generation progress
- **Event Types**: `CHAPTER_COMPLETED`, `SECTION_COMPLETED`, `GENERATION_FAILED`, `API_CALL_FAILED`
- **Progress Tracking**: Real-time monitoring with optional webhook callbacks

#### Book Generation Pipeline
1. Title generation/translation
2. Table of contents generation (JSON format)
3. Chapter-by-chapter generation with context management
4. Section generation within chapters
5. Incremental saving after each step for recovery

#### Error Handling
- **Custom Exceptions** (`exceptions.py`): Hierarchical exception structure
- **Graceful Degradation**: Partial book saving, resume from interruptions
- **Retry Strategy**: Provider-level exponential backoff with jitter

### Important Patterns

#### Thread Safety
The codebase uses thread-safe implementations for:
- Background task processing (`background_tasks.py`)
- Cache management (`cache_manager.py`)
- Event system with proper locking

#### Context Management
- **Token Optimizer** (`token_optimizer.py`): Sliding window for managing conversation history
- **History Tracking**: Maintains proper chronological order in conversations
- **Safe JSON Parsing**: Robust extraction from LLM responses, handles markdown code blocks

#### Project Structure
Books are saved in project-based structure:
- `projects/<project-id>/content/` - Book JSON and markdown files
- `projects/<project-id>/exports/` - Generated EPUB, PDF, DOCX files
- `projects/<project-id>/cache/` - Project-specific cache

## Configuration

### Environment Variables
Create a `.env` file from `env.example` with:
- `LLM_PROVIDER`: Choose provider (openai, anthropic, cohere, gemini, openrouter)
- `*_API_KEY`: Provider-specific API keys
- `TEMPERATURE`: Generation creativity (0.0-1.0, default 0.2)
- `TOKEN_LIMIT`: Context window limit
- `BOOK_LANGUAGE`, `BOOK_TITLE`, `BOOK_INSTRUCTIONS`: Default book settings

### Python Version
The project targets Python 3.9+ but is developed on Python 3.11 (Windows environment uses Python 3.13.5).

## Testing Strategy

### Test Structure
- `tests/conftest.py`: Shared fixtures and test configuration
- `tests/test_integration.py`: End-to-end book generation tests
- `tests/unit/`: Unit tests for individual components

### Test Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Long-running tests

### Mock Strategy
Tests use mocked API calls to avoid network dependencies. Dummy API keys are provided in CI environment.

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):
1. Install dependencies (requirements.txt + requirements-dev.txt)
2. Check formatting with black
3. Lint with ruff
4. Type check with mypy
5. Run pytest suite

## Code Style

### Formatting Rules (pyproject.toml)
- Line length: 100 characters
- Black formatting for consistency
- Ruff linting with extended rules (E, F, W, I, N, UP, B, C4, SIM)
- MyPy for type checking (optional typing, ignore missing imports)

### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports for clarity

## Recent Refactoring (December 2024)

Major improvements made:
- Eliminated code duplication via `BookGenerator` class pattern
- Centralized retry logic in base provider class
- Fixed token counting accuracy across all providers
- Fixed conversation history ordering bug
- Added safe JSON parsing for various LLM response formats
- Improved thread safety and callback mechanisms