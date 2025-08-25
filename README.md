# GhostWriter AI: Advanced Multi-Provider Book Writing System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Production-ready AI-powered book writing application with support for latest LLM models, advanced RAG capabilities, and enterprise-grade architecture patterns. Built with clean code principles, comprehensive error handling, and no backward compatibility constraints - focusing on modern, maintainable solutions.

## Project Overview

### Key Capabilities
- **Multi-Provider LLM Support**: OpenAI GPT-5, Anthropic Claude 4, Google Gemini 2.5, Cohere, OpenRouter
- **Production-Ready Infrastructure**: Circuit breakers, connection pooling, retry logic, distributed tracing
- **Advanced RAG System**: FAISS vector search, smart summarization, context optimization
- **CLI/TUI Interface**: Terminal-based operation, no web dependencies
- **Clean Architecture**: SOLID principles, dependency injection, event-driven design
- **Modern Python**: 3.9+ with type hints, async/await, Pydantic v2

### Current Status (January 2025)
- **43 tasks completed** - Core functionality operational with clean architecture
- **26 tasks pending** - See TODO.md for roadmap
- **Legacy code removed** - All backward compatibility eliminated (Jan 26)
- **Modern codebase** - No deprecated features, clean architecture

## Recent Improvements (January 2025)

### ✅ Legacy Code Cleanup (Jan 26, 2025)
- **Removed all backward compatibility code** - Clean, modern architecture
- **Deleted deprecated files**: `migrate_to_prompt_service.py`, `prompts_templated.py`, `FIXES_IMPLEMENTED.md`, `primer.md`
- **Eliminated legacy fallbacks** - All systems now use modern implementations
- **Updated RAG system** - Removed DISABLED mode, now always uses modern RAG
- **Cleaned test suite** - Removed backward compatibility tests

### ✅ Enhanced RAG System (Jan 25, 2025)
- **Hybrid search** - Combined dense (FAISS) and sparse (BM25, TF-IDF) retrieval
- **Knowledge graphs** - Entity relationships and context tracking
- **Incremental indexing** - Real-time content updates
- **Semantic caching** - Similarity-based retrieval optimization
- **Quality metrics** - Feedback loop system for continuous improvement

### ✅ Error Recovery & Resilience (Jan 25, 2025)
- **Distributed tracing** - OpenTelemetry integration for debugging
- **Saga pattern** - Multi-step transactional operations with rollback
- **Health monitoring** - Comprehensive health checks for all services
- **Fallback strategies** - 6 different content generation fallback methods
- **Dead letter queue** - Persistent storage and retry for failed operations

## Architecture & Design Patterns

### 1. Dependency Injection (containers.py)
- **Thread-safe singleton management** with double-checked locking
- **Service locator pattern** for centralized service access
- **Lazy initialization** for resource efficiency
- **Atomic operations** for concurrent safety

### 2. Provider Pattern (providers/)
- **Abstract base class** with circuit breaker and connection pooling
- **Exponential backoff retry** with jitter for rate limit handling
- **Token management** with accurate counting per provider
- **Streaming support** for real-time content generation

### 3. Service Layer (services/)
- **GenerationService**: Orchestrates book generation workflow
- **PromptService**: Template-based prompt management with YAML
- **Clear separation** of business logic from infrastructure
- **Testable design** with dependency injection

### 4. Event System (events.py)
- **Observer pattern** for decoupled progress tracking
- **Webhook support** for external integrations
- **Real-time updates** without blocking main workflow
- **Event replay** capability for debugging

### 5. RAG System (token_optimizer_rag.py)
- **FAISS vector search** for semantic similarity
- **Smart summarization** with LLM-powered compression
- **Hybrid context management** (40% core, 40% RAG, 20% summaries)
- **Persistent storage** in .rag/ directories
- **Upgrade path** to knowledge graphs planned

## Complete Project Structure

### Core System Files

| File | Purpose | Responsibilities |
|------|---------|------------------|
| **main.py** | Application entry point | CLI orchestration, workflow coordination, user interaction |
| **app_config.py** | Configuration management | Pydantic v2 settings, env vars, fallback defaults, validation |
| **containers.py** | Dependency injection | Thread-safe singletons, service registry, lazy loading |
| **exceptions.py** | Error handling | Custom exception hierarchy, error codes, recovery strategies |
| **events.py** | Event system | Observer pattern, event bus, webhook dispatch |
| **service_initializer.py** | Service bootstrap | Dependency wiring, provider setup, cache initialization |

### Services Layer

| File | Purpose | Key Features |
|------|---------|--------------|  
| **services/generation_service.py** | Book generation | Chapter workflow, section management, progress tracking |
| **services/prompt_service.py** | Prompt management | Template loading, variable substitution, style injection |
| **services/validation_service.py** | Content validation | Structure checks, coherence validation, quality assurance |

### LLM Provider System

| File | Purpose | Key Features |
|------|---------|--------------|
| **providers/base.py** | Abstract base provider | Circuit breaker, connection pool, retry logic, token counting |
| **providers/factory.py** | Provider instantiation | Factory pattern, runtime switching, config validation |
| **providers/openai_provider.py** | OpenAI GPT-5 | 256k context, thinking mode, function calling |
| **providers/anthropic_provider.py** | Claude 4 Opus | Hybrid reasoning, 200k context, best for code |
| **providers/gemini_provider.py** | Gemini 2.5 Pro | 2M context, Deep Think, multimodal support |
| **providers/cohere_provider.py** | Cohere Command R+ | 128k context, multilingual focus, RAG optimization |
| **providers/openrouter_provider.py** | Universal gateway | Access all models, automatic fallback, cost optimization |

### Book Generation Components

| File | Purpose | Key Features |
|------|---------|--------------|
| **book_generator.py** | Book orchestration | Complete workflow, chapter coordination, progress saving |
| **character_development.py** | Character system | Profiles, arcs, relationships, dialogue consistency |
| **style_templates.py** | Writing styles | 15+ templates, custom styles, content ratings |
| **export_formats.py** | Multi-format export | EPUB, PDF, DOCX, HTML with metadata preservation |

### RAG & Context Management

| File | Purpose | Key Features |
|------|---------|--------------|  
| **token_optimizer_rag.py** | Base hybrid RAG | FAISS indexing, semantic search, smart summarization |
| **rag_enhanced_system.py** | Enhanced RAG orchestrator | Integrates all advanced RAG components |
| **rag_hybrid_search.py** | Hybrid search engine | Dense+sparse retrieval, BM25, TF-IDF, RRF fusion |
| **rag_knowledge_graph.py** | Knowledge graph builder | Entity extraction, relationship mapping, graph queries |
| **rag_incremental_indexing.py** | Incremental indexer | Real-time updates, batch processing, delta encoding |
| **rag_semantic_cache.py** | Semantic cache | Similarity-based caching, adaptive TTL, LRU eviction |
| **rag_metrics.py** | Quality metrics | Performance tracking, feedback loops, A/B testing |
| **rag_retriever.py** | Base retriever | Document chunking, embedding, retrieval |
| **rag_integration.py** | RAG pipeline | Document processing, retrieval optimization |
| **context_manager.py** | Context window | Token allocation, sliding window, priority management |
| **tokenizer.py** | Token utilities | Accurate counting, chunking, truncation strategies |

### Infrastructure Components

| File | Purpose | Key Features |
|------|---------|--------------|
| **cache_manager.py** | Caching system | Multi-backend (memory/Redis/file), TTL, invalidation |
| **streaming.py** | Real-time streaming | SSE support, chunk processing, backpressure handling |
| **background_tasks.py** | Async processing | Celery/RQ integration, task queues, retries |
| **file_operations.py** | File management | Atomic writes, path validation, cleanup routines |
| **project_manager.py** | Project isolation | Workspace management, archiving, metadata tracking |
| **tracing.py** | Distributed tracing | OpenTelemetry spans, event recording, trace context |
| **saga_pattern.py** | Transaction management | Compensating actions, multi-step workflows, auto-rollback |
| **health_check.py** | Health monitoring | Service health checks, overall status, HTTP endpoints |
| **fallback_strategies.py** | Fallback generation | 6 fallback methods, provider switching, content adaptation |
| **dead_letter_queue.py** | Failed operation queue | Persistent storage, auto-retry, exponential backoff |

### Testing Infrastructure

| Directory | Purpose | Coverage |
|-----------|---------|----------|
| **tests/unit/** | Unit tests | Services, providers, utilities |
| **tests/integration/** | Integration tests | End-to-end workflows, provider integration |
| **tests/fixtures/** | Test data | Sample books, mock responses, test configs |
| **tests/conftest.py** | Pytest configuration | Shared fixtures, mocks, test utilities |

### Script Files

| File | Purpose | Usage |
|------|---------|-------|
| **generate_full_book.py** | Standalone generator | Direct book generation without UI |
| **generate_fantasy_book.py** | Fantasy genre demo | Example fantasy book generation |
| **test_gemini_book.py** | Gemini testing | Provider-specific integration test |
| **test_gemini_direct.py** | Direct API test | Low-level Gemini API validation |

## Features

### Multiple LLM Provider Support (Latest 2025 Models)
- **OpenAI**: GPT-5, GPT-5 Mini/Nano (256k context, 94.6% AIME)
- **Anthropic**: Claude 4 Opus/Sonnet (72.5% SWE-bench, hybrid reasoning)
- **Google**: Gemini 2.5 Pro/Flash (2M context, thinking capabilities)
- **Cohere**: Command R+, Command R (128k context)
- **OpenRouter**: Access all latest models through one API

### Advanced Error Recovery & Resilience
- **Circuit Breaker Pattern** - Automatic failure detection and recovery
- **Dead Letter Queue** - Persistent storage of failed operations with auto-retry
- **Saga Pattern** - Multi-step transactions with automatic compensation
- **Fallback Strategies** - 6 different fallback methods for content generation
- **Health Monitoring** - Real-time health checks for all critical services
- **Distributed Tracing** - OpenTelemetry integration for debugging workflows
- Custom exception hierarchy for different error types
- Automatic retry with exponential backoff
- Graceful degradation on failures
- Partial book saving for recovery
- Token limit management

### Event-Driven Progress Tracking
- Real-time generation progress monitoring
- Event emission for all major operations
- Customizable event handlers
- Progress statistics and reporting
- Optional webhook integration

### Smart Book Generation
- Generates complete books with chapters and sections
- Maintains context throughout generation
- Incremental saving after each step
- Resume from interruptions
- Multiple language support

### Project Management
- **Complete project isolation** - Each book is a separate project
- **No data mixing** - Projects are completely isolated from each other
- **Save progress** - Automatic progress tracking per project
- **Delete old projects** - Clean up unused projects
- **Archive projects** - Compress and archive completed books
- **Project switching** - Easy switch between multiple books
- **Metadata tracking** - Track creation date, word count, status

### Style Templates (15+ Predefined Styles)
- **Fiction**: Literary, Thriller, Romance, Fantasy, Sci-Fi, Mystery, Horror
- **Non-Fiction**: Academic, Technical, Business, Self-Help, Biography
- **Special**: Children's, Young Adult, Erotic Romance (18+)
- **Custom styles** - Create your own writing styles
- **Age ratings** - Content appropriate ratings
- **Content warnings** - Automatic content labeling

### Character Development (Fiction)
- **Character profiles** - Complete character sheets
- **Personality tracking** - Traits, strengths, weaknesses, fears
- **Speech patterns** - Unique dialogue for each character
- **Relationship matrix** - Track character relationships
- **Character arcs** - Growth and development tracking
- **Plot tracking** - Monitor plot points and resolutions
- **Dialogue consistency** - Maintain character voice

### Multiple Export Formats
- **EPUB** - Professional e-book format
- **PDF** - Print-ready documents
- **DOCX** - Microsoft Word compatible
- **HTML** - Web-ready format
- **Metadata support** - Author, title, language
- **Custom styling** - Format-specific layouts
- **Batch export** - Export to all formats at once

### Performance Optimizations
- **Streaming responses** - Real-time content generation
- **Smart caching** - Avoid regenerating identical content
- **Token optimization** - Sliding window context management
- **Background processing** - Non-blocking book generation
- **Multiple cache backends** - Memory, Redis, File-based
- **Progress tracking** - Monitor generation in real-time

### Advanced Hybrid RAG System (Enhanced January 2025)
- **Hybrid Search** - Combines dense (FAISS) and sparse (BM25, TF-IDF) retrieval for superior accuracy
- **Knowledge Graph** - Entity extraction and relationship mapping for contextual understanding
- **Incremental Indexing** - Real-time index updates without full reindexing
- **Semantic Caching** - Intelligent query result caching based on semantic similarity
- **Quality Metrics** - Comprehensive RAG performance tracking with feedback loops
- **Smart Summarization** - LLM-based chapter summaries for better context
- **Hybrid Context** - Intelligent token allocation (40% core, 40% RAG, 20% summaries)
- **Vector Indexing** - Automatic book content indexing with sentence-transformers
- **Persistent Storage** - Vector stores saved in `.rag/` directory per book
- **Configurable Modes** - Choose from disabled, basic, hybrid, full, or enhanced RAG modes

## Development Guidelines

### Core Principles
- **KISS (Keep It Simple)**: Choose straightforward solutions over complex ones
- **YAGNI (You Aren't Gonna Need It)**: Build only what's needed now
- **SOLID Principles**: Single responsibility, open/closed, dependency inversion
- **Clean Code**: Self-documenting, minimal comments, clear naming
- **No Backward Compatibility**: Delete old code when refactoring

### Code Standards
- **Python 3.9+** with type hints everywhere
- **Ruff** for linting (100 char line limit)
- **File limits**: 500 lines max per file
- **Function limits**: 50 lines max per function
- **Class limits**: 100 lines max per class
- **Test coverage**: 80%+ for critical paths

### Testing Requirements
- Unit tests for all services and utilities
- Integration tests for provider interactions
- End-to-end tests for complete workflows
- Mocked external dependencies
- Fixtures for test data isolation

## Priority Improvements (from TODO.md)

### ✅ Completed (January 2025)
1. **Extended Error Recovery & Resilience** ✅
   - ✅ Distributed tracing with OpenTelemetry for debugging complex workflows (`tracing.py`)
   - ✅ Saga pattern implementation for multi-step transactional operations (`saga_pattern.py`)
   - ✅ Health check endpoints for all critical services (`health_check.py`)
   - ✅ Fallback strategies for content generation failures (`fallback_strategies.py`)
   - ✅ Dead letter queue for failed operations with retry mechanism (`dead_letter_queue.py`)
   - ✅ **REMOVED**: Legacy error handling code without circuit breaker pattern in `book_generator.py`

2. **Enhanced RAG System** ✅ (January 25, 2025)
   - ✅ Hybrid search combining dense and sparse retrieval (`rag_hybrid_search.py`)
   - ✅ Knowledge graph for entity relationships (`rag_knowledge_graph.py`)
   - ✅ Incremental indexing for real-time updates (`rag_incremental_indexing.py`)
   - ✅ Semantic caching layer for RAG queries (`rag_semantic_cache.py`)
   - ✅ Quality metrics and feedback loops (`rag_metrics.py`)
   - ✅ **REMOVED**: Old simple RAG implementation replaced with enhanced system

3. **Advanced Token Management** (Next Priority)
   - ML-based token prediction
   - Dynamic context window optimization
   - Cross-provider token pooling

4. **Legacy Code Cleanup**
   - Remove deprecated functions
   - Consolidate duplicate logic
   - Modernize remaining Python 3.8 code
   - **PRIORITY**: Remove old error handling implementations lacking circuit breaker

### Next Phase
- GraphQL API layer
- Kubernetes deployment manifests
- Prometheus metrics integration
- Multi-language content generation
- Collaborative editing features

## Installation

### System Requirements
- **Python**: 3.9 or higher (3.11+ recommended for performance)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large books)
- **Disk Space**: 2GB for base installation + 500MB per book project
- **Operating System**: Windows 10+, macOS 11+, Ubuntu 20.04+

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ghostwriter-ai.git
cd ghostwriter-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your environment:
```bash
cp env.example .env
# Edit .env with your API keys and preferences
```

### Dependencies and Requirements

#### Core Dependencies
- **openai** (>=1.0.0) - OpenAI GPT models integration
- **anthropic** (>=0.25.0) - Claude models integration  
- **google-generativeai** (>=0.5.0) - Gemini models integration
- **cohere** (>=5.0.0) - Cohere Command models
- **pydantic** (>=2.0.0) - Data validation and settings
- **pydantic-settings** (>=2.0.0) - Environment configuration

#### RAG & Vector Search
- **faiss-cpu** (>=1.7.4) - Vector similarity search
- **sentence-transformers** (>=2.2.0) - Text embeddings
- **numpy** (>=1.24.0) - Numerical operations
- **scikit-learn** (>=1.3.0) - ML utilities

#### Optional Dependencies
- **redis** (>=5.0.0) - Distributed caching backend
- **celery** (>=5.3.0) - Background task processing
- **prometheus-client** (>=0.19.0) - Metrics monitoring
- **opentelemetry-api** (>=1.21.0) - Distributed tracing

#### Development Dependencies
- **pytest** (>=7.4.0) - Testing framework
- **pytest-cov** (>=4.1.0) - Code coverage
- **pytest-asyncio** (>=0.21.0) - Async test support
- **ruff** (>=0.1.0) - Fast Python linter
- **mypy** (>=1.5.0) - Static type checking
- **ipdb** (>=0.13.0) - Interactive debugging

## Configuration

### Security & API Keys Management

#### Best Practices
- **Never commit .env files** - Always add `.env` to `.gitignore`
- **Use environment variables** - Store API keys in environment variables, not in code
- **Rotate keys regularly** - Update API keys periodically for security
- **Use secret managers** - For production, use AWS Secrets Manager, Azure Key Vault, or similar
- **Restrict key permissions** - Use API keys with minimal required permissions
- **Monitor usage** - Set up alerts for unusual API usage patterns

#### Rate Limiting Considerations
Each provider has different rate limits:
- **OpenAI**: 10,000 TPM (tokens per minute) for GPT-5
- **Anthropic**: 100,000 TPM for Claude Opus
- **Google**: 2,000 RPM (requests per minute) for Gemini Pro
- **Cohere**: 10,000 calls/month for free tier
- **OpenRouter**: Varies by model, automatic fallback on limits

The application automatically handles rate limits with exponential backoff and retry logic.

### Quick Start

1. Choose your LLM provider in `.env`:
```env
LLM_PROVIDER=openai  # or anthropic, cohere, gemini, openrouter
```

2. Set your API key:
```env
OPENAI_API_KEY=your-api-key-here
```

3. Configure book defaults (optional):
```env
BOOK_LANGUAGE=English
BOOK_TITLE=My Amazing Book
BOOK_INSTRUCTIONS=Write about artificial intelligence
```

### Provider-Specific Setup

#### OpenAI GPT-5
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5  # Latest GPT-5 with built-in thinking
```

#### Anthropic Claude 4
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4.1-20250805  # Best coding model
```

#### Google Gemini 2.5
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro  # 2M context, thinking mode
```

#### OpenRouter (Access All Latest Models)
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-opus-4.1  # Or openai/gpt-5, google/gemini-2.5-pro
```

## Database & Storage Architecture

### Storage Locations
The application uses a hierarchical storage structure:

```
ghostwriter-ai/
├── projects/                    # All book projects (gitignored)
│   └── <project-id>/           # Unique project directory
│       ├── project.json        # Project metadata and configuration
│       ├── content/            # Book content storage
│       │   ├── book.json      # Complete book data
│       │   └── chapters/      # Individual chapter files
│       ├── .rag/              # Vector store for RAG
│       │   ├── index.faiss   # FAISS vector index
│       │   ├── chunks.pkl    # Text chunks
│       │   └── metadata.json # RAG configuration
│       ├── cache/             # Project-specific cache
│       │   ├── memory/       # In-memory cache dumps
│       │   └── responses/    # Cached LLM responses
│       └── exports/           # Generated book formats
├── .cache/                     # Global application cache
│   ├── providers/             # Provider-specific caches
│   └── templates/             # Compiled prompt templates
└── books/                      # Legacy storage (deprecated)
```

### Cache Configuration
- **Memory Cache**: Default, fastest, limited by RAM
- **Redis Cache**: Distributed, persistent, requires Redis server
- **File Cache**: Persistent, slower, unlimited size

### Vector Store (RAG)
- **Location**: `projects/<id>/.rag/` directory
- **Index Type**: FAISS with L2 distance metric
- **Embeddings**: Stored with sentence-transformers
- **Chunk Size**: Configurable (default 512 tokens)
- **Persistence**: Automatic save/load on project access

## Usage

### Basic Usage

Run the application:
```bash
python main.py
```

The app will prompt for:
- Book language (if not set in .env)
- Book title (if not set in .env)
- Book instructions (if not set in .env)

### CLI Commands Reference

#### Main Application
```bash
# Basic book generation
python main.py

# With environment overrides
LLM_PROVIDER=anthropic python main.py

# Debug mode
python main.py --debug

# Specify project
python main.py --project-id <id>

# Resume interrupted generation
python main.py --resume

# Export only (no generation)
python main.py --export-only --format epub
```

#### Standalone Scripts
```bash
# Generate complete book programmatically
python generate_full_book.py --title "My Book" --instructions "Write about AI"

# Generate fantasy book example
python generate_fantasy_book.py

# Test specific provider
python test_gemini_book.py
python test_gemini_direct.py
```

#### Environment Variable Overrides
All configuration can be overridden via environment variables:
```bash
# Override provider
LLM_PROVIDER=gemini python main.py

# Override model
OPENAI_MODEL=gpt-5-mini python main.py

# Enable debug logging
DEBUG=true python main.py

# Set cache backend
CACHE_BACKEND=redis python main.py

# Configure RAG
ENABLE_RAG=true RAG_MODE=full python main.py
```

### Project Management

```python
from project_manager import get_project_manager

pm = get_project_manager()

# Create a new book project
project_id = pm.create_project(
    title="My Amazing Novel",
    language="English",
    style="thriller"
)

# Switch between projects
pm.switch_project(project_id)

# List all projects
projects = pm.list_projects(status="draft")

# Archive old project
pm.archive_project(old_project_id)

# Delete project (with confirmation)
pm.delete_project(old_project_id, confirm=True)

# Clean up old drafts (30+ days old)
pm.cleanup_old_projects(days=30, status="draft", dry_run=False)
```

### Style Templates

```python
from style_templates import get_style_manager, apply_style

manager = get_style_manager()

# List available styles
styles = manager.list_styles(category="fiction")

# Apply style to generation
styled_prompt = apply_style(
    "Write a chapter about the discovery",
    style="thriller",
    prompt_type="chapter"
)

# Create custom style
custom = manager.create_custom_style(
    name="my_style",
    base_style="romance",
    tone="mysterious, sensual",
    vocabulary_level="advanced"
)
```

### Character Development

```python
from character_development import get_character_manager, CharacterRole

cm = get_character_manager()

# Create character
protagonist = cm.create_character(
    name="Sarah Connor",
    role=CharacterRole.PROTAGONIST,
    personality_traits=["brave", "determined", "protective"],
    speech_pattern="Direct, military-influenced"
)

# Track relationships
cm.add_relationship(
    "Sarah Connor",
    "John Connor",
    RelationshipType.FAMILY,
    description="Mother and son"
)

# Generate character-appropriate dialogue
dialogue_prompt = cm.generate_dialogue(
    "Sarah Connor",
    context="Facing the terminator",
    emotion="determined"
)
```

### Export Formats

```python
from export_formats import get_exporter

exporter = get_exporter()

# Export to specific format
epub_path = exporter.export(book_data, "epub", {
    "author": "Your Name",
    "cover_image": "cover.jpg"
})

# Export to all formats
paths = exporter.export_all_formats(book_data, {
    "author": "Your Name"
})
```

### Advanced Usage

#### Resume Interrupted Generation
If generation is interrupted, simply run again with the same title. The app will resume from the last saved checkpoint.

#### Custom Event Handlers
Add custom event handlers in your code:
```python
from events import event_manager, EventType

def on_chapter_complete(event):
    print(f"Chapter {event.data['chapter_number']} completed!")

event_manager.subscribe(EventType.CHAPTER_COMPLETED, on_chapter_complete)
```

#### Switch Providers at Runtime
```python
from ai import switch_provider

switch_provider('anthropic', {
    'api_key': 'your-key',
    'model': 'claude-3-5-sonnet-20241022'
})
```

## Testing

### Running Tests

#### Basic Test Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_generation_service.py -v

# Run tests matching pattern
python -m pytest tests/ -k "test_provider" -v

# Run with verbose output
python -m pytest tests/ -vv

# Run with minimal output
python -m pytest tests/ -q
```

#### Code Coverage
```bash
# Run tests with coverage report
python -m pytest --cov=. --cov-report=html

# Coverage with terminal report
python -m pytest --cov=. --cov-report=term-missing

# Coverage for specific module
python -m pytest --cov=providers --cov-report=html tests/unit/test_providers.py

# Generate XML coverage for CI/CD
python -m pytest --cov=. --cov-report=xml
```

#### Code Quality Checks
```bash
# Run linting
python -m ruff check .

# Auto-fix linting issues
python -m ruff check . --fix

# Format code
python -m ruff format .

# Type checking
python -m mypy . --ignore-missing-imports

# Type check specific module
python -m mypy providers/ --strict
```

#### Integration Testing
```bash
# Test with real API (requires API keys)
INTEGRATION_TEST=true python -m pytest tests/integration/ -v

# Test specific provider integration
python -m pytest tests/integration/test_openai_integration.py -v

# Load test with concurrent requests
python -m pytest tests/performance/ -v --workers 4
```

#### Debugging Tests
```bash
# Run with debugger on failure
python -m pytest tests/ --pdb

# Run with ipdb debugger
python -m pytest tests/ --pdbcls=IPython.terminal.debugger:TerminalPdb

# Show local variables on failure
python -m pytest tests/ -l

# Capture print statements
python -m pytest tests/ -s

# Run specific test by name
python -m pytest tests/unit/test_cache.py::TestCacheManager::test_memory_cache
```

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (no external dependencies)
│   ├── test_generation_service.py
│   ├── test_providers.py
│   ├── test_cache_manager.py
│   └── test_rag_integration.py
├── integration/            # Integration tests (require external services)
│   ├── test_book_generation.py
│   └── test_provider_switching.py
├── fixtures/               # Test data and mocks
│   ├── sample_data.py
│   └── mock_responses.json
└── performance/            # Performance and load tests
    └── test_concurrent_generation.py
```

## Complete File Structure

```
ghostwriter-ai/
├── Core Modules
│   ├── main.py                        # CLI entry point, user interaction
│   ├── app_config.py                  # Pydantic v2 configuration management
│   ├── containers.py                  # Thread-safe dependency injection
│   ├── exceptions.py                  # Custom exception hierarchy
│   ├── events.py                      # Event-driven architecture
│   ├── service_initializer.py         # Service bootstrap and wiring
│   └── book_config.json               # Book generation configuration
│
├── Services (services/)
│   ├── generation_service.py          # Core book generation workflow
│   ├── prompt_service.py              # Template-based prompt management
│   └── validation_service.py          # Content validation and QA
│
├── LLM Providers (providers/)
│   ├── base.py                        # Abstract base with exponential backoff retry
│   ├── factory.py                     # Factory pattern for provider instantiation
│   ├── openai_provider.py             # GPT-5 with built-in thinking (256k context)
│   ├── anthropic_provider.py          # Claude 4 Opus/Sonnet (hybrid reasoning)
│   ├── gemini_provider.py             # Gemini 2.5 Pro/Flash (2M context)
│   ├── cohere_provider.py             # Command R+ (multilingual focus)
│   └── openrouter_provider.py         # Universal access to all models
│
├── Book Generation
│   ├── book_generator.py              # Main book generation orchestrator
│   ├── character_development.py       # Character profiles, arcs, relationships
│   ├── style_templates.py             # 15+ writing styles with customization
│   ├── export_formats.py              # Multi-format export with metadata
│   └── prompts_templated.py           # YAML-based prompt templates
│
├── RAG & Context Management
│   ├── token_optimizer_rag.py         # Hybrid RAG with FAISS indexing
│   ├── rag_integration.py             # RAG pipeline and retrieval
│   ├── context_manager.py             # Context window optimization
│   └── tokenizer.py                   # Token counting and chunking
│
├── Infrastructure & Utilities
│   ├── cache_manager.py               # Multi-backend caching system
│   ├── streaming.py                   # Real-time SSE streaming
│   ├── background_tasks.py            # Async task queue processing
│   ├── file_operations.py             # File I/O utilities
│   ├── project_manager.py             # Project workspace management
│   ├── tracing.py                     # OpenTelemetry distributed tracing
│   ├── saga_pattern.py                # Saga pattern for transactions
│   ├── health_check.py                # Health monitoring system
│   ├── fallback_strategies.py         # Fallback generation strategies
│   └── dead_letter_queue.py           # DLQ for failed operations
│
├── Scripts & Testing
│   ├── generate_full_book.py          # Standalone book generator
│   ├── generate_fantasy_book.py       # Fantasy genre example
│   ├── test_gemini_book.py           # Gemini provider test
│   └── test_gemini_direct.py         # Direct API validation
│
├── Configuration
│   ├── pyproject.toml                 # Project config, tool settings
│   ├── requirements.txt               # Core dependencies
│   ├── requirements-dev.txt           # Development dependencies
│   ├── env.example                    # Environment template
│   ├── .env                          # Local environment (gitignored)
│   ├── book_config.json              # Book generation config
│   └── templates/
│       └── prompts.yaml              # Customizable prompts
│
├── Testing (tests/)
│   ├── conftest.py                   # Shared pytest fixtures
│   ├── test_integration.py           # End-to-end tests
│   ├── unit/
│   │   ├── test_generation_service.py
│   │   ├── test_providers.py
│   │   ├── test_cache_manager.py
│   │   └── test_rag_integration.py
│   ├── integration/
│   │   ├── test_book_generation.py
│   │   └── test_provider_switching.py
│   └── fixtures/
│       └── sample_data.py
│
├── Documentation (docs/)
│   ├── README.md                      # This file (main documentation)
│   ├── CLAUDE.md                      # AI assistant instructions
│   ├── TODO.md                        # Development roadmap (44+ tasks)
│   ├── PERFORMANCE.md                 # Optimization guide
│   ├── API.md                         # API documentation
│   └── primer.md                      # Quick start guide
│
├── Hidden Directories
│   └── .serena/                       # Serena MCP server files
│
└── Generated Content (gitignored)
    ├── projects/                      # Isolated book projects
    │   └── <project-id>/
    │       ├── project.json          # Project metadata
    │       ├── content/              # Book chapters and sections
    │       ├── .rag/                 # Vector stores for RAG
    │       ├── cache/                # Project-specific cache
    │       └── exports/              # Generated formats
    └── books/                         # Legacy book storage
```

## Recent Improvements (December 2024 - January 2025)

### 🚀 Enhanced RAG System Implementation (January 25, 2025) - COMPLETED
- **Hybrid Search System** (`rag_hybrid_search.py`) - Combines dense (FAISS) and sparse (BM25, TF-IDF) retrieval with Reciprocal Rank Fusion
- **Knowledge Graph** (`rag_knowledge_graph.py`) - Entity extraction, relationship mapping, graph-based context retrieval
- **Incremental Indexing** (`rag_incremental_indexing.py`) - Real-time index updates, batch processing, checkpoint recovery
- **Semantic Caching** (`rag_semantic_cache.py`) - Query result caching based on semantic similarity, adaptive TTL
- **Quality Metrics** (`rag_metrics.py`) - Comprehensive performance tracking, feedback loops, A/B testing support
- **Unified System** (`rag_enhanced_system.py`) - Orchestrates all RAG components with backward compatibility
- **Legacy Removal** - Replaced old simple RAG with advanced hybrid system

### 🚀 Enhanced Error Recovery & Resilience (January 2025) - COMPLETED
- **Distributed tracing implemented** (`tracing.py`) - OpenTelemetry integration with span tracking, event recording, and automatic instrumentation
- **Saga pattern implemented** (`saga_pattern.py`) - Multi-step transactional operations with automatic compensation on failure
- **Health check system added** (`health_check.py`) - Comprehensive monitoring for providers, cache, RAG, and filesystem with health endpoints
- **Fallback strategies system** (`fallback_strategies.py`) - Six different fallback methods including provider switching and content adaptation
- **Dead letter queue implemented** (`dead_letter_queue.py`) - Persistent failed operation storage with automatic retry and exponential backoff
- **Circuit breaker pattern** - Already implemented in base provider class for fault tolerance
- **Legacy code removed** - Cleaned up old retry logic in `book_generator.py`, now using circuit breaker pattern exclusively

### Critical Bug Fixes (January 2025)
- **Added missing _call_with_retry method** - Implemented comprehensive retry logic with circuit breaker protection in base provider class
- **Fixed API key configuration** - Ensured consistent API key handling across all providers (OpenAI, Anthropic, Gemini, Cohere, OpenRouter)
- **Cache Manager improvements** - MemoryCache now properly accepts and uses expire parameter for TTL management
- **Fixed Gemini Provider retry logic** - Now properly uses _call_with_retry with circuit breaker for all API calls
- **Fixed Gemini error handling** - _handle_error now correctly raises exceptions instead of returning them
- **Fixed token counting in Gemini** - Tokenizer now accepts API key parameter and properly handles authentication
- **Added UTF-8 encoding** - All file operations now explicitly use UTF-8 encoding to prevent Windows encoding issues
- **Improved response handling** - Gemini provider now handles empty responses gracefully without crashes

### Code Quality & Architecture
- **Refactored generate.py** - New `BookGenerator` class eliminates code duplication
- **Centralized retry logic** - Base `LLMProvider` class now handles all retry logic with exponential backoff
- **Template-based prompts** - Prompts moved to external YAML files for easy customization
- **Dependency injection** - Removed global singletons for better testing and isolation
- **Accurate token counting** - All providers now use official SDK methods for token counting
- **Fixed history management** - Maintains proper chronological order in conversations
- **Safe JSON parsing** - Robust extraction from LLM responses with markdown code blocks

### Critical Fixes
- Fixed token counting accuracy for Anthropic, Cohere, and Gemini providers
- Fixed conversation history ordering bug that was breaking context
- Added safe JSON parsing to handle various LLM response formats
- Improved error handling and logging throughout

## Architectural Design

### 1. Clean Architecture
The codebase follows clean architecture principles with clear separation of concerns:

- **Domain Layer**: Core business logic (book generation, character management)
- **Application Layer**: Use cases and services (GenerationService)
- **Infrastructure Layer**: External interfaces (LLM providers, cache, storage)
- **Presentation Layer**: CLI interface and export formats

### 2. Design Patterns Implemented

#### Factory Pattern
```python
# providers/factory.py
provider = ProviderFactory.create_provider("openai", config)
```

#### Dependency Injection
```python
# containers.py - Thread-safe singleton management
container = AppContainer()
service = container.generation_service()
```

#### Observer Pattern
```python
# events.py - Event-driven architecture
event_manager.subscribe(EventType.CHAPTER_COMPLETED, handler)
event_manager.emit(Event(EventType.CHAPTER_COMPLETED, data))
```

#### Strategy Pattern
```python
# Multiple LLM providers implementing common interface
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: pass
```

### 3. SOLID Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extensible via base classes, closed for modification
- **Liskov Substitution**: All providers are interchangeable
- **Interface Segregation**: Focused interfaces for each feature
- **Dependency Inversion**: Depend on abstractions, not implementations

### 4. Key Architectural Features

#### Hybrid RAG System
Combines three context strategies for optimal performance:
- **40% Core Context**: Recent chapters and book structure
- **40% RAG Retrieved**: Semantically similar content via FAISS
- **20% Summaries**: LLM-generated chapter summaries

#### Thread-Safe Implementation
- Double-checked locking for singletons
- Thread-safe cache operations
- Concurrent request handling

#### Graceful Degradation
- Fallback when optional dependencies missing
- Partial book saving on failures
- Automatic retry with exponential backoff

## Code Highlights

### BookGenerator Class
The refactored `BookGenerator` class provides a clean, DRY approach to book generation:

```python
from services.generation_service import GenerationService

# Unified generation interface
service = GenerationService(provider, cache, token_optimizer)
book = service.generate_book(title, instructions, language)
```

### Centralized Retry Logic
All providers inherit robust retry logic from the base class:

```python
class LLMProvider(ABC):
    def _call_with_retry(self, api_call, max_retries=3, 
                        exponential_base=2.0, jitter=True):
        # Handles rate limits, timeouts, connection errors
        # Exponential backoff with jitter
        # Provider-specific exception handling
```

### Template-Based Prompts
Prompts are now external and easily customizable:

```yaml
# templates/prompts.yaml
chapter:
  template: |
    Write Chapter {chapter_number}: {chapter_title}
    Book: "{title}"
    Topics to cover: {topics}
    Style: {style}
    Previous context: {context}
```

### Project Isolation
Complete project isolation through dependency injection:

```python
# Each project has isolated resources
pm = ProjectManager()
project = pm.create_project(title="My Book")
cache = project.get_cache_manager()     # Project-specific cache
rag = project.get_rag_manager()         # Project-specific vectors
exporter = project.get_exporter()       # Project-specific exports
```

## Generated Output

### Project-Based Structure
Books are organized by project in `projects/<project-id>/`:
```
projects/
├── <project-id>/
│   ├── project.json       # Project configuration
│   ├── content/           # Book content
│   │   ├── book.json     # Complete book data
│   │   ├── README.md     # Table of contents
│   │   └── chapters/     # Individual chapters
│   ├── characters/        # Character profiles (fiction)
│   ├── exports/          # Generated formats
│   │   ├── book.epub    # EPUB export
│   │   ├── book.pdf     # PDF export
│   │   ├── book.docx    # Word document
│   │   └── book.html    # HTML version
│   ├── cache/            # Project-specific cache
│   └── assets/           # Images, covers, etc.
```

## Model Comparison (2025 Latest)

| Provider | Model | Context Window | Performance | Best For |
|----------|-------|---------------|-------------|----------|
| OpenAI | GPT-5 | 256K | 94.6% AIME, 74.9% SWE-bench | General purpose, built-in thinking |
| Anthropic | Claude Opus 4.1 | 200K | 72.5% SWE-bench, 43.2% Terminal | Best coding, hybrid reasoning |
| Google | Gemini 2.5 Pro | 2M | 63.8% SWE-bench, Deep Think mode | Very long books, thinking mode |
| Cohere | Command R+ | 128K | Strong multilingual | Multilingual content |
| OpenRouter | All models | Varies | Access latest models | Model flexibility |

## Environment Variables

### Essential
- `LLM_PROVIDER` - Provider selection
- `*_API_KEY` - Provider API key

### Generation Control
- `TEMPERATURE` - Creativity (0.0-1.0, default: 0.2)
- `TOKEN_LIMIT` - Context window limit
- `MAX_TOKENS` - Max response length

### Book Settings
- `BOOK_LANGUAGE` - Target language
- `BOOK_TITLE` - Book title
- `BOOK_INSTRUCTIONS` - Generation instructions

### Features
- `ENABLE_PROGRESS_TRACKING` - Progress monitoring
- `PROGRESS_CALLBACK_URL` - Webhook for updates

### RAG Configuration
- `ENABLE_RAG` - Enable/disable RAG features (default: true)
- `RAG_MODE` - RAG operation mode: disabled, basic, hybrid, full (default: hybrid)
- `RAG_EMBEDDING_MODEL` - Sentence transformer model (default: all-MiniLM-L6-v2)
- `RAG_CHUNK_SIZE` - Text chunk size for indexing (default: 512)
- `RAG_TOP_K` - Number of similar chunks to retrieve (default: 10)
- `RAG_SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.5)
- `RAG_CORE_CONTEXT_RATIO` - Token allocation for core context (default: 0.4)
- `RAG_RETRIEVED_CONTEXT_RATIO` - Token allocation for RAG content (default: 0.4)
- `RAG_SUMMARY_CONTEXT_RATIO` - Token allocation for summaries (default: 0.2)

### Error Recovery & Resilience Configuration
- `TRACING_ENABLED` - Enable OpenTelemetry distributed tracing (default: false)
- `TRACING_CONSOLE_EXPORT` - Export traces to console (default: false)
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OpenTelemetry collector endpoint
- `OTEL_EXPORTER_OTLP_INSECURE` - Use insecure connection (default: true)
- `DLQ_STORAGE_PATH` - Dead letter queue storage location (default: dlq_storage)
- `DLQ_MAX_RETRIES` - Maximum retry attempts for failed operations (default: 3)
- `DLQ_RETRY_POLICY` - Retry policy: exponential_backoff, linear_backoff, fixed_delay (default: exponential_backoff)
- `HEALTH_CHECK_CACHE_TTL` - Health check cache duration in seconds (default: 30)
- `FALLBACK_STRATEGIES` - Comma-separated list of enabled fallback strategies

## Performance Tuning

### Memory Optimization
```env
# Limit concurrent chapter generation
MAX_CONCURRENT_CHAPTERS=2

# Reduce cache size
CACHE_MAX_SIZE=100
CACHE_TTL=3600

# Use file-based cache for low memory systems
CACHE_BACKEND=file

# Limit RAG vector store size
RAG_MAX_CHUNKS=1000
RAG_CHUNK_SIZE=256
```

### Token Budget Management
```env
# Set conservative token limits
TOKEN_LIMIT=50000
MAX_TOKENS=2000

# Optimize context allocation
RAG_CORE_CONTEXT_RATIO=0.3
RAG_RETRIEVED_CONTEXT_RATIO=0.5
RAG_SUMMARY_CONTEXT_RATIO=0.2

# Enable aggressive summarization
ENABLE_AGGRESSIVE_SUMMARY=true
SUMMARY_MAX_LENGTH=500
```

### Concurrent Generation Limits
```python
# In app_config.py
MAX_WORKERS = 4  # Limit thread pool size
MAX_RETRIES = 3  # Reduce retry attempts
TIMEOUT_SECONDS = 120  # Set request timeout
```

### Cache Size Configuration
```python
# Memory cache limits
MEMORY_CACHE_MAX_SIZE = 1000  # Maximum items
MEMORY_CACHE_SIZE_MB = 512    # Maximum memory usage

# Redis cache configuration
REDIS_MAX_CONNECTIONS = 50
REDIS_SOCKET_TIMEOUT = 10
REDIS_CONNECTION_POOL_CLASS = "BlockingConnectionPool"
```

## Monitoring & Logging

### Log Configuration

#### Log Levels
```env
# Set global log level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Provider-specific logging
OPENAI_LOG_LEVEL=DEBUG
ANTHROPIC_LOG_LEVEL=WARNING

# Component logging
RAG_LOG_LEVEL=DEBUG
CACHE_LOG_LEVEL=INFO
```

#### Log Output
```bash
# Enable debug logging
DEBUG=true python main.py

# Log to file
python main.py 2>&1 | tee generation.log

# Structured logging (JSON format)
LOG_FORMAT=json python main.py

# Verbose logging with timestamps
python main.py --log-level DEBUG --log-timestamp
```

### Log File Locations
```
logs/
├── app.log              # Main application log
├── providers.log        # Provider interactions
├── rag.log             # RAG system operations
├── cache.log           # Cache operations
├── errors.log          # Error-only log
└── debug.log           # Debug-level details
```

### Monitoring Webhooks
```env
# Progress tracking webhook
PROGRESS_CALLBACK_URL=https://your-server.com/webhook

# Error reporting webhook
ERROR_WEBHOOK_URL=https://your-server.com/errors

# Completion notification
COMPLETION_WEBHOOK_URL=https://your-server.com/complete
```

### Metrics Collection
```python
# Prometheus metrics endpoint
METRICS_ENABLED=true
METRICS_PORT=9090

# Available metrics:
# - generation_duration_seconds
# - tokens_used_total
# - cache_hit_ratio
# - provider_errors_total
# - rag_retrieval_duration_seconds
```

## Troubleshooting

### Common Errors and Solutions

#### 1. API Key Errors
```
Error: Invalid API key for provider
```
**Solution**:
- Verify API key in `.env` file
- Check key hasn't expired
- Ensure correct provider selected
- Test key with provider's playground

#### 2. Token Limit Exceeded
```
Error: Context length exceeded maximum tokens
```
**Solution**:
- Reduce `TOKEN_LIMIT` in configuration
- Enable more aggressive summarization
- Use a model with larger context window
- Split book into smaller sections

#### 3. Rate Limit Errors
```
Error: Rate limit exceeded, please retry
```
**Solution**:
- Application auto-retries with exponential backoff
- Reduce `MAX_CONCURRENT_CHAPTERS`
- Add delays between requests
- Upgrade to higher tier API plan

#### 4. Memory Errors
```
Error: Out of memory during generation
```
**Solution**:
- Switch to file-based cache
- Reduce cache size limits
- Disable RAG or reduce chunk count
- Process chapters sequentially

#### 5. Connection Errors
```
Error: Connection timeout to provider API
```
**Solution**:
- Check internet connection
- Verify firewall settings
- Test provider API status
- Increase timeout values

### Debug Mode Activation
```bash
# Enable full debug output
DEBUG=true LOG_LEVEL=DEBUG python main.py

# Debug specific component
DEBUG_PROVIDER=true python main.py
DEBUG_RAG=true python main.py
DEBUG_CACHE=true python main.py

# Save debug output
python main.py --debug 2>&1 | tee debug.log
```

### Log File Analysis
```bash
# Check for errors
grep ERROR logs/app.log

# Monitor generation progress
tail -f logs/app.log

# Count provider errors
grep "provider_error" logs/providers.log | wc -l

# Find slow operations
grep "duration" logs/app.log | sort -k3 -n
```

### Recovery from Failures

#### Resume Interrupted Generation
```bash
# Automatic resume from last checkpoint
python main.py --resume

# Resume specific project
python main.py --project-id <id> --resume

# Force restart (ignore checkpoint)
python main.py --force-restart
```

#### Recover Partial Books
```python
# Check for partial saves
ls projects/*/book.json.partial

# Load partial book
from file_operations import load_json
partial = load_json("projects/<id>/book.json.partial")

# Continue from last chapter
last_chapter = partial.get("last_completed_chapter", 0)
```

### Reporting Issues
When reporting issues, include:
1. **Error message** and full stack trace
2. **Configuration**: Provider, model, settings
3. **Debug log**: Run with `--debug` flag
4. **Environment**: OS, Python version, dependencies
5. **Steps to reproduce**: Exact commands run
6. **Sample data**: If applicable (redact sensitive info)

## Migration Guide

### Upgrading from Legacy Structure

If you have books in the old `books/` directory structure, follow these steps:

#### 1. Backup Existing Books
```bash
# Create backup
cp -r books/ books_backup/

# Or create archive
tar -czf books_backup.tar.gz books/
```

#### 2. Run Migration Script
```python
# migrate_books.py
from project_manager import ProjectManager
from file_operations import load_json, save_json
import os

pm = ProjectManager()

# Migrate each book
for book_dir in os.listdir("books/"):
    book_path = f"books/{book_dir}/book.json"
    if os.path.exists(book_path):
        book_data = load_json(book_path)
        
        # Create new project
        project_id = pm.create_project(
            title=book_data.get("title"),
            language=book_data.get("language", "English")
        )
        
        # Copy book data to new structure
        pm.import_book(project_id, book_data)
        print(f"Migrated: {book_dir} -> {project_id}")
```

#### 3. Verify Migration
```bash
# List new projects
python -c "from project_manager import ProjectManager; pm = ProjectManager(); print(pm.list_projects())"

# Test opening a migrated book
python main.py --project-id <migrated-id>
```

#### 4. Clean Up Old Structure
```bash
# After verifying migration success
rm -rf books/
```

### Configuration Migration

#### Old Environment Variables (deprecated)
```env
# Old format
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4
```

#### New Environment Variables
```env
# New format
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-5
```

### API Changes

#### Old API (deprecated)
```python
from ai import AI
ai = AI(provider="openai")
response = ai.generate(prompt)
```

#### New API
```python
from providers.factory import ProviderFactory
provider = ProviderFactory.create("openai", config)
response = provider.generate(prompt, context)
```

## Known Limitations

### Provider-Specific Limitations

#### OpenAI
- **Rate Limits**: 10,000 TPM for standard tier
- **Context Window**: 256K tokens maximum
- **Response Length**: 16K tokens per response
- **Thinking Time**: Built-in thinking adds latency

#### Anthropic Claude
- **Rate Limits**: 100,000 TPM 
- **Context Window**: 200K tokens
- **Response Length**: 8K tokens typical
- **API Availability**: May have occasional outages

#### Google Gemini
- **Rate Limits**: 2,000 RPM
- **Context Window**: 2M tokens (but slower with full context)
- **Response Length**: 32K tokens
- **Region Restrictions**: Not available in all countries

#### Cohere
- **Free Tier**: 10,000 calls/month limit
- **Context Window**: 128K tokens
- **Language Support**: Best for English, variable for others

### Book Generation Limitations

#### Size Constraints
- **Maximum Chapters**: 100 chapters per book
- **Maximum Sections**: 20 sections per chapter
- **Maximum Words**: ~500,000 words per book
- **Generation Time**: 2-8 hours for full novel

#### Content Limitations
- **Fiction Quality**: Varies by model and prompt quality
- **Consistency**: May have minor inconsistencies in long books
- **Character Depth**: Limited by context window
- **Plot Complexity**: Best with structured outlines

### Language Support

#### Fully Supported Languages
- English, Spanish, French, German, Italian
- Portuguese, Dutch, Russian, Japanese, Chinese

#### Partially Supported Languages
- Arabic, Hebrew (RTL issues)
- Hindi, Bengali (formatting issues)
- Korean (character rendering)

#### Limited Support
- Minor languages may have quality issues
- Technical terminology may default to English
- Cultural nuances may be missed

### Platform-Specific Issues

#### Windows
- **Path Length**: Maximum path length of 260 characters
- **Unicode**: Some Unicode characters in filenames may fail
- **Permissions**: May need admin rights for some operations
- **Line Endings**: CRLF vs LF issues with Git

#### macOS
- **Gatekeeper**: May block first run
- **Permissions**: Need to allow terminal access
- **Case Sensitivity**: File system case handling

#### Linux
- **Dependencies**: May need to install additional packages
- **Permissions**: File permission issues in Docker
- **Encoding**: UTF-8 locale required

### Performance Limitations

#### Memory Usage
- **Base**: ~500MB for application
- **Per Book**: ~100MB in memory during generation
- **RAG Index**: ~200MB per 100K words
- **Cache**: Grows unbounded without limits

#### CPU Usage
- **Single-threaded**: Main generation loop
- **Multi-threaded**: Provider calls, RAG indexing
- **High Load**: During embedding generation

#### Network Requirements
- **Bandwidth**: 1-10 MB per chapter generation
- **Latency**: <100ms recommended to providers
- **Stability**: Requires stable connection

### Technical Debt

#### Known Issues
- Legacy code in some utility functions
- Incomplete test coverage (~60%)
- Some hardcoded values need configuration
- Circular import potential in some modules

#### Planned Improvements
- Full async/await implementation
- GraphQL API layer
- Kubernetes deployment support
- Plugin architecture for extensions

## Contributing

Contributions are welcome! To add a new provider:

1. Create provider class in `providers/`
2. Inherit from `LLMProvider` base class
3. Implement required methods
4. Register in `providers/factory.py`
5. Add configuration in `config.py`

## License

MIT License - see LICENSE file for details

## Acknowledgments

Inspired by various AI writing projects and the amazing capabilities of modern LLMs.