# GhostWriter AI: Multi-Provider AI Book Writer

An advanced AI-powered book writing application that supports multiple LLM providers, comprehensive error handling, and real-time progress tracking.

## Recent Improvements (December 2024)

### 🔧 Code Quality & Architecture
- **Refactored generate.py** - New `BookGenerator` class eliminates code duplication
- **Centralized retry logic** - Base `LLMProvider` class now handles all retry logic with exponential backoff
- **Template-based prompts** - Prompts moved to external YAML files for easy customization
- **Dependency injection** - Removed global singletons for better testing and isolation
- **Accurate token counting** - All providers now use official SDK methods for token counting
- **Fixed history management** - Maintains proper chronological order in conversations
- **Safe JSON parsing** - Robust extraction from LLM responses with markdown code blocks

### 🐛 Critical Fixes
- Fixed token counting accuracy for Anthropic, Cohere, and Gemini providers
- Fixed conversation history ordering bug that was breaking context
- Added safe JSON parsing to handle various LLM response formats
- Improved error handling and logging throughout

## Key Components & File Descriptions

### Core System Files

| File | Purpose | Key Features |
|------|---------|--------------|
| **main.py** | Application entry point | CLI interface, orchestrates generation workflow, event handler setup |
| **app_config.py** | Configuration management | Pydantic settings with fallback, environment variable loading |
| **containers.py** | Dependency injection | Thread-safe singleton management, service locator pattern |
| **services/generation_service.py** | Core generation logic | Book generation workflow, chapter/section management |

### LLM Provider System

| File | Purpose | Key Features |
|------|---------|--------------|
| **providers/base.py** | Abstract base provider | Exponential backoff retry, rate limit handling, token counting |
| **providers/factory.py** | Provider instantiation | Factory pattern, dynamic provider loading |
| **providers/openai_provider.py** | OpenAI GPT-5 integration | 256k context, built-in thinking mode |
| **providers/anthropic_provider.py** | Claude 4 integration | Hybrid reasoning, 200k context |
| **providers/gemini_provider.py** | Gemini 2.5 integration | 2M context, Deep Think mode |

### Advanced Features

| File | Purpose | Key Features |
|------|---------|--------------|
| **token_optimizer_rag.py** | Hybrid RAG system | FAISS vector search, smart summarization, context optimization |
| **character_development.py** | Character management | Profile tracking, relationship matrix, dialogue consistency |
| **style_templates.py** | Writing styles | 15+ predefined styles, custom style creation, age ratings |
| **export_formats.py** | Multi-format export | EPUB, PDF, DOCX, HTML generation with metadata |
| **project_manager.py** | Project isolation | Complete project separation, archiving, metadata tracking |

### Infrastructure Components

| File | Purpose | Key Features |
|------|---------|--------------|
| **cache_manager.py** | Caching system | Multi-backend support (memory/Redis/file), TTL management |
| **events.py** | Event system | Observer pattern, progress tracking, webhook support |
| **exceptions.py** | Error handling | Custom exception hierarchy, graceful degradation |
| **streaming.py** | Real-time streaming | Async content generation, progress updates |
| **background_tasks.py** | Async processing | Celery/RQ integration, non-blocking operations |

## Features

### 🚀 Multiple LLM Provider Support (Latest 2025 Models)
- **OpenAI**: GPT-5, GPT-5 Mini/Nano (256k context, 94.6% AIME)
- **Anthropic**: Claude 4 Opus/Sonnet (72.5% SWE-bench, hybrid reasoning)
- **Google**: Gemini 2.5 Pro/Flash (2M context, thinking capabilities)
- **Cohere**: Command R+, Command R (128k context)
- **OpenRouter**: Access all latest models through one API

### 🛡️ Robust Error Handling
- Custom exception hierarchy for different error types
- Automatic retry with exponential backoff
- Graceful degradation on failures
- Partial book saving for recovery
- Token limit management

### 📊 Event-Driven Progress Tracking
- Real-time generation progress monitoring
- Event emission for all major operations
- Customizable event handlers
- Progress statistics and reporting
- Optional webhook integration

### 📚 Smart Book Generation
- Generates complete books with chapters and sections
- Maintains context throughout generation
- Incremental saving after each step
- Resume from interruptions
- Multiple language support

### 🎯 Project Management
- **Complete project isolation** - Each book is a separate project
- **No data mixing** - Projects are completely isolated from each other
- **Save progress** - Automatic progress tracking per project
- **Delete old projects** - Clean up unused projects
- **Archive projects** - Compress and archive completed books
- **Project switching** - Easy switch between multiple books
- **Metadata tracking** - Track creation date, word count, status

### 🎨 Style Templates (15+ Predefined Styles)
- **Fiction**: Literary, Thriller, Romance, Fantasy, Sci-Fi, Mystery, Horror
- **Non-Fiction**: Academic, Technical, Business, Self-Help, Biography
- **Special**: Children's, Young Adult, Erotic Romance (18+)
- **Custom styles** - Create your own writing styles
- **Age ratings** - Content appropriate ratings
- **Content warnings** - Automatic content labeling

### 👥 Character Development (Fiction)
- **Character profiles** - Complete character sheets
- **Personality tracking** - Traits, strengths, weaknesses, fears
- **Speech patterns** - Unique dialogue for each character
- **Relationship matrix** - Track character relationships
- **Character arcs** - Growth and development tracking
- **Plot tracking** - Monitor plot points and resolutions
- **Dialogue consistency** - Maintain character voice

### 📤 Multiple Export Formats
- **EPUB** - Professional e-book format
- **PDF** - Print-ready documents
- **DOCX** - Microsoft Word compatible
- **HTML** - Web-ready format
- **Metadata support** - Author, title, language
- **Custom styling** - Format-specific layouts
- **Batch export** - Export to all formats at once

### ⚡ Performance Optimizations
- **Streaming responses** - Real-time content generation
- **Smart caching** - Avoid regenerating identical content
- **Token optimization** - Sliding window context management
- **Background processing** - Non-blocking book generation
- **Multiple cache backends** - Memory, Redis, File-based
- **Progress tracking** - Monitor generation in real-time

### 🧠 Hybrid RAG System (NEW)
- **Semantic Search** - FAISS-powered vector search for relevant context retrieval
- **Smart Summarization** - LLM-based chapter summaries for better context
- **Hybrid Context** - Intelligent token allocation (40% core, 40% RAG, 20% summaries)
- **Vector Indexing** - Automatic book content indexing with sentence-transformers
- **Persistent Storage** - Vector stores saved in `.rag/` directory per book
- **Backward Compatible** - Seamless fallback to legacy system when needed
- **Configurable Modes** - Choose from disabled, basic, hybrid, or full RAG modes

## Installation

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

## Configuration

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

### Advanced Features

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

## Project Structure

```
ghostwriter-ai/
├── Core Modules
│   ├── main.py                        # Application entry point - orchestrates book generation
│   ├── app_config.py                  # Settings management with Pydantic/fallback
│   ├── containers.py                  # Dependency injection with thread safety
│   └── services/
│       └── generation_service.py      # Core book generation logic and workflow
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
├── Advanced Features
│   ├── token_optimizer_rag.py         # Hybrid RAG with FAISS + smart summarization
│   ├── character_development.py       # Character profiles, arcs, relationships
│   ├── style_templates.py             # 15+ writing styles (fiction/non-fiction)
│   ├── export_formats.py              # Multi-format export (EPUB/PDF/DOCX/HTML)
│   ├── project_manager.py             # Project isolation and lifecycle management
│   └── prompts_templated.py           # Template-based prompt management
│
├── Infrastructure
│   ├── cache_manager.py               # Multi-backend caching (memory/Redis/file)
│   ├── events.py                      # Event-driven architecture for monitoring
│   ├── exceptions.py                  # Custom exception hierarchy
│   ├── streaming.py                   # Real-time content streaming
│   ├── background_tasks.py            # Async task processing with Celery/RQ
│   └── tokenizer.py                   # Token counting utilities
│
├── Configuration
│   ├── pyproject.toml                 # Project config, Ruff, MyPy, Pytest settings
│   ├── requirements.txt               # Core dependencies
│   ├── requirements-dev.txt           # Development dependencies
│   ├── env.example                    # Environment variables template
│   └── templates/
│       └── prompts.yaml               # Customizable prompt templates
│
├── Testing
│   ├── tests/
│   │   ├── conftest.py               # Shared pytest fixtures
│   │   ├── test_integration.py       # End-to-end integration tests
│   │   └── unit/
│   │       ├── test_generation_service.py
│   │       └── test_rag_integration.py
│   └── Makefile                       # Test automation commands
│
├── Documentation
│   ├── README.md                      # This file
│   ├── CLAUDE.md                      # AI assistant instructions
│   ├── PERFORMANCE.md                 # Performance optimization guide
│   └── primer.md                      # Quick start guide
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

## Architectural Patterns & Design Principles

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

## Code Architecture Highlights

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

## Troubleshooting

### Rate Limits
The app automatically handles rate limits with exponential backoff. No action needed.

### Token Limits
History is automatically managed to fit within token limits. Adjust `TOKEN_LIMIT` if needed.

### Partial Generation
If generation fails, check `books/<title>/book.json.partial` for recovery.

### Provider Issues
- Ensure API keys are valid
- Check provider service status
- Verify model availability
- Review error logs for details

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