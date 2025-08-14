# GhostWriter AI: Multi-Provider AI Book Writer

An advanced AI-powered book writing application that supports multiple LLM providers, comprehensive error handling, and real-time progress tracking.

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
├── providers/              # LLM provider implementations
│   ├── base.py            # Base provider interface
│   ├── openai_provider.py # OpenAI implementation
│   ├── anthropic_provider.py
│   ├── cohere_provider.py
│   ├── gemini_provider.py
│   └── openrouter_provider.py
├── main.py                # Main application entry
├── generate.py            # Book generation logic
├── ai.py                  # AI interface layer
├── ai_enhanced.py         # Enhanced AI with optimizations
├── config.py              # Configuration management
├── events.py              # Event system
├── exceptions.py          # Custom exceptions
├── prompts.py             # Prompt templates
├── bookprinter.py         # Markdown output generation
├── project_manager.py     # Project isolation & management
├── style_templates.py     # Writing style templates
├── character_development.py # Character tracking (fiction)
├── export_formats.py      # EPUB/PDF/DOCX/HTML export
├── streaming.py           # Real-time streaming
├── cache_manager.py       # Smart caching system
├── token_optimizer.py     # Context window management
├── background_tasks.py    # Async task processing
├── PERFORMANCE.md         # Performance optimization guide
└── requirements.txt       # Python dependencies
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