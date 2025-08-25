# GhostWriter AI - System Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [LLM Provider Architecture](#llm-provider-architecture)
5. [RAG System Architecture](#rag-system-architecture)
6. [Event-Driven Architecture](#event-driven-architecture)
7. [Database/Storage Schema](#databasestorage-schema)
8. [API Structure](#api-structure)
9. [Deployment Architecture](#deployment-architecture)
10. [Technology Stack](#technology-stack)

## System Overview

GhostWriter AI is a sophisticated AI-powered book generation system that leverages multiple LLM providers, advanced RAG capabilities, and event-driven architecture to create complete books with coherent narratives, rich character development, and consistent style.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interface                              │
│                      (CLI Handler / Web API)                         │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                       Main Orchestrator                              │
│                         (main.py)                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │Event Manager │  │  DI Container │  │  Service Initializer     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                    Book Generation Layer                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Book Generator                             │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │  │
│  │  │ Checkpoint │  │Character Dev │  │  Style Templates  │   │  │
│  │  │  Manager   │  │   Manager    │  │     Manager       │   │  │
│  │  └────────────┘  └──────────────┘  └───────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                    AI Generation Service                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Generation Service Orchestrator                  │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │  │
│  │  │   Token    │  │     RAG      │  │   Prompt Service  │   │  │
│  │  │ Optimizer  │  │   Manager    │  │                   │   │  │
│  │  └────────────┘  └──────────────┘  └───────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                     LLM Provider Layer                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Provider Factory                             │  │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │  │
│  │  │ OpenAI  │ │Anthropic │ │ Gemini │ │ Cohere │ │  Open  │ │  │
│  │  │Provider │ │ Provider │ │Provider│ │Provider│ │ Router │ │  │
│  │  └─────────┘ └──────────┘ └────────┘ └────────┘ └────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                    Infrastructure Layer                              │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │   Cache    │  │    Storage   │  │   Background Tasks         │  │
│  │  Manager   │  │   (Files)    │  │      Manager               │  │
│  └────────────┘  └──────────────┘  └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Detailed Component Relationships

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Handler]
        API[Web API<br/>Future]
    end

    subgraph "Orchestration Layer"
        MAIN[Main Entry Point]
        DI[DI Container]
        EM[Event Manager]
        SI[Service Initializer]
    end

    subgraph "Book Generation Core"
        BG[Book Generator]
        CM[Character Manager]
        SM[Style Manager]
        CHK[Checkpoint Manager]
        PM[Project Manager]
    end

    subgraph "AI Services"
        GS[Generation Service]
        PS[Prompt Service]
        TO[Token Optimizer]
        RAG[RAG System]
    end

    subgraph "RAG Components"
        HCM[Hybrid Context Manager]
        VDB[FAISS Vector DB]
        EMB[Embeddings Model]
        RET[Retriever]
        SUM[Summarizer]
        CHU[Chunker]
        IND[Indexer]
    end

    subgraph "Provider System"
        PF[Provider Factory]
        BP[Base Provider]
        OP[OpenAI]
        AP[Anthropic]
        GP[Gemini]
        CP[Cohere]
        OR[OpenRouter]
    end

    subgraph "Infrastructure"
        CACHE[Cache Manager]
        FS[File System]
        BTM[Background Tasks]
        EXP[Export Formats]
    end

    %% User interactions
    CLI --> MAIN
    API -.-> MAIN

    %% Main orchestration
    MAIN --> DI
    MAIN --> EM
    MAIN --> SI
    SI --> GS
    DI --> GS
    DI --> CACHE
    DI --> PF

    %% Book generation flow
    MAIN --> BG
    BG --> CM
    BG --> SM
    BG --> CHK
    BG --> PM
    BG --> GS

    %% Generation service connections
    GS --> PS
    GS --> TO
    GS --> RAG
    GS --> PF

    %% RAG system connections
    RAG --> HCM
    HCM --> VDB
    HCM --> EMB
    HCM --> RET
    HCM --> SUM
    HCM --> CHU
    HCM --> IND

    %% Provider connections
    PF --> BP
    BP --> OP
    BP --> AP
    BP --> GP
    BP --> CP
    BP --> OR

    %% Infrastructure connections
    GS --> CACHE
    BG --> FS
    BG --> BTM
    BG --> EXP

    %% Event system connections
    EM -.-> BG
    EM -.-> GS
    EM -.-> CHK

    classDef interface fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef ai fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef provider fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef infra fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class CLI,API interface
    class BG,CM,SM,CHK,PM core
    class GS,PS,TO,RAG,HCM,VDB,EMB,RET,SUM,CHU,IND ai
    class PF,BP,OP,AP,GP,CP,OR provider
    class CACHE,FS,BTM,EXP infra
```

## Data Flow Diagrams

### Book Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Main
    participant BookGen
    participant GenService
    participant RAG
    participant Provider
    participant Cache
    participant Storage

    User->>CLI: Enter book parameters
    CLI->>Main: Initialize system
    Main->>BookGen: Start generation
    
    loop For each chapter
        BookGen->>GenService: Generate outline
        GenService->>Cache: Check cache
        alt Cache miss
            GenService->>RAG: Prepare context
            RAG->>RAG: Index content
            RAG->>RAG: Retrieve relevant chunks
            RAG->>RAG: Summarize if needed
            RAG-->>GenService: Return context
            GenService->>Provider: Send prompt
            Provider-->>GenService: Return content
            GenService->>Cache: Store result
        else Cache hit
            Cache-->>GenService: Return cached
        end
        GenService-->>BookGen: Return outline
        
        loop For each section
            BookGen->>GenService: Generate content
            Note over GenService,RAG: Similar flow as outline
            GenService-->>BookGen: Return content
        end
        
        BookGen->>Storage: Save chapter
        BookGen->>BookGen: Update checkpoint
    end
    
    BookGen->>BookGen: Export formats
    BookGen-->>User: Book complete
```

### RAG Context Preparation Flow

```mermaid
flowchart LR
    subgraph "Input"
        Q[Query/Prompt]
        BC[Book Content]
        CC[Current Chapter]
    end

    subgraph "RAG Processing"
        subgraph "Indexing"
            CH[Chunker]
            EMB[Embedder]
            IDX[FAISS Index]
        end
        
        subgraph "Retrieval"
            SEM[Semantic Search]
            RANK[Re-ranking]
            FILT[Filtering]
        end
        
        subgraph "Context Building"
            CORE[Core Context]
            RAGC[RAG Context]
            SUMC[Summary Context]
            MERGE[Context Merger]
        end
    end

    subgraph "Output"
        CTX[Final Context]
    end

    Q --> SEM
    BC --> CH
    CC --> CORE
    
    CH --> EMB
    EMB --> IDX
    IDX --> SEM
    SEM --> RANK
    RANK --> FILT
    FILT --> RAGC
    
    BC --> SUMC
    
    CORE --> MERGE
    RAGC --> MERGE
    SUMC --> MERGE
    MERGE --> CTX
```

## LLM Provider Architecture

### Provider Factory Pattern

```mermaid
classDiagram
    class LLMProvider {
        <<abstract>>
        +generate(prompt: str, **kwargs)
        +stream(prompt: str, **kwargs)
        +validate_config()
        +get_info()
        #_execute_with_retry()
    }

    class BaseProvider {
        <<abstract>>
        -max_retries: int
        -retry_delay: float
        -timeout: int
        +generate(prompt, **kwargs)
        +stream(prompt, **kwargs)
        #_make_request()
        #_handle_error()
    }

    class ProviderFactory {
        -providers: Dict
        +register_provider(name, class)
        +create_provider(name, config)
        +list_providers()
    }

    class OpenAIProvider {
        -client: OpenAI
        -model: str
        +generate(prompt, **kwargs)
        +stream(prompt, **kwargs)
    }

    class AnthropicProvider {
        -client: Anthropic
        -model: str
        +generate(prompt, **kwargs)
        +stream(prompt, **kwargs)
    }

    class GeminiProvider {
        -model: GenerativeModel
        +generate(prompt, **kwargs)
        +stream(prompt, **kwargs)
    }

    class CohereProvider {
        -client: Client
        -model: str
        +generate(prompt, **kwargs)
        +stream(prompt, **kwargs)
    }

    class OpenRouterProvider {
        -base_url: str
        -api_key: str
        +generate(prompt, **kwargs)
        +stream(prompt, **kwargs)
    }

    LLMProvider <|-- BaseProvider
    BaseProvider <|-- OpenAIProvider
    BaseProvider <|-- AnthropicProvider
    BaseProvider <|-- GeminiProvider
    BaseProvider <|-- CohereProvider
    BaseProvider <|-- OpenRouterProvider
    ProviderFactory ..> LLMProvider : creates
    ProviderFactory o-- OpenAIProvider
    ProviderFactory o-- AnthropicProvider
    ProviderFactory o-- GeminiProvider
    ProviderFactory o-- CohereProvider
    ProviderFactory o-- OpenRouterProvider
```

### Provider Features

| Provider | Streaming | Retry Logic | Token Counting | Error Recovery | Rate Limiting |
|----------|-----------|-------------|----------------|----------------|---------------|
| OpenAI | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anthropic | ✓ | ✓ | ✓ | ✓ | ✓ |
| Gemini | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cohere | ✓ | ✓ | ✓ | ✓ | ✓ |
| OpenRouter | ✓ | ✓ | ✓ | ✓ | ✓ |

## RAG System Architecture

### Hybrid RAG Components

```mermaid
graph TB
    subgraph "RAG Configuration"
        CONFIG[RAGConfig]
        MODE[RAG Modes<br/>- Disabled<br/>- Basic<br/>- Hybrid<br/>- Full]
    end

    subgraph "Document Processing"
        DOC[Document Input]
        CHUNK[Smart Chunker<br/>- Sentence boundary<br/>- Token-aware<br/>- Metadata preservation]
        META[Chunk Metadata<br/>- Chapter/Section<br/>- Position<br/>- Importance]
    end

    subgraph "Embedding & Indexing"
        EMBED[Sentence Transformer<br/>- all-MiniLM-L6-v2<br/>- GPU accelerated]
        FAISS[FAISS Index<br/>- IVF indexing<br/>- GPU support<br/>- Batch processing]
        VCACHE[Vector Cache<br/>- LRU with TTL<br/>- Frequent queries]
    end

    subgraph "Retrieval System"
        SEARCH[Semantic Search<br/>- Top-K retrieval<br/>- Similarity threshold]
        RERANK[Re-ranking<br/>- MMR diversity<br/>- Relevance scoring]
        FILTER[Context Filtering<br/>- Deduplication<br/>- Length constraints]
    end

    subgraph "Context Optimization"
        SUMM[Smart Summarizer<br/>- LLM-based<br/>- Importance weighting]
        ALLOC[Token Allocator<br/>- Core: 40%<br/>- RAG: 30%<br/>- Summary: 30%]
        BUILD[Context Builder<br/>- Structured output<br/>- Priority ordering]
    end

    CONFIG --> MODE
    MODE --> DOC
    DOC --> CHUNK
    CHUNK --> META
    CHUNK --> EMBED
    EMBED --> FAISS
    FAISS --> VCACHE
    
    VCACHE --> SEARCH
    SEARCH --> RERANK
    RERANK --> FILTER
    
    FILTER --> ALLOC
    META --> SUMM
    SUMM --> ALLOC
    ALLOC --> BUILD

    style CONFIG fill:#e3f2fd
    style FAISS fill:#fff9c4
    style SUMM fill:#f3e5f5
```

### RAG Performance Optimizations

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG Performance Features                   │
├─────────────────────────────────────────────────────────────┤
│ • IVF Indexing: 100x faster search for large documents      │
│ • GPU Acceleration: 10x faster embeddings with CUDA         │
│ • Batch Processing: Process multiple queries simultaneously │
│ • Vector Caching: LRU cache with TTL for frequent queries   │
│ • Smart Chunking: Context-aware splitting at boundaries     │
│ • Parallel Processing: Multi-threaded document processing   │
│ • Memory Mapping: Efficient handling of large indices       │
│ • Incremental Indexing: Add new content without rebuild     │
└─────────────────────────────────────────────────────────────┘
```

## Event-Driven Architecture

### Event System

```mermaid
stateDiagram-v2
    [*] --> EventManager
    
    EventManager --> EventEmit
    EventEmit --> EventHandlers
    
    state EventHandlers {
        [*] --> ProgressTracker
        [*] --> LoggingHandler
        [*] --> CheckpointHandler
        [*] --> ErrorHandler
        [*] --> MetricsCollector
    }
    
    EventHandlers --> EventProcessing
    
    state EventProcessing {
        Synchronous --> CallbackExecution
        Asynchronous --> TaskQueue
        TaskQueue --> BackgroundWorker
    }
    
    EventProcessing --> EventCompletion
    EventCompletion --> [*]
```

### Event Types and Handlers

```python
EventType Enum:
├── BOOK_STARTED
├── BOOK_COMPLETED
├── BOOK_FAILED
├── BOOK_EXPORTED
├── CHAPTER_STARTED
├── CHAPTER_COMPLETED
├── CHAPTER_FAILED
├── SECTION_STARTED
├── SECTION_COMPLETED
├── GENERATION_STARTED
├── GENERATION_COMPLETED
├── GENERATION_RETRY
├── CACHE_HIT
├── CACHE_MISS
├── RAG_INDEXING_STARTED
├── RAG_INDEXING_COMPLETED
├── RAG_RETRIEVAL_STARTED
├── RAG_RETRIEVAL_COMPLETED
├── CHECKPOINT_SAVED
├── CHECKPOINT_LOADED
└── ERROR_OCCURRED
```

### Observer Pattern Implementation

```mermaid
classDiagram
    class EventManager {
        -subscribers: Dict
        -event_queue: Queue
        -thread_pool: ThreadPoolExecutor
        +subscribe(event_type, handler)
        +unsubscribe(event_type, handler)
        +emit(event)
        +emit_async(event)
        +subscribe_all(handler)
    }

    class Event {
        +type: EventType
        +data: Dict
        +timestamp: datetime
        +source: str
    }

    class EventHandler {
        <<interface>>
        +handle(event)
    }

    class ProgressTracker {
        -progress: Dict
        +track_progress(event)
        +get_progress()
        +get_stats()
    }

    class CheckpointHandler {
        -checkpoint_manager: CheckpointManager
        +handle(event)
        +save_checkpoint()
        +load_checkpoint()
    }

    class MetricsCollector {
        -metrics: Dict
        +collect(event)
        +export_metrics()
    }

    EventManager --> Event : emits
    EventManager --> EventHandler : notifies
    EventHandler <|-- ProgressTracker
    EventHandler <|-- CheckpointHandler
    EventHandler <|-- MetricsCollector
```

## Database/Storage Schema

### File System Organization

```
projects/
├── {book_title}/
│   ├── book.json                 # Book metadata and structure
│   ├── chapters/
│   │   ├── chapter_01/
│   │   │   ├── metadata.json     # Chapter metadata
│   │   │   ├── outline.json      # Chapter outline
│   │   │   ├── content.md        # Generated content
│   │   │   └── sections/
│   │   │       ├── section_01.md
│   │   │       └── section_02.md
│   │   └── chapter_02/
│   │       └── ...
│   ├── characters/
│   │   ├── profiles.json         # Character profiles
│   │   └── relationships.json    # Character relationships
│   ├── checkpoints/
│   │   ├── checkpoint_latest.json
│   │   └── checkpoint_{timestamp}.json
│   ├── cache/
│   │   ├── embeddings/           # Cached embeddings
│   │   ├── summaries/            # Cached summaries
│   │   └── generations/          # Cached LLM outputs
│   ├── rag/
│   │   ├── index.faiss          # FAISS vector index
│   │   ├── chunks.json          # Document chunks
│   │   └── metadata.json        # RAG metadata
│   └── exports/
│       ├── book.epub
│       ├── book.pdf
│       ├── book.docx
│       └── book.html
```

### Data Models

```python
# Book Structure
Book:
    title: str
    original_title: str
    language: str
    instructions: str
    style: str
    chapters: List[Chapter]
    characters: List[Character]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# Chapter Structure
Chapter:
    number: int
    title: str
    outline: ChapterOutline
    sections: List[Section]
    summary: str
    word_count: int
    status: ChapterStatus

# Character Model
Character:
    name: str
    description: str
    role: CharacterRole
    traits: List[str]
    backstory: str
    arc: str
    relationships: List[Relationship]

# RAG Chunk Model
Chunk:
    id: str
    content: str
    embedding: np.ndarray
    metadata: ChunkMetadata
    chapter_num: int
    section_num: int
    importance_score: float
```

## API Structure

### Main API Endpoints (Future Web API)

```yaml
/api/v1:
  /books:
    POST: Create new book project
    GET: List all book projects
    
  /books/{book_id}:
    GET: Get book details
    PUT: Update book settings
    DELETE: Delete book project
    
  /books/{book_id}/generate:
    POST: Start book generation
    GET: Get generation status
    
  /books/{book_id}/chapters:
    GET: List all chapters
    POST: Generate specific chapter
    
  /books/{book_id}/chapters/{chapter_id}:
    GET: Get chapter content
    PUT: Edit chapter content
    DELETE: Delete chapter
    
  /books/{book_id}/export:
    POST: Export book to format
    GET: Get export status
    
  /providers:
    GET: List available LLM providers
    POST: Test provider configuration
    
  /styles:
    GET: List available writing styles
    
  /templates:
    GET: List prompt templates
    PUT: Update prompt template
```

### Internal Service APIs

```python
# Generation Service API
class GenerationService:
    def generate_text(provider_name, prompt, **kwargs) -> str
    def stream_text(provider_name, prompt, **kwargs) -> Generator
    def generate_with_context(book, chapter, prompt) -> str
    def summarize_content(content, max_tokens) -> str

# RAG Service API  
class RAGService:
    def index_content(content, metadata) -> None
    def search(query, top_k) -> List[Chunk]
    def prepare_context(book, chapter, query) -> str
    def update_index(new_content) -> None

# Character Service API
class CharacterService:
    def create_character(profile) -> Character
    def develop_character(character, context) -> Character
    def get_character_context(character, chapter) -> str
    def update_relationships(characters) -> None
```

## Deployment Architecture

### Container Architecture (Docker)

```yaml
version: '3.8'

services:
  ghostwriter-app:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ENABLE_RAG=true
      - RAG_MODE=hybrid
    volumes:
      - ./projects:/app/projects
      - ./cache:/app/cache
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - ghostwriter-app

volumes:
  redis-data:
```

### Production Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│                    (Nginx / HAProxy)                        │
└──────────┬────────────────────────────────┬─────────────────┘
           │                                │
    ┌──────▼──────┐                 ┌──────▼──────┐
    │   App Node 1│                 │   App Node 2│
    │  (Primary)  │                 │  (Secondary)│
    └──────┬──────┘                 └──────┬──────┘
           │                                │
           └────────────┬───────────────────┘
                        │
            ┌───────────▼───────────┐
            │    Shared Storage     │
            │   (NFS / S3 / GCS)    │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
  │   Redis   │  │PostgreSQL │  │   FAISS   │
  │   Cache   │  │ (Future)  │  │   Index   │
  └───────────┘  └───────────┘  └───────────┘
```

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.11+ | Primary development language |
| **Package Manager** | UV | Latest | Fast Python package management |
| **Framework** | FastAPI | 0.100+ | Web API framework (future) |
| **CLI** | Click | 8.0+ | Command-line interface |

### AI/ML Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Providers** | OpenAI API | GPT-4/GPT-3.5 text generation |
| | Anthropic API | Claude models |
| | Google Gemini | Gemini Pro models |
| | Cohere API | Command models |
| | OpenRouter | Multi-provider gateway |
| **Embeddings** | Sentence-Transformers | Document embeddings |
| **Vector DB** | FAISS | Similarity search |
| **ML Framework** | PyTorch | Deep learning operations |
| **NLP** | spaCy | Text processing |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Caching** | Redis | Distributed cache |
| **Queue** | Celery | Background tasks |
| **Storage** | Local FS / S3 | File storage |
| **Container** | Docker | Containerization |
| **Orchestration** | Kubernetes | Container orchestration |
| **Monitoring** | Prometheus | Metrics collection |
| **Logging** | ELK Stack | Log aggregation |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Linting** | Ruff | Fast Python linter |
| **Formatting** | Black | Code formatting |
| **Type Checking** | mypy | Static type checking |
| **Testing** | pytest | Unit/integration testing |
| **Documentation** | Sphinx | API documentation |
| **CI/CD** | GitHub Actions | Continuous integration |

### Key Libraries

```toml
[dependencies]
# Core
pydantic = "^2.0"
dependency-injector = "^4.41"

# AI/LLM
openai = "^1.0"
anthropic = "^0.25"
google-generativeai = "^0.5"
cohere = "^5.0"

# RAG System
sentence-transformers = "^2.5"
faiss-cpu = "^1.7"  # or faiss-gpu
torch = "^2.0"

# Export Formats
pypandoc = "^1.12"
ebooklib = "^0.18"
python-docx = "^1.0"
reportlab = "^4.0"

# Infrastructure
redis = "^5.0"
celery = "^5.3"
structlog = "^24.0"

# Development
pytest = "^8.0"
pytest-asyncio = "^0.23"
pytest-cov = "^4.0"
```

### Performance Characteristics

| Metric | Target | Current |
|--------|--------|---------|
| **Chapter Generation** | < 60s | ~45s |
| **RAG Query Time** | < 500ms | ~200ms |
| **Vector Search (1M docs)** | < 100ms | ~50ms |
| **Cache Hit Rate** | > 60% | ~70% |
| **Concurrent Books** | 10+ | 15 |
| **Memory Usage** | < 4GB | ~3GB |
| **GPU Utilization** | > 70% | ~80% |

---

## Architecture Principles

1. **Modularity**: Each component has a single responsibility
2. **Scalability**: Horizontal scaling through stateless services
3. **Resilience**: Retry logic, circuit breakers, and graceful degradation
4. **Performance**: Caching, async operations, and GPU acceleration
5. **Maintainability**: Clean architecture, dependency injection, and comprehensive testing
6. **Extensibility**: Plugin architecture for providers and export formats
7. **Observability**: Comprehensive logging, metrics, and event tracking

## Future Enhancements

- **Web UI**: React-based frontend for browser access
- **Collaborative Editing**: Multi-user book projects
- **Fine-tuning**: Custom model training on genre-specific data
- **Real-time Generation**: WebSocket-based streaming
- **Advanced RAG**: Knowledge graphs and hybrid search
- **Multi-modal**: Image generation for book illustrations
- **Voice Integration**: Text-to-speech for audiobook generation

---

*Last Updated: 2025-08-15*
*Version: 1.0.0*