"""
Application Configuration with fallback for missing pydantic_settings
"""
import os
from typing import Optional

# Try to use pydantic_settings if available, otherwise use simple configuration
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    try:
        from pydantic import BaseSettings
        PYDANTIC_AVAILABLE = True
        SettingsConfigDict = dict  # Fallback for older pydantic versions
    except ImportError:
        PYDANTIC_AVAILABLE = False

if PYDANTIC_AVAILABLE:
    class AppSettings(BaseSettings):
        """
        Defines the application's configuration settings.
        Settings are loaded from environment variables or a.env file.
        """
        # API Keys
        OPENAI_API_KEY: Optional[str] = None
        ANTHROPIC_API_KEY: Optional[str] = None
        GEMINI_API_KEY: Optional[str] = None
        COHERE_API_KEY: Optional[str] = None
        OPENROUTER_API_KEY: Optional[str] = None

        # Provider settings
        LLM_PROVIDER: str = "openai"  # Default LLM provider
        
        # Generation parameters
        MAX_TOKENS: int = 1024
        TEMPERATURE: float = 0.7
        TOKEN_LIMIT: int = 4096

        # Optional settings with default values
        CACHE_TYPE: str = "memory"
        CACHE_TTL_SECONDS: int = 3600
        LOG_LEVEL: str = "INFO"

        # Book settings
        BASE_DIR: str = "."
        BOOK_INSTRUCTIONS: Optional[str] = None
        BOOK_LANGUAGE: Optional[str] = None
        BOOK_TITLE: Optional[str] = None
        ENABLE_PROGRESS_TRACKING: bool = False

        # RAG settings
        ENABLE_RAG: bool = True  # Enable RAG features
        RAG_MODE: str = "hybrid"  # Options: disabled, basic, hybrid, full
        RAG_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        RAG_CHUNK_SIZE: int = 512
        RAG_TOP_K: int = 10
        RAG_SIMILARITY_THRESHOLD: float = 0.5
        RAG_CORE_CONTEXT_RATIO: float = 0.4
        RAG_RETRIEVED_CONTEXT_RATIO: float = 0.4
        RAG_SUMMARY_CONTEXT_RATIO: float = 0.2

        # This tells Pydantic to look for a.env file
        if hasattr(BaseSettings, 'Config'):
            # Pydantic v1 style
            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"
        else:
            # Pydantic v2 style
            model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
else:
    # Fallback configuration class when pydantic is not available
    class AppSettings:
        """
        Simple configuration class that reads from environment variables
        """
        def __init__(self):
            # Load .env file if it exists
            self._load_env_file()

            # API Keys
            self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
            self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            self.COHERE_API_KEY = os.getenv('COHERE_API_KEY')
            self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

            # Provider settings
            self.LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
            
            # Generation parameters
            self.MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1024'))
            self.TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
            self.TOKEN_LIMIT = int(os.getenv('TOKEN_LIMIT', '4096'))

            # Optional settings with default values
            self.CACHE_TYPE = os.getenv('CACHE_TYPE', 'memory')
            self.CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))
            self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

            # Book settings
            self.BASE_DIR = os.getenv('BASE_DIR', '.')
            self.BOOK_INSTRUCTIONS = os.getenv('BOOK_INSTRUCTIONS')
            self.BOOK_LANGUAGE = os.getenv('BOOK_LANGUAGE')
            self.BOOK_TITLE = os.getenv('BOOK_TITLE')
            progress_track = os.getenv('ENABLE_PROGRESS_TRACKING', 'False')
            self.ENABLE_PROGRESS_TRACKING = progress_track.lower() == 'true'

            # RAG settings
            self.ENABLE_RAG = os.getenv('ENABLE_RAG', 'True').lower() == 'true'
            self.RAG_MODE = os.getenv('RAG_MODE', 'hybrid')
            self.RAG_EMBEDDING_MODEL = os.getenv('RAG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.RAG_CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '512'))
            self.RAG_TOP_K = int(os.getenv('RAG_TOP_K', '10'))
            self.RAG_SIMILARITY_THRESHOLD = float(os.getenv('RAG_SIMILARITY_THRESHOLD', '0.5'))
            self.RAG_CORE_CONTEXT_RATIO = float(os.getenv('RAG_CORE_CONTEXT_RATIO', '0.4'))
            retrieved_ratio = os.getenv('RAG_RETRIEVED_CONTEXT_RATIO', '0.4')
            self.RAG_RETRIEVED_CONTEXT_RATIO = float(retrieved_ratio)
            self.RAG_SUMMARY_CONTEXT_RATIO = float(os.getenv('RAG_SUMMARY_CONTEXT_RATIO', '0.2'))

        def _load_env_file(self):
            """Load environment variables from .env file if it exists"""
            env_file = '.env'
            if os.path.exists(env_file):
                try:
                    with open(env_file, encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                if key not in os.environ:
                                    os.environ[key] = value
                except Exception:
                    pass  # Ignore errors in .env file

# Create a single, global instance of the settings
try:
    settings = AppSettings()
except Exception:
    # If initialization fails (e.g., missing required env vars in strict mode),
    # create with defaults
    if PYDANTIC_AVAILABLE:
        settings = AppSettings(
            OPENAI_API_KEY=os.getenv('OPENAI_API_KEY'),
            ANTHROPIC_API_KEY=os.getenv('ANTHROPIC_API_KEY')
        )
    else:
        settings = AppSettings()
