"""
Enhanced configuration with provider selection and error handling
"""
import logging
import os

from dotenv import load_dotenv

from exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO, format='[%(levelname)s][%(module)s.%(funcName)s] %(message)s')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(f'{BASE_DIR}/.env', verbose=True)

def get_int_env(name, default):
    """Get integer environment variable with error handling"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigurationError(f"Invalid integer value for {name}: {value}")

def get_float_env(name, default):
    """Get float environment variable with error handling"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigurationError(f"Invalid float value for {name}: {value}")

# Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Provider-specific configurations
PROVIDER_CONFIG = {}

if LLM_PROVIDER == "openai":
    PROVIDER_CONFIG = {
        'provider': 'openai',
        'api_key': os.getenv("OPENAI_API_KEY"),
        'model': os.getenv("OPENAI_MODEL"),
        'engine': os.getenv("OPENAI_ENGINE"),
        'api_base': os.getenv("OPENAI_API_BASE"),
        'api_type': os.getenv("OPENAI_API_TYPE"),
        'api_version': os.getenv("OPENAI_API_VERSION"),
    }

    if not PROVIDER_CONFIG['api_key'] and not os.getenv('PYTEST_CURRENT_TEST'):
        raise ConfigurationError("OPENAI_API_KEY is required for OpenAI provider")
    if not PROVIDER_CONFIG['model'] and not PROVIDER_CONFIG['engine'] and not os.getenv('PYTEST_CURRENT_TEST'):
        raise ConfigurationError("Either OPENAI_MODEL or OPENAI_ENGINE must be set")

elif LLM_PROVIDER == "anthropic":
    PROVIDER_CONFIG = {
        'provider': 'anthropic',
        'api_key': os.getenv("ANTHROPIC_API_KEY"),
        'model': os.getenv("ANTHROPIC_MODEL", "claude-opus-4.1-20250805"),  # Claude 4.1 - best coding model
    }

    if not PROVIDER_CONFIG['api_key'] and not os.getenv('PYTEST_CURRENT_TEST'):
        raise ConfigurationError("ANTHROPIC_API_KEY is required for Anthropic provider")

elif LLM_PROVIDER == "cohere":
    PROVIDER_CONFIG = {
        'provider': 'cohere',
        'api_key': os.getenv("COHERE_API_KEY"),
        'model': os.getenv("COHERE_MODEL", "command-r-plus"),
    }

    if not PROVIDER_CONFIG['api_key'] and not os.getenv('PYTEST_CURRENT_TEST'):
        raise ConfigurationError("COHERE_API_KEY is required for Cohere provider")

elif LLM_PROVIDER == "gemini":
    PROVIDER_CONFIG = {
        'provider': 'gemini',
        'api_key': os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        'model': os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),  # Gemini 2.5 Pro with thinking
    }

    if not PROVIDER_CONFIG['api_key'] and not os.getenv('PYTEST_CURRENT_TEST'):
        raise ConfigurationError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini provider")

elif LLM_PROVIDER == "openrouter":
    PROVIDER_CONFIG = {
        'provider': 'openrouter',
        'api_key': os.getenv("OPENROUTER_API_KEY"),
        'model': os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.1"),  # Claude 4.1 via OpenRouter
        'site_url': os.getenv("OPENROUTER_SITE_URL", "https://github.com/ghostwriter-ai"),
        'site_name': os.getenv("OPENROUTER_SITE_NAME", "GhostWriter AI"),
    }

    if not PROVIDER_CONFIG['api_key'] and not os.getenv('PYTEST_CURRENT_TEST'):
        raise ConfigurationError("OPENROUTER_API_KEY is required for OpenRouter provider")

else:
    raise ConfigurationError(f"Unknown LLM provider: {LLM_PROVIDER}")

# Common configuration
TEMPERATURE = get_float_env("TEMPERATURE", 0.2)
TOKEN_LIMIT = get_int_env("TOKEN_LIMIT", 4096)
MAX_TOKENS = get_int_env("MAX_TOKENS", TOKEN_LIMIT // 4)
MAX_TOKENS_SHORT = get_int_env("MAX_TOKENS_SHORT", MAX_TOKENS // 4)

# Add to provider config
PROVIDER_CONFIG['token_limit'] = TOKEN_LIMIT

# Book configuration
BOOK_LANGUAGE = os.getenv("BOOK_LANGUAGE")
BOOK_TITLE = os.getenv("BOOK_TITLE")
BOOK_INSTRUCTIONS = os.getenv("BOOK_INSTRUCTIONS")

# Event configuration
ENABLE_PROGRESS_TRACKING = os.getenv("ENABLE_PROGRESS_TRACKING", "true").lower() == "true"
PROGRESS_CALLBACK_URL = os.getenv("PROGRESS_CALLBACK_URL")  # Optional webhook for progress updates

logging.info(f'''
>> Config:
LLM_PROVIDER: {LLM_PROVIDER}
MODEL: {PROVIDER_CONFIG.get('model') or PROVIDER_CONFIG.get('engine')}
TEMPERATURE: {TEMPERATURE}
TOKEN_LIMIT: {TOKEN_LIMIT}
MAX_TOKENS: {MAX_TOKENS}
MAX_TOKENS_SHORT: {MAX_TOKENS_SHORT}
PROGRESS_TRACKING: {ENABLE_PROGRESS_TRACKING}
'''.strip())
