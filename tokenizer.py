"""
Unified tokenization module for all LLM providers
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

# Import provider-specific tokenizers
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages"""
        total = 0
        for message in messages:
            # Add tokens for role and content
            if 'role' in message:
                total += self.count_tokens(message['role'])
            if 'content' in message:
                total += self.count_tokens(message['content'])
            # Add overhead for message structure (approximation)
            total += 4  # Typical overhead per message
        return total


class DefaultTokenizer(BaseTokenizer):
    """Default tokenizer with rough approximation"""

    def count_tokens(self, text: str) -> int:
        """Approximate token count (1 token ≈ 4 characters)"""
        if not text:
            return 0
        # More accurate approximation considering spaces and punctuation
        words = text.split()
        # Average: 1.3 tokens per word
        return int(len(words) * 1.3)


class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer using tiktoken"""

    def __init__(self, model_name: str = "gpt-4"):
        super().__init__(model_name)
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for OpenAI tokenizer")

        try:
            # Try to get encoding for specific model
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding (GPT-4, GPT-3.5-turbo)
            self.logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        if not text:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            self.logger.warning(f"Tiktoken encoding failed: {e}, using fallback")
            return DefaultTokenizer().count_tokens(text)


class AnthropicTokenizer(BaseTokenizer):
    """Anthropic tokenizer"""

    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        super().__init__(model_name)
        self.client = None

        if ANTHROPIC_AVAILABLE:
            try:
                # Try to initialize client for token counting
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                self.logger.warning(f"Could not initialize Anthropic client: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens for Anthropic models"""
        if not text:
            return 0

        # If we have a client, try to use official token counting
        if self.client and hasattr(self.client, 'count_tokens'):
            try:
                return self.client.count_tokens(text)
            except Exception as e:
                self.logger.debug(f"Anthropic token counting failed: {e}")

        # Fallback: Anthropic's Claude models use similar tokenization to GPT
        # Use a slightly different ratio
        words = text.split()
        return int(len(words) * 1.2)  # Slightly fewer tokens per word than OpenAI


class GeminiTokenizer(BaseTokenizer):
    """Google Gemini tokenizer"""

    def __init__(self, model_name: str = "gemini-pro"):
        super().__init__(model_name)
        self.model = None

        try:
            import google.generativeai as genai
            # Try to get the model for token counting
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            self.logger.warning(f"Could not initialize Gemini model: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens for Gemini models"""
        if not text:
            return 0

        # If we have a model instance, try to use official token counting
        if self.model and hasattr(self.model, 'count_tokens'):
            try:
                result = self.model.count_tokens(text)
                return result.total_tokens if hasattr(result, 'total_tokens') else result
            except Exception as e:
                self.logger.debug(f"Gemini token counting failed: {e}")

        # Fallback: Gemini uses SentencePiece tokenization
        # Approximation based on character count
        return len(text) // 4


class CohereTokenizer(BaseTokenizer):
    """Cohere tokenizer"""

    def __init__(self, model_name: str = "command"):
        super().__init__(model_name)
        self.client = None

        try:
            import os

            import cohere
            api_key = os.getenv("COHERE_API_KEY")
            if api_key:
                self.client = cohere.Client(api_key)
        except Exception as e:
            self.logger.warning(f"Could not initialize Cohere client: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens for Cohere models"""
        if not text:
            return 0

        # If we have a client, try to use tokenize endpoint
        if self.client and hasattr(self.client, 'tokenize'):
            try:
                response = self.client.tokenize(text=text, model=self.model_name)
                return len(response.tokens) if hasattr(response, 'tokens') else len(response)
            except Exception as e:
                self.logger.debug(f"Cohere tokenization failed: {e}")

        # Fallback approximation
        return len(text) // 4


class TokenizerFactory:
    """Factory for creating appropriate tokenizers"""

    _tokenizers = {
        'openai': OpenAITokenizer,
        'anthropic': AnthropicTokenizer,
        'gemini': GeminiTokenizer,
        'cohere': CohereTokenizer,
        'openrouter': DefaultTokenizer,  # OpenRouter uses various models
        'default': DefaultTokenizer
    }

    @classmethod
    def create(cls, provider: str, model_name: str = None) -> BaseTokenizer:
        """
        Create a tokenizer for the specified provider
        
        Args:
            provider: Provider name
            model_name: Optional model name
            
        Returns:
            Appropriate tokenizer instance
        """
        provider_lower = provider.lower()

        # Special handling for OpenRouter - detect underlying model
        if provider_lower == 'openrouter' and model_name:
            if 'gpt' in model_name.lower() or 'openai' in model_name.lower():
                tokenizer_class = cls._tokenizers['openai']
            elif 'claude' in model_name.lower() or 'anthropic' in model_name.lower():
                tokenizer_class = cls._tokenizers['anthropic']
            elif 'gemini' in model_name.lower() or 'google' in model_name.lower():
                tokenizer_class = cls._tokenizers['gemini']
            elif 'command' in model_name.lower() or 'cohere' in model_name.lower():
                tokenizer_class = cls._tokenizers['cohere']
            else:
                tokenizer_class = cls._tokenizers['default']
        else:
            tokenizer_class = cls._tokenizers.get(provider_lower, cls._tokenizers['default'])

        try:
            return tokenizer_class(model_name)
        except Exception as e:
            logging.warning(f"Failed to create {tokenizer_class.__name__}: {e}, using default")
            return DefaultTokenizer(model_name)

    @classmethod
    def register(cls, provider: str, tokenizer_class: type):
        """Register a custom tokenizer for a provider"""
        cls._tokenizers[provider.lower()] = tokenizer_class
