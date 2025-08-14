"""
Base LLM Provider Interface
"""
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LLMResponse:
    """Standard response format from LLM providers"""
    content: str
    tokens_used: int
    finish_reason: str
    model: str
    raw_response: Optional[Dict[str, Any]] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider with configuration
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """Validate provider configuration"""
        pass

    @abstractmethod
    def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """
        Generate text completion
        
        Args:
            prompt: The input prompt
            history: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        pass

    def validate_token_limit(self, messages: List[Dict[str, str]], max_tokens: int) -> bool:
        """
        Check if messages fit within token limit
        
        Args:
            messages: List of messages
            max_tokens: Maximum tokens for response
            
        Returns:
            True if within limit, False otherwise
        """
        total_tokens = sum(self.count_tokens(msg.get('content', '')) for msg in messages)
        return total_tokens + max_tokens <= self.get_model_info().get('max_tokens', 4096)

    def prepare_messages(self, prompt: str, history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Prepare messages for API call
        
        Args:
            prompt: Current prompt
            history: Conversation history
            
        Returns:
            Formatted messages list
        """
        messages = history.copy() if history else []
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """
        Generate text with streaming (optional implementation)
        
        Args:
            prompt: Input prompt
            history: Conversation history  
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Text chunks as they're generated
        """
        # Default implementation: yield complete response
        response = self.generate(prompt, history, max_tokens, temperature, **kwargs)
        yield response.content

    def _call_with_retry(self,
                        api_call: Callable,
                        max_retries: int = 3,
                        base_delay: float = 1.0,
                        max_delay: float = 60.0,
                        exponential_base: float = 2.0,
                        jitter: bool = True,
                        retry_on: Optional[List[type]] = None,
                        **kwargs) -> Any:
        """
        Generic retry logic for API calls
        
        Args:
            api_call: Function to call
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
            retry_on: List of exception types to retry on (None = all)
            **kwargs: Arguments to pass to api_call
            
        Returns:
            Result from successful API call
            
        Raises:
            Last exception if all retries failed
        """
        retry_on = retry_on or [Exception]
        last_exception = None

        for attempt in range(max_retries):
            try:
                return api_call(**kwargs)

            except tuple(retry_on) as e:
                last_exception = e

                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter if requested
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    time.sleep(delay)
                else:
                    self.logger.error(f"All {max_retries} attempts failed. Last error: {e}")

        raise last_exception

    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry
        Can be overridden by subclasses for provider-specific logic
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        # Common retryable conditions
        error_str = str(exception).lower()
        retryable_patterns = [
            'rate limit',
            'quota',
            'timeout',
            'connection',
            'temporary',
            '429',  # Too Many Requests
            '503',  # Service Unavailable
            '504',  # Gateway Timeout
        ]

        return any(pattern in error_str for pattern in retryable_patterns)
