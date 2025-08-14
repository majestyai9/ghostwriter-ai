"""
Base LLM Provider Interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging

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