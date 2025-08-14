"""
Provider Factory for dynamic provider selection
"""
from typing import Dict, Any, Type
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .cohere_provider import CohereProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider
from ..exceptions import ConfigurationError

class ProviderFactory:
    """Factory class for creating LLM provider instances"""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'cohere': CohereProvider,
        'gemini': GeminiProvider,
        'openrouter': OpenRouterProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]):
        """
        Register a new provider
        
        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls._providers[name.lower()] = provider_class
        
    @classmethod
    def create_provider(cls, provider_name: str, config: Dict[str, Any]) -> LLMProvider:
        """
        Create a provider instance
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            LLMProvider instance
            
        Raises:
            ConfigurationError: If provider is not found
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ConfigurationError(
                f"Provider '{provider_name}' not found. Available providers: {available}"
            )
            
        provider_class = cls._providers[provider_name]
        return provider_class(config)
        
    @classmethod
    def list_providers(cls) -> list:
        """List available providers"""
        return list(cls._providers.keys())

def get_provider(provider_name: str = None, config: Dict[str, Any] = None) -> LLMProvider:
    """
    Convenience function to get a provider instance
    
    Args:
        provider_name: Name of the provider (if None, uses config['provider'])
        config: Provider configuration
        
    Returns:
        LLMProvider instance
    """
    if config is None:
        raise ConfigurationError("Configuration is required")
        
    if provider_name is None:
        provider_name = config.get('provider')
        if not provider_name:
            raise ConfigurationError("Provider name must be specified")
            
    return ProviderFactory.create_provider(provider_name, config)