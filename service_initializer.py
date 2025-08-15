"""
Service initialization module for setting up dependencies.
"""

import logging
from typing import Tuple

from app_config import settings
from cache_manager import CacheManager
from exceptions import GhostwriterException
from providers.factory import ProviderFactory
from services.generation_service import GenerationService
from token_optimizer import TokenOptimizer


class ServiceInitializer:
    """Initialize and configure services for book generation."""

    def __init__(self) -> None:
        """Initialize the service initializer."""
        self.logger = logging.getLogger(__name__)

    def initialize_services(self) -> GenerationService:
        """
        Initialize all required services with error handling.

        Returns:
            Configured GenerationService instance

        Raises:
            GhostwriterException: If service initialization fails
        """
        try:
            # Create provider factory
            provider_factory = ProviderFactory()
            
            # Get appropriate API key based on provider
            api_key = self._get_api_key()
            
            # Create provider
            provider = provider_factory.create_provider(
                settings.LLM_PROVIDER,
                {"api_key": api_key}
            )
            
            # Initialize cache manager
            cache_manager = CacheManager(
                backend=settings.CACHE_TYPE,
                expire=settings.CACHE_TTL_SECONDS
            )
            
            # Initialize token optimizer
            token_optimizer = TokenOptimizer(provider=provider)
            
            # Create generation service
            generation_service = GenerationService(
                provider_factory, cache_manager, token_optimizer
            )
            
            self.logger.info("All services initialized successfully")
            return generation_service
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise GhostwriterException(
                "Service initialization failed. Please check your API keys and configuration."
            ) from e

    def _get_api_key(self) -> str:
        """
        Get the appropriate API key based on the configured provider.

        Returns:
            API key for the configured provider

        Raises:
            ValueError: If no API key is configured for the provider
        """
        if settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is not configured")
            return settings.OPENAI_API_KEY
        elif settings.LLM_PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key is not configured")
            return settings.ANTHROPIC_API_KEY
        else:
            # For other providers, try to get a generic API key
            # This might need to be extended based on actual provider requirements
            if hasattr(settings, f"{settings.LLM_PROVIDER.upper()}_API_KEY"):
                return getattr(settings, f"{settings.LLM_PROVIDER.upper()}_API_KEY")
            else:
                raise ValueError(f"No API key configured for provider: {settings.LLM_PROVIDER}")