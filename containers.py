"""
Dependency injection container for the application.

This module provides a robust dependency injection system using the dependency-injector
framework with thread-safe singleton patterns and proper factory configurations.
"""

import logging
import threading
from typing import Any, Dict, Optional

from dependency_injector import containers, providers

import app_config
from background_tasks import BackgroundTaskManager
from cache_manager import CacheManager
from character_development import CharacterManager
from events import event_manager
from export_formats import BookExporter
from project_manager import ProjectManager
from providers.factory import ProviderFactory
from services.generation_service import GenerationService
from style_templates import StyleManager
from token_optimizer import TokenOptimizer

# Configure logging
logger = logging.getLogger(__name__)


class Container(containers.DeclarativeContainer):
    """
    Main dependency injection container with thread-safe providers.
    
    Features:
    - Thread-safe singleton providers for stateful services
    - Factory providers for stateless/per-request services
    - Configuration management through dependency-injector
    - Lazy initialization for optimal performance
    - Proper resource cleanup and lifecycle management
    """
    
    # Configuration provider
    config = providers.Configuration()
    
    # Initialize with defaults from app_config
    config.from_dict({
        "openai_api_key": getattr(app_config.settings, "OPENAI_API_KEY", None),
        "anthropic_api_key": getattr(app_config.settings, "ANTHROPIC_API_KEY", None),
        "gemini_api_key": getattr(app_config.settings, "GEMINI_API_KEY", None),
        "cache_type": getattr(app_config.settings, "CACHE_TYPE", "memory"),
        "cache_ttl": getattr(app_config.settings, "CACHE_TTL_SECONDS", 3600),
        "log_level": getattr(app_config.settings, "LOG_LEVEL", "INFO"),
        "base_dir": getattr(app_config.settings, "BASE_DIR", "./projects"),
        "enable_progress_tracking": getattr(
            app_config.settings, "ENABLE_PROGRESS_TRACKING", False
        ),
        "enable_rag": getattr(app_config.settings, "ENABLE_RAG", False),
        "provider_name": "openai",
        "cache_max_size": 1000,
        "cache_cleanup_interval": 300,
        "task_backend": "thread",
        "output_dir": "./output",
        "project_dir": "./projects/current",
        "token_cache_size": 2048,
        "max_tokens": 4096,
        "temperature": 0.7,
    })
    
    # Provider Factory - Singleton for managing provider instances
    provider_factory = providers.ThreadSafeSingleton(
        ProviderFactory
    )
    
    # LLM Provider - Created through factory
    llm_provider = providers.ThreadSafeSingleton(
        providers.Callable(
            lambda factory, config_dict: factory.create_provider(
                config_dict["provider_name"],
                {
                    "provider": config_dict["provider_name"],
                    "api_key": config_dict.get(
                        f"{config_dict['provider_name']}_api_key"
                    ),
                    "model": config_dict.get("model", "gpt-4"),
                    "temperature": config_dict.get("temperature", 0.7),
                    "max_tokens": config_dict.get("max_tokens", 4096),
                }
            ),
            factory=provider_factory,
            config_dict=config.as_dict(),
        )
    )
    
    # Event Manager - Global singleton
    event_manager_service = providers.ThreadSafeSingleton(
        lambda: event_manager
    )
    
    # Project Manager - Thread-safe singleton
    project_manager = providers.ThreadSafeSingleton(
        ProjectManager,
        base_dir=config.base_dir,
    )
    
    # Cache Manager - Factory for flexibility (can have multiple instances)
    cache_manager = providers.Factory(
        CacheManager,
        backend=config.cache_type,
        max_size=config.cache_max_size,
        cleanup_interval=config.cache_cleanup_interval,
    )
    
    # Background Task Manager - Factory pattern
    background_task_manager = providers.Factory(
        BackgroundTaskManager,
        backend=config.task_backend,
    )
    
    # Book Exporter - Factory for per-export instances
    book_exporter = providers.Factory(
        BookExporter,
        output_dir=config.output_dir,
    )
    
    # Style Manager - Singleton for consistent style management
    style_manager = providers.ThreadSafeSingleton(
        StyleManager
    )
    
    # Character Manager - Factory for project-specific instances
    character_manager = providers.Factory(
        CharacterManager,
        project_dir=config.project_dir,
    )
    
    # Token Optimizer - Factory with provider injection
    token_optimizer = providers.Factory(
        TokenOptimizer,
        provider=llm_provider,
        cache_size=config.token_cache_size,
    )
    
    # Generation Service - Factory with dependencies
    generation_service = providers.Factory(
        GenerationService,
        provider_factory=provider_factory,
        cache_manager=cache_manager,
        token_optimizer=token_optimizer,
        enable_rag=config.enable_rag,
    )


# Global container instance management
_container: Optional[Container] = None
_container_lock = threading.RLock()


def get_container() -> Container:
    """
    Get the global container instance with thread-safe initialization.
    
    Uses double-checked locking pattern for optimal performance.
    
    Returns:
        Container: The global dependency injection container
    """
    global _container
    
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
                logger.info("Dependency injection container initialized")
    
    return _container


def init_container(custom_config: Optional[Dict[str, Any]] = None) -> Container:
    """
    Initialize or update the container with custom configuration.
    
    Thread-safe configuration update with proper validation.
    
    Args:
        custom_config: Optional custom configuration dictionary
        
    Returns:
        Container: Configured container instance
        
    Example:
        >>> container = init_container({
        ...     "provider_name": "anthropic",
        ...     "anthropic_api_key": "sk-...",
        ...     "cache_type": "redis",
        ...     "enable_rag": True
        ... })
    """
    container = get_container()
    
    if custom_config:
        with _container_lock:
            # Validate and update configuration
            validated_config = _validate_config(custom_config)
            container.config.update(validated_config)
            
            # Log configuration changes
            logger.info(f"Container configuration updated with {len(validated_config)} settings")
            
            # Reset any cached providers if provider changed
            if "provider_name" in validated_config:
                container.llm_provider.reset()
                logger.info(f"LLM provider reset to: {validated_config['provider_name']}")
    
    return container


def reset_container() -> None:
    """
    Reset the global container and cleanup resources.
    
    Thread-safe container reset with proper resource cleanup.
    Useful for testing and reinitialization scenarios.
    """
    global _container
    
    with _container_lock:
        if _container is not None:
            try:
                # Attempt to clean up resources
                _container.shutdown_resources()
                logger.info("Container resources cleaned up")
            except Exception as e:
                logger.warning(f"Error during container cleanup: {e}")
            finally:
                _container = None
                logger.info("Container reset completed")


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize configuration values.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Dict[str, Any]: Validated configuration
        
    Raises:
        ValueError: If configuration contains invalid values
    """
    validated = {}
    
    # Validate provider name
    if "provider_name" in config:
        valid_providers = ["openai", "anthropic", "gemini", "cohere", "openrouter"]
        if config["provider_name"] not in valid_providers:
            raise ValueError(
                f"Invalid provider_name: {config['provider_name']}. "
                f"Must be one of: {', '.join(valid_providers)}"
            )
        validated["provider_name"] = config["provider_name"]
        
        # Ensure corresponding API key exists
        api_key_field = f"{config['provider_name']}_api_key"
        if api_key_field in config:
            validated[api_key_field] = config[api_key_field]
    
    # Validate cache settings
    if "cache_type" in config:
        valid_cache_types = ["memory", "redis", "file"]
        if config["cache_type"] not in valid_cache_types:
            raise ValueError(
                f"Invalid cache_type: {config['cache_type']}. "
                f"Must be one of: {', '.join(valid_cache_types)}"
            )
        validated["cache_type"] = config["cache_type"]
    
    # Validate numeric settings
    numeric_fields = [
        "cache_max_size", "cache_cleanup_interval", "token_cache_size",
        "max_tokens", "cache_ttl"
    ]
    for field in numeric_fields:
        if field in config:
            try:
                value = int(config[field])
                if value <= 0:
                    raise ValueError(f"{field} must be positive")
                validated[field] = value
            except ValueError as e:
                if "must be positive" in str(e):
                    raise e
                raise ValueError(f"Invalid {field}: {config[field]}") from e
            except TypeError as e:
                raise ValueError(f"Invalid {field}: {config[field]}") from e
    
    # Validate boolean settings
    boolean_fields = ["enable_progress_tracking", "enable_rag"]
    for field in boolean_fields:
        if field in config:
            validated[field] = bool(config[field])
    
    # Validate temperature
    if "temperature" in config:
        try:
            temp = float(config["temperature"])
            if not 0.0 <= temp <= 2.0:
                raise ValueError("temperature must be between 0.0 and 2.0")
            validated["temperature"] = temp
        except ValueError as e:
            if "must be between" in str(e):
                raise e
            raise ValueError(f"Invalid temperature: {config['temperature']}") from e
        except TypeError as e:
            raise ValueError(f"Invalid temperature: {config['temperature']}") from e
    
    # Pass through other valid string settings
    string_fields = ["base_dir", "output_dir", "project_dir", "task_backend", "log_level"]
    for field in string_fields:
        if field in config:
            validated[field] = str(config[field])
    
    return validated


# Convenience functions for common services
def get_generation_service() -> GenerationService:
    """
    Get a generation service instance with all dependencies wired.
    
    Returns:
        GenerationService: Configured generation service
    """
    container = get_container()
    return container.generation_service()


def get_cache_manager() -> CacheManager:
    """
    Get a cache manager instance.
    
    Returns:
        CacheManager: Configured cache manager
    """
    container = get_container()
    return container.cache_manager()


def get_project_manager() -> ProjectManager:
    """
    Get the project manager singleton.
    
    Returns:
        ProjectManager: Project manager instance
    """
    container = get_container()
    return container.project_manager()


def get_provider_factory() -> ProviderFactory:
    """
    Get the provider factory singleton.
    
    Returns:
        ProviderFactory: Provider factory instance
    """
    container = get_container()
    return container.provider_factory()


# Export main components
__all__ = [
    "Container",
    "get_container",
    "init_container",
    "reset_container",
    "get_generation_service",
    "get_cache_manager",
    "get_project_manager",
    "get_provider_factory",
]