"""
Dependency injection container for the application with thread safety

This version works with or without dependency-injector library installed.
"""
import threading
from typing import Any, Dict, Optional, Callable

# Try to import dependency_injector, fall back to custom implementation if not available
try:
    from dependency_injector import containers, providers
    DEPENDENCY_INJECTOR_AVAILABLE = True
except ImportError:
    DEPENDENCY_INJECTOR_AVAILABLE = False
    # Custom implementation will be used below

# Fixed imports with error handling
import app_config  # Changed from 'config' to 'app_config'

# Import with graceful fallback for missing dependencies
try:
    from background_tasks import BackgroundTaskManager
except ImportError:
    BackgroundTaskManager = None

try:
    from cache_manager import CacheManager
except ImportError:
    CacheManager = None

try:
    from character_development import CharacterManager
except ImportError:
    CharacterManager = None

try:
    from events import event_manager
except ImportError:
    event_manager = None

try:
    from export_formats import BookExporter
except ImportError:
    BookExporter = None

try:
    from project_manager import ProjectManager
except ImportError:
    ProjectManager = None

try:
    from providers.factory import ProviderFactory  # Fixed import path
except ImportError:
    ProviderFactory = None

try:
    from style_templates import StyleManager
except ImportError:
    StyleManager = None

try:
    from token_optimizer import TokenOptimizer
except ImportError:
    TokenOptimizer = None


class ThreadSafeSingleton:
    """
    Thread-safe singleton pattern implementation using double-checked locking
    """
    _instances: Dict[type, Any] = {}
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, factory_func: Callable, *args, **kwargs):
        """
        Get or create a singleton instance with thread safety
        
        Args:
            factory_func: Factory function to create the instance
            *args: Positional arguments for factory
            **kwargs: Keyword arguments for factory
            
        Returns:
            Singleton instance
        """
        # First check without locking for performance
        if factory_func not in cls._instances:
            with cls._lock:
                # Double-check pattern after acquiring lock
                if factory_func not in cls._instances:
                    cls._instances[factory_func] = factory_func(*args, **kwargs)
        
        return cls._instances[factory_func]
    
    @classmethod
    def clear(cls):
        """Clear all singleton instances (useful for testing)"""
        with cls._lock:
            cls._instances.clear()


# Custom container implementation for when dependency-injector is not available
class SimpleContainer:
    """
    Simple dependency injection container without external dependencies
    
    Provides thread-safe singleton and factory patterns for dependency management.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._services = {}
        self._factories = {}
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from app_config"""
        return {
            'openai_api_key': getattr(app_config.settings, 'OPENAI_API_KEY', None),
            'anthropic_api_key': getattr(app_config.settings, 'ANTHROPIC_API_KEY', None),
            'cache_type': getattr(app_config.settings, 'CACHE_TYPE', 'memory'),
            'cache_ttl': getattr(app_config.settings, 'CACHE_TTL_SECONDS', 3600),
            'log_level': getattr(app_config.settings, 'LOG_LEVEL', 'INFO'),
            'base_dir': getattr(app_config.settings, 'BASE_DIR', '.'),
            'enable_progress_tracking': getattr(app_config.settings, 'ENABLE_PROGRESS_TRACKING', False),
            'provider_name': 'openai',
            'cache_max_size': 1000,
            'cache_cleanup_interval': 300,
            'task_backend': 'thread',
            'output_dir': './output',
            'project_dir': './projects/current',
            'token_cache_size': 2048,
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def update_config(self, config: Dict[str, Any]):
        """Update configuration"""
        with self._lock:
            self._config.update(config)
    
    def get_llm_provider(self):
        """Get or create LLM provider singleton"""
        return ThreadSafeSingleton.get_instance(
            self._create_llm_provider
        )
    
    def _create_llm_provider(self):
        """Create LLM provider instance"""
        if ProviderFactory is None:
            # Return a dummy provider if ProviderFactory is not available
            return None
        
        provider_config = {
            'provider': self._config.get('provider_name', 'openai'),
            'api_key': self._config.get('openai_api_key'),
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 4096
        }
        
        try:
            return ProviderFactory.create_provider(
                self._config.get('provider_name', 'openai'),
                provider_config
            )
        except Exception:
            # Return None if provider creation fails
            return None
    
    def get_event_manager(self):
        """Get event manager singleton"""
        return ThreadSafeSingleton.get_instance(lambda: event_manager)
    
    def get_project_manager(self):
        """Get or create project manager singleton"""
        if ProjectManager is None:
            return None
        return ThreadSafeSingleton.get_instance(
            ProjectManager,
            base_dir=self._config.get('base_dir', './projects')
        )
    
    def create_cache_manager(self):
        """Create a new cache manager instance"""
        if CacheManager is None:
            return None
        return CacheManager(
            backend=self._config.get('cache_type', 'memory'),
            max_size=self._config.get('cache_max_size', 1000),
            cleanup_interval=self._config.get('cache_cleanup_interval', 300)
        )
    
    def create_background_task_manager(self):
        """Create a new background task manager instance"""
        if BackgroundTaskManager is None:
            return None
        return BackgroundTaskManager(
            backend=self._config.get('task_backend', 'thread')
        )
    
    def create_book_exporter(self):
        """Create a new book exporter instance"""
        if BookExporter is None:
            return None
        return BookExporter(
            output_dir=self._config.get('output_dir', './output')
        )
    
    def get_style_manager(self):
        """Get or create style manager singleton"""
        if StyleManager is None:
            return None
        return ThreadSafeSingleton.get_instance(StyleManager)
    
    def create_character_manager(self):
        """Create a new character manager instance"""
        if CharacterManager is None:
            return None
        return CharacterManager(
            project_dir=self._config.get('project_dir', './projects/current')
        )
    
    def create_token_optimizer(self):
        """Create a new token optimizer instance"""
        if TokenOptimizer is None:
            return None
        provider = self.get_llm_provider() if ProviderFactory else None
        return TokenOptimizer(
            provider=provider,
            cache_size=self._config.get('token_cache_size', 2048)
        )
    
    def reset(self):
        """Reset all singletons and clear cache"""
        ThreadSafeSingleton.clear()
        self._services.clear()
        self._factories.clear()


# Use dependency-injector if available, otherwise use simple implementation
if DEPENDENCY_INJECTOR_AVAILABLE:
    class Container(containers.DeclarativeContainer):
        """
        Main dependency injection container with thread-safe singleton providers
        
        Features:
        - Thread-safe singleton pattern with double-checked locking
        - Proper provider factory usage
        - Configuration from app_config module
        - Lazy initialization for performance
        """
        
        # Configuration provider
        config_provider = providers.Configuration()
        
        # Initialize config from app_config module
        config_provider.from_dict({
            'openai_api_key': getattr(app_config.settings, 'OPENAI_API_KEY', None),
            'anthropic_api_key': getattr(app_config.settings, 'ANTHROPIC_API_KEY', None),
            'cache_type': getattr(app_config.settings, 'CACHE_TYPE', 'in_memory'),
            'cache_ttl': getattr(app_config.settings, 'CACHE_TTL_SECONDS', 3600),
            'log_level': getattr(app_config.settings, 'LOG_LEVEL', 'INFO'),
            'base_dir': getattr(app_config.settings, 'BASE_DIR', '.'),
            'enable_progress_tracking': getattr(app_config.settings, 'ENABLE_PROGRESS_TRACKING', False),
        })
        
        # Thread-safe LLM Provider using ProviderFactory
        llm_provider = providers.ThreadSafeSingleton(
            ProviderFactory.create_provider,
            provider_name=config_provider.provider_name,
            config=config_provider.provider_config
        )
        
        # Thread-safe Event Manager
        event_manager_service = providers.ThreadSafeSingleton(
            lambda: event_manager
        )
        
        # Thread-safe Project Manager
        project_manager = providers.ThreadSafeSingleton(
            ProjectManager,
            base_dir=config_provider.base_dir
        )
        
        # Cache Manager (Factory pattern for multiple instances if needed)
        cache_manager = providers.Factory(
            CacheManager,
            backend=config_provider.cache_type,
            max_size=providers.Configuration().cache_max_size,
            cleanup_interval=providers.Configuration().cache_cleanup_interval
        )
        
        # Background Task Manager (Factory for flexibility)
        background_task_manager = providers.Factory(
            BackgroundTaskManager,
            backend=providers.Configuration().task_backend
        )
        
        # Book Exporter (Factory pattern)
        book_exporter = providers.Factory(
            BookExporter,
            output_dir=providers.Configuration().output_dir
        )
        
        # Thread-safe Style Manager
        style_manager = providers.ThreadSafeSingleton(
            StyleManager
        )
        
        # Character Manager (Factory for project-specific instances)
        character_manager = providers.Factory(
            CharacterManager,
            project_dir=providers.Configuration().project_dir
        )
        
        # Token Optimizer with provider integration
        token_optimizer = providers.Factory(
            TokenOptimizer,
            provider=llm_provider,
            cache_size=providers.Configuration().token_cache_size
        )
else:
    # Use simple container when dependency-injector is not available
    Container = SimpleContainer


# Global container instance with thread safety
_container: Optional[Container] = None
_container_lock = threading.RLock()


def get_container() -> Container:
    """
    Get the global container instance with double-checked locking for thread safety
    
    Returns:
        Container: The global dependency injection container
    """
    global _container
    
    # First check without locking (performance optimization)
    if _container is None:
        with _container_lock:
            # Double-check after acquiring lock
            if _container is None:
                _container = Container()
                
                if DEPENDENCY_INJECTOR_AVAILABLE:
                    # Set default configuration values for dependency-injector
                    _container.config_provider.base_dir.from_value('./projects')
                    _container.config_provider.cache_type.from_value('memory')
                    _container.config_provider.cache_max_size.from_value(1000)
                    _container.config_provider.cache_cleanup_interval.from_value(300)
                    _container.config_provider.task_backend.from_value('thread')
                    _container.config_provider.output_dir.from_value('./output')
                    _container.config_provider.project_dir.from_value('./projects/current')
                    _container.config_provider.token_cache_size.from_value(2048)
                    
                    # Default provider configuration
                    _container.config_provider.provider_name.from_value('openai')
                    _container.config_provider.provider_config.from_value({
                        'provider': 'openai',
                        'api_key': getattr(app_config.settings, 'OPENAI_API_KEY', None),
                        'model': 'gpt-4',
                        'temperature': 0.7,
                        'max_tokens': 4096
                    })
    
    return _container


def init_container(custom_config: Dict[str, Any] = None) -> Container:
    """
    Initialize container with custom configuration
    
    Thread-safe initialization with proper locking.
    
    Args:
        custom_config: Optional custom configuration dictionary
        
    Returns:
        Container: Configured container instance
    """
    container = get_container()
    
    if custom_config:
        with _container_lock:
            if DEPENDENCY_INJECTOR_AVAILABLE:
                # Update configuration for dependency-injector
                container.config_provider.update(custom_config)
                
                # Update provider config if specified
                if 'provider_name' in custom_config:
                    container.config_provider.provider_name.from_value(custom_config['provider_name'])
                
                if 'provider_config' in custom_config:
                    container.config_provider.provider_config.from_value(custom_config['provider_config'])
            else:
                # Update configuration for simple container
                container.update_config(custom_config)
    
    return container


def reset_container():
    """
    Reset the global container (useful for testing)
    
    Thread-safe container reset.
    """
    global _container
    
    with _container_lock:
        if _container is not None:
            # Clean up resources if needed
            try:
                if DEPENDENCY_INJECTOR_AVAILABLE:
                    _container.shutdown_resources()
                else:
                    _container.reset()
            except:
                pass
            
            _container = None