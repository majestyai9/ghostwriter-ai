"""
Comprehensive unit tests for dependency injection container.

This module tests the Container class, configuration management,
thread safety, and service instantiation.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from containers import (
    Container,
    get_container,
    init_container,
    reset_container,
    get_generation_service,
    get_cache_manager,
    get_project_manager,
    get_provider_factory,
    _validate_config
)


class TestContainer:
    """Test suite for Container class."""
    
    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before and after each test."""
        reset_container()
        yield
        reset_container()
    
    @pytest.fixture
    def mock_app_config(self):
        """Mock app_config module."""
        with patch("containers.app_config") as mock_config:
            mock_settings = MagicMock()
            mock_settings.OPENAI_API_KEY = "test-openai-key"
            mock_settings.ANTHROPIC_API_KEY = "test-anthropic-key"
            mock_settings.GEMINI_API_KEY = "test-gemini-key"
            mock_settings.CACHE_TYPE = "memory"
            mock_settings.CACHE_TTL_SECONDS = 3600
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.BASE_DIR = "./test-projects"
            mock_settings.ENABLE_PROGRESS_TRACKING = False
            mock_settings.ENABLE_RAG = False
            mock_config.settings = mock_settings
            yield mock_config
    
    def test_container_initialization(self, mock_app_config):
        """Test container initialization with default config."""
        container = Container()
        
        # Check configuration is loaded
        config_dict = container.config()
        assert config_dict["openai_api_key"] == "test-openai-key"
        assert config_dict["anthropic_api_key"] == "test-anthropic-key"
        assert config_dict["cache_type"] == "memory"
        assert config_dict["cache_ttl"] == 3600
        assert config_dict["base_dir"] == "./test-projects"
    
    def test_get_container_singleton(self, mock_app_config):
        """Test that get_container returns singleton instance."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2
        assert isinstance(container1, Container)
    
    def test_init_container_with_custom_config(self, mock_app_config):
        """Test initializing container with custom configuration."""
        custom_config = {
            "provider_name": "anthropic",
            "cache_type": "redis",
            "cache_max_size": 2000,
            "temperature": 0.5,
            "enable_rag": True
        }
        
        container = init_container(custom_config)
        
        config_dict = container.config()
        assert config_dict["provider_name"] == "anthropic"
        assert config_dict["cache_type"] == "redis"
        assert config_dict["cache_max_size"] == 2000
        assert config_dict["temperature"] == 0.5
        assert config_dict["enable_rag"] is True
    
    def test_reset_container(self, mock_app_config):
        """Test container reset functionality."""
        # Create and configure container
        container1 = get_container()
        init_container({"provider_name": "gemini"})
        
        # Reset container
        reset_container()
        
        # Get new container
        container2 = get_container()
        
        # Should be different instance
        assert container1 is not container2
        
        # Should have default config
        config_dict = container2.config()
        assert config_dict["provider_name"] == "openai"  # Default
    
    @patch("containers.ProviderFactory")
    def test_provider_factory_singleton(self, mock_factory_class, mock_app_config):
        """Test that provider factory is a singleton."""
        container = get_container()
        
        factory1 = container.provider_factory()
        factory2 = container.provider_factory()
        
        assert factory1 is factory2
    
    @patch("containers.CacheManager")
    def test_cache_manager_factory(self, mock_cache_class, mock_app_config):
        """Test that cache manager is created as factory (new instance each time)."""
        container = get_container()
        
        cache1 = container.cache_manager()
        cache2 = container.cache_manager()
        
        # Should be different instances
        assert cache1 is not cache2
        
        # Should be called with correct parameters
        mock_cache_class.assert_called_with(
            backend="memory",
            max_size=1000,
            cleanup_interval=300
        )
    
    @patch("containers.GenerationService")
    @patch("containers.TokenOptimizer")
    @patch("containers.CacheManager")
    @patch("containers.ProviderFactory")
    def test_generation_service_creation(
        self,
        mock_factory_class,
        mock_cache_class,
        mock_optimizer_class,
        mock_service_class,
        mock_app_config
    ):
        """Test generation service creation with dependencies."""
        container = get_container()
        
        # Create service
        service = container.generation_service()
        
        # Verify service was created with correct dependencies
        mock_service_class.assert_called_once()
        call_kwargs = mock_service_class.call_args.kwargs
        
        assert "provider_factory" in call_kwargs
        assert "cache_manager" in call_kwargs
        assert "token_optimizer" in call_kwargs
        assert call_kwargs["enable_rag"] is False
    
    @patch("containers.ProjectManager")
    def test_project_manager_singleton(self, mock_manager_class, mock_app_config):
        """Test that project manager is a singleton."""
        container = get_container()
        
        manager1 = container.project_manager()
        manager2 = container.project_manager()
        
        assert manager1 is manager2
        
        # Should be called only once
        mock_manager_class.assert_called_once_with(
            base_dir="./test-projects"
        )


class TestConfigValidation:
    """Test suite for configuration validation."""
    
    def test_validate_valid_provider_name(self):
        """Test validation of valid provider names."""
        config = {"provider_name": "openai"}
        validated = _validate_config(config)
        assert validated["provider_name"] == "openai"
        
        config = {"provider_name": "anthropic"}
        validated = _validate_config(config)
        assert validated["provider_name"] == "anthropic"
    
    def test_validate_invalid_provider_name(self):
        """Test validation rejects invalid provider names."""
        config = {"provider_name": "invalid_provider"}
        
        with pytest.raises(ValueError) as exc_info:
            _validate_config(config)
        
        assert "Invalid provider_name" in str(exc_info.value)
    
    def test_validate_cache_type(self):
        """Test validation of cache types."""
        config = {"cache_type": "redis"}
        validated = _validate_config(config)
        assert validated["cache_type"] == "redis"
        
        config = {"cache_type": "invalid_cache"}
        with pytest.raises(ValueError) as exc_info:
            _validate_config(config)
        assert "Invalid cache_type" in str(exc_info.value)
    
    def test_validate_numeric_fields(self):
        """Test validation of numeric configuration fields."""
        config = {
            "cache_max_size": 2000,
            "cache_cleanup_interval": 600,
            "token_cache_size": 4096,
            "max_tokens": 8192,
            "cache_ttl": 7200
        }
        
        validated = _validate_config(config)
        assert validated["cache_max_size"] == 2000
        assert validated["cache_cleanup_interval"] == 600
        assert validated["token_cache_size"] == 4096
        assert validated["max_tokens"] == 8192
        assert validated["cache_ttl"] == 7200
    
    def test_validate_invalid_numeric_fields(self):
        """Test validation rejects invalid numeric values."""
        # Negative value
        config = {"cache_max_size": -100}
        with pytest.raises(ValueError) as exc_info:
            _validate_config(config)
        assert "must be positive" in str(exc_info.value)
        
        # Non-numeric value
        config = {"max_tokens": "not_a_number"}
        with pytest.raises(ValueError) as exc_info:
            _validate_config(config)
        assert "Invalid max_tokens" in str(exc_info.value)
    
    def test_validate_boolean_fields(self):
        """Test validation of boolean fields."""
        config = {
            "enable_progress_tracking": True,
            "enable_rag": False
        }
        
        validated = _validate_config(config)
        assert validated["enable_progress_tracking"] is True
        assert validated["enable_rag"] is False
        
        # Test conversion
        config = {
            "enable_progress_tracking": 1,
            "enable_rag": 0
        }
        
        validated = _validate_config(config)
        assert validated["enable_progress_tracking"] is True
        assert validated["enable_rag"] is False
    
    def test_validate_temperature(self):
        """Test validation of temperature parameter."""
        # Valid temperatures
        config = {"temperature": 0.0}
        validated = _validate_config(config)
        assert validated["temperature"] == 0.0
        
        config = {"temperature": 1.5}
        validated = _validate_config(config)
        assert validated["temperature"] == 1.5
        
        config = {"temperature": 2.0}
        validated = _validate_config(config)
        assert validated["temperature"] == 2.0
    
    def test_validate_invalid_temperature(self):
        """Test validation rejects invalid temperature values."""
        # Too low
        config = {"temperature": -0.1}
        with pytest.raises(ValueError) as exc_info:
            _validate_config(config)
        assert "temperature must be between 0.0 and 2.0" in str(exc_info.value)
        
        # Too high
        config = {"temperature": 2.1}
        with pytest.raises(ValueError) as exc_info:
            _validate_config(config)
        assert "temperature must be between 0.0 and 2.0" in str(exc_info.value)
    
    def test_validate_string_fields(self):
        """Test validation of string fields."""
        config = {
            "base_dir": "/custom/base",
            "output_dir": "/custom/output",
            "project_dir": "/custom/project",
            "task_backend": "celery",
            "log_level": "DEBUG"
        }
        
        validated = _validate_config(config)
        assert validated["base_dir"] == "/custom/base"
        assert validated["output_dir"] == "/custom/output"
        assert validated["project_dir"] == "/custom/project"
        assert validated["task_backend"] == "celery"
        assert validated["log_level"] == "DEBUG"
    
    def test_validate_api_key_with_provider(self):
        """Test that API key is included when provider is specified."""
        config = {
            "provider_name": "openai",
            "openai_api_key": "sk-test-key"
        }
        
        validated = _validate_config(config)
        assert validated["provider_name"] == "openai"
        assert validated["openai_api_key"] == "sk-test-key"


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before and after each test."""
        reset_container()
        yield
        reset_container()
    
    @patch("containers.GenerationService")
    @patch("containers.TokenOptimizer")
    @patch("containers.CacheManager")
    @patch("containers.ProviderFactory")
    @patch("containers.app_config")
    def test_get_generation_service(
        self,
        mock_app_config,
        mock_factory_class,
        mock_cache_class,
        mock_optimizer_class,
        mock_service_class
    ):
        """Test get_generation_service convenience function."""
        mock_app_config.settings = MagicMock()
        
        service = get_generation_service()
        
        assert mock_service_class.called
        assert service is not None
    
    @patch("containers.CacheManager")
    @patch("containers.app_config")
    def test_get_cache_manager(self, mock_app_config, mock_cache_class):
        """Test get_cache_manager convenience function."""
        mock_app_config.settings = MagicMock()
        
        cache = get_cache_manager()
        
        assert mock_cache_class.called
        assert cache is not None
    
    @patch("containers.ProjectManager")
    @patch("containers.app_config")
    def test_get_project_manager(self, mock_app_config, mock_manager_class):
        """Test get_project_manager convenience function."""
        mock_app_config.settings = MagicMock()
        
        manager = get_project_manager()
        
        assert mock_manager_class.called
        assert manager is not None
    
    @patch("containers.ProviderFactory")
    @patch("containers.app_config")
    def test_get_provider_factory(self, mock_app_config, mock_factory_class):
        """Test get_provider_factory convenience function."""
        mock_app_config.settings = MagicMock()
        
        factory = get_provider_factory()
        
        assert mock_factory_class.called
        assert factory is not None


class TestThreadSafety:
    """Test thread safety of container operations."""
    
    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before and after each test."""
        reset_container()
        yield
        reset_container()
    
    @pytest.mark.slow
    @patch("containers.app_config")
    def test_concurrent_get_container(self, mock_app_config):
        """Test concurrent access to get_container."""
        mock_app_config.settings = MagicMock()
        
        containers = []
        errors = []
        
        def get_container_thread():
            try:
                container = get_container()
                containers.append(container)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(100):
            thread = threading.Thread(target=get_container_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should get the same container instance
        assert len(errors) == 0
        assert len(containers) == 100
        assert all(c is containers[0] for c in containers)
    
    @pytest.mark.slow
    @patch("containers.app_config")
    def test_concurrent_init_container(self, mock_app_config):
        """Test concurrent configuration updates."""
        mock_app_config.settings = MagicMock()
        
        results = []
        errors = []
        
        def update_config(provider_name):
            try:
                container = init_container({
                    "provider_name": provider_name,
                    "temperature": 0.5
                })
                config = container.config()
                results.append(config["provider_name"])
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                provider = "openai" if i % 2 == 0 else "anthropic"
                future = executor.submit(update_config, provider)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 50
    
    @pytest.mark.slow
    @patch("containers.CacheManager")
    @patch("containers.app_config")
    def test_concurrent_service_creation(self, mock_app_config, mock_cache_class):
        """Test concurrent service instantiation."""
        mock_app_config.settings = MagicMock()
        
        services = []
        errors = []
        
        def create_cache_manager():
            try:
                cache = get_cache_manager()
                services.append(cache)
            except Exception as e:
                errors.append(str(e))
        
        # Create services concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(create_cache_manager)
                for _ in range(100)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # Should have no errors
        assert len(errors) == 0
        assert len(services) == 100
        
        # Each should be a unique instance (factory pattern)
        assert len(set(id(s) for s in services)) == 100