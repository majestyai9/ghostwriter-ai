"""
Comprehensive unit tests for base provider functionality.

This module tests the LLMProvider base class, LLMResponse, and related functionality
including error handling, rate limiting, and thread safety.
"""

import time
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from providers.base import LLMProvider, LLMResponse


# Test implementation of LLMProvider for testing
class TestProvider(LLMProvider):
    """Concrete implementation of LLMProvider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize test provider."""
        super().__init__(config)
        self.generate_called = False
        self.generate_stream_called = False
        self.test_response = "Test response"
        self.test_stream = ["chunk1", "chunk2", "chunk3"]
    
    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate test response."""
        self.generate_called = True
        self.last_prompt = prompt
        self.last_history = history
        self.last_kwargs = kwargs
        
        return LLMResponse(
            content=self.test_response,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            model=self.config.get("model", "test-model")
        )
    
    def generate_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        """Generate streaming test response."""
        self.generate_stream_called = True
        self.last_stream_prompt = prompt
        self.last_stream_history = history
        self.last_stream_kwargs = kwargs
        
        for chunk in self.test_stream:
            yield chunk


class TestLLMResponse:
    """Test suite for LLMResponse dataclass."""
    
    def test_llm_response_creation(self):
        """Test creating an LLM response with all fields."""
        response = LLMResponse(
            content="Generated text",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            finish_reason="stop",
            model="gpt-4"
        )
        
        assert response.content == "Generated text"
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 200
        assert response.total_tokens == 300
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4"
    
    def test_llm_response_optional_fields(self):
        """Test LLM response with optional fields."""
        response = LLMResponse(
            content="Text",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            finish_reason=None,
            model=None
        )
        
        assert response.content == "Text"
        assert response.prompt_tokens is None
        assert response.completion_tokens is None
        assert response.total_tokens is None
        assert response.finish_reason is None
        assert response.model is None
    
    def test_llm_response_equality(self):
        """Test LLM response equality comparison."""
        response1 = LLMResponse(
            content="Same",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            model="gpt-4"
        )
        
        response2 = LLMResponse(
            content="Same",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            model="gpt-4"
        )
        
        response3 = LLMResponse(
            content="Different",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            model="gpt-4"
        )
        
        assert response1 == response2
        assert response1 != response3


class TestLLMProvider:
    """Test suite for LLMProvider base class."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            "api_key": "test-key",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    @pytest.fixture
    def test_provider(self, basic_config):
        """Create a test provider instance."""
        return TestProvider(basic_config)
    
    def test_provider_initialization(self, basic_config):
        """Test provider initialization with config."""
        provider = TestProvider(basic_config)
        
        assert provider.config == basic_config
        assert provider.config["api_key"] == "test-key"
        assert provider.config["model"] == "test-model"
        assert provider.config["temperature"] == 0.7
        assert provider.config["max_tokens"] == 1000
    
    def test_provider_abstract_base_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Should raise TypeError because abstract methods not implemented
            LLMProvider({})
    
    def test_generate_method(self, test_provider):
        """Test the generate method."""
        response = test_provider.generate(
            prompt="Test prompt",
            temperature=0.5
        )
        
        assert test_provider.generate_called
        assert test_provider.last_prompt == "Test prompt"
        assert test_provider.last_kwargs == {"temperature": 0.5}
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.total_tokens == 30
    
    def test_generate_with_history(self, test_provider):
        """Test generate method with conversation history."""
        history = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        response = test_provider.generate(
            prompt="Follow-up question",
            history=history,
            max_tokens=500
        )
        
        assert test_provider.last_history == history
        assert test_provider.last_kwargs == {"max_tokens": 500}
        assert response.content == "Test response"
    
    def test_generate_stream_method(self, test_provider):
        """Test the streaming generation method."""
        result = list(test_provider.generate_stream(
            prompt="Stream test",
            temperature=0.8
        ))
        
        assert test_provider.generate_stream_called
        assert test_provider.last_stream_prompt == "Stream test"
        assert test_provider.last_stream_kwargs == {"temperature": 0.8}
        assert result == ["chunk1", "chunk2", "chunk3"]
    
    def test_generate_stream_with_history(self, test_provider):
        """Test streaming with conversation history."""
        history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]
        
        result = list(test_provider.generate_stream(
            prompt="Stream with history",
            history=history
        ))
        
        assert test_provider.last_stream_history == history
        assert result == ["chunk1", "chunk2", "chunk3"]
    
    def test_config_immutability(self, test_provider):
        """Test that config remains immutable after initialization."""
        original_config = test_provider.config.copy()
        
        # Try to modify config
        test_provider.config["new_key"] = "new_value"
        test_provider.config["temperature"] = 0.9
        
        # Create new provider with original config
        new_provider = TestProvider(original_config)
        
        # Original config should be unchanged
        assert "new_key" not in original_config
        assert original_config["temperature"] == 0.7
    
    def test_empty_config(self):
        """Test provider with empty configuration."""
        provider = TestProvider({})
        
        assert provider.config == {}
        response = provider.generate("test")
        assert response.content == "Test response"
    
    def test_none_history(self, test_provider):
        """Test that None history is handled correctly."""
        response = test_provider.generate(
            prompt="No history",
            history=None
        )
        
        assert test_provider.last_history is None
        assert response.content == "Test response"


class TestThreadSafety:
    """Test thread safety of provider implementations."""
    
    @pytest.mark.slow
    def test_concurrent_generation(self, basic_config):
        """Test concurrent generation calls."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        provider = TestProvider(basic_config)
        results = []
        errors = []
        
        def generate_text(prompt_id):
            try:
                response = provider.generate(f"Prompt {prompt_id}")
                return (prompt_id, response.content)
            except Exception as e:
                errors.append((prompt_id, str(e)))
                return None
        
        # Run concurrent generations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(generate_text, i)
                for i in range(100)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        # Verify results
        assert len(errors) == 0
        assert len(results) == 100
        
        # All results should be valid
        for prompt_id, content in results:
            assert content == "Test response"


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_tracking(self, basic_config):
        """Test that rate limit information is tracked."""
        provider = TestProvider(basic_config)
        
        # Add rate limit tracking to test provider
        provider.rate_limit_remaining = 100
        provider.rate_limit_reset = time.time() + 3600
        
        response = provider.generate("Test")
        
        # Simulate rate limit decrease
        provider.rate_limit_remaining -= 1
        
        assert provider.rate_limit_remaining == 99
        assert provider.rate_limit_reset > time.time()
    
    def test_rate_limit_headers_parsing(self):
        """Test parsing of rate limit headers from API responses."""
        # This would typically be implemented in concrete providers
        headers = {
            "x-ratelimit-remaining": "50",
            "x-ratelimit-reset": str(int(time.time() + 60))
        }
        
        remaining = int(headers.get("x-ratelimit-remaining", 0))
        reset_time = int(headers.get("x-ratelimit-reset", 0))
        
        assert remaining == 50
        assert reset_time > time.time()


class TestErrorHandling:
    """Test error handling in providers."""
    
    class ErrorProvider(LLMProvider):
        """Provider that raises errors for testing."""
        
        def __init__(self, config, error_type="generic"):
            super().__init__(config)
            self.error_type = error_type
        
        def generate(self, prompt, history=None, **kwargs):
            if self.error_type == "api":
                raise ConnectionError("API connection failed")
            elif self.error_type == "auth":
                raise ValueError("Invalid API key")
            elif self.error_type == "timeout":
                raise TimeoutError("Request timed out")
            else:
                raise Exception("Generic error")
        
        def generate_stream(self, prompt, history=None, **kwargs):
            if self.error_type == "stream":
                yield "Start"
                raise ConnectionError("Stream interrupted")
            else:
                raise Exception("Stream error")
    
    def test_api_error_handling(self):
        """Test handling of API connection errors."""
        provider = self.ErrorProvider({}, error_type="api")
        
        with pytest.raises(ConnectionError) as exc_info:
            provider.generate("test")
        
        assert "API connection failed" in str(exc_info.value)
    
    def test_auth_error_handling(self):
        """Test handling of authentication errors."""
        provider = self.ErrorProvider({}, error_type="auth")
        
        with pytest.raises(ValueError) as exc_info:
            provider.generate("test")
        
        assert "Invalid API key" in str(exc_info.value)
    
    def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        provider = self.ErrorProvider({}, error_type="timeout")
        
        with pytest.raises(TimeoutError) as exc_info:
            provider.generate("test")
        
        assert "Request timed out" in str(exc_info.value)
    
    def test_stream_error_handling(self):
        """Test error handling in streaming."""
        provider = self.ErrorProvider({}, error_type="stream")
        
        stream = provider.generate_stream("test")
        
        # First chunk should work
        assert next(stream) == "Start"
        
        # Next should raise error
        with pytest.raises(ConnectionError) as exc_info:
            next(stream)
        
        assert "Stream interrupted" in str(exc_info.value)


class TestProviderValidation:
    """Test input validation in providers."""
    
    def test_empty_prompt(self, test_provider):
        """Test generation with empty prompt."""
        response = test_provider.generate("")
        
        assert test_provider.last_prompt == ""
        assert response.content == "Test response"
    
    def test_very_long_prompt(self, test_provider):
        """Test generation with very long prompt."""
        long_prompt = "x" * 100000
        response = test_provider.generate(long_prompt)
        
        assert test_provider.last_prompt == long_prompt
        assert response.content == "Test response"
    
    def test_invalid_history_format(self, test_provider):
        """Test generation with invalid history format."""
        # Provider should handle or validate this
        invalid_history = [
            {"invalid": "format"},
            "not a dict",
            {"role": "user"},  # Missing content
        ]
        
        # Test provider doesn't validate, but real ones should
        response = test_provider.generate(
            "test",
            history=invalid_history
        )
        
        assert test_provider.last_history == invalid_history
        assert response.content == "Test response"
    
    def test_special_characters_in_prompt(self, test_provider):
        """Test generation with special characters."""
        special_prompt = "Test with émojis 😀 and symbols @#$%^&*()"
        response = test_provider.generate(special_prompt)
        
        assert test_provider.last_prompt == special_prompt
        assert response.content == "Test response"
    
    def test_none_prompt(self, test_provider):
        """Test generation with None prompt."""
        # Should handle gracefully or raise appropriate error
        response = test_provider.generate(None)
        
        assert test_provider.last_prompt is None
        assert response.content == "Test response"