"""
Base LLM Provider Interface with Circuit Breaker and Connection Pooling.

This module provides the abstract base class for all LLM providers, implementing
critical reliability patterns including circuit breakers for fault tolerance and
connection pooling for performance optimization.
"""
import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from exceptions import ProviderError
from tokenizer import BaseTokenizer, TokenizerFactory


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance.
    
    Attributes:
        CLOSED: Normal operation, all requests allowed
        OPEN: Circuit broken due to failures, requests blocked
        HALF_OPEN: Testing recovery, limited requests allowed
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.
    
    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Successes needed in HALF_OPEN to close circuit
        timeout: Seconds to wait before attempting recovery
        half_open_requests: Max requests allowed in HALF_OPEN state
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_requests: int = 3


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by the circuit breaker.
    
    Attributes:
        failure_count: Total number of failures
        success_count: Total number of successes
        consecutive_failures: Current consecutive failure streak
        consecutive_successes: Current consecutive success streak
        last_failure_time: Timestamp of most recent failure
        last_success_time: Timestamp of most recent success
        state_changes: History of state transitions
        total_calls: Total number of API calls attempted
        blocked_calls: Number of calls blocked by open circuit
        half_open_attempts: Current attempts in HALF_OPEN state
    """
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: List[tuple[CircuitBreakerState, datetime]] = field(
        default_factory=list
    )
    total_calls: int = 0
    blocked_calls: int = 0
    half_open_attempts: int = 0


class CircuitBreaker:
    """Thread-safe circuit breaker implementation.
    
    Prevents cascading failures by temporarily blocking requests to failing
    services and allowing gradual recovery testing.
    """

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker with configuration.
        
        Args:
            config: Circuit breaker configuration parameters
        """
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from function execution
            
        Raises:
            ProviderError: When circuit is open or function fails
        """
        with self.lock:
            self.metrics.total_calls += 1

            if self._should_attempt_reset():
                self._transition_to_half_open()

            if self.state == CircuitBreakerState.OPEN:
                self.metrics.blocked_calls += 1
                raise ProviderError(
                    f"Circuit breaker is OPEN. Service unavailable. "
                    f"Retry after {self._time_until_retry()} seconds"
                )

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.half_open_attempts >= self.config.half_open_requests:
                    self.metrics.blocked_calls += 1
                    raise ProviderError(
                        "Circuit breaker is HALF_OPEN with max attempts reached"
                    )
                self.metrics.half_open_attempts += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from function execution
            
        Raises:
            ProviderError: When circuit is open or function fails
        """
        with self.lock:
            self.metrics.total_calls += 1

            if self._should_attempt_reset():
                self._transition_to_half_open()

            if self.state == CircuitBreakerState.OPEN:
                self.metrics.blocked_calls += 1
                raise ProviderError(
                    f"Circuit breaker is OPEN. Service unavailable. "
                    f"Retry after {self._time_until_retry()} seconds"
                )

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.half_open_attempts >= self.config.half_open_requests:
                    self.metrics.blocked_calls += 1
                    raise ProviderError(
                        "Circuit breaker is HALF_OPEN with max attempts reached"
                    )
                self.metrics.half_open_attempts += 1

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    def _record_success(self) -> None:
        """Record successful call and update state if needed."""
        with self.lock:
            self.metrics.success_count += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.now()

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()

    def _record_failure(self) -> None:
        """Record failed call and update state if needed."""
        with self.lock:
            self.metrics.failure_count += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.now()

            if self.state == CircuitBreakerState.HALF_OPEN or (self.state == CircuitBreakerState.CLOSED and
                  self.metrics.consecutive_failures >= self.config.failure_threshold):
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt recovery.
        
        Returns:
            True if timeout has elapsed and reset should be attempted
        """
        return (
            self.state == CircuitBreakerState.OPEN and
            self.metrics.last_failure_time and
            (datetime.now() - self.metrics.last_failure_time).total_seconds()
            >= self.config.timeout
        )

    def _time_until_retry(self) -> float:
        """Calculate seconds until retry is allowed.
        
        Returns:
            Seconds remaining until circuit can attempt recovery
        """
        if not self.metrics.last_failure_time:
            return 0
        elapsed = (datetime.now() - self.metrics.last_failure_time).total_seconds()
        return max(0, self.config.timeout - elapsed)

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.metrics.consecutive_failures = 0
        self.metrics.half_open_attempts = 0
        self.metrics.state_changes.append((self.state, datetime.now()))
        self.logger.info("Circuit breaker transitioned to CLOSED")

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.metrics.consecutive_successes = 0
        self.metrics.half_open_attempts = 0
        self.metrics.state_changes.append((self.state, datetime.now()))
        self.logger.warning(
            f"Circuit breaker transitioned to OPEN after "
            f"{self.metrics.consecutive_failures} failures"
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes = 0
        self.metrics.half_open_attempts = 0
        self.metrics.state_changes.append((self.state, datetime.now()))
        self.logger.info("Circuit breaker transitioned to HALF_OPEN")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.
        
        Returns:
            Current state of the circuit breaker
        """
        with self.lock:
            return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics.
        
        Returns:
            Copy of current metrics
        """
        with self.lock:
            return CircuitBreakerMetrics(
                failure_count=self.metrics.failure_count,
                success_count=self.metrics.success_count,
                consecutive_failures=self.metrics.consecutive_failures,
                consecutive_successes=self.metrics.consecutive_successes,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                state_changes=self.metrics.state_changes.copy(),
                total_calls=self.metrics.total_calls,
                blocked_calls=self.metrics.blocked_calls,
                half_open_attempts=self.metrics.half_open_attempts
            )


class ConnectionPool:
    """Thread-safe HTTP connection pool manager using aiohttp.
    
    Manages persistent HTTP sessions with connection pooling to reduce
    latency and improve performance for API calls.
    """

    def __init__(self,
                 connector_limit: int = 100,
                 connector_limit_per_host: int = 30,
                 timeout: float = 30.0):
        """Initialize connection pool with configuration.
        
        Args:
            connector_limit: Total connection pool size
            connector_limit_per_host: Max connections per host
            timeout: Default timeout for requests in seconds
        """
        self.connector_limit = connector_limit
        self.connector_limit_per_host = connector_limit_per_host
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.ConnectionPool")

    @asynccontextmanager
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling.
        
        Yields:
            Configured aiohttp ClientSession
        """
        if self.session is None or self.session.closed:
            with self.lock:
                if self.session is None or self.session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self.connector_limit,
                        limit_per_host=self.connector_limit_per_host,
                        force_close=False,
                        enable_cleanup_closed=True
                    )
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=self.timeout
                    )
                    self.logger.info(
                        f"Created connection pool with limit={self.connector_limit}, "
                        f"per_host={self.connector_limit_per_host}"
                    )
        yield self.session

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Connection pool closed")


@dataclass
class LLMResponse:
    """Standard response format from LLM providers.
    
    Attributes:
        content: Generated text content
        tokens_used: Number of tokens consumed
        finish_reason: Reason for completion (e.g., 'stop', 'length')
        model: Model identifier used for generation
        raw_response: Optional raw API response for debugging
    """
    content: str
    tokens_used: int
    finish_reason: str
    model: str
    raw_response: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers with reliability features.
    
    Provides circuit breaker pattern for fault tolerance and connection
    pooling for performance optimization. All provider implementations
    should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.tokenizer: Optional[BaseTokenizer] = None

        # Initialize circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=config.get('circuit_breaker_failures', 5),
            success_threshold=config.get('circuit_breaker_successes', 2),
            timeout=config.get('circuit_breaker_timeout', 60),
            half_open_requests=config.get('circuit_breaker_half_open', 3)
        )
        self.circuit_breaker = CircuitBreaker(cb_config)

        # Initialize connection pool
        self.connection_pool = ConnectionPool(
            connector_limit=config.get('pool_connections', 100),
            connector_limit_per_host=config.get('pool_connections_per_host', 30),
            timeout=config.get('request_timeout', 30.0)
        )

        self._validate_config()
        self._init_tokenizer()

    @abstractmethod
    def _validate_config(self):
        """Validate provider configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """Generate text completion with circuit breaker protection.
        
        Args:
            prompt: The input prompt for generation
            history: Optional conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse containing generated text and metadata
            
        Raises:
            ProviderError: If generation fails or circuit is open
        """
        pass

    @abstractmethod
    async def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming support.
        
        Args:
            prompt: Input prompt for generation
            history: Optional conversation history  
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Text chunks as they're generated
            
        Raises:
            ProviderError: If generation fails or circuit is open
        """
        pass

    def _init_tokenizer(self) -> None:
        """Initialize tokenizer for the provider."""
        provider_name = self.config.get(
            'provider',
            self.__class__.__name__.replace('Provider', '').lower()
        )
        model_name = getattr(self, 'model', None) or self.config.get('model')
        self.tokenizer = TokenizerFactory.create(provider_name, model_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using provider-specific tokenizer.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Number of tokens in the text
        """
        if self.tokenizer:
            return self.tokenizer.count_tokens(text)
        # Fallback to rough approximation
        return len(text) // 4

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model metadata including:
            - name: Model identifier
            - max_tokens: Maximum context length
            - capabilities: List of supported features
        """
        pass

    def validate_token_limit(self,
                            messages: List[Dict[str, str]],
                            max_tokens: int) -> bool:
        """Check if messages fit within model's token limit.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response
            
        Returns:
            True if within limit, False otherwise
        """
        total_tokens = sum(
            self.count_tokens(msg.get('content', ''))
            for msg in messages
        )
        model_limit = self.get_model_info().get('max_tokens', 4096)
        return total_tokens + max_tokens <= model_limit

    def prepare_messages(self,
                        prompt: str,
                        history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Prepare messages for API call.
        
        Args:
            prompt: Current user prompt
            history: Optional conversation history
            
        Returns:
            Formatted list of message dictionaries
        """
        messages = history.copy() if history else []
        messages.append({"role": "user", "content": prompt})
        return messages

    def _handle_error(self, error: Exception) -> ProviderError:
        """Translate provider exceptions to standardized format.

        Args:
            error: The provider-specific exception

        Returns:
            Standardized ProviderError instance
        """
        # Basic implementation - subclasses should override for specific handling
        return ProviderError(str(error))

    async def _call_with_retry(self,
                        api_call: callable,
                        max_retries: int = 3,
                        exponential_base: float = 2.0,
                        jitter: bool = True,
                        **kwargs) -> Any:
        """Execute API call with circuit breaker protection and retry logic.
        
        Implements exponential backoff with optional jitter for rate limit
        handling and transient error recovery.
        
        Args:
            api_call: The API function to execute
            max_retries: Maximum number of retry attempts
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add randomization to backoff delays
            **kwargs: Arguments to pass to the API call
            
        Returns:
            Result from the successful API call
            
        Raises:
            ProviderError: If all retries fail or circuit breaker is open
        """
        # Check circuit breaker state
        if self.circuit_breaker.get_state() == CircuitBreakerState.OPEN:
            raise ProviderError("Circuit breaker is open, service unavailable")
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Execute with circuit breaker protection
                if asyncio.iscoroutinefunction(api_call):
                    result = await api_call(**kwargs)
                else:
                    result = api_call(**kwargs)
                self.circuit_breaker._record_success()
                
                # Reset circuit breaker on success
                if attempt > 0:
                    self.logger.info(f"API call succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failure in circuit breaker
                self.circuit_breaker._record_failure()
                
                # Check if we should retry
                if attempt >= max_retries:
                    self.logger.error(f"All {max_retries} retries failed: {e}")
                    break
                
                # Calculate backoff delay
                delay = exponential_base ** attempt
                if jitter:
                    import random
                    delay *= (0.5 + random.random())  # Add 50-150% jitter
                
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.2f} seconds: {e}"
                )
                
                # Wait before retry using async sleep
                await asyncio.sleep(delay)
        
        # All retries failed
        if last_exception:
            raise self._handle_error(last_exception)
        else:
            raise ProviderError("API call failed without exception")

    def get_circuit_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.
        
        Returns:
            Current state of the circuit breaker
        """
        return self.circuit_breaker.get_state()

    def get_circuit_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics for monitoring.
        
        Returns:
            Current circuit breaker metrics
        """
        return self.circuit_breaker.get_metrics()

    async def cleanup(self) -> None:
        """Cleanup resources including connection pool.
        
        Should be called when provider is no longer needed.
        """
        await self.connection_pool.close()
