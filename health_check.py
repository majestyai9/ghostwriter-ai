"""
Health check endpoints for monitoring critical services.
Provides comprehensive health status for all components.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import time
import json
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a service."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health information for a service."""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: Optional[float] = None
    last_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": self.response_time_ms,
            "last_check": self.last_check.isoformat(),
            "metadata": self.metadata
        }


class HealthChecker:
    """Base class for health checkers."""
    
    def check(self) -> ServiceHealth:
        """
        Check the health of a service.
        
        Returns:
            ServiceHealth object
        """
        raise NotImplementedError


class ProviderHealthChecker(HealthChecker):
    """Health checker for LLM providers."""
    
    def __init__(self, provider_name: str):
        """
        Initialize provider health checker.
        
        Args:
            provider_name: Name of the provider to check
        """
        self.provider_name = provider_name
    
    def check(self) -> ServiceHealth:
        """Check provider health."""
        start_time = time.time()
        
        try:
            from providers.factory import ProviderFactory
            from app_config import settings
            
            # Try to create provider instance
            provider = ProviderFactory.create_provider(
                self.provider_name,
                {"api_key": getattr(settings, f"{self.provider_name.upper()}_API_KEY", None)}
            )
            
            # Check if provider has circuit breaker and its state
            if hasattr(provider, "circuit_breaker"):
                cb_state = provider.circuit_breaker.state
                if cb_state == "open":
                    return ServiceHealth(
                        name=f"provider_{self.provider_name}",
                        status=HealthStatus.UNHEALTHY,
                        message="Circuit breaker is open",
                        response_time_ms=(time.time() - start_time) * 1000,
                        metadata={"circuit_breaker_state": cb_state}
                    )
                elif cb_state == "half_open":
                    return ServiceHealth(
                        name=f"provider_{self.provider_name}",
                        status=HealthStatus.DEGRADED,
                        message="Circuit breaker is half-open",
                        response_time_ms=(time.time() - start_time) * 1000,
                        metadata={"circuit_breaker_state": cb_state}
                    )
            
            # Try a simple token count to verify API key
            try:
                token_count = provider.count_tokens("test")
                response_time = (time.time() - start_time) * 1000
                
                return ServiceHealth(
                    name=f"provider_{self.provider_name}",
                    status=HealthStatus.HEALTHY,
                    message="Provider is operational",
                    response_time_ms=response_time,
                    metadata={
                        "model": getattr(provider, "model", "unknown"),
                        "token_count_test": token_count
                    }
                )
            except Exception as e:
                # API key might be invalid or service down
                return ServiceHealth(
                    name=f"provider_{self.provider_name}",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Provider check failed: {str(e)}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            return ServiceHealth(
                name=f"provider_{self.provider_name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to initialize provider: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )


class CacheHealthChecker(HealthChecker):
    """Health checker for cache services."""
    
    def check(self) -> ServiceHealth:
        """Check cache health."""
        start_time = time.time()
        
        try:
            from cache_manager import CacheManager
            
            # Test cache operations
            cache = CacheManager(backend="memory")
            test_key = "_health_check_test"
            test_value = {"timestamp": datetime.now().isoformat()}
            
            # Test set operation
            cache.set(test_key, test_value, expire=10)
            
            # Test get operation
            retrieved = cache.get(test_key)
            
            # Test delete operation
            cache.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            if retrieved == test_value:
                return ServiceHealth(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache is operational",
                    response_time_ms=response_time,
                    metadata={
                        "backend": "memory",
                        "operations_tested": ["set", "get", "delete"]
                    }
                )
            else:
                return ServiceHealth(
                    name="cache",
                    status=HealthStatus.DEGRADED,
                    message="Cache operations completed but data mismatch",
                    response_time_ms=response_time
                )
                
        except Exception as e:
            return ServiceHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )


class RAGHealthChecker(HealthChecker):
    """Health checker for RAG system."""
    
    def check(self) -> ServiceHealth:
        """Check RAG system health."""
        start_time = time.time()
        
        try:
            from token_optimizer_rag import HybridTokenOptimizer
            from app_config import settings
            
            if not getattr(settings, "ENABLE_RAG", True):
                return ServiceHealth(
                    name="rag",
                    status=HealthStatus.UNKNOWN,
                    message="RAG is disabled",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Try to create RAG instance
            rag = HybridTokenOptimizer()
            
            # Check if FAISS is available
            if not rag.faiss_available:
                return ServiceHealth(
                    name="rag",
                    status=HealthStatus.DEGRADED,
                    message="RAG running without FAISS (using fallback)",
                    response_time_ms=(time.time() - start_time) * 1000,
                    metadata={"faiss_available": False}
                )
            
            # Test basic RAG operations
            test_text = "This is a health check test."
            rag.add_to_index("test", test_text)
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="rag",
                status=HealthStatus.HEALTHY,
                message="RAG system is operational",
                response_time_ms=response_time,
                metadata={
                    "faiss_available": True,
                    "embedding_model": getattr(settings, "RAG_EMBEDDING_MODEL", "unknown")
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                name="rag",
                status=HealthStatus.UNHEALTHY,
                message=f"RAG check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )


class FileSystemHealthChecker(HealthChecker):
    """Health checker for file system operations."""
    
    def check(self) -> ServiceHealth:
        """Check file system health."""
        start_time = time.time()
        
        try:
            import os
            import tempfile
            
            # Test write/read/delete operations
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                test_file = f.name
                f.write("health check test")
            
            # Read the file
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Delete the file
            os.unlink(test_file)
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="filesystem",
                status=HealthStatus.HEALTHY,
                message="File system is operational",
                response_time_ms=response_time,
                metadata={
                    "operations_tested": ["write", "read", "delete"],
                    "temp_dir": tempfile.gettempdir()
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                name="filesystem",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.checkers: Dict[str, HealthChecker] = {}
        self.health_cache: Dict[str, ServiceHealth] = {}
        self.cache_ttl = 30  # Cache health results for 30 seconds
        
        # Register default checkers
        self._register_default_checkers()
    
    def _register_default_checkers(self):
        """Register default health checkers."""
        # Provider checkers
        providers = ["openai", "anthropic", "gemini", "cohere", "openrouter"]
        for provider in providers:
            self.register_checker(
                f"provider_{provider}",
                ProviderHealthChecker(provider)
            )
        
        # System checkers
        self.register_checker("cache", CacheHealthChecker())
        self.register_checker("rag", RAGHealthChecker())
        self.register_checker("filesystem", FileSystemHealthChecker())
    
    def register_checker(self, name: str, checker: HealthChecker):
        """
        Register a health checker.
        
        Args:
            name: Name of the checker
            checker: HealthChecker instance
        """
        self.checkers[name] = checker
    
    def check_service(self, name: str, use_cache: bool = True) -> ServiceHealth:
        """
        Check health of a specific service.
        
        Args:
            name: Service name
            use_cache: Whether to use cached results
            
        Returns:
            ServiceHealth object
        """
        # Check cache first
        if use_cache and name in self.health_cache:
            cached = self.health_cache[name]
            if (datetime.now() - cached.last_check).seconds < self.cache_ttl:
                return cached
        
        # Perform health check
        if name in self.checkers:
            health = self.checkers[name].check()
            self.health_cache[name] = health
            return health
        
        return ServiceHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            message="No health checker registered"
        )
    
    def check_all(self, use_cache: bool = True) -> Dict[str, ServiceHealth]:
        """
        Check health of all services.
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary of service health statuses
        """
        results = {}
        for name in self.checkers:
            results[name] = self.check_service(name, use_cache)
        return results
    
    def get_overall_status(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """
        Get overall system health status.
        
        Returns:
            Tuple of (overall status, detailed report)
        """
        all_health = self.check_all()
        
        # Count statuses
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        critical_services = ["filesystem", "cache"]
        critical_unhealthy = False
        
        for name, health in all_health.items():
            status_counts[health.status] += 1
            
            # Check if any critical service is unhealthy
            if name in critical_services and health.status == HealthStatus.UNHEALTHY:
                critical_unhealthy = True
        
        # Determine overall status
        if critical_unhealthy or status_counts[HealthStatus.UNHEALTHY] > len(all_health) / 2:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0 or status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        report = {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "services": {name: health.to_dict() for name, health in all_health.items()},
            "summary": {
                "total_services": len(all_health),
                "healthy": status_counts[HealthStatus.HEALTHY],
                "degraded": status_counts[HealthStatus.DEGRADED],
                "unhealthy": status_counts[HealthStatus.UNHEALTHY],
                "unknown": status_counts[HealthStatus.UNKNOWN]
            }
        }
        
        return overall_status, report
    
    def get_health_endpoint_response(self) -> Dict[str, Any]:
        """
        Get health endpoint response for API.
        
        Returns:
            Health status response
        """
        status, report = self.get_overall_status()
        
        # Add HTTP status code recommendation
        if status == HealthStatus.HEALTHY:
            report["http_status"] = 200
        elif status == HealthStatus.DEGRADED:
            report["http_status"] = 200  # Still operational
        else:
            report["http_status"] = 503  # Service unavailable
        
        return report


# Global health monitor instance
health_monitor = HealthMonitor()


# Convenience functions
def check_health(service: Optional[str] = None) -> Dict[str, Any]:
    """
    Check health of service(s).
    
    Args:
        service: Optional specific service name
        
    Returns:
        Health status dictionary
    """
    if service:
        health = health_monitor.check_service(service)
        return health.to_dict()
    else:
        return health_monitor.get_health_endpoint_response()


def is_healthy() -> bool:
    """
    Check if system is healthy.
    
    Returns:
        True if healthy, False otherwise
    """
    status, _ = health_monitor.get_overall_status()
    return status == HealthStatus.HEALTHY


def register_health_checker(name: str, checker: HealthChecker):
    """
    Register a custom health checker.
    
    Args:
        name: Checker name
        checker: HealthChecker instance
    """
    health_monitor.register_checker(name, checker)