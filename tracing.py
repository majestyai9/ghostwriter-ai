"""
Distributed tracing implementation using OpenTelemetry.
Provides comprehensive tracing for debugging complex workflows.
"""

import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager
import json
import time

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

from app_config import settings

logger = logging.getLogger(__name__)


class TracingManager:
    """Manages distributed tracing for the application."""
    
    def __init__(self):
        """Initialize the tracing manager."""
        self.enabled = getattr(settings, "TRACING_ENABLED", False) and OTEL_AVAILABLE
        self.tracer = None
        self.provider = None
        
        if self.enabled:
            self._initialize_tracing()
    
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing."""
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": "ghostwriter-ai",
                "service.version": "1.0.0",
                "deployment.environment": getattr(settings, "ENVIRONMENT", "development")
            })
            
            # Create tracer provider
            self.provider = TracerProvider(resource=resource)
            
            # Add exporters based on configuration
            if getattr(settings, "TRACING_CONSOLE_EXPORT", False):
                # Console exporter for development
                console_exporter = ConsoleSpanExporter()
                self.provider.add_span_processor(
                    BatchSpanProcessor(console_exporter)
                )
            
            if hasattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT"):
                # OTLP exporter for production
                otlp_exporter = OTLPSpanExporter(
                    endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                    insecure=getattr(settings, "OTEL_EXPORTER_OTLP_INSECURE", True)
                )
                self.provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
            
            # Set the tracer provider
            trace.set_tracer_provider(self.provider)
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument HTTP requests automatically
            if getattr(settings, "TRACING_INSTRUMENT_REQUESTS", True):
                RequestsInstrumentor().instrument()
            
            logger.info("OpenTelemetry tracing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.enabled = False
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a tracing span context manager.
        
        Args:
            name: Name of the span
            attributes: Optional attributes to add to the span
        """
        if not self.enabled or not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            try:
                # Add attributes if provided
                if attributes:
                    for key, value in attributes.items():
                        # Convert complex types to strings
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        span.set_attribute(key, str(value))
                
                yield span
                
                # Mark span as successful
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Record exception in span
                if span:
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                raise
    
    def trace_function(self, name: Optional[str] = None):
        """
        Decorator to trace function execution.
        
        Args:
            name: Optional custom name for the span
        """
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Create attributes from function arguments
                attributes = {
                    "function.module": func.__module__,
                    "function.name": func.__name__,
                }
                
                # Add selected kwargs as attributes (avoid sensitive data)
                safe_kwargs = ["provider", "model", "chapter_number", "title", "language"]
                for key in safe_kwargs:
                    if key in kwargs:
                        attributes[f"function.arg.{key}"] = str(kwargs[key])
                
                with self.span(span_name, attributes):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def record_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Record an event in the current span.
        
        Args:
            name: Name of the event
            attributes: Optional attributes for the event
        """
        if not self.enabled or not trace:
            return
        
        current_span = trace.get_current_span()
        if current_span:
            event_attributes = attributes or {}
            # Convert complex types to strings
            for key, value in event_attributes.items():
                if isinstance(value, (dict, list)):
                    event_attributes[key] = json.dumps(value)
                else:
                    event_attributes[key] = str(value)
            
            current_span.add_event(name, attributes=event_attributes)
    
    def set_attribute(self, key: str, value: Any):
        """
        Set an attribute on the current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled or not trace:
            return
        
        current_span = trace.get_current_span()
        if current_span:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            current_span.set_attribute(key, str(value))
    
    def get_trace_context(self) -> Optional[Dict[str, str]]:
        """
        Get the current trace context for propagation.
        
        Returns:
            Trace context dictionary or None
        """
        if not self.enabled or not trace:
            return None
        
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            return {
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
                "trace_flags": format(span_context.trace_flags, "02x")
            }
        return None


# Global tracing manager instance
tracing_manager = TracingManager()


# Convenience functions
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a tracing span context manager."""
    return tracing_manager.span(name, attributes)


def trace_function(name: Optional[str] = None):
    """Decorator to trace function execution."""
    return tracing_manager.trace_function(name)


def record_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Record an event in the current span."""
    tracing_manager.record_event(name, attributes)


def set_trace_attribute(key: str, value: Any):
    """Set an attribute on the current span."""
    tracing_manager.set_attribute(key, value)


def get_trace_context() -> Optional[Dict[str, str]]:
    """Get the current trace context."""
    return tracing_manager.get_trace_context()


# Trace common operations
class TracedOperations:
    """Common traced operations for the application."""
    
    @staticmethod
    @contextmanager
    def trace_book_generation(title: str, language: str, provider: str):
        """Trace book generation workflow."""
        with trace_span("book.generation", {
            "book.title": title,
            "book.language": language,
            "provider.name": provider,
            "operation.type": "full_book"
        }):
            yield
    
    @staticmethod
    @contextmanager
    def trace_chapter_generation(chapter_number: int, chapter_title: str):
        """Trace chapter generation."""
        with trace_span("chapter.generation", {
            "chapter.number": chapter_number,
            "chapter.title": chapter_title,
            "operation.type": "chapter"
        }):
            yield
    
    @staticmethod
    @contextmanager
    def trace_llm_call(provider: str, model: str, tokens: int):
        """Trace LLM API call."""
        with trace_span("llm.call", {
            "provider.name": provider,
            "model.name": model,
            "tokens.count": tokens,
            "operation.type": "llm_api"
        }):
            yield
    
    @staticmethod
    @contextmanager
    def trace_cache_operation(operation: str, key: str, hit: bool = False):
        """Trace cache operation."""
        with trace_span(f"cache.{operation}", {
            "cache.key": key,
            "cache.hit": hit,
            "operation.type": "cache"
        }):
            yield
    
    @staticmethod
    @contextmanager
    def trace_rag_operation(operation: str, chunks: int = 0):
        """Trace RAG operation."""
        with trace_span(f"rag.{operation}", {
            "rag.chunks": chunks,
            "operation.type": "rag"
        }):
            yield