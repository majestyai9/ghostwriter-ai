"""
Event system for progress tracking and extensibility
"""
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List


class EventType(Enum):
    """Event types for book generation process"""
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"

    TITLE_STARTED = "title_started"
    TITLE_COMPLETED = "title_completed"

    TOC_STARTED = "toc_started"
    TOC_COMPLETED = "toc_completed"

    SUMMARY_STARTED = "summary_started"
    SUMMARY_COMPLETED = "summary_completed"

    CHAPTER_STARTED = "chapter_started"
    CHAPTER_COMPLETED = "chapter_completed"

    SECTION_STARTED = "section_started"
    SECTION_COMPLETED = "section_completed"

    BOOK_SAVED = "book_saved"
    BOOK_EXPORTED = "book_exported"

    API_CALL_STARTED = "api_call_started"
    API_CALL_COMPLETED = "api_call_completed"
    API_CALL_FAILED = "api_call_failed"

    RETRY_ATTEMPTED = "retry_attempted"
    TOKEN_LIMIT_REACHED = "token_limit_reached"
    RATE_LIMIT_HIT = "rate_limit_hit"

class UIEventType(Enum):
    """UI-specific event types for Gradio interface"""
    # Project events
    PROJECT_CREATED = "ui_project_created"
    PROJECT_DELETED = "ui_project_deleted"
    PROJECT_ARCHIVED = "ui_project_archived"
    PROJECT_SELECTED = "ui_project_selected"
    
    # Character events
    CHARACTER_CREATED = "ui_character_created"
    CHARACTER_UPDATED = "ui_character_updated"
    CHARACTER_DELETED = "ui_character_deleted"
    
    # Style events
    STYLE_SELECTED = "ui_style_selected"
    STYLE_CREATED = "ui_style_created"
    STYLE_PREVIEW = "ui_style_preview"
    
    # Export events
    EXPORT_STARTED = "ui_export_started"
    EXPORT_COMPLETED = "ui_export_completed"
    EXPORT_FAILED = "ui_export_failed"
    
    # Batch operations
    BATCH_OPERATION_STARTED = "ui_batch_started"
    BATCH_OPERATION_PROGRESS = "ui_batch_progress"
    BATCH_OPERATION_COMPLETED = "ui_batch_completed"
    BATCH_OPERATION_FAILED = "ui_batch_failed"
    
    # Performance events
    PERFORMANCE_WARNING = "ui_performance_warning"
    MEMORY_WARNING = "ui_memory_warning"
    CACHE_CLEARED = "ui_cache_cleared"
    
    # Error events
    ERROR_OCCURRED = "ui_error_occurred"
    ERROR_RECOVERED = "ui_error_recovered"
    RETRY_STARTED = "ui_retry_started"

class Event:
    """Event data structure"""
    def __init__(self, event_type: EventType, data: Dict[str, Any] = None):
        self.type = event_type
        self.data = data or {}

    def __str__(self):
        return f"Event({self.type.value}, {self.data})"

class EventManager:
    """Thread-safe event manager for subscriptions and dispatching"""

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._global_listeners: List[Callable] = []
        self._lock = threading.RLock()  # RLock for nested locking safety
        self.logger = logging.getLogger(__name__)

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to a specific event type with thread safety"""
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(callback)
            self.logger.debug(f"Subscribed {callback.__name__} to {event_type.value}")

    def subscribe_all(self, callback: Callable):
        """Subscribe to all events with thread safety"""
        with self._lock:
            self._global_listeners.append(callback)
            self.logger.debug(f"Subscribed {callback.__name__} to all events")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from a specific event type with thread safety"""
        with self._lock:
            if event_type in self._listeners and callback in self._listeners[event_type]:
                self._listeners[event_type].remove(callback)
                msg = f"Unsubscribed {callback.__name__} from {event_type.value}"
                self.logger.debug(msg)

    def emit(self, event: Event):
        """Emit an event to all subscribers with thread safety"""
        self.logger.debug(f"Emitting event: {event}")

        # Create a copy of listeners to avoid modification during iteration
        with self._lock:
            specific_listeners = self._listeners.get(event.type, []).copy()
            global_listeners = self._global_listeners.copy()

        # Notify specific listeners (outside lock to prevent deadlock)
        for callback in specific_listeners:
            try:
                callback(event)
            except Exception as e:
                error_msg = f"Error in event callback {callback.__name__}: {e}"
                self.logger.error(error_msg)

        # Notify global listeners
        for callback in global_listeners:
            try:
                callback(event)
            except Exception as e:
                error_msg = f"Error in global callback {callback.__name__}: {e}"
                self.logger.error(error_msg)

# Global event manager instance
event_manager = EventManager()

# Convenience decorator for event handlers
def on_event(event_type: EventType):
    """Decorator to register event handlers"""
    def decorator(func):
        event_manager.subscribe(event_type, func)
        return func
    return decorator

# Built-in event handlers for logging
class ProgressTracker:
    """Built-in progress tracking handler"""

    def __init__(self):
        self.total_chapters = 0
        self.completed_chapters = 0
        self.total_sections = 0
        self.completed_sections = 0
        self.current_chapter = None
        self.current_section = None

    def track_progress(self, event: Event):
        """Track generation progress"""
        if event.type == EventType.TOC_COMPLETED:
            toc = event.data.get('toc', {})
            self.total_chapters = len(toc.get('chapters', []))
            self.total_sections = sum(len(ch.get('sections', []))
                                     for ch in toc.get('chapters', []))

        elif event.type == EventType.CHAPTER_STARTED:
            self.current_chapter = event.data.get('chapter_number')

        elif event.type == EventType.CHAPTER_COMPLETED:
            self.completed_chapters += 1
            if self.total_chapters > 0:
                percentage = (self.completed_chapters / self.total_chapters * 100)
            else:
                percentage = 0
            progress_msg = (f"Progress: {self.completed_chapters}/{self.total_chapters} "
                          f"chapters ({percentage:.1f}%)")
            logging.info(progress_msg)

        elif event.type == EventType.SECTION_STARTED:
            self.current_section = event.data.get('section_number')

        elif event.type == EventType.SECTION_COMPLETED:
            self.completed_sections += 1

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        return {
            'chapters': {'completed': self.completed_chapters, 'total': self.total_chapters},
            'sections': {'completed': self.completed_sections, 'total': self.total_sections},
            'current_chapter': self.current_chapter,
            'current_section': self.current_section,
            'percentage': ((self.completed_chapters / self.total_chapters * 100)
                          if self.total_chapters > 0 else 0)
        }
