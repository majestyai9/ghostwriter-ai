"""
Event system for progress tracking and extensibility
"""
from typing import Dict, List, Callable, Any
from enum import Enum
import logging

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

class Event:
    """Event data structure"""
    def __init__(self, event_type: EventType, data: Dict[str, Any] = None):
        self.type = event_type
        self.data = data or {}
        
    def __str__(self):
        return f"Event({self.type.value}, {self.data})"

class EventManager:
    """Manages event subscriptions and dispatching"""
    
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._global_listeners: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to a specific event type"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        self.logger.debug(f"Subscribed {callback.__name__} to {event_type.value}")
        
    def subscribe_all(self, callback: Callable):
        """Subscribe to all events"""
        self._global_listeners.append(callback)
        self.logger.debug(f"Subscribed {callback.__name__} to all events")
        
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from a specific event type"""
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed {callback.__name__} from {event_type.value}")
            
    def emit(self, event: Event):
        """Emit an event to all subscribers"""
        self.logger.debug(f"Emitting event: {event}")
        
        # Notify specific listeners
        if event.type in self._listeners:
            for callback in self._listeners[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback {callback.__name__}: {e}")
                    
        # Notify global listeners
        for callback in self._global_listeners:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in global callback {callback.__name__}: {e}")

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
            percentage = (self.completed_chapters / self.total_chapters * 100) if self.total_chapters > 0 else 0
            logging.info(f"Progress: {self.completed_chapters}/{self.total_chapters} chapters ({percentage:.1f}%)")
            
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
            'percentage': (self.completed_chapters / self.total_chapters * 100) if self.total_chapters > 0 else 0
        }