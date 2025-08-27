"""
Gradio Handlers: Business logic for Gradio interface.
Separates UI components from actual functionality.
"""

import logging
import json
import asyncio
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import deque
import traceback

from containers import get_container
from project_manager import ProjectManager
from style_templates import StyleManager
from character_tracker import CharacterDatabase
from events import event_manager, Event, EventType, UIEventType
from book_generator import BookGenerator
from export_formats import BookExporter

logger = logging.getLogger(__name__)
# Gradio-specific logger configuration
class GradioLogger:
    """Enhanced logger for Gradio UI with user-friendly messages."""
    
    def __init__(self, base_logger):
        self.base_logger = base_logger
        self.ui_logs = deque(maxlen=100)  # Keep last 100 UI logs
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup custom handlers for UI logging."""
        # Create UI handler that stores logs for display
        ui_handler = logging.Handler()
        ui_handler.emit = self._ui_handler_emit
        self.base_logger.addHandler(ui_handler)
    
    def _ui_handler_emit(self, record):
        """Custom handler that stores logs for UI display."""
        timestamp = time.strftime('%H:%M:%S', time.localtime(record.created))
        level_emoji = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }.get(record.levelname, '📝')
        
        user_message = self._get_user_friendly_message(record.getMessage())
        
        self.ui_logs.append({
            'timestamp': timestamp,
            'level': record.levelname,
            'emoji': level_emoji,
            'message': user_message,
            'raw_message': record.getMessage()
        })
    
    def _get_user_friendly_message(self, message: str) -> str:
        """Convert technical messages to user-friendly ones."""
        replacements = {
            'API_CALL_FAILED': 'Connection issue with AI service. Retrying...',
            'RATE_LIMIT': 'Too many requests. Waiting a moment...',
            'TOKEN_LIMIT': 'Content too long. Optimizing...',
            'CONNECTION_ERROR': 'Network issue detected. Reconnecting...',
            'TIMEOUT': 'Operation taking longer than expected...',
            'RETRY': 'Attempting to recover...',
            'CACHE_HIT': 'Using cached data for faster response',
            'CACHE_MISS': 'Fetching fresh data...'
        }
        
        for key, friendly_msg in replacements.items():
            if key.lower() in message.lower():
                return friendly_msg
        
        return message
    
    def get_ui_logs(self, count: int = 10) -> List[str]:
        """Get recent UI logs formatted for display."""
        recent_logs = list(self.ui_logs)[-count:]
        formatted_logs = []
        
        for log in recent_logs:
            formatted_logs.append(
                f"{log['timestamp']} {log['emoji']} {log['message']}"
            )
        
        return formatted_logs
    
    def info(self, message: str, ui_message: str = None):
        """Log info with optional UI-specific message."""
        self.base_logger.info(message)
        if ui_message:
            self._add_ui_log('INFO', ui_message)
    
    def warning(self, message: str, ui_message: str = None):
        """Log warning with optional UI-specific message."""
        self.base_logger.warning(message)
        if ui_message:
            self._add_ui_log('WARNING', ui_message)
    
    def error(self, message: str, ui_message: str = None):
        """Log error with optional UI-specific message."""
        self.base_logger.error(message)
        if ui_message:
            self._add_ui_log('ERROR', ui_message)
    
    def _add_ui_log(self, level: str, message: str):
        """Add a log entry specifically for UI display."""
        timestamp = time.strftime('%H:%M:%S')
        level_emoji = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌'
        }.get(level, '📝')
        
        self.ui_logs.append({
            'timestamp': timestamp,
            'level': level,
            'emoji': level_emoji,
            'message': message,
            'raw_message': message
        })

# Replace the standard logger with Gradio-enhanced logger
from functools import lru_cache, wraps
import threading
import psutil

gradio_logger = GradioLogger(logger)

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self._lock = threading.Lock()
        
    def record_metric(self, operation: str, duration: float, memory_mb: float):
        """Record a performance metric."""
        with self._lock:
            self.metrics_history.append({
                'timestamp': time.time(),
                'operation': operation,
                'duration': duration,
                'memory_mb': memory_mb
            })
    
    def get_metrics(self) -> List[Dict]:
        """Get recent metrics."""
        with self._lock:
            return list(self.metrics_history)
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            operations = {}
            for metric in self.metrics_history:
                op = metric['operation']
                if op not in operations:
                    operations[op] = {'count': 0, 'total_duration': 0, 'avg_memory': 0}
                operations[op]['count'] += 1
                operations[op]['total_duration'] += metric['duration']
                operations[op]['avg_memory'] += metric['memory_mb']
            
            for op in operations:
                operations[op]['avg_duration'] = operations[op]['total_duration'] / operations[op]['count']
                operations[op]['avg_memory'] /= operations[op]['count']
            
            return operations

# Cache decorator with TTL
def timed_cache(seconds: int = 300):
    """Cache decorator with TTL (time-to-live)."""
    def decorator(func):
        cache = {}
        cache_time = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key = str(args) + str(kwargs)
            
            # Check if cached value exists and is still valid
            if key in cache and key in cache_time:
                if time.time() - cache_time[key] < seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = time.time()
            logger.debug(f"Cache miss for {func.__name__}, caching result")
            
            return result
        
        wrapper.clear_cache = lambda: cache.clear() or cache_time.clear()
        return wrapper
    
    return decorator

# Debounce decorator
def debounce(wait: float):
    """Debounce decorator to prevent frequent calls."""
    def decorator(func):
        func._timer = None
        func._lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            def call_func():
                with func._lock:
                    func._timer = None
                return func(*args, **kwargs)
            
            with func._lock:
                if func._timer is not None:
                    func._timer.cancel()
                func._timer = threading.Timer(wait, call_func)
                func._timer.start()
        
        return wrapper
    
    return decorator

# Performance tracking decorator
def track_performance(monitor: Optional['PerformanceMonitor'] = None):
    """Track performance of function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                # Record metrics
                duration = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = end_memory - start_memory
                
                if monitor:
                    monitor.record_metric(func.__name__, duration, memory_used)
                
                logger.debug(f"{func.__name__} took {duration:.3f}s, used {memory_used:.2f}MB")
                
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator

class GradioHandlers:
    """
    Handles business logic for Gradio interface.
    Separates UI concerns from application logic.
    """
    
    def __init__(self):
        """Initialize handlers with dependencies from DI container."""
        self.container = get_container()
        self.project_manager = self.container.project_manager()
        self.style_manager = StyleManager()
        self.character_db = None  # Initialized per project
        self.book_exporter = BookExporter()
        self.generation_active = False
        self.current_generation = None
        self.event_logs = []
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Cache for expensive operations
        self._project_cache = {}
        self._style_cache = {}
        self._character_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Batch operation queues
        self.batch_queue = deque()
        self.batch_processing = False
        self._batch_lock = threading.Lock()
        
        # Subscribe to events for progress tracking
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for real-time updates"""
        from events import event_manager, EventType
        
        def log_event(event):
            """Log generation events"""
            self.event_logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "data": event.data
            })
            logger.info(f"Event: {event.type} - {event.data}")
        
        # Subscribe to relevant events
        event_manager.subscribe(EventType.CHAPTER_COMPLETED, log_event)
        event_manager.subscribe(EventType.GENERATION_COMPLETED, log_event)
        event_manager.subscribe(EventType.GENERATION_FAILED, log_event)
    
    # ===== Project Management =====
    
    @timed_cache(seconds=60)  # Cache for 1 minute
    @track_performance()
    def list_projects(self) -> List[Dict[str, Any]]:
        """Get list of all projects with metadata."""
        try:
            projects = self.project_manager.list_projects()
            formatted_projects = []
            for p in projects:
                # Convert ProjectMetadata to dict
                if hasattr(p, 'to_dict'):
                    project_dict = p.to_dict()
                else:
                    project_dict = p
                formatted_projects.append(self._format_project_info(project_dict))
            return formatted_projects
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return []
    
    def create_project(
        self,
        title: str,
        language: str,
        style: str,
        instructions: str,
        chapters: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Create a new book project.
        
        Returns:
            Tuple of (success, message, project_info)
        """
        try:
            # Validate inputs
            if not title or not title.strip():
                return False, "Title is required", {}
            
            if chapters < 1 or chapters > 100:
                return False, "Chapters must be between 1 and 100", {}
            
            # Create project
            project_id = self.project_manager.create_project(
                title=title.strip(),
                language=language or "English",
                style=style or "general",
                metadata={
                    "instructions": instructions,
                    "chapters": chapters,
                    "created_at": datetime.now().isoformat(),
                    "status": "draft"
                }
            )
            
            # Get project info from list
            projects = self.project_manager.list_projects()
            project_info = None
            for p in projects:
                p_id = p.project_id if hasattr(p, 'project_id') else p.get('id', '')
                if p_id == project_id:
                    project_info = p.to_dict() if hasattr(p, 'to_dict') else p
                    break
            
            if not project_info:
                # If not found in list, create minimal info
                project_info = {
                    'id': project_id,
                    'title': title,
                    'language': language,
                    'style': style,
                    'metadata': metadata
                }
            
            return True, f"Project '{title}' created successfully", self._format_project_info(project_info)
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return False, f"Error creating project: {str(e)}", {}
    
    def delete_project(self, project_id: str) -> Tuple[bool, str]:
        """Delete a project."""
        try:
            self.project_manager.delete_project(project_id)
            return True, "Project deleted successfully"
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            return False, f"Error deleting project: {str(e)}"
    
    def archive_project(self, project_id: str) -> Tuple[bool, str]:
        """Archive a project."""
        try:
            archive_path = self.project_manager.archive_project(project_id)
            return True, f"Project archived to {archive_path}"
        except Exception as e:
            logger.error(f"Error archiving project {project_id}: {e}")
            return False, f"Error archiving project: {str(e)}"
    
    def get_project_details(self, project_id: str) -> Dict[str, Any]:
        """Get detailed information about a project."""
        try:
            projects = self.project_manager.list_projects()
            for p in projects:
                p_id = p.project_id if hasattr(p, 'project_id') else p.get('id', '')
                if p_id == project_id:
                    project_dict = p.to_dict() if hasattr(p, 'to_dict') else p
                    return self._format_project_info(project_dict)
            return {}
        except Exception as e:
            logger.error(f"Error getting project details: {e}")
            return {}
    
    # ===== Book Generation =====
    
    async def generate_book(
        self,
        project_id: str,
        provider: str,
        model: str,
        temperature: float,
        enable_rag: bool,
        enable_quality: bool,
        progress_callback=None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Tuple[bool, str]:
        """
        Generate a book for the specified project with retry mechanism.
        
        Args:
            project_id: Project ID
            provider: LLM provider name
            model: Model name
            temperature: Generation temperature
            enable_rag: Enable RAG system
            enable_quality: Enable quality checks
            progress_callback: Optional progress callback
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries (exponential backoff)
        
        Returns:
            Tuple of (success, message)
        """
        if self.generation_active:
            return False, "Generation already in progress"
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self.generation_active = True
                self.event_logs = []
                
                # Get project details with caching
                cache_key = f"project_{project_id}"
                if cache_key in self._project_cache and \
                   time.time() - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl:
                    project = self._project_cache[cache_key]
                else:
                    project = self.project_manager.get_project(project_id)
                    self._project_cache[cache_key] = project
                    self._cache_timestamps[cache_key] = time.time()
                
                if not project:
                    return False, "Project not found"
                
                # Create provider configuration with validation
                api_key = self._get_api_key(provider)
                if not api_key:
                    return False, f"API key not found for provider {provider}"
                
                provider_config = {
                    "provider": provider,
                    "model": model,
                    "temperature": max(0, min(1, temperature)),  # Ensure valid range
                    "api_key": api_key
                }
                
                # Create generation configuration
                generation_config = {
                    "title": project["title"],
                    "language": project.get("language", "English"),
                    "style": project.get("style", "general"),
                    "instructions": project.get("metadata", {}).get("instructions", ""),
                    "chapters": project.get("metadata", {}).get("chapters", 10),
                    "enable_rag": enable_rag,
                    "enable_quality_checks": enable_quality,
                    "project_id": project_id
                }
                
                # Log generation start
                logger.info(f"Starting book generation for project {project_id} (attempt {retry_count + 1})")
                
                # Initialize book generator with error handling
                try:
                    generator = BookGenerator(
                        provider_config=provider_config,
                        generation_config=generation_config,
                        project_manager=self.project_manager
                    )
                except Exception as init_error:
                    logger.error(f"Failed to initialize generator: {init_error}")
                    if retry_count < max_retries:
                        retry_count += 1
                        await asyncio.sleep(retry_delay * (2 ** retry_count))  # Exponential backoff
                        continue
                    else:
                        raise
                
                # Set current generation for tracking
                self.current_generation = generator
                
                # Generate the book with timeout
                try:
                    result = await asyncio.wait_for(
                        generator.generate_async(),
                        timeout=3600  # 1 hour timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Book generation timed out")
                    if retry_count < max_retries:
                        retry_count += 1
                        await asyncio.sleep(retry_delay * (2 ** retry_count))
                        continue
                    else:
                        return False, "Generation timed out after maximum retries"
                
                if result["success"]:
                    # Update project status
                    self.project_manager.update_project(project_id, {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "generation_attempts": retry_count + 1
                    })
                    
                    # Clear cache for this project
                    if cache_key in self._project_cache:
                        del self._project_cache[cache_key]
                    
                    # Record performance
                    if hasattr(self, 'performance_monitor'):
                        self.performance_monitor.record_metric(
                            "book_generation",
                            result.get("duration", 0),
                            result.get("memory_used", 0)
                        )
                    
                    return True, f"Book generated successfully: {result['book_path']}"
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"Generation failed (attempt {retry_count + 1}): {error_msg}")
                    
                    # Check if error is retryable
                    retryable_errors = ['rate limit', 'timeout', 'connection', 'temporary']
                    is_retryable = any(err in error_msg.lower() for err in retryable_errors)
                    
                    if is_retryable and retry_count < max_retries:
                        retry_count += 1
                        await asyncio.sleep(retry_delay * (2 ** retry_count))
                        continue
                    else:
                        return False, f"Generation failed: {error_msg}"
                        
            except Exception as e:
                last_error = e
                logger.error(f"Error generating book (attempt {retry_count + 1}): {e}\n{traceback.format_exc()}")
                
                # Determine if error is retryable
                error_str = str(e).lower()
                retryable = any(term in error_str for term in 
                              ['rate', 'timeout', 'connection', 'temporary', 'network'])
                
                if retryable and retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"Retrying generation in {retry_delay * (2 ** retry_count)} seconds...")
                    await asyncio.sleep(retry_delay * (2 ** retry_count))
                    continue
                else:
                    return False, f"Error generating book after {retry_count + 1} attempts: {str(last_error)}"
            finally:
                self.generation_active = False
                self.current_generation = None
        
        # Should not reach here, but just in case
        return False, f"Maximum retries ({max_retries}) exceeded. Last error: {str(last_error)}"
    
    def stop_generation(self) -> Tuple[bool, str]:
        """Stop the current book generation."""
        if not self.generation_active or not self.current_generation:
            return False, "No generation in progress"
        
        try:
            self.current_generation.stop()
            self.generation_active = False
            self.current_generation = None
            return True, "Generation stopped"
        except Exception as e:
            logger.error(f"Error stopping generation: {e}")
            return False, f"Error stopping generation: {str(e)}"
    
    def get_generation_progress(self) -> Dict[str, Any]:
        """Get current generation progress."""
        if not self.generation_active or not self.current_generation:
            return {
                "active": False,
                "progress": 0,
                "message": "No generation in progress",
                "logs": self.event_logs[-20:]  # Last 20 log entries
            }
        
        return {
            "active": True,
            "progress": self.current_generation.get_progress(),
            "message": self.current_generation.get_status(),
            "logs": self.event_logs[-20:]
        }
    
    # ===== Character Management =====
    
    def list_characters(self, project_id: str) -> List[Dict[str, Any]]:
        """Get list of characters for a project."""
        try:
            db_path = self._get_character_db_path(project_id)
            if not db_path.exists():
                return []
            
            db = CharacterDatabase(db_path)
            characters = db.get_all_characters()
            return characters
        except Exception as e:
            logger.error(f"Error listing characters: {e}")
            return []
    
    def create_character(
        self,
        project_id: str,
        name: str,
        role: str,
        traits: Dict[str, float],
        description: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Create a new character."""
        try:
            from character_tracker import Character, CharacterRole, PersonalityTraits
            
            db_path = self._get_character_db_path(project_id)
            db = CharacterDatabase(db_path)
            
            # Convert role string to CharacterRole enum
            role_map = {
                "protagonist": CharacterRole.PROTAGONIST,
                "antagonist": CharacterRole.ANTAGONIST,
                "supporting": CharacterRole.SUPPORTING,
                "minor": CharacterRole.MINOR
            }
            char_role = role_map.get(role.lower(), CharacterRole.MINOR)
            
            # Create PersonalityTraits from OCEAN scores
            # Convert OCEAN scores to trait descriptions
            trait_descriptions = []
            if traits.get('openness', 50) > 70:
                trait_descriptions.append("Creative and open-minded")
            if traits.get('conscientiousness', 50) > 70:
                trait_descriptions.append("Organized and dependable")
            if traits.get('extraversion', 50) > 70:
                trait_descriptions.append("Outgoing and energetic")
            if traits.get('agreeableness', 50) > 70:
                trait_descriptions.append("Cooperative and trusting")
            if traits.get('neuroticism', 50) > 70:
                trait_descriptions.append("Emotionally reactive")
            
            personality = PersonalityTraits(
                traits=trait_descriptions if trait_descriptions else ["Balanced personality"]
            )
            
            # Create Character object
            character = Character(
                name=name,
                role=char_role,
                personality=personality,
                background=description
            )
            
            db.add_character(character)
            
            # Return character as dict for UI
            return True, f"Character '{name}' created", {
                "name": character.name,
                "role": character.role.value,
                "description": character.background,
                "personality_traits": traits
            }
            
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            return False, f"Error creating character: {str(e)}", {}
    
    def update_character(
        self,
        project_id: str,
        character_id: int,
        updates: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Update a character."""
        try:
            db_path = self._get_character_db_path(project_id)
            db = CharacterDatabase(db_path)
            
            db.update_character(character_id, updates)
            return True, "Character updated successfully"
            
        except Exception as e:
            logger.error(f"Error updating character: {e}")
            return False, f"Error updating character: {str(e)}"
    
    def delete_character(self, project_id: str, character_id: int) -> Tuple[bool, str]:
        """Delete a character."""
        try:
            db_path = self._get_character_db_path(project_id)
            db = CharacterDatabase(db_path)
            
            db.delete_character(character_id)
            return True, "Character deleted successfully"
            
        except Exception as e:
            logger.error(f"Error deleting character: {e}")
            return False, f"Error deleting character: {str(e)}"
    
    # ===== Style Management =====
    
    def list_styles(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available styles."""
        try:
            style_names = self.style_manager.list_styles(category)
            styles = []
            for name in style_names:
                style = self.style_manager.get_style(name)
                if style:
                    styles.append({
                        'name': style.name,
                        'category': style.category,
                        'description': f"{style.tone} writing with {style.vocabulary_level} vocabulary"
                    })
            return styles
        except Exception as e:
            logger.error(f"Error listing styles: {e}")
            return []
    
    def get_style_details(self, style_name: str) -> Dict[str, Any]:
        """Get details about a specific style."""
        try:
            style = self.style_manager.get_style(style_name)
            if style:
                return {
                    'name': style.name,
                    'category': style.category,
                    'description': f"{style.tone} writing with {style.vocabulary_level} vocabulary",
                    'tone': style.tone,
                    'vocabulary_level': style.vocabulary_level,
                    'pacing': style.paragraph_length,  # Use paragraph_length as pacing
                    'example': style.features[0] if style.features else "No example available"
                }
            return self.style_manager.get_style_info(style_name)
        except Exception as e:
            logger.error(f"Error getting style details: {e}")
            return {}
    
    def create_custom_style(
        self,
        name: str,
        base_style: str,
        modifications: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Create a custom style."""
        try:
            self.style_manager.create_custom_style(name, base_style, modifications)
            return True, f"Custom style '{name}' created successfully"
        except Exception as e:
            logger.error(f"Error creating custom style: {e}")
            return False, f"Error creating style: {str(e)}"
    
    # ===== Export Functions =====
    
    def export_book(
        self,
        project_id: str,
        format: str,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export a book to specified format.
        
        Returns:
            Tuple of (success, message, file_path)
        """
        try:
            # Get book data
            book_path = Path(f"projects/{project_id}/content/book.json")
            if not book_path.exists():
                return False, "Book not found. Generate the book first.", None
            
            with open(book_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Export to specified format
            export_path = self.book_exporter.export(book_data, format, metadata)
            
            return True, f"Book exported successfully", str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting book: {e}")
            return False, f"Error exporting: {str(e)}", None

    # Batch operations
    
    async def batch_export_books(self, project_ids: List[str], formats: List[str], 
                                 progress_callback=None) -> Dict[str, Any]:
        """Export multiple books in batch with progress tracking.
        
        Args:
            project_ids: List of project IDs to export
            formats: List of formats to export to (epub, pdf, docx, html)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with results for each project
        """
        results = {}
        total = len(project_ids) * len(formats)
        completed = 0
        
        with self._batch_lock:
            self.batch_processing = True
            
            try:
                for project_id in project_ids:
                    results[project_id] = {"success": [], "failed": []}
                    
                    # Load project
                    try:
                        project = self.project_manager.get_project(project_id)
                        if not project:
                            results[project_id]["failed"] = formats
                            logger.error(f"Project {project_id} not found")
                            continue
                    except Exception as e:
                        results[project_id]["failed"] = formats
                        logger.error(f"Error loading project {project_id}: {e}")
                        continue
                    
                    # Export to each format
                    for format_type in formats:
                        try:
                            export_path = await self._export_single_book(
                                project_id, project, format_type
                            )
                            results[project_id]["success"].append({
                                "format": format_type,
                                "path": export_path
                            })
                        except Exception as e:
                            results[project_id]["failed"].append({
                                "format": format_type,
                                "error": str(e)
                            })
                            logger.error(f"Failed to export {project_id} as {format_type}: {e}")
                        
                        completed += 1
                        if progress_callback:
                            progress_callback(completed / total * 100, 
                                            f"Exported {completed}/{total}")
                
                return {
                    "results": results,
                    "summary": {
                        "total_projects": len(project_ids),
                        "total_formats": len(formats),
                        "successful_exports": sum(
                            len(r["success"]) for r in results.values()
                        ),
                        "failed_exports": sum(
                            len(r["failed"]) for r in results.values()
                        )
                    }
                }
            finally:
                self.batch_processing = False
    
    async def _export_single_book(self, project_id: str, project: Any, 
                                  format_type: str) -> str:
        """Export a single book to a specific format."""
        # Load book content
        book_path = f"projects/{project_id}/content/book.json"
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"Book content not found for {project_id}")
        
        with open(book_path, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        
        # Export using book exporter
        export_path = self.book_exporter.export(
            book_data, 
            format_type,
            output_dir=f"projects/{project_id}/exports"
        )
        
        return export_path
    
    def batch_import_characters(self, source_project_id: str, 
                               target_project_ids: List[str],
                               character_names: Optional[List[str]] = None) -> Dict:
        """Import characters from one project to multiple projects.
        
        Args:
            source_project_id: Project to copy characters from
            target_project_ids: Projects to copy characters to
            character_names: Optional list of specific characters to import
            
        Returns:
            Dict with import results for each target project
        """
        results = {}
        
        # Load source characters
        source_db_path = self._get_character_db_path(source_project_id)
        if not os.path.exists(source_db_path):
            return {"error": f"No characters found in source project {source_project_id}"}
        
        source_db = CharacterDatabase(source_db_path)
        
        # Get characters to import
        if character_names:
            characters = [source_db.get_character(name) for name in character_names 
                         if source_db.get_character(name)]
        else:
            characters = source_db.list_characters()
        
        if not characters:
            return {"error": "No characters to import"}
        
        # Import to each target project
        for target_id in target_project_ids:
            results[target_id] = {"imported": [], "failed": []}
            
            target_db_path = self._get_character_db_path(target_id)
            target_db = CharacterDatabase(target_db_path)
            
            for character in characters:
                try:
                    # Check if character already exists
                    existing = target_db.get_character(character['name'])
                    if existing:
                        # Update instead of create
                        target_db.update_character(character['name'], character)
                    else:
                        target_db.add_character(character)
                    
                    results[target_id]["imported"].append(character['name'])
                except Exception as e:
                    results[target_id]["failed"].append({
                        "name": character['name'],
                        "error": str(e)
                    })
                    logger.error(f"Failed to import {character['name']} to {target_id}: {e}")
        
        return {
            "results": results,
            "summary": {
                "source_project": source_project_id,
                "characters_count": len(characters),
                "target_projects": len(target_project_ids),
                "total_imported": sum(len(r["imported"]) for r in results.values()),
                "total_failed": sum(len(r["failed"]) for r in results.values())
            }
        }
    
    def batch_delete_projects(self, project_ids: List[str], 
                             skip_archived: bool = True) -> Dict:
        """Delete multiple projects in batch.
        
        Args:
            project_ids: List of project IDs to delete
            skip_archived: Skip deletion of archived projects
            
        Returns:
            Dict with deletion results
        """
        results = {"deleted": [], "skipped": [], "failed": []}
        
        for project_id in project_ids:
            try:
                project = self.project_manager.get_project(project_id)
                
                if not project:
                    results["failed"].append({
                        "id": project_id,
                        "error": "Project not found"
                    })
                    continue
                
                # Check if archived and should skip
                if skip_archived and project.get('status') == 'archived':
                    results["skipped"].append(project_id)
                    continue
                
                # Delete project
                self.project_manager.delete_project(project_id, confirm=True)
                results["deleted"].append(project_id)
                
                # Clear cache for this project
                if project_id in self._project_cache:
                    del self._project_cache[project_id]
                
            except Exception as e:
                results["failed"].append({
                    "id": project_id,
                    "error": str(e)
                })
                logger.error(f"Failed to delete project {project_id}: {e}")
        
        return {
            "results": results,
            "summary": {
                "total": len(project_ids),
                "deleted": len(results["deleted"]),
                "skipped": len(results["skipped"]),
                "failed": len(results["failed"])
            }
        }
    
    # ===== Analytics =====
    
    def get_book_statistics(self, project_id: str) -> Dict[str, Any]:
        """Get statistics about a generated book."""
        try:
            book_path = Path(f"projects/{project_id}/content/book.json")
            if not book_path.exists():
                return {"error": "Book not found"}
            
            with open(book_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Calculate statistics
            total_words = 0
            total_chapters = len(book_data.get("chapters", []))
            chapter_lengths = []
            
            for chapter in book_data.get("chapters", []):
                words = len(chapter.get("content", "").split())
                total_words += words
                chapter_lengths.append(words)
            
            return {
                "title": book_data.get("title", "Unknown"),
                "total_chapters": total_chapters,
                "total_words": total_words,
                "average_chapter_length": total_words // total_chapters if total_chapters > 0 else 0,
                "shortest_chapter": min(chapter_lengths) if chapter_lengths else 0,
                "longest_chapter": max(chapter_lengths) if chapter_lengths else 0,
                "language": book_data.get("language", "Unknown"),
                "style": book_data.get("style", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting book statistics: {e}")
            return {"error": str(e)}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics for monitoring dashboard.
        
        Returns:
            Dict containing performance metrics and system stats
        """
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent(interval=0.1)
            
            # Get performance monitor summary
            perf_summary = {}
            if hasattr(self, 'performance_monitor'):
                perf_summary = self.performance_monitor.get_summary()
            
            # Get cache statistics
            cache_stats = {
                'project_cache_size': len(self._project_cache),
                'style_cache_size': len(self._style_cache),
                'character_cache_size': len(self._character_cache),
                'total_cache_items': len(self._project_cache) + len(self._style_cache) + len(self._character_cache)
            }
            
            # Get generation statistics
            generation_stats = {
                'active': self.generation_active,
                'current_project': self.current_generation.project_id if self.current_generation else None,
                'event_log_size': len(self.event_logs)
            }
            
            # Calculate health score (0-100)
            health_score = 100
            if cpu_percent > 90:
                health_score -= 20
            if memory.percent > 90:
                health_score -= 20
            if disk.percent > 95:
                health_score -= 10
            if process_memory > 1000:  # Over 1GB
                health_score -= 10
            
            return {
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / 1024 / 1024,
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / 1024 / 1024 / 1024
                },
                'process': {
                    'memory_mb': process_memory,
                    'cpu_percent': process_cpu,
                    'threads': process.num_threads()
                },
                'performance': perf_summary,
                'cache': cache_stats,
                'generation': generation_stats,
                'health': {
                    'score': health_score,
                    'status': 'healthy' if health_score > 70 else 'warning' if health_score > 40 else 'critical',
                    'warnings': self._get_performance_warnings(cpu_percent, memory.percent, process_memory)
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                'error': str(e),
                'health': {
                    'score': 0,
                    'status': 'error'
                }
            }
    
    def _get_performance_warnings(self, cpu: float, memory: float, process_mem: float) -> List[str]:
        """Get performance warnings based on current metrics."""
        warnings = []
        
        if cpu > 90:
            warnings.append(f"High CPU usage: {cpu:.1f}%")
        if memory > 90:
            warnings.append(f"High memory usage: {memory:.1f}%")
        if process_mem > 1000:
            warnings.append(f"Process using {process_mem:.0f}MB memory")
        if len(self._project_cache) > 50:
            warnings.append(f"Large project cache: {len(self._project_cache)} items")
        
        return warnings
    
    def get_provider_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance metrics across different LLM providers.
        
        Returns:
            Dict with provider comparison data
        """
        try:
            # Get metrics from performance monitor
            if not hasattr(self, 'performance_monitor'):
                return {'error': 'Performance monitor not initialized'}
            
            metrics = self.performance_monitor.get_metrics()
            
            # Group by provider
            provider_stats = {}
            for metric in metrics:
                if 'provider' in metric.get('operation', ''):
                    provider = metric['operation'].split('_')[0]
                    if provider not in provider_stats:
                        provider_stats[provider] = {
                            'calls': 0,
                            'total_duration': 0,
                            'avg_duration': 0,
                            'errors': 0,
                            'success_rate': 0
                        }
                    
                    provider_stats[provider]['calls'] += 1
                    provider_stats[provider]['total_duration'] += metric['duration']
            
            # Calculate averages and rates
            for provider in provider_stats:
                stats = provider_stats[provider]
                if stats['calls'] > 0:
                    stats['avg_duration'] = stats['total_duration'] / stats['calls']
                    stats['success_rate'] = ((stats['calls'] - stats['errors']) / stats['calls']) * 100
            
            # Add cost estimates
            cost_per_1k_tokens = {
                'openai': 0.01,
                'anthropic': 0.008,
                'gemini': 0.005,
                'cohere': 0.004,
                'openrouter': 0.006
            }
            
            for provider in provider_stats:
                provider_key = provider.lower()
                if provider_key in cost_per_1k_tokens:
                    # Estimate based on average tokens per call
                    avg_tokens = 2000  # Rough estimate
                    calls = provider_stats[provider]['calls']
                    estimated_cost = (calls * avg_tokens / 1000) * cost_per_1k_tokens[provider_key]
                    provider_stats[provider]['estimated_cost'] = round(estimated_cost, 2)
            
            return {
                'providers': provider_stats,
                'best_performance': min(provider_stats.items(), 
                                       key=lambda x: x[1].get('avg_duration', float('inf')))[0]
                                       if provider_stats else None,
                'most_reliable': max(provider_stats.items(),
                                    key=lambda x: x[1].get('success_rate', 0))[0]
                                    if provider_stats else None
            }
        except Exception as e:
            logger.error(f"Error comparing provider performance: {e}")
            return {'error': str(e)}
    
    def clear_cache(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """Clear cache to free memory.
        
        Args:
            cache_type: Optional specific cache to clear ('project', 'style', 'character')
                       If None, clears all caches
        
        Returns:
            Dict with number of items cleared from each cache
        """
        cleared = {}
        
        if cache_type in [None, 'project']:
            cleared['project'] = len(self._project_cache)
            self._project_cache.clear()
        
        if cache_type in [None, 'style']:
            cleared['style'] = len(self._style_cache)
            self._style_cache.clear()
        
        if cache_type in [None, 'character']:
            cleared['character'] = len(self._character_cache)
            self._character_cache.clear()
        
        # Clear timestamps
        if cache_type is None:
            self._cache_timestamps.clear()
        
        # Clear any cached functions
        if hasattr(self.list_projects, 'clear_cache'):
            self.list_projects.clear_cache()
        
        # Log the action
        total_cleared = sum(cleared.values())
        gradio_logger.info(
            f"Cleared {total_cleared} cached items",
            ui_message=f"Cache cleared: {total_cleared} items removed"
        )
        
        # Emit UI event
        if hasattr(self, 'container') and hasattr(self.container, 'event_manager'):
            event_manager = self.container.event_manager()
            event_manager.emit(Event(
                UIEventType.CACHE_CLEARED,
                {'cleared': cleared, 'total': total_cleared}
            ))
        
        return cleared
    
    # ===== Helper Methods =====
    
    def _format_project_info(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Format project information for display."""
        # Handle both dict and object forms
        if isinstance(project, dict):
            return {
                "id": project.get("project_id", project.get("id", "")),
                "title": project.get("title", "Unknown"),
                "language": project.get("language", "English"),
                "style": project.get("style", "general"),
                "status": project.get("status", project.get("metadata", {}).get("status", "draft")),
                "created_at": project.get("created_at", project.get("metadata", {}).get("created_at", "")),
                "chapters": project.get("chapter_count", project.get("metadata", {}).get("chapters", 0)),
                "instructions": project.get("metadata", {}).get("instructions", "") if "metadata" in project else ""
            }
        else:
            # Handle object directly
            return {
                "id": getattr(project, "project_id", ""),
                "title": getattr(project, "title", "Unknown"),
                "language": getattr(project, "language", "English"),
                "style": getattr(project, "style", "general"),
                "status": getattr(project, "status", "draft"),
                "created_at": getattr(project, "created_at", ""),
                "chapters": getattr(project, "chapter_count", 0),
                "instructions": ""
            }
    
    def _get_character_db_path(self, project_id: str) -> Path:
        """Get path to character database for a project."""
        project_dir = self.project_manager.get_project_dir(project_id)
        return project_dir / "characters.db"
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        from app_config import settings
        
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"
        }
        
        key_name = key_mapping.get(provider.lower())
        if key_name:
            return getattr(settings, key_name, None)
        return None


# Global instance for easy access
handlers = GradioHandlers()