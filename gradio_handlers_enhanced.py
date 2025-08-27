"""
Enhanced Gradio Handlers with Security and Performance Optimizations
Integrates secure API key storage, path validation, rate limiting, and performance caching
"""

import logging
import json
import asyncio
import os
import time
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import traceback
from concurrent.futures import ThreadPoolExecutor, Future

# Import existing modules
from containers import get_container
from project_manager import ProjectManager
from style_templates import StyleManager
from character_tracker import CharacterDatabase
from events import event_manager, Event, EventType
from book_generator import BookGenerator
from export_formats import BookExporter
from providers.factory import ProviderFactory

# Import new security and performance modules
from security_manager import (
    SecureKeyStorage,
    PathValidator,
    RateLimiter,
    SecurityError,
    PathTraversalError,
    RateLimitError,
    get_secure_storage,
    get_rate_limiter,
    rate_limit
)
from performance_optimizer import (
    EnhancedCache,
    StreamProcessor,
    TaskOptimizer,
    cached_with_ttl,
    get_cache,
    get_stream_processor,
    get_task_optimizer
)
from background_tasks import BackgroundTaskManager, TaskStatus

logger = logging.getLogger(__name__)


class EnhancedGradioHandlers:
    """
    Enhanced Gradio handlers with security and performance optimizations
    """
    
    def __init__(self):
        """Initialize enhanced handlers with security and performance features"""
        # Core components
        self.project_manager = ProjectManager()
        self.style_manager = StyleManager()
        self.character_db = None
        self.book_generator = None
        self.book_exporter = BookExporter()
        
        # Security components
        self.secure_storage = get_secure_storage()
        self.rate_limiter = get_rate_limiter()
        
        # Performance components
        self.cache = get_cache()
        self.stream_processor = get_stream_processor()
        self.task_optimizer = get_task_optimizer()
        self.background_tasks = BackgroundTaskManager(backend='thread')
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.generation_tasks: Dict[str, Future] = {}
        
        # Initialize character database
        self._init_character_db()
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Event loop for async operations
        self._loop = None
        self._ensure_event_loop()
        
        logger.info("Enhanced Gradio handlers initialized with security and performance features")
    
    def _ensure_event_loop(self):
        """Ensure an event loop exists for async operations"""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
    
    def _init_character_db(self):
        """Initialize character database with proper path"""
        try:
            db_path = Path("characters.db")
            self.character_db = CharacterDatabase(str(db_path))
            logger.info(f"Character database initialized at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize character database: {e}")
            self.character_db = None
    
    # ================== Security Methods ==================
    
    def store_api_key_securely(
        self,
        provider: str,
        api_key: str,
        session_id: str
    ) -> Tuple[bool, str]:
        """
        Securely store an API key with encryption
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            api_key: The API key to store
            session_id: Session identifier for rate limiting
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate provider name (basic path validation)
            PathValidator.validate_project_id(provider)
            
            # Check rate limit
            self.rate_limiter.check_rate_limit(
                "api_key_storage",
                session_id,
                max_requests=10,
                time_window=60
            )
            
            # Store encrypted key
            self.secure_storage.store_api_key(provider, api_key)
            
            # Clear provider cache to force re-initialization
            self.cache.invalidate_pattern(f"provider:{provider}:*")
            
            logger.info(f"API key for {provider} stored securely")
            return True, f"API key for {provider} stored securely"
            
        except PathTraversalError as e:
            logger.error(f"Invalid provider name: {e}")
            return False, f"Invalid provider name: {e}"
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False, f"Failed to store API key: {e}"
    
    def get_api_key_secure(self, provider: str) -> Optional[str]:
        """
        Retrieve a decrypted API key for a provider
        
        Args:
            provider: Provider name
            
        Returns:
            Decrypted API key or None
        """
        try:
            return self.secure_storage.get_api_key(provider)
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            return None
    
    def validate_project_path(self, project_id: str) -> Tuple[bool, str]:
        """
        Validate a project ID for path safety
        
        Args:
            project_id: Project identifier
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            PathValidator.validate_project_id(project_id)
            return True, ""
        except PathTraversalError as e:
            return False, str(e)
    
    # ================== Performance Methods ==================
    
    @cached_with_ttl(ttl=300)  # Cache for 5 minutes
    def get_project_list(self) -> List[Dict[str, Any]]:
        """
        Get cached list of projects
        
        Returns:
            List of project dictionaries
        """
        projects = []
        for project_id in self.project_manager.list_projects():
            project = self.project_manager.load_project(project_id)
            if project:
                projects.append({
                    'id': project_id,
                    'title': project.book_title,
                    'created_at': project.created_at.isoformat(),
                    'chapters': len(project.chapters)
                })
        return projects
    
    async def stream_book_generation(
        self,
        project_id: str,
        session_id: str,
        **generation_params
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream book generation progress
        
        Args:
            project_id: Project identifier
            session_id: Session identifier
            **generation_params: Generation parameters
            
        Yields:
            Progress updates
        """
        try:
            # Validate project ID
            is_valid, error_msg = self.validate_project_path(project_id)
            if not is_valid:
                yield {
                    'type': 'error',
                    'message': f"Invalid project ID: {error_msg}"
                }
                return
            
            # Check rate limit
            self.rate_limiter.check_rate_limit(
                "generation",
                session_id,
                max_requests=5,
                time_window=60
            )
            
            # Create progress queue
            progress_queue = asyncio.Queue()
            
            # Submit generation task
            task_id = await self._submit_generation_task(
                project_id,
                generation_params,
                progress_queue
            )
            
            # Stream progress updates
            async for update in self.stream_processor.stream_generation_progress(
                task_id,
                progress_queue
            ):
                yield update
                
        except RateLimitError as e:
            yield {
                'type': 'error',
                'message': str(e)
            }
        except Exception as e:
            logger.error(f"Generation streaming failed: {e}")
            yield {
                'type': 'error',
                'message': f"Generation failed: {e}"
            }
    
    async def _submit_generation_task(
        self,
        project_id: str,
        params: Dict[str, Any],
        progress_queue: asyncio.Queue
    ) -> str:
        """
        Submit a book generation task
        
        Args:
            project_id: Project identifier
            params: Generation parameters
            progress_queue: Queue for progress updates
            
        Returns:
            Task ID
        """
        def generation_worker():
            try:
                # Initialize book generator with secure API keys
                provider_name = params.get('provider', 'openai')
                api_key = self.get_api_key_secure(provider_name)
                
                if not api_key:
                    asyncio.run(progress_queue.put({
                        'type': 'error',
                        'message': f"No API key found for {provider_name}"
                    }))
                    return None
                
                # Create provider with secure key
                provider_factory = ProviderFactory()
                provider = provider_factory.create_provider(
                    provider_name,
                    api_key=api_key,
                    **params.get('provider_config', {})
                )
                
                # Initialize book generator
                self.book_generator = BookGenerator(
                    provider=provider,
                    **params.get('generator_config', {})
                )
                
                # Generate book with progress updates
                project = self.project_manager.load_project(project_id)
                
                for i in range(project.total_chapters):
                    # Generate chapter
                    chapter = self.book_generator.generate_chapter(
                        chapter_number=i + 1,
                        previous_content=project.get_previous_content(i)
                    )
                    
                    # Update project
                    project.add_chapter(chapter)
                    self.project_manager.save_project(project)
                    
                    # Send progress update
                    progress = ((i + 1) / project.total_chapters) * 100
                    asyncio.run(progress_queue.put({
                        'type': 'progress',
                        'progress': progress,
                        'chapter': i + 1,
                        'total_chapters': project.total_chapters,
                        'message': f"Generated chapter {i + 1}/{project.total_chapters}"
                    }))
                
                # Send completion
                asyncio.run(progress_queue.put({
                    'type': 'complete',
                    'message': 'Book generation completed successfully'
                }))
                
                return project
                
            except Exception as e:
                logger.error(f"Generation worker failed: {e}")
                asyncio.run(progress_queue.put({
                    'type': 'error',
                    'message': str(e)
                }))
                return None
            finally:
                # Send sentinel to close stream
                asyncio.run(progress_queue.put(None))
        
        # Submit task to optimizer
        future = self.task_optimizer.submit_task(generation_worker)
        task_id = str(id(future))
        self.generation_tasks[task_id] = future
        
        return task_id
    
    @rate_limit("export", max_requests=20, time_window=60)
    async def export_book_streaming(
        self,
        project_id: str,
        format: str,
        session_id: str
    ) -> AsyncIterator[bytes]:
        """
        Stream book export in chunks
        
        Args:
            project_id: Project identifier
            format: Export format
            session_id: Session identifier
            
        Yields:
            Export data chunks
        """
        try:
            # Validate project ID
            is_valid, error_msg = self.validate_project_path(project_id)
            if not is_valid:
                raise ValueError(f"Invalid project ID: {error_msg}")
            
            # Load project
            project = self.project_manager.load_project(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Get full book content
            book_content = project.get_full_book()
            
            # Stream export
            async for chunk in self.stream_processor.stream_book_export(
                book_content,
                format=format
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Export streaming failed: {e}")
            raise
    
    def batch_import_characters(
        self,
        characters_data: List[Dict[str, Any]],
        session_id: str
    ) -> Tuple[bool, str]:
        """
        Import multiple characters in batch
        
        Args:
            characters_data: List of character data
            session_id: Session identifier
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check rate limit
            self.rate_limiter.check_rate_limit(
                "character_creation",
                session_id,
                max_requests=50,
                time_window=60
            )
            
            def batch_processor(batch):
                results = []
                for char_data in batch:
                    try:
                        # Validate character name
                        name = PathValidator.sanitize_filename(char_data.get('name', ''))
                        char_data['name'] = name
                        
                        # Add to database
                        self.character_db.add_character(char_data)
                        results.append({'name': name, 'success': True})
                    except Exception as e:
                        results.append({
                            'name': char_data.get('name', 'Unknown'),
                            'success': False,
                            'error': str(e)
                        })
                return results
            
            # Process in batches
            future = self.task_optimizer.batch_operation(
                "character_import",
                characters_data,
                batch_processor
            )
            
            # Force processing if needed
            if not future.done():
                future = self.task_optimizer.force_batch_processing(
                    "character_import",
                    batch_processor
                )
            
            if future:
                results = future.result(timeout=10)
                successful = sum(1 for r in results if r['success'])
                return True, f"Imported {successful}/{len(characters_data)} characters"
            
            return False, "Batch processing failed"
            
        except RateLimitError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Batch import failed: {e}")
            return False, f"Import failed: {e}"
    
    # ================== Cache Management ==================
    
    def clear_project_cache(self, project_id: str = None) -> bool:
        """
        Clear cache for a specific project or all projects
        
        Args:
            project_id: Optional project ID to clear
            
        Returns:
            Success status
        """
        try:
            if project_id:
                # Clear specific project cache
                count = self.cache.invalidate_pattern(f"project:{project_id}:*")
                logger.info(f"Cleared {count} cache entries for project {project_id}")
            else:
                # Clear all project caches
                count = self.cache.invalidate_pattern("project:*")
                logger.info(f"Cleared {count} project cache entries")
            
            # Also clear the project list cache
            self.get_project_list.invalidate_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = {
            'cache': self.cache.get_statistics(),
            'tasks': self.task_optimizer.get_statistics(),
            'rate_limits': self.rate_limiter.get_statistics(),
            'active_streams': len(self.stream_processor.active_streams),
            'active_generations': len(self.generation_tasks),
            'sessions': len(self.active_sessions)
        }
        
        # Add cache hit rate
        cache_stats = metrics['cache']
        if cache_stats['hits'] + cache_stats['misses'] > 0:
            hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses'])
            metrics['cache_hit_rate_percent'] = round(hit_rate * 100, 2)
        
        return metrics
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get security status and statistics
        
        Returns:
            Dictionary with security information
        """
        stored_providers = self.secure_storage.list_providers()
        rate_limit_stats = self.rate_limiter.get_statistics()
        
        return {
            'stored_api_keys': len(stored_providers),
            'providers': [p['provider'] for p in stored_providers],
            'rate_limiters_active': rate_limit_stats['active_limiters'],
            'total_requests': rate_limit_stats['total_requests'],
            'encryption_enabled': True,
            'path_validation_enabled': True
        }
    
    # ================== Session Management ==================
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Create a new session with security context
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session information
        """
        session = {
            'id': session_id,
            'created_at': datetime.now().isoformat(),
            'rate_limits': {},
            'active_tasks': [],
            'cache_keys': []
        }
        
        self.active_sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        
        return session
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up session resources
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Cancel active tasks
                for task_id in session.get('active_tasks', []):
                    if task_id in self.generation_tasks:
                        future = self.generation_tasks[task_id]
                        if not future.done():
                            future.cancel()
                        del self.generation_tasks[task_id]
                
                # Clear session cache entries
                for cache_key in session.get('cache_keys', []):
                    self.cache.invalidate(cache_key)
                
                # Reset rate limits for session
                for resource in ['generation', 'export', 'api_call']:
                    self.rate_limiter.reset_limit(resource, session_id)
                
                # Remove session
                del self.active_sessions[session_id]
                
                logger.info(f"Cleaned up session: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {e}")
            return False
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Shutdown executor
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True, cancel_futures=True)
            
            # Shutdown task optimizer
            if hasattr(self, 'task_optimizer'):
                self.task_optimizer.shutdown(wait=False)
            
            # Clean up all sessions
            for session_id in list(self.active_sessions.keys()):
                self.cleanup_session(session_id)
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")