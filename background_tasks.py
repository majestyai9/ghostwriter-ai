"""
Background task processing with Celery and RQ support
"""
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

# Try to import task queue libraries
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    from redis import Redis
    from rq import Connection, Queue, Worker
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False

class TaskStatus(Enum):
    """Task status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskResult:
    """Task result wrapper"""

    def __init__(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.progress = 0
        self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'progress': self.progress,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# Celery configuration and tasks
if CELERY_AVAILABLE:
    # Initialize Celery
    celery_app = Celery(
        'ghostwriter',
        broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    )

    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour
        task_soft_time_limit=3300,  # 55 minutes
    )

    @celery_app.task(bind=True, name='generate_book_async')
    def generate_book_celery(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate book asynchronously with Celery"""
        from generate import write_book

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting generation'})

        book = book_data.get('book', {})
        title = book_data['title']
        instructions = book_data['instructions']
        language = book_data['language']

        try:
            # Track progress
            total_chapters = len(book.get('toc', {}).get('chapters', []))
            current_chapter = 0

            for updated_book in write_book(book, title, instructions, language):
                # Update progress
                if 'toc' in updated_book:
                    completed_chapters = sum(
                        1 for ch in updated_book['toc']['chapters']
                        if ch.get('content')
                    )
                    if completed_chapters > current_chapter:
                        current_chapter = completed_chapters
                        progress = int((current_chapter / total_chapters) * 100) if total_chapters > 0 else 0

                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'progress': progress,
                                'status': f'Generated chapter {current_chapter}/{total_chapters}',
                                'current_chapter': current_chapter,
                                'total_chapters': total_chapters
                            }
                        )

                book = updated_book

            return {
                'status': 'completed',
                'book': book,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'status': 'Generation failed'}
            )
            raise

    @celery_app.task(bind=True, name='generate_chapter_async')
    def generate_chapter_celery(self, chapter_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single chapter asynchronously"""
        from generate import _write_chapter

        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting chapter generation'})

        try:
            book = chapter_data['book']
            chapter = chapter_data['chapter']
            history = chapter_data.get('history', [])

            for updated_book in _write_chapter(book, chapter, history):
                self.update_state(
                    state='PROGRESS',
                    meta={'progress': 50, 'status': 'Generating content'}
                )
                book = updated_book

            return {
                'status': 'completed',
                'chapter': chapter,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'status': 'Chapter generation failed'}
            )
            raise

# RQ configuration and tasks
if RQ_AVAILABLE:
    # Initialize Redis connection
    redis_conn = Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0))
    )

    # Create queue
    task_queue = Queue('ghostwriter', connection=redis_conn)

    def generate_book_rq(book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate book with RQ"""
        from rq import get_current_job

        from generate import write_book

        job = get_current_job()

        book = book_data.get('book', {})
        title = book_data['title']
        instructions = book_data['instructions']
        language = book_data['language']

        try:
            # Track progress
            total_chapters = len(book.get('toc', {}).get('chapters', []))
            current_chapter = 0

            for updated_book in write_book(book, title, instructions, language):
                # Update progress in job meta
                if 'toc' in updated_book:
                    completed_chapters = sum(
                        1 for ch in updated_book['toc']['chapters']
                        if ch.get('content')
                    )
                    if completed_chapters > current_chapter:
                        current_chapter = completed_chapters
                        progress = int((current_chapter / total_chapters) * 100) if total_chapters > 0 else 0

                        job.meta['progress'] = progress
                        job.meta['status'] = f'Generated chapter {current_chapter}/{total_chapters}'
                        job.save_meta()

                book = updated_book

            return {
                'status': 'completed',
                'book': book,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            job.meta['error'] = str(e)
            job.meta['status'] = 'Generation failed'
            job.save_meta()
            raise

class BackgroundTaskManager:
    """Unified interface for background task management"""

    def __init__(self, backend: str = 'celery'):
        """
        Initialize task manager
        
        Args:
            backend: Task backend ('celery', 'rq', 'thread')
        """
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        if backend == 'celery' and not CELERY_AVAILABLE:
            self.logger.warning("Celery not available, falling back to thread backend")
            self.backend = 'thread'
        elif backend == 'rq' and not RQ_AVAILABLE:
            self.logger.warning("RQ not available, falling back to thread backend")
            self.backend = 'thread'

        # Store for thread-based tasks
        self.tasks = {}

    def submit_task(self,
                   task_name: str,
                   task_data: Dict[str, Any],
                   callback: Optional[Callable] = None) -> str:
        """
        Submit a task for background processing
        
        Args:
            task_name: Name of the task
            task_data: Task data/parameters
            callback: Optional callback for completion
            
        Returns:
            Task ID
        """
        if self.backend == 'celery':
            return self._submit_celery_task(task_name, task_data, callback)
        elif self.backend == 'rq':
            return self._submit_rq_task(task_name, task_data, callback)
        else:
            return self._submit_thread_task(task_name, task_data, callback)

    def _submit_celery_task(self, task_name: str, task_data: Dict[str, Any], callback: Optional[Callable]) -> str:
        """Submit task using Celery"""
        if task_name == 'generate_book':
            task = generate_book_celery.delay(task_data)
        elif task_name == 'generate_chapter':
            task = generate_chapter_celery.delay(task_data)
        else:
            raise ValueError(f"Unknown task: {task_name}")

        self.logger.info(f"Submitted Celery task {task.id}")

        if callback:
            # Store callback for later execution
            self.tasks[task.id] = {'callback': callback, 'celery_task': task}

        return task.id

    def _submit_rq_task(self, task_name: str, task_data: Dict[str, Any], callback: Optional[Callable]) -> str:
        """Submit task using RQ"""
        if task_name == 'generate_book':
            job = task_queue.enqueue(generate_book_rq, task_data, job_timeout='1h')
        else:
            raise ValueError(f"Unknown task: {task_name}")

        self.logger.info(f"Submitted RQ job {job.id}")

        if callback:
            self.tasks[job.id] = {'callback': callback, 'rq_job': job}

        return job.id

    def _submit_thread_task(self, task_name: str, task_data: Dict[str, Any], callback: Optional[Callable]) -> str:
        """Submit task using threading (fallback)"""
        import threading
        import uuid

        task_id = str(uuid.uuid4())

        def run_task():
            try:
                self.tasks[task_id]['status'] = TaskStatus.RUNNING

                if task_name == 'generate_book':
                    from generate import write_book
                    book = task_data.get('book', {})

                    for updated_book in write_book(
                        book,
                        task_data['title'],
                        task_data['instructions'],
                        task_data['language']
                    ):
                        book = updated_book
                        self.tasks[task_id]['progress'] = 50  # Rough estimate

                    result = {'status': 'completed', 'book': book}
                else:
                    raise ValueError(f"Unknown task: {task_name}")

                self.tasks[task_id]['status'] = TaskStatus.COMPLETED
                self.tasks[task_id]['result'] = result

                if callback:
                    callback(result)

            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                self.tasks[task_id]['status'] = TaskStatus.FAILED
                self.tasks[task_id]['error'] = str(e)

        self.tasks[task_id] = {
            'status': TaskStatus.PENDING,
            'progress': 0,
            'thread': threading.Thread(target=run_task)
        }

        self.tasks[task_id]['thread'].start()
        self.logger.info(f"Started thread task {task_id}")

        return task_id

    def get_task_status(self, task_id: str) -> TaskResult:
        """
        Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskResult object
        """
        if self.backend == 'celery':
            return self._get_celery_status(task_id)
        elif self.backend == 'rq':
            return self._get_rq_status(task_id)
        else:
            return self._get_thread_status(task_id)

    def _get_celery_status(self, task_id: str) -> TaskResult:
        """Get Celery task status"""
        from celery.result import AsyncResult

        task = AsyncResult(task_id, app=celery_app)

        if task.state == 'PENDING':
            status = TaskStatus.PENDING
        elif task.state == 'PROGRESS':
            status = TaskStatus.RUNNING
        elif task.state == 'SUCCESS':
            status = TaskStatus.COMPLETED
        elif task.state == 'FAILURE':
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.PENDING

        result = TaskResult(task_id, status)

        if task.info:
            if isinstance(task.info, dict):
                result.progress = task.info.get('progress', 0)
                result.metadata = task.info
            else:
                result.result = task.info

        return result

    def _get_rq_status(self, task_id: str) -> TaskResult:
        """Get RQ job status"""
        from rq.job import Job

        job = Job.fetch(task_id, connection=redis_conn)

        if job.is_queued:
            status = TaskStatus.PENDING
        elif job.is_started:
            status = TaskStatus.RUNNING
        elif job.is_finished:
            status = TaskStatus.COMPLETED
        elif job.is_failed:
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.PENDING

        result = TaskResult(task_id, status)

        if job.meta:
            result.progress = job.meta.get('progress', 0)
            result.metadata = job.meta

        if job.result:
            result.result = job.result

        return result

    def _get_thread_status(self, task_id: str) -> TaskResult:
        """Get thread task status"""
        if task_id not in self.tasks:
            return TaskResult(task_id, TaskStatus.PENDING)

        task = self.tasks[task_id]
        result = TaskResult(task_id, task.get('status', TaskStatus.PENDING))
        result.progress = task.get('progress', 0)
        result.result = task.get('result')
        result.error = task.get('error')

        return result

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled successfully
        """
        if self.backend == 'celery':
            from celery.result import AsyncResult
            task = AsyncResult(task_id, app=celery_app)
            task.revoke(terminate=True)
            return True
        elif self.backend == 'rq':
            from rq.job import Job
            job = Job.fetch(task_id, connection=redis_conn)
            job.cancel()
            return True
        else:
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = TaskStatus.CANCELLED
                return True
            return False

    def wait_for_task(self, task_id: str, timeout: int = None) -> TaskResult:
        """
        Wait for task completion
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            Final TaskResult
        """
        start_time = time.time()

        while True:
            result = self.get_task_status(task_id)

            if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return result

            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Task {task_id} timed out after {timeout}s")
                return result

            time.sleep(1)

# Global task manager
task_manager = None

def initialize_tasks(backend: str = 'celery'):
    """Initialize global task manager"""
    global task_manager
    task_manager = BackgroundTaskManager(backend)
    return task_manager

def get_task_manager() -> BackgroundTaskManager:
    """Get global task manager"""
    global task_manager
    if task_manager is None:
        # Auto-detect backend
        if CELERY_AVAILABLE:
            backend = 'celery'
        elif RQ_AVAILABLE:
            backend = 'rq'
        else:
            backend = 'thread'
        task_manager = BackgroundTaskManager(backend)
    return task_manager
