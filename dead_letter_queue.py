"""
Dead Letter Queue implementation for failed operations.
Captures, stores, and retries failed operations with exponential backoff.
"""

import logging
import json
import time
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from threading import Thread, Lock, Event
import pickle
from pathlib import Path

from tracing import trace_span, record_event

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be queued."""
    CHAPTER_GENERATION = "chapter_generation"
    SECTION_GENERATION = "section_generation"
    OUTLINE_GENERATION = "outline_generation"
    EXPORT_GENERATION = "export_generation"
    RAG_INDEXING = "rag_indexing"
    CACHE_OPERATION = "cache_operation"
    API_CALL = "api_call"
    VALIDATION = "validation"


class RetryPolicy(Enum):
    """Retry policies for failed operations."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class FailedOperation:
    """Represents a failed operation in the DLQ."""
    id: str
    operation_type: OperationType
    payload: Dict[str, Any]
    error_message: str
    error_type: str
    created_at: datetime = field(default_factory=datetime.now)
    last_retry_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, processing, completed, failed_permanently
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "operation_type": self.operation_type.value,
            "payload": self.payload,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "created_at": self.created_at.isoformat(),
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_policy": self.retry_policy.value,
            "metadata": self.metadata,
            "status": self.status
        }
    
    def can_retry(self) -> bool:
        """Check if operation can be retried."""
        return self.retry_count < self.max_retries and self.status != "failed_permanently"
    
    def get_next_retry_delay(self) -> float:
        """Calculate next retry delay based on policy."""
        if self.retry_policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            # 2^retry_count seconds with max of 300 seconds (5 minutes)
            return min(2 ** self.retry_count, 300)
        elif self.retry_policy == RetryPolicy.LINEAR_BACKOFF:
            # 10 seconds * retry_count with max of 300 seconds
            return min(10 * (self.retry_count + 1), 300)
        elif self.retry_policy == RetryPolicy.FIXED_DELAY:
            return 30  # Fixed 30 seconds
        else:  # IMMEDIATE
            return 0


class DeadLetterQueue:
    """
    Dead Letter Queue for managing failed operations.
    Provides persistent storage and automatic retry mechanisms.
    """
    
    def __init__(self, storage_path: str = "dlq_storage"):
        """
        Initialize the Dead Letter Queue.
        
        Args:
            storage_path: Path to store DLQ data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.queue: List[FailedOperation] = []
        self.queue_lock = Lock()
        
        self.handlers: Dict[OperationType, Callable] = {}
        self.processing = False
        self.stop_event = Event()
        self.processor_thread: Optional[Thread] = None
        
        # Load persisted queue
        self._load_queue()
        
        # Start processor thread
        self._start_processor()
    
    def _load_queue(self):
        """Load persisted queue from storage."""
        queue_file = self.storage_path / "queue.pkl"
        
        if queue_file.exists():
            try:
                with open(queue_file, "rb") as f:
                    self.queue = pickle.load(f)
                logger.info(f"Loaded {len(self.queue)} operations from DLQ storage")
            except Exception as e:
                logger.error(f"Failed to load DLQ storage: {e}")
                self.queue = []
    
    def _save_queue(self):
        """Persist queue to storage."""
        queue_file = self.storage_path / "queue.pkl"
        
        try:
            with self.queue_lock:
                with open(queue_file, "wb") as f:
                    pickle.dump(self.queue, f)
        except Exception as e:
            logger.error(f"Failed to save DLQ storage: {e}")
    
    def add_operation(
        self,
        operation_type: OperationType,
        payload: Dict[str, Any],
        error: Exception,
        max_retries: int = 3,
        retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a failed operation to the DLQ.
        
        Args:
            operation_type: Type of operation
            payload: Operation payload
            error: The exception that caused the failure
            max_retries: Maximum retry attempts
            retry_policy: Retry policy to use
            metadata: Additional metadata
            
        Returns:
            Operation ID
        """
        import uuid
        
        operation_id = str(uuid.uuid4())
        
        failed_op = FailedOperation(
            id=operation_id,
            operation_type=operation_type,
            payload=payload,
            error_message=str(error),
            error_type=type(error).__name__,
            max_retries=max_retries,
            retry_policy=retry_policy,
            metadata=metadata or {}
        )
        
        with self.queue_lock:
            self.queue.append(failed_op)
        
        self._save_queue()
        
        logger.info(f"Added operation {operation_id} to DLQ: {operation_type.value}")
        record_event("dlq.operation.added", {
            "operation_id": operation_id,
            "operation_type": operation_type.value,
            "error_type": failed_op.error_type
        })
        
        return operation_id
    
    def register_handler(self, operation_type: OperationType, handler: Callable):
        """
        Register a handler for an operation type.
        
        Args:
            operation_type: Type of operation
            handler: Handler function that takes the payload and returns success boolean
        """
        self.handlers[operation_type] = handler
        logger.info(f"Registered handler for {operation_type.value}")
    
    def _start_processor(self):
        """Start the background processor thread."""
        if not self.processing:
            self.processing = True
            self.processor_thread = Thread(target=self._process_queue, daemon=True)
            self.processor_thread.start()
            logger.info("DLQ processor started")
    
    def stop_processor(self):
        """Stop the background processor thread."""
        self.processing = False
        self.stop_event.set()
        
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        
        logger.info("DLQ processor stopped")
    
    def _process_queue(self):
        """Background thread to process the queue."""
        while self.processing:
            try:
                # Process operations that are due for retry
                self._process_pending_operations()
                
                # Sleep for a short interval
                self.stop_event.wait(timeout=10)
                
            except Exception as e:
                logger.error(f"Error in DLQ processor: {e}")
    
    def _process_pending_operations(self):
        """Process operations that are due for retry."""
        with self.queue_lock:
            pending_ops = [
                op for op in self.queue
                if op.status == "pending" and op.can_retry()
            ]
        
        for op in pending_ops:
            # Check if enough time has passed for retry
            if op.last_retry_at:
                delay = op.get_next_retry_delay()
                if (datetime.now() - op.last_retry_at).seconds < delay:
                    continue
            
            # Try to process the operation
            self._retry_operation(op)
    
    def _retry_operation(self, operation: FailedOperation):
        """
        Retry a failed operation.
        
        Args:
            operation: The operation to retry
        """
        with trace_span("dlq.retry", {
            "operation_id": operation.id,
            "operation_type": operation.operation_type.value,
            "retry_count": operation.retry_count
        }):
            logger.info(f"Retrying operation {operation.id} (attempt {operation.retry_count + 1})")
            
            # Update operation status
            with self.queue_lock:
                operation.status = "processing"
                operation.last_retry_at = datetime.now()
                operation.retry_count += 1
            
            try:
                # Get handler for this operation type
                handler = self.handlers.get(operation.operation_type)
                
                if not handler:
                    logger.error(f"No handler registered for {operation.operation_type.value}")
                    with self.queue_lock:
                        operation.status = "failed_permanently"
                    return
                
                # Execute the handler
                success = handler(operation.payload)
                
                if success:
                    # Operation succeeded
                    logger.info(f"Operation {operation.id} succeeded after {operation.retry_count} retries")
                    record_event("dlq.retry.success", {
                        "operation_id": operation.id,
                        "retry_count": operation.retry_count
                    })
                    
                    with self.queue_lock:
                        operation.status = "completed"
                        # Remove from queue if completed
                        self.queue.remove(operation)
                else:
                    # Operation failed again
                    self._handle_retry_failure(operation)
                    
            except Exception as e:
                logger.error(f"Error retrying operation {operation.id}: {e}")
                self._handle_retry_failure(operation, str(e))
            
            finally:
                self._save_queue()
    
    def _handle_retry_failure(self, operation: FailedOperation, error: Optional[str] = None):
        """
        Handle a failed retry attempt.
        
        Args:
            operation: The operation that failed
            error: Optional error message
        """
        if error:
            operation.error_message = error
        
        if operation.can_retry():
            # Can retry again later
            with self.queue_lock:
                operation.status = "pending"
            
            logger.warning(f"Operation {operation.id} failed, will retry again")
            record_event("dlq.retry.failed", {
                "operation_id": operation.id,
                "retry_count": operation.retry_count,
                "can_retry": True
            })
        else:
            # Max retries reached
            with self.queue_lock:
                operation.status = "failed_permanently"
            
            logger.error(f"Operation {operation.id} failed permanently after {operation.retry_count} retries")
            record_event("dlq.retry.failed_permanently", {
                "operation_id": operation.id,
                "retry_count": operation.retry_count
            })
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get DLQ statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.queue_lock:
            total = len(self.queue)
            pending = sum(1 for op in self.queue if op.status == "pending")
            processing = sum(1 for op in self.queue if op.status == "processing")
            failed = sum(1 for op in self.queue if op.status == "failed_permanently")
            
            by_type = {}
            for op in self.queue:
                op_type = op.operation_type.value
                if op_type not in by_type:
                    by_type[op_type] = 0
                by_type[op_type] += 1
        
        return {
            "total_operations": total,
            "pending": pending,
            "processing": processing,
            "failed_permanently": failed,
            "by_operation_type": by_type,
            "storage_path": str(self.storage_path)
        }
    
    def get_failed_operations(
        self,
        operation_type: Optional[OperationType] = None,
        status: Optional[str] = None
    ) -> List[FailedOperation]:
        """
        Get failed operations from the queue.
        
        Args:
            operation_type: Filter by operation type
            status: Filter by status
            
        Returns:
            List of failed operations
        """
        with self.queue_lock:
            operations = self.queue.copy()
        
        if operation_type:
            operations = [op for op in operations if op.operation_type == operation_type]
        
        if status:
            operations = [op for op in operations if op.status == status]
        
        return operations
    
    def retry_operation_manually(self, operation_id: str) -> bool:
        """
        Manually retry a specific operation.
        
        Args:
            operation_id: ID of the operation to retry
            
        Returns:
            True if retry was initiated
        """
        with self.queue_lock:
            operation = next((op for op in self.queue if op.id == operation_id), None)
        
        if operation:
            # Reset retry count for manual retry
            operation.retry_count = 0
            operation.status = "pending"
            operation.last_retry_at = None
            
            # Process immediately
            self._retry_operation(operation)
            return True
        
        return False
    
    def remove_operation(self, operation_id: str) -> bool:
        """
        Remove an operation from the queue.
        
        Args:
            operation_id: ID of the operation to remove
            
        Returns:
            True if operation was removed
        """
        with self.queue_lock:
            operation = next((op for op in self.queue if op.id == operation_id), None)
            
            if operation:
                self.queue.remove(operation)
                self._save_queue()
                logger.info(f"Removed operation {operation_id} from DLQ")
                return True
        
        return False


# Global DLQ instance
dlq = DeadLetterQueue()


# Example handlers for common operations
def chapter_generation_handler(payload: Dict[str, Any]) -> bool:
    """Handler for retrying chapter generation."""
    try:
        from services.generation_service import GenerationService
        from containers import get_container
        
        container = get_container()
        service = container.generation_service()
        
        # Retry chapter generation
        result = service.generate_chapter(
            chapter_number=payload.get("chapter_number"),
            chapter_title=payload.get("chapter_title"),
            context=payload.get("context", {})
        )
        
        return result is not None
        
    except Exception as e:
        logger.error(f"Chapter generation handler failed: {e}")
        return False


# Register default handlers
dlq.register_handler(OperationType.CHAPTER_GENERATION, chapter_generation_handler)


# Convenience functions
def add_to_dlq(
    operation_type: OperationType,
    payload: Dict[str, Any],
    error: Exception,
    **kwargs
) -> str:
    """Add a failed operation to the DLQ."""
    return dlq.add_operation(operation_type, payload, error, **kwargs)


def get_dlq_stats() -> Dict[str, Any]:
    """Get DLQ statistics."""
    return dlq.get_statistics()