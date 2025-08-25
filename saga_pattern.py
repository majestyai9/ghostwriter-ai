"""
Saga pattern implementation for managing multi-step transactional operations.
Ensures consistency in complex book generation workflows with compensating actions.
"""

import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import traceback
from abc import ABC, abstractmethod

from tracing import trace_span, record_event

logger = logging.getLogger(__name__)


class SagaStepStatus(Enum):
    """Status of a saga step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


@dataclass
class SagaStep:
    """Represents a single step in a saga."""
    name: str
    action: Callable
    compensation: Optional[Callable] = None
    retries: int = 3
    timeout: Optional[float] = None
    critical: bool = True  # If True, failure stops the saga
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaStepResult:
    """Result of a saga step execution."""
    step_name: str
    status: SagaStepStatus
    result: Any = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    attempts: int = 0
    compensation_result: Any = None


class SagaContext:
    """Context shared across saga steps."""
    
    def __init__(self):
        """Initialize saga context."""
        self.data: Dict[str, Any] = {}
        self.step_results: List[SagaStepResult] = []
        self.metadata: Dict[str, Any] = {}
        self.transaction_id: str = ""
    
    def set(self, key: str, value: Any):
        """Set a value in the context."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)
    
    def add_result(self, result: SagaStepResult):
        """Add a step result to the context."""
        self.step_results.append(result)
    
    def get_last_result(self) -> Optional[SagaStepResult]:
        """Get the last step result."""
        return self.step_results[-1] if self.step_results else None


class Saga:
    """
    Saga orchestrator for managing multi-step transactions.
    Implements the saga pattern with compensating transactions.
    """
    
    def __init__(self, name: str, steps: List[SagaStep]):
        """
        Initialize a saga.
        
        Args:
            name: Name of the saga
            steps: List of saga steps to execute
        """
        self.name = name
        self.steps = steps
        self.context = SagaContext()
        self.completed_steps: List[SagaStep] = []
        self.failed_step: Optional[SagaStep] = None
        
    def execute(self) -> Tuple[bool, SagaContext]:
        """
        Execute the saga.
        
        Returns:
            Tuple of (success, context)
        """
        with trace_span(f"saga.{self.name}", {"saga.name": self.name}):
            logger.info(f"Starting saga: {self.name}")
            
            try:
                # Execute each step
                for step in self.steps:
                    success = self._execute_step(step)
                    
                    if not success:
                        if step.critical:
                            # Critical step failed, start compensation
                            logger.error(f"Critical step {step.name} failed, starting compensation")
                            self._compensate()
                            return False, self.context
                        else:
                            # Non-critical step failed, continue
                            logger.warning(f"Non-critical step {step.name} failed, continuing")
                    else:
                        self.completed_steps.append(step)
                
                logger.info(f"Saga {self.name} completed successfully")
                record_event("saga.completed", {"saga.name": self.name})
                return True, self.context
                
            except Exception as e:
                logger.error(f"Saga {self.name} failed with exception: {e}")
                record_event("saga.failed", {
                    "saga.name": self.name,
                    "error": str(e)
                })
                self._compensate()
                return False, self.context
    
    def _execute_step(self, step: SagaStep) -> bool:
        """
        Execute a single saga step.
        
        Args:
            step: The step to execute
            
        Returns:
            True if successful, False otherwise
        """
        with trace_span(f"saga.step.{step.name}", {
            "step.name": step.name,
            "step.critical": step.critical
        }):
            result = SagaStepResult(
                step_name=step.name,
                status=SagaStepStatus.RUNNING
            )
            
            logger.info(f"Executing saga step: {step.name}")
            
            for attempt in range(step.retries):
                try:
                    result.attempts = attempt + 1
                    
                    # Execute the step action
                    step_result = step.action(self.context)
                    
                    result.status = SagaStepStatus.COMPLETED
                    result.result = step_result
                    result.completed_at = datetime.now()
                    
                    self.context.add_result(result)
                    
                    logger.info(f"Saga step {step.name} completed successfully")
                    record_event("saga.step.completed", {
                        "step.name": step.name,
                        "attempts": result.attempts
                    })
                    
                    return True
                    
                except Exception as e:
                    logger.warning(f"Saga step {step.name} failed on attempt {attempt + 1}: {e}")
                    
                    if attempt == step.retries - 1:
                        # Final attempt failed
                        result.status = SagaStepStatus.FAILED
                        result.error = str(e)
                        result.completed_at = datetime.now()
                        
                        self.context.add_result(result)
                        self.failed_step = step
                        
                        logger.error(f"Saga step {step.name} failed after {step.retries} attempts")
                        record_event("saga.step.failed", {
                            "step.name": step.name,
                            "error": str(e),
                            "attempts": result.attempts
                        })
                        
                        return False
            
            return False
    
    def _compensate(self):
        """Execute compensating transactions for completed steps."""
        logger.info(f"Starting compensation for saga: {self.name}")
        record_event("saga.compensation.started", {"saga.name": self.name})
        
        # Compensate in reverse order
        for step in reversed(self.completed_steps):
            if step.compensation:
                self._execute_compensation(step)
    
    def _execute_compensation(self, step: SagaStep):
        """
        Execute compensation for a step.
        
        Args:
            step: The step to compensate
        """
        with trace_span(f"saga.compensation.{step.name}", {
            "step.name": step.name
        }):
            logger.info(f"Executing compensation for step: {step.name}")
            
            try:
                compensation_result = step.compensation(self.context)
                
                # Update the step result with compensation info
                for result in self.context.step_results:
                    if result.step_name == step.name:
                        result.status = SagaStepStatus.COMPENSATED
                        result.compensation_result = compensation_result
                        break
                
                logger.info(f"Compensation for step {step.name} completed")
                record_event("saga.compensation.completed", {
                    "step.name": step.name
                })
                
            except Exception as e:
                logger.error(f"Compensation for step {step.name} failed: {e}")
                
                # Mark compensation as failed
                for result in self.context.step_results:
                    if result.step_name == step.name:
                        result.status = SagaStepStatus.COMPENSATION_FAILED
                        break
                
                record_event("saga.compensation.failed", {
                    "step.name": step.name,
                    "error": str(e)
                })


# Book generation saga implementation
class BookGenerationSaga:
    """Saga for book generation workflow."""
    
    @staticmethod
    def create_book_saga(title: str, language: str, instructions: str) -> Saga:
        """
        Create a saga for book generation.
        
        Args:
            title: Book title
            language: Book language
            instructions: Generation instructions
            
        Returns:
            Configured saga instance
        """
        steps = [
            SagaStep(
                name="initialize_project",
                action=BookGenerationSaga._initialize_project,
                compensation=BookGenerationSaga._cleanup_project,
                critical=True
            ),
            SagaStep(
                name="generate_outline",
                action=BookGenerationSaga._generate_outline,
                compensation=BookGenerationSaga._remove_outline,
                critical=True
            ),
            SagaStep(
                name="setup_rag_index",
                action=BookGenerationSaga._setup_rag_index,
                compensation=BookGenerationSaga._cleanup_rag_index,
                critical=False  # RAG is optional
            ),
            SagaStep(
                name="generate_chapters",
                action=BookGenerationSaga._generate_chapters,
                compensation=BookGenerationSaga._cleanup_chapters,
                critical=True
            ),
            SagaStep(
                name="validate_content",
                action=BookGenerationSaga._validate_content,
                compensation=None,  # No compensation needed
                critical=False
            ),
            SagaStep(
                name="export_book",
                action=BookGenerationSaga._export_book,
                compensation=BookGenerationSaga._cleanup_exports,
                critical=False
            )
        ]
        
        saga = Saga("book_generation", steps)
        saga.context.set("title", title)
        saga.context.set("language", language)
        saga.context.set("instructions", instructions)
        
        return saga
    
    @staticmethod
    def _initialize_project(context: SagaContext) -> Dict[str, Any]:
        """Initialize book project."""
        from project_manager import get_project_manager
        
        pm = get_project_manager()
        project_id = pm.create_project(
            title=context.get("title"),
            language=context.get("language")
        )
        
        context.set("project_id", project_id)
        logger.info(f"Project initialized: {project_id}")
        
        return {"project_id": project_id}
    
    @staticmethod
    def _cleanup_project(context: SagaContext):
        """Clean up project on failure."""
        from project_manager import get_project_manager
        
        project_id = context.get("project_id")
        if project_id:
            pm = get_project_manager()
            pm.delete_project(project_id, confirm=True)
            logger.info(f"Project cleaned up: {project_id}")
    
    @staticmethod
    def _generate_outline(context: SagaContext) -> Dict[str, Any]:
        """Generate book outline."""
        # This would call the actual outline generation logic
        logger.info("Generating book outline")
        
        # Placeholder for actual implementation
        outline = {
            "chapters": [
                {"number": i, "title": f"Chapter {i}"}
                for i in range(1, 11)
            ]
        }
        
        context.set("outline", outline)
        return outline
    
    @staticmethod
    def _remove_outline(context: SagaContext):
        """Remove generated outline."""
        context.set("outline", None)
        logger.info("Outline removed")
    
    @staticmethod
    def _setup_rag_index(context: SagaContext) -> Dict[str, Any]:
        """Setup RAG index for the book."""
        logger.info("Setting up RAG index")
        
        # Placeholder for actual RAG setup
        context.set("rag_enabled", True)
        return {"rag_enabled": True}
    
    @staticmethod
    def _cleanup_rag_index(context: SagaContext):
        """Clean up RAG index."""
        context.set("rag_enabled", False)
        logger.info("RAG index cleaned up")
    
    @staticmethod
    def _generate_chapters(context: SagaContext) -> Dict[str, Any]:
        """Generate book chapters."""
        outline = context.get("outline", {})
        chapters = []
        
        for chapter in outline.get("chapters", []):
            logger.info(f"Generating chapter {chapter['number']}")
            # Placeholder for actual chapter generation
            chapters.append({
                "number": chapter["number"],
                "title": chapter["title"],
                "content": f"Content for {chapter['title']}"
            })
        
        context.set("chapters", chapters)
        return {"chapters_count": len(chapters)}
    
    @staticmethod
    def _cleanup_chapters(context: SagaContext):
        """Clean up generated chapters."""
        context.set("chapters", [])
        logger.info("Chapters cleaned up")
    
    @staticmethod
    def _validate_content(context: SagaContext) -> Dict[str, Any]:
        """Validate generated content."""
        chapters = context.get("chapters", [])
        
        # Placeholder for validation logic
        is_valid = len(chapters) > 0
        
        context.set("validation_passed", is_valid)
        return {"valid": is_valid}
    
    @staticmethod
    def _export_book(context: SagaContext) -> Dict[str, Any]:
        """Export book to various formats."""
        logger.info("Exporting book")
        
        # Placeholder for export logic
        exports = ["epub", "pdf", "docx"]
        
        context.set("exports", exports)
        return {"formats": exports}
    
    @staticmethod
    def _cleanup_exports(context: SagaContext):
        """Clean up exported files."""
        context.set("exports", [])
        logger.info("Exports cleaned up")