"""
Book generation orchestration module.
"""

import json
import logging
import os
import time
from typing import Any, Dict

from app_config import settings
from checkpoint_manager import CheckpointManager
from cli_handler import CLIHandler
from events import Event, EventType, event_manager
from exceptions import ContentGenerationError
from file_operations import FileOperations
from services.generation_service import GenerationService


class BookGenerator:
    """Orchestrate the book generation process."""

    def __init__(
        self,
        generation_service: GenerationService,
        checkpoint_manager: CheckpointManager,
        file_ops: FileOperations,
        cli_handler: CLIHandler
    ) -> None:
        """
        Initialize the book generator.

        Args:
            generation_service: Service for generating content
            checkpoint_manager: Manager for checkpoints
            file_ops: File operations handler
            cli_handler: CLI handler for user interaction
        """
        self.generation_service = generation_service
        self.checkpoint_manager = checkpoint_manager
        self.file_ops = file_ops
        self.cli_handler = cli_handler
        self.logger = logging.getLogger(__name__)

    def generate_book(
        self,
        book_base_dir: str,
        title: str,
        instructions: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Load or generate book with enhanced error handling and checkpointing.

        Args:
            book_base_dir: Base directory for book files
            title: Book title
            instructions: Generation instructions
            language: Target language

        Returns:
            Generated book data

        Raises:
            ContentGenerationError: If critical generation steps fail
        """
        book_json_path = f'{book_base_dir}/book.json'
        book = self._load_existing_book(book_json_path, book_base_dir)

        # Generate title if needed
        if not book.get('title'):
            book = self._generate_title(book, title, language, book_json_path, book_base_dir)

        # Generate table of contents if needed
        if not book.get('toc'):
            book = self._generate_toc(book, instructions, book_json_path, book_base_dir)

        # Generate chapters
        book = self._generate_chapters(book, book_json_path, book_base_dir)

        return book

    def _load_existing_book(
        self,
        book_json_path: str,
        book_base_dir: str
    ) -> Dict[str, Any]:
        """
        Load existing book or restore from checkpoint.

        Args:
            book_json_path: Path to book.json
            book_base_dir: Base directory for the book

        Returns:
            Loaded book data or empty dict
        """
        if os.path.exists(book_json_path):
            try:
                self.logger.info(f"Reading book {book_json_path}...")
                return self.file_ops.load_json_safely(book_json_path)
            except Exception as e:
                self.logger.error(f"Failed to read book.json: {e}")
                # Try to restore from checkpoint
                book = self.checkpoint_manager.restore_from_checkpoint(book_base_dir)
                if book is None:
                    self.logger.error("No valid checkpoint found, starting fresh")
                    return {}
                return book
        return {}

    def _generate_title(
        self,
        book: Dict[str, Any],
        original_title: str,
        language: str,
        book_json_path: str,
        book_base_dir: str
    ) -> Dict[str, Any]:
        """
        Generate translated title for the book.

        Args:
            book: Current book data
            original_title: Original title
            language: Target language
            book_json_path: Path to save book.json
            book_base_dir: Base directory for the book

        Returns:
            Updated book data with title
        """
        try:
            prompt = f"Translate the title \"{original_title}\" to {language} and normalize it."
            book['title'] = self.generation_service.generate_text(
                settings.LLM_PROVIDER, prompt
            )
            self.file_ops.save_json_atomically(book, book_json_path)
            self.checkpoint_manager.create_checkpoint(
                book, book_base_dir, "title_generated"
            )
        except Exception as e:
            self.logger.error(f"Failed to generate title: {e}")
            # Use original title as fallback
            book['title'] = original_title
            self.file_ops.save_json_atomically(book, book_json_path)
        
        return book

    def _generate_toc(
        self,
        book: Dict[str, Any],
        instructions: str,
        book_json_path: str,
        book_base_dir: str
    ) -> Dict[str, Any]:
        """
        Generate table of contents for the book.

        Args:
            book: Current book data
            instructions: Generation instructions
            book_json_path: Path to save book.json
            book_base_dir: Base directory for the book

        Returns:
            Updated book data with TOC

        Raises:
            ContentGenerationError: If TOC generation fails
        """
        try:
            toc_prompt = (
                f"Generate a table of contents for a book titled "
                f"\"{book['title']}\". {instructions}"
            )
            toc_json = self.generation_service.generate_text(
                settings.LLM_PROVIDER, toc_prompt
            )
            book['toc'] = json.loads(toc_json)
            self.file_ops.save_json_atomically(book, book_json_path)
            self.checkpoint_manager.create_checkpoint(
                book, book_base_dir, "toc_generated"
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse TOC JSON: {e}")
            raise ContentGenerationError("Invalid TOC format received")
        except Exception as e:
            self.logger.error(f"Failed to generate TOC: {e}")
            raise ContentGenerationError(f"TOC generation failed: {e}")
        
        return book

    def _generate_chapters(
        self,
        book: Dict[str, Any],
        book_json_path: str,
        book_base_dir: str
    ) -> Dict[str, Any]:
        """
        Generate all chapters with error recovery.

        Args:
            book: Current book data
            book_json_path: Path to save book.json
            book_base_dir: Base directory for the book

        Returns:
            Updated book data with chapters
        """
        total_chapters = len(book['toc']['chapters'])
        
        for i, chapter in enumerate(book['toc']['chapters']):
            if not chapter.get('content'):
                self._generate_single_chapter(
                    book, i, chapter, total_chapters,
                    book_json_path, book_base_dir
                )
        
        return book

    def _generate_single_chapter(
        self,
        book: Dict[str, Any],
        chapter_index: int,
        chapter: Dict[str, Any],
        total_chapters: int,
        book_json_path: str,
        book_base_dir: str
    ) -> None:
        """
        Generate a single chapter with retry logic.

        Args:
            book: Current book data
            chapter_index: Index of the chapter (0-based)
            chapter: Chapter data
            total_chapters: Total number of chapters
            book_json_path: Path to save book.json
            book_base_dir: Base directory for the book
        """
        chapter_num = chapter_index + 1
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.logger.info(
                    f"Generating chapter {chapter_num}/{total_chapters}: "
                    f"{chapter.get('title', 'Untitled')}"
                )

                chapter['content'] = self.generation_service.generate_book_chapter(
                    settings.LLM_PROVIDER,
                    book,
                    chapter_index,
                    book_dir=book_base_dir  # Pass book directory for RAG
                )

                # Save after each successful chapter
                self.file_ops.save_json_atomically(book, book_json_path)

                # Create periodic checkpoint
                self.checkpoint_manager.create_periodic_checkpoint(
                    book, book_base_dir, chapter_index, total_chapters
                )

                # Emit progress event
                event_manager.emit(Event(EventType.CHAPTER_COMPLETED, {
                    'chapter_number': chapter_num,
                    'chapter_title': chapter.get('title', 'Untitled'),
                    'total_chapters': total_chapters
                }))
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"Failed to generate chapter {chapter_num} "
                    f"(attempt {retry_count}/{max_retries}): {e}"
                )

                if retry_count >= max_retries:
                    self._handle_chapter_failure(
                        book, chapter, chapter_num, chapter_index,
                        book_json_path, book_base_dir, e
                    )
                    break
                else:
                    # Wait before retry with exponential backoff
                    wait_time = retry_count * 2
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

    def _handle_chapter_failure(
        self,
        book: Dict[str, Any],
        chapter: Dict[str, Any],
        chapter_num: int,
        chapter_index: int,
        book_json_path: str,
        book_base_dir: str,
        error: Exception
    ) -> None:
        """
        Handle chapter generation failure.

        Args:
            book: Current book data
            chapter: Chapter data
            chapter_num: Chapter number (1-based)
            chapter_index: Chapter index (0-based)
            book_json_path: Path to save book.json
            book_base_dir: Base directory for the book
            error: The exception that caused the failure

        Raises:
            ContentGenerationError: If user chooses not to continue
        """
        # Save partial book before failing
        self.logger.error(f"Max retries reached for chapter {chapter_num}")
        chapter['content'] = f"[ERROR: Failed to generate this chapter: {error}]"
        self.file_ops.save_json_atomically(book, book_json_path)
        self.checkpoint_manager.create_checkpoint(
            book, book_base_dir, f"partial_chapter_{chapter_index}"
        )

        # Ask user if they want to continue
        if not self.cli_handler.confirm_continuation(chapter_num):
            raise ContentGenerationError(
                f"Chapter {chapter_num} generation failed after maximum attempts"
            )