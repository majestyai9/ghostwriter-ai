"""
Checkpoint manager for saving and restoring book generation progress.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from file_operations import FileOperations


class CheckpointManager:
    """Manage checkpoints for book generation progress."""

    def __init__(self) -> None:
        """Initialize the checkpoint manager."""
        self.logger = logging.getLogger(__name__)
        self.file_ops = FileOperations()

    def create_checkpoint(
        self,
        book: Dict[str, Any],
        book_base_dir: str,
        checkpoint_name: str
    ) -> None:
        """
        Create a checkpoint of the current book state.

        Args:
            book: Current book data
            book_base_dir: Base directory for the book
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_dir = os.path.join(book_base_dir, '.checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
        self.file_ops.save_json_atomically(book, checkpoint_path)
        self.logger.info(f"Checkpoint created: {checkpoint_name}")

    def restore_from_checkpoint(
        self,
        book_base_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Restore book from the latest checkpoint if available.

        Args:
            book_base_dir: Base directory for the book

        Returns:
            Restored book data or None if no checkpoint exists
        """
        checkpoint_dir = os.path.join(book_base_dir, '.checkpoints')
        if not os.path.exists(checkpoint_dir):
            return None

        # Find latest checkpoint
        checkpoints = sorted(
            Path(checkpoint_dir).glob('*.json'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not checkpoints:
            return None

        latest_checkpoint = checkpoints[0]
        try:
            with open(latest_checkpoint, encoding='utf-8') as f:
                book = json.load(f)
            self.logger.info(f"Restored from checkpoint: {latest_checkpoint.name}")
            return book
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}")
            return None

    def create_periodic_checkpoint(
        self,
        book: Dict[str, Any],
        book_base_dir: str,
        chapter_index: int,
        total_chapters: int,
        checkpoint_interval: int = 3
    ) -> None:
        """
        Create checkpoint periodically or at completion.

        Args:
            book: Current book data
            book_base_dir: Base directory for the book
            chapter_index: Current chapter index (0-based)
            total_chapters: Total number of chapters
            checkpoint_interval: Create checkpoint every N chapters
        """
        chapter_num = chapter_index + 1
        
        # Create checkpoint every N chapters or at the end
        if chapter_num % checkpoint_interval == 0 or chapter_index == total_chapters - 1:
            checkpoint_name = f"chapter_{chapter_num}_of_{total_chapters}"
            self.create_checkpoint(book, book_base_dir, checkpoint_name)