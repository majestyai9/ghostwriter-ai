"""
CLI handler for parsing and validating command-line arguments and user input.
"""

import logging
import os
from typing import Tuple

import slugify

from app_config import settings


class CLIHandler:
    """Handle command-line interface interactions and input validation."""

    def __init__(self) -> None:
        """Initialize the CLI handler."""
        self.logger = logging.getLogger(__name__)

    def get_book_parameters(self) -> Tuple[str, str, str, str]:
        """
        Get book generation parameters from environment or user input.

        Returns:
            Tuple containing (language, original_title, instructions, book_base_dir)

        Raises:
            ValueError: If required parameters are missing
        """
        # Get language
        language = settings.BOOK_LANGUAGE
        if not language:
            language = input(
                "In which language do you want to write the book? "
                "(you can use BOOK_LANGUAGE env variable): "
            )

        # Get title
        original_title = settings.BOOK_TITLE
        if not original_title:
            original_title = input(
                "What is the title of the book? "
                "(you can use BOOK_TITLE env variable): "
            )

        # Get instructions
        instructions = settings.BOOK_INSTRUCTIONS
        if not instructions:
            instructions = input(
                "What are the instructions for the book? "
                "(you can use BOOK_INSTRUCTIONS env variable): "
            )

        # Validate required parameters
        if not language or not original_title:
            raise ValueError("Language and title are required")

        # Generate book directory path
        book_base_dir = self._create_book_directory(original_title)

        self.logger.info(
            f"Book parameters: language={language}, "
            f"title={original_title}, dir={book_base_dir}"
        )

        return language, original_title, instructions, book_base_dir

    def _create_book_directory(self, title: str) -> str:
        """
        Create and return the book base directory path.

        Args:
            title: Book title to create directory for

        Returns:
            Path to the book base directory
        """
        book_base_dir = f"{settings.BASE_DIR}/books/{slugify.slugify(title)}"
        os.makedirs(book_base_dir, exist_ok=True)
        return book_base_dir

    def confirm_continuation(self, chapter_num: int) -> bool:
        """
        Ask user whether to continue after a chapter failure.

        Args:
            chapter_num: Number of the failed chapter

        Returns:
            True if user wants to continue, False otherwise
        """
        response = input(
            f"Chapter {chapter_num} failed. Continue with next chapter? (y/n): "
        )
        return response.lower() == 'y'