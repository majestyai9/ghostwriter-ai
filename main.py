"""
Main application entry point - thin orchestrator for book generation.
"""

import logging
import sys
import traceback
from typing import NoReturn

from app_config import settings
from book_generator import BookGenerator
from bookprinter import print_book
from checkpoint_manager import CheckpointManager
from cli_handler import CLIHandler
from event_setup import EventSetup
from events import Event, EventType, ProgressTracker, event_manager
from exceptions import GhostwriterException
from file_operations import FileOperations
from service_initializer import ServiceInitializer


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main() -> int:
    """
    Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(">> Book Writer AI (Enhanced Edition with Error Recovery)")

    # Initialize components
    event_setup = EventSetup()
    event_setup.setup_event_handlers()

    # Add progress tracker
    progress_tracker = ProgressTracker()
    event_manager.subscribe_all(progress_tracker.track_progress)

    # Initialize handlers and managers
    cli_handler = CLIHandler()
    checkpoint_manager = CheckpointManager()
    file_ops = FileOperations()
    service_initializer = ServiceInitializer()

    try:
        # Get book parameters from user
        language, original_title, instructions, book_base_dir = (
            cli_handler.get_book_parameters()
        )

        # Initialize services
        generation_service = service_initializer.initialize_services()

        # Initialize book generator
        book_generator = BookGenerator(
            generation_service=generation_service,
            checkpoint_manager=checkpoint_manager,
            file_ops=file_ops,
            cli_handler=cli_handler
        )

        # Generate the book
        book = book_generator.generate_book(
            book_base_dir=book_base_dir,
            title=original_title,
            instructions=instructions,
            language=language
        )

        # Export the book
        try:
            print_book(book_base_dir, book)
            event_manager.emit(Event(EventType.BOOK_EXPORTED, {
                'path': book_base_dir,
                'format': 'markdown'
            }))
            logger.info(f"✓ Book successfully generated at: {book_base_dir}")

            # Show final statistics
            stats = progress_tracker.get_progress()
            logger.info(
                f"Final statistics: {stats['chapters']['completed']}/"
                f"{stats['chapters']['total']} chapters completed"
            )

        except Exception as e:
            logger.error(f"Failed to export book: {e}")
            logger.info("Book data has been saved and can be exported manually")
            return 1

    except GhostwriterException as e:
        logger.error(f"Application error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nGeneration interrupted by user")
        logger.info("Progress has been saved. You can resume by running the program again.")
        return 2
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        logger.info("An unexpected error occurred. Please check the logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())