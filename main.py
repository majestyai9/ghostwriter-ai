"""
Main application entry point with enhanced error recovery and checkpointing.
"""
import json
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import slugify

from app_config import settings
from bookprinter import print_book
from cache_manager import CacheManager
from events import Event, EventType, ProgressTracker, event_manager
from exceptions import ContentGenerationError, FileOperationError, GhostwriterException
from providers.factory import ProviderFactory
from services.generation_service import GenerationService
from token_optimizer import TokenOptimizer


def setup_event_handlers() -> None:
    """Setup custom event handlers for monitoring"""

    def log_progress(event: Event):
        """Log progress events"""
        if event.type == EventType.CHAPTER_COMPLETED:
            chapter_num = event.data.get('chapter_number')
            chapter_title = event.data.get('chapter_title')
            logging.info(f"✓ Chapter {chapter_num}: {chapter_title} completed")

        elif event.type == EventType.SECTION_COMPLETED:
            section_num = event.data.get('section_number')
            section_title = event.data.get('section_title')
            logging.info(f"  ✓ Section {section_num}: {section_title} completed")

    def handle_errors(event: Event):
        """Handle error events"""
        if event.type == EventType.GENERATION_FAILED:
            stage = event.data.get('stage')
            error = event.data.get('error')
            logging.error(f"Generation failed at {stage}: {error}")

        elif event.type == EventType.API_CALL_FAILED:
            provider = event.data.get('provider')
            error = event.data.get('error')
            logging.error(f"API call failed for {provider}: {error}")

    # Subscribe handlers
    event_manager.subscribe(EventType.CHAPTER_COMPLETED, log_progress)
    event_manager.subscribe(EventType.SECTION_COMPLETED, log_progress)
    event_manager.subscribe(EventType.GENERATION_FAILED, handle_errors)
    event_manager.subscribe(EventType.API_CALL_FAILED, handle_errors)


def save_book_atomically(book: Dict[str, Any], book_json_path: str) -> None:
    """
    Save book data atomically using temp file and rename.
    
    Args:
        book: Book data to save
        book_json_path: Target path for book.json
    """
    # Create temp file in same directory for atomic rename
    dir_path = os.path.dirname(book_json_path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                     dir=dir_path, suffix='.tmp',
                                     delete=False) as tmp_file:
        try:
            json.dump(book, tmp_file, indent=4, ensure_ascii=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Ensure data is written to disk
            tmp_path = tmp_file.name
        except Exception as e:
            os.unlink(tmp_file.name)
            raise FileOperationError(f"Failed to write temp file: {e}")
    
    # Atomic rename
    try:
        if os.name == 'nt':  # Windows
            # Windows doesn't support atomic rename if target exists
            if os.path.exists(book_json_path):
                backup_path = f"{book_json_path}.backup"
                shutil.copy2(book_json_path, backup_path)
            shutil.move(tmp_path, book_json_path)
        else:  # Unix-like
            os.rename(tmp_path, book_json_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise FileOperationError(f"Failed to save book atomically: {e}")


def create_checkpoint(book: Dict[str, Any], book_base_dir: str, 
                      checkpoint_name: str) -> None:
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
    save_book_atomically(book, checkpoint_path)
    logging.info(f"Checkpoint created: {checkpoint_name}")


def restore_from_checkpoint(book_base_dir: str) -> Optional[Dict[str, Any]]:
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
    checkpoints = sorted(Path(checkpoint_dir).glob('*.json'), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not checkpoints:
        return None
    
    latest_checkpoint = checkpoints[0]
    try:
        with open(latest_checkpoint, encoding='utf-8') as f:
            book = json.load(f)
        logging.info(f"Restored from checkpoint: {latest_checkpoint.name}")
        return book
    except Exception as e:
        logging.error(f"Failed to restore from checkpoint: {e}")
        return None


def generate_book(generation_service: GenerationService, book_base_dir: str, 
                 title: str, instructions: str, language: str) -> Dict[str, Any]:
    """
    Load or generate book with enhanced error handling and checkpointing.
    
    Args:
        generation_service: Service for generating content
        book_base_dir: Base directory for book files
        title: Book title
        instructions: Generation instructions
        language: Target language
        
    Returns:
        Generated book data
    """
    book_json_path = f'{book_base_dir}/book.json'
    book = {}
    
    # Try to load existing book
    if os.path.exists(book_json_path):
        try:
            logging.info(f"Reading book {book_json_path}...")
            with open(book_json_path, encoding='utf-8') as f:
                book = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Failed to read book.json: {e}")
            # Try to restore from checkpoint
            book = restore_from_checkpoint(book_base_dir)
            if book is None:
                logging.error("No valid checkpoint found, starting fresh")
                book = {}

    # Generate title if needed
    if not book.get('title'):
        try:
            prompt = f"Translate the title \"{title}\" to {language} and normalize it."
            book['title'] = generation_service.generate_text(settings.LLM_PROVIDER, prompt)
            save_book_atomically(book, book_json_path)
            create_checkpoint(book, book_base_dir, "title_generated")
        except Exception as e:
            logging.error(f"Failed to generate title: {e}")
            # Use original title as fallback
            book['title'] = title
            save_book_atomically(book, book_json_path)

    # Generate table of contents if needed
    if not book.get('toc'):
        try:
            toc_prompt = (f"Generate a table of contents for a book titled "
                         f"\"{book['title']}\". {instructions}")
            toc_json = generation_service.generate_text(settings.LLM_PROVIDER, toc_prompt)
            book['toc'] = json.loads(toc_json)
            save_book_atomically(book, book_json_path)
            create_checkpoint(book, book_base_dir, "toc_generated")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse TOC JSON: {e}")
            raise ContentGenerationError("Invalid TOC format received")
        except Exception as e:
            logging.error(f"Failed to generate TOC: {e}")
            raise ContentGenerationError(f"TOC generation failed: {e}")

    # Generate chapters with error recovery
    total_chapters = len(book['toc']['chapters'])
    for i, chapter in enumerate(book['toc']['chapters']):
        if not chapter.get('content'):
            chapter_num = i + 1
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logging.info(f"Generating chapter {chapter_num}/{total_chapters}: "
                               f"{chapter.get('title', 'Untitled')}")
                    
                    chapter['content'] = generation_service.generate_book_chapter(
                        settings.LLM_PROVIDER, 
                        book, 
                        i,
                        book_dir=book_base_dir  # Pass book directory for RAG
                    )
                    
                    # Save after each successful chapter
                    save_book_atomically(book, book_json_path)
                    
                    # Create checkpoint every 3 chapters or at the end
                    if (i + 1) % 3 == 0 or i == total_chapters - 1:
                        checkpoint_name = f"chapter_{i+1}_of_{total_chapters}"
                        create_checkpoint(book, book_base_dir, checkpoint_name)
                    
                    # Emit progress event
                    event_manager.emit(Event(EventType.CHAPTER_COMPLETED, {
                        'chapter_number': chapter_num,
                        'chapter_title': chapter.get('title', 'Untitled'),
                        'total_chapters': total_chapters
                    }))
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    logging.error(f"Failed to generate chapter {chapter_num} "
                                f"(attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count >= max_retries:
                        # Save partial book before failing
                        logging.error(f"Max retries reached for chapter {chapter_num}")
                        chapter['content'] = f"[ERROR: Failed to generate this chapter: {e}]"
                        save_book_atomically(book, book_json_path)
                        create_checkpoint(book, book_base_dir, f"partial_chapter_{i}")
                        
                        # Ask user if they want to continue
                        response = input(f"Chapter {chapter_num} failed. "
                                       f"Continue with next chapter? (y/n): ")
                        if response.lower() != 'y':
                            raise ContentGenerationError(
                                f"Chapter {chapter_num} generation failed after "
                                f"{max_retries} attempts"
                            )
                    else:
                        # Wait before retry
                        import time
                        wait_time = retry_count * 2  # Exponential backoff
                        logging.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)

    return book

def main() -> int:
    """Enhanced main function with comprehensive error handling and recovery.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info(">> Book Writer AI (Enhanced Edition with Error Recovery)")

    setup_event_handlers()
    
    # Add progress tracker
    progress_tracker = ProgressTracker()
    event_manager.subscribe_all(progress_tracker.track_progress)

    try:
        # Get input parameters with validation
        language = settings.BOOK_LANGUAGE or input(
            "In which language do you want to write the book? "
            "(you can use BOOK_LANGUAGE env variable): "
        )
        original_title = settings.BOOK_TITLE or input(
            "What is the title of the book? "
            "(you can use BOOK_TITLE env variable): "
        )
        instructions = settings.BOOK_INSTRUCTIONS or input(
            "What are the instructions for the book? "
            "(you can use BOOK_INSTRUCTIONS env variable): "
        )

        if not language or not original_title:
            raise ValueError("Language and title are required")

        book_base_dir = f"{settings.BASE_DIR}/books/{slugify.slugify(original_title)}"
        os.makedirs(book_base_dir, exist_ok=True)

        # Initialize services with error handling
        try:
            provider_factory = ProviderFactory()
            api_key = (settings.OPENAI_API_KEY if settings.LLM_PROVIDER == "openai" 
                      else settings.ANTHROPIC_API_KEY)
            provider = provider_factory.create_provider(
                settings.LLM_PROVIDER, 
                {"api_key": api_key}
            )
            cache_manager = CacheManager(
                backend=settings.CACHE_TYPE, 
                expire=settings.CACHE_TTL_SECONDS
            )
            token_optimizer = TokenOptimizer(provider=provider)
            generation_service = GenerationService(
                provider_factory, cache_manager, token_optimizer
            )
        except Exception as e:
            logging.error(f"Failed to initialize services: {e}")
            logging.error("Please check your API keys and configuration")
            return 1

        # Generate book with comprehensive error handling
        book = generate_book(
            generation_service, book_base_dir, 
            original_title, instructions, language
        )

        # Export book
        try:
            print_book(book_base_dir, book)
            event_manager.emit(Event(EventType.BOOK_EXPORTED, {
                'path': book_base_dir,
                'format': 'markdown'
            }))
            logging.info(f"✓ Book successfully generated at: {book_base_dir}")
            
            # Show final statistics
            stats = progress_tracker.get_progress()
            logging.info(f"Final statistics: {stats['chapters']['completed']}/"
                        f"{stats['chapters']['total']} chapters completed")
            
        except Exception as e:
            logging.error(f"Failed to export book: {e}")
            logging.info("Book data has been saved and can be exported manually")
            return 1

    except (GhostwriterException, ValueError) as e:
        logging.error(f"Application error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.info("\nGeneration interrupted by user")
        logging.info("Progress has been saved. You can resume by running the program again.")
        return 2
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        logging.info("An unexpected error occurred. Please check the logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())