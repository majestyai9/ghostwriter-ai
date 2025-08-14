"""
Main application entry point.
"""
import json
import logging
import os

import slugify

from app_config import settings
from bookprinter import print_book
from cache_manager import CacheManager
from events import Event, EventType, ProgressTracker, event_manager
from exceptions import ContentGenerationError, FileOperationError, GhostwriterException
from providers.factory import ProviderFactory
from services.generation_service import GenerationService
from token_optimizer import TokenOptimizer


def setup_event_handlers():
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


def generate_book(generation_service: GenerationService, book_base_dir: str, title: str, instructions: str, language: str):
    """
    Load or generate book with error handling
    """
    book_json_path = f'{book_base_dir}/book.json'

    book = {}

    if os.path.exists(book_json_path):
        try:
            logging.info(f"Reading book {book_json_path}...")
            with open(book_json_path, encoding='utf-8') as f:
                book = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Failed to read or parse book.json: {e}")
            raise FileOperationError(f"Invalid book.json format: {e}")

    if not book.get('title'):
        book['title'] = generation_service.generate_text(settings.LLM_PROVIDER, f"Translate the title \"{title}\" to {language} and normalize it.")

    if not book.get('toc'):
        toc_prompt = f"Generate a table of contents for a book titled \"{book['title']}\". {instructions}"
        toc_json = generation_service.generate_text(settings.LLM_PROVIDER, toc_prompt)
        book['toc'] = json.loads(toc_json)

    for i, chapter in enumerate(book['toc']['chapters']):
        if not chapter.get('content'):
            chapter['content'] = generation_service.generate_book_chapter(
                settings.LLM_PROVIDER, 
                book, 
                i,
                book_dir=book_base_dir  # Pass book directory for RAG
            )

        # Save after each chapter
        with open(book_json_path, 'w', encoding='utf-8') as f:
            json.dump(book, f, indent=4, ensure_ascii=False)

    return book

def main():
    """Enhanced main function with error handling"""
    logging.basicConfig(level=settings.LOG_LEVEL.upper())
    logging.info(">> Book Writer AI (Enhanced Edition)")

    setup_event_handlers()

    try:
        language = settings.BOOK_LANGUAGE or input("In which language do you want to write the book? (you can use BOOK_LANGUAGE env variable): ")
        original_title = settings.BOOK_TITLE or input("What is the title of the book? (you can use BOOK_TITLE env variable): ")
        instructions = settings.BOOK_INSTRUCTIONS or input("What are the instructions for the book? (you can use BOOK_INSTRUCTIONS env variable): ")

        if not language or not original_title:
            raise ValueError("Language and title are required")

        book_base_dir = f"{settings.BASE_DIR}/books/{slugify.slugify(original_title)}"
        os.makedirs(book_base_dir, exist_ok=True)

        # Initialize services
        provider_factory = ProviderFactory()
        provider = provider_factory.create_provider(settings.LLM_PROVIDER, {"api_key": settings.OPENAI_API_KEY if settings.LLM_PROVIDER == "openai" else settings.ANTHROPIC_API_KEY})
        cache_manager = CacheManager(backend=settings.CACHE_TYPE, expire=settings.CACHE_TTL_SECONDS)
        token_optimizer = TokenOptimizer(provider=provider)
        generation_service = GenerationService(provider_factory, cache_manager, token_optimizer)

        book = generate_book(generation_service, book_base_dir, original_title, instructions, language)

        print_book(book_base_dir, book)
        event_manager.emit(Event(EventType.BOOK_EXPORTED, {
            'path': book_base_dir,
            'format': 'markdown'
        }))
        logging.info(f"✓ Book successfully generated at: {book_base_dir}")

    except (GhostwriterException, ValueError) as e:
        logging.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.info("\nGeneration interrupted by user")
        return 2
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())