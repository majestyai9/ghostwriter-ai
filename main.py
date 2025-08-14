"""
Enhanced main module with error handling and event system
"""
import json
import logging
import os

import slugify

import generate
from bookprinter import print_book
from config import BASE_DIR, BOOK_INSTRUCTIONS, BOOK_LANGUAGE, BOOK_TITLE, ENABLE_PROGRESS_TRACKING
from events import Event, EventType, ProgressTracker, event_manager
from exceptions import ContentGenerationError, FileOperationError, GhostwriterException

# Setup progress tracking if enabled
progress_tracker = None
if ENABLE_PROGRESS_TRACKING:
    progress_tracker = ProgressTracker()
    event_manager.subscribe_all(progress_tracker.track_progress)

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

def get_book(book_base_dir, title, instructions, language):
    """
    Load or generate book with error handling
    
    Args:
        book_base_dir: Base directory for book files
        title: Book title
        instructions: Generation instructions
        language: Target language
        
    Returns:
        Complete book dictionary
        
    Raises:
        FileOperationError: If file operations fail
        ContentGenerationError: If generation fails
    """
    book_json_path = f'{book_base_dir}/book.json'

    book = {}

    # Try to load existing book
    if os.path.exists(book_json_path):
        try:
            logging.info(f"Reading book {book_json_path}...")
            with open(book_json_path, encoding='utf-8') as f:
                book = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse book.json: {e}")
            raise FileOperationError(f"Invalid book.json format: {e}")
        except OSError as e:
            logging.error(f"Failed to read book.json: {e}")
            raise FileOperationError(f"Cannot read book.json: {e}")

    # Generate book content
    try:
        for b in generate.write_book(book, title, instructions, language):
            # Save after each generation step
            try:
                with open(book_json_path, 'w', encoding='utf-8') as f:
                    json.dump(b, f, indent=4, ensure_ascii=False)
                    logging.info(">> Book saved to book.json")
                    event_manager.emit(Event(EventType.BOOK_SAVED, {
                        'path': book_json_path,
                        'size': os.path.getsize(book_json_path)
                    }))
            except OSError as e:
                logging.error(f"Failed to save book.json: {e}")
                # Continue generation even if save fails

    except ContentGenerationError as e:
        logging.error(f"Book generation failed: {e}")
        # Save partial progress if available
        if book:
            try:
                with open(f"{book_json_path}.partial", 'w', encoding='utf-8') as f:
                    json.dump(book, f, indent=4, ensure_ascii=False)
                logging.info(f"Partial book saved to {book_json_path}.partial")
            except:
                pass
        raise

    # Add file paths to chapters
    for chapter in book['toc']['chapters']:
        nstr = "{:02}".format(chapter['number'])
        chapter_slug = slugify.slugify(chapter['title'])
        chapter["file"] = f"{nstr}-{chapter_slug}.md"

    return book

def main():
    """Enhanced main function with error handling"""
    logging.info(">> Book Writer AI (Enhanced Edition)")

    # Setup event handlers
    setup_event_handlers()

    try:
        # Get book configuration
        language = BOOK_LANGUAGE or input("In which language do you want to write the book? (you can use BOOK_LANGUAGE env variable): ")
        original_title = BOOK_TITLE or input("What is the title of the book? (you can use BOOK_TITLE env variable): ")
        instructions = BOOK_INSTRUCTIONS or input("What are the instructions for the book? (you can use BOOK_INSTRUCTIONS env variable): ")

        if not language or not original_title:
            raise ValueError("Language and title are required")

        # Create book directory
        book_base_dir = f"{BASE_DIR}/books/{slugify.slugify(original_title)}"
        try:
            os.makedirs(book_base_dir, exist_ok=True)
        except OSError as e:
            raise FileOperationError(f"Failed to create book directory: {e}")

        # Generate book
        book = get_book(book_base_dir, original_title, instructions, language)

        # Print book to files
        try:
            print_book(book_base_dir, book)
            event_manager.emit(Event(EventType.BOOK_EXPORTED, {
                'path': book_base_dir,
                'format': 'markdown'
            }))
            logging.info(f"✓ Book successfully generated at: {book_base_dir}")

            # Print final progress if tracking
            if progress_tracker:
                progress = progress_tracker.get_progress()
                logging.info(f"""
                Final Progress:
                - Chapters: {progress['chapters']['completed']}/{progress['chapters']['total']}
                - Sections: {progress['sections']['completed']}/{progress['sections']['total']}
                - Completion: {progress['percentage']:.1f}%
                """)

        except Exception as e:
            raise FileOperationError(f"Failed to export book: {e}")

    except GhostwriterException as e:
        logging.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.info("\nGeneration interrupted by user")
        return 2
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
