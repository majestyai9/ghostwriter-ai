"""
Enhanced generate module with error handling and event system
"""
import json
import logging
import re
from collections.abc import Generator
from typing import Any, Dict

import prompts
from ai import callLLM
from events import Event, EventType, event_manager
from exceptions import ContentGenerationError, ValidationError


def _limit_text(text, limit=200):
    if len(text) > limit:
        return text[:limit] + "..."
    else:
        return text

def _extract_json(text: str) -> str:
    """
    Extract JSON from text that may contain additional content
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Clean JSON string
    """
    # Try to find JSON block with markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find raw JSON object or array
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # If no match, return original text (will likely fail JSON parsing)
    return text

def _toc_2_text(toc, highlightChapter=None):
    text = ""
    for chapter in toc["chapters"]:
        text += f"{chapter['number']}. {chapter['title']}\n"
        if highlightChapter is None or highlightChapter == chapter:
            for section in chapter["sections"]:
                text += f"    {chapter['number']}.{section['number']}. {section['title']}\n"
        else:
            text += f"    ({len(chapter['sections'])} sections here...)\n"
    return text

def _write_title(book, original_title, history):
    """Generate or validate book title with error handling"""
    try:
        if not book.get('title'):
            logging.info("# Generating a new title...")
            event_manager.emit(Event(EventType.TITLE_STARTED, {'original_title': original_title}))

            book["title"] = callLLM(
                prompts.title(original_title),
                history,
                waitingShortAnswer=True
            )

            logging.info(f"# New title: {book['title']}")
            event_manager.emit(Event(EventType.TITLE_COMPLETED, {'title': book['title']}))
            yield book
        else:
            logging.info(f"# Writing book {book['title']}...")
            history.append({"role": "system", "content": f'The title of the book is: "{book["title"]}".'})

        history[0]["content"] += f' The title of the book is "{book["title"]}"'
        logging.info("")

    except Exception as e:
        logging.error(f"Failed to generate title: {e}")
        event_manager.emit(Event(EventType.GENERATION_FAILED, {'stage': 'title', 'error': str(e)}))
        raise ContentGenerationError(f"Title generation failed: {e}")

def _write_toc(book, instructions, history):
    """Generate table of contents with error handling"""
    try:
        if not book.get('toc'):
            logging.info("# Generating a new table of contents...")
            event_manager.emit(Event(EventType.TOC_STARTED, {'instructions': instructions}))

            toc_str = callLLM(prompts.table_of_contents(instructions), history)

            try:
                # Clean the JSON string before parsing
                cleaned_json = _extract_json(toc_str)
                book["toc"] = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse TOC JSON: {e}")
                logging.error(f"Problematic string: {toc_str[:500]}")  # Log first 500 chars for debugging
                raise ValidationError(f"Invalid TOC format: {e}")

            # Validate TOC structure
            if not isinstance(book["toc"], dict) or "chapters" not in book["toc"]:
                raise ValidationError("TOC must contain 'chapters' key")

            logging.info(f"Table of Contents:\n{_toc_2_text(book['toc'])}")
            event_manager.emit(Event(EventType.TOC_COMPLETED, {'toc': book['toc']}))
            yield book
        else:
            logging.info("# Table of contents already defined...")
            history.append({"role": "system", "content": f"The book has the following table of contents:\n{_toc_2_text(book['toc'])}"})
            logging.info(f"Table of Contents:\n{_toc_2_text(book['toc'])}")

        logging.info("")

    except ContentGenerationError:
        raise
    except Exception as e:
        logging.error(f"Failed to generate TOC: {e}")
        event_manager.emit(Event(EventType.GENERATION_FAILED, {'stage': 'toc', 'error': str(e)}))
        raise ContentGenerationError(f"TOC generation failed: {e}")

def _write_summary(book, instructions, history):
    """Generate book summary with error handling"""
    try:
        if not book.get('summary'):
            logging.info("# Generating a new summary...")
            event_manager.emit(Event(EventType.SUMMARY_STARTED, {'title': book.get('title')}))

            book["summary"] = callLLM(
                prompts.summary(book, instructions),
                history,
                waitingShortAnswer=True,
                forceMaximum=True
            )

            logging.info(f'# New summary:\n{book["summary"]}')
            event_manager.emit(Event(EventType.SUMMARY_COMPLETED, {'summary': book['summary']}))
            yield book
        else:
            logging.info(f"# Summary is already defined: {_limit_text(book['summary'])}")

        history[0]["content"] += f' The summary of the book is: "{book["summary"]}"'
        logging.info("")

    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
        event_manager.emit(Event(EventType.GENERATION_FAILED, {'stage': 'summary', 'error': str(e)}))
        raise ContentGenerationError(f"Summary generation failed: {e}")

def _write_chapter(book, chapter, history):
    """Generate chapter content with error handling"""
    chapter_desc = f'"{chapter["number"]}. {chapter["title"]}"'

    try:
        if not chapter.get('content'):
            logging.info(f"# Generating a new content for chapter {chapter_desc}:")
            event_manager.emit(Event(EventType.CHAPTER_STARTED, {
                'chapter_number': chapter["number"],
                'chapter_title': chapter["title"]
            }))

            chapter_highlighted_toc = _toc_2_text(book["toc"], chapter)

            # Generate chapter topics
            chapter["topics"] = callLLM(
                prompts.chapter_topics(book, chapter, chapter_highlighted_toc),
                history,
                waitingShortAnswer=True
            )
            logging.info(f"Topics: {chapter['topics']}")

            # Generate chapter content
            chapter["content"] = callLLM(
                prompts.chapter(book, chapter, chapter_highlighted_toc),
                history,
                forceMaximum=True
            )
            logging.info(chapter["content"])

            event_manager.emit(Event(EventType.CHAPTER_COMPLETED, {
                'chapter_number': chapter["number"],
                'chapter_title': chapter["title"],
                'content_length': len(chapter["content"])
            }))
            yield book
        else:
            logging.info(f"# Content for chapter {chapter_desc} is already defined: {_limit_text(chapter['content'])}.\nChapter Topics: {chapter['topics']}")
            history.append({"role": "system", "content": f'The book content for chapter {chapter_desc} is:\n{chapter["content"]}'})

        logging.info("")

    except Exception as e:
        logging.error(f"Failed to generate chapter {chapter_desc}: {e}")
        event_manager.emit(Event(EventType.GENERATION_FAILED, {
            'stage': 'chapter',
            'chapter': chapter_desc,
            'error': str(e)
        }))
        raise ContentGenerationError(f"Chapter generation failed for {chapter_desc}: {e}")

def _write_section(book, chapter, section, history):
    """Generate section content with error handling"""
    section_desc = f'"{chapter["number"]}.{section["number"]}. {section["title"]}"'

    try:
        if not section.get('content'):
            logging.info(f"# Generating a new content for section {section_desc}:")
            event_manager.emit(Event(EventType.SECTION_STARTED, {
                'chapter_number': chapter["number"],
                'section_number': section["number"],
                'section_title': section["title"]
            }))

            chapter_highlighted_toc = _toc_2_text(book["toc"], chapter)

            # Generate section topics
            section["topics"] = callLLM(
                prompts.section_topics(book, chapter, section, chapter_highlighted_toc),
                history,
                waitingShortAnswer=True
            )
            logging.info(f"Topics: {section['topics']}")

            # Generate section content
            section["content"] = callLLM(
                prompts.section(book, chapter, section),
                history,
                forceMaximum=True
            )
            logging.info(section["content"])

            event_manager.emit(Event(EventType.SECTION_COMPLETED, {
                'chapter_number': chapter["number"],
                'section_number': section["number"],
                'section_title': section["title"],
                'content_length': len(section["content"])
            }))
            yield book
        else:
            logging.info(f"# Content for section {section_desc} is already defined: {_limit_text(section['content'])}.\nSection Topics: {section['topics']}")
            history.append({"role": "system", "content": f'The book content for section {section_desc} is:\n{section["content"]}'})

        logging.info("")

    except Exception as e:
        logging.error(f"Failed to generate section {section_desc}: {e}")
        event_manager.emit(Event(EventType.GENERATION_FAILED, {
            'stage': 'section',
            'section': section_desc,
            'error': str(e)
        }))
        raise ContentGenerationError(f"Section generation failed for {section_desc}: {e}")

def write_book(book: Dict[str, Any],
              original_title: str,
              instructions: str = "",
              language: str = "Brazilian Portuguese") -> Generator[Dict[str, Any], None, None]:
    """
    Enhanced book writing with error handling and event system
    
    Args:
        book: Existing book data (if any)
        original_title: Original book title
        instructions: Generation instructions
        language: Target language
        
    Yields:
        Updated book dictionary after each generation step
        
    Raises:
        ContentGenerationError: If any generation step fails
    """
    original_message = {
        "role": "system",
        "content": f"You are a book writer, writing a new book in {language} refered in the future as BookLanguage."
    }
    history = [original_message]

    # Emit generation started event
    event_manager.emit(Event(EventType.GENERATION_STARTED, {
        'title': original_title,
        'language': language,
        'instructions': instructions
    }))

    try:
        # Generate each component with error handling
        for b in _write_title(book, original_title, history):
            yield b

        for b in _write_toc(book, instructions, history):
            yield b

        for b in _write_summary(book, instructions, history):
            yield b

        # Generate chapters and sections
        for chapter in book['toc']['chapters']:
            for b in _write_chapter(book, chapter, history):
                yield b

            for section in chapter['sections']:
                for b in _write_section(book, chapter, section, history):
                    yield b

        logging.info("")
        logging.info("# Book is finished!")

        # Emit generation completed event
        event_manager.emit(Event(EventType.GENERATION_COMPLETED, {
            'title': book.get('title'),
            'chapters': len(book['toc']['chapters']),
            'total_sections': sum(len(ch['sections']) for ch in book['toc']['chapters'])
        }))

        yield book

    except ContentGenerationError:
        # Re-raise content generation errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        logging.error(f"Unexpected error during book generation: {e}")
        event_manager.emit(Event(EventType.GENERATION_FAILED, {
            'stage': 'unknown',
            'error': str(e)
        }))
        raise ContentGenerationError(f"Book generation failed: {e}")
