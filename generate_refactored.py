"""
Refactored generate module with BookGenerator class
"""
import json
import logging
import re
from collections.abc import Generator
from typing import Any, Dict, List, Optional

import prompts
from ai import callLLM
from events import Event, EventType, event_manager
from exceptions import ContentGenerationError, ValidationError


class BookGenerator:
    """Class to manage book generation with reduced code duplication"""

    def __init__(self, book: Dict[str, Any], history: List[Dict[str, str]]):
        """
        Initialize BookGenerator
        
        Args:
            book: Book data dictionary
            history: Conversation history
        """
        self.book = book
        self.history = history
        self.logger = logging.getLogger(__name__)

    def _limit_text(self, text: str, limit: int = 200) -> str:
        """Helper to limit text length for logging"""
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain additional content"""
        # Try to find JSON block with markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # If no match, return original text
        return text

    def _toc_to_text(self, toc: Dict, highlight_chapter: Optional[Dict] = None) -> str:
        """Convert TOC to text format"""
        text = ""
        for chapter in toc["chapters"]:
            text += f"{chapter['number']}. {chapter['title']}\n"
            if highlight_chapter is None or highlight_chapter == chapter:
                for section in chapter["sections"]:
                    text += f"    {chapter['number']}.{section['number']}. {section['title']}\n"
            else:
                text += f"    ({len(chapter['sections'])} sections here...)\n"
        return text

    def _generate_part(self,
                      part_name: str,
                      part_key: str,
                      prompt_func,
                      event_started: EventType,
                      event_completed: EventType,
                      event_data: Dict[str, Any] = None,
                      existing_check_key: Optional[str] = None,
                      wait_short: bool = False,
                      force_max: bool = False,
                      parse_json: bool = False,
                      validate_func: Optional[callable] = None,
                      update_history: bool = True) -> Generator:
        """
        Generic method to generate any part of the book
        
        Args:
            part_name: Human-readable name of the part
            part_key: Key in book dictionary
            prompt_func: Function to generate prompt
            event_started: Event type for start
            event_completed: Event type for completion
            event_data: Additional event data
            existing_check_key: Key to check if part exists
            wait_short: Use waitingShortAnswer
            force_max: Use forceMaximum
            parse_json: Parse response as JSON
            validate_func: Optional validation function
            update_history: Whether to update history
            
        Yields:
            Updated book
        """
        check_key = existing_check_key or part_key
        event_data = event_data or {}

        try:
            if not self._get_nested_value(check_key):
                self.logger.info(f"# Generating new {part_name}...")
                event_manager.emit(Event(event_started, event_data))

                # Call LLM
                response = callLLM(
                    prompt_func(),
                    self.history,
                    waitingShortAnswer=wait_short,
                    forceMaximum=force_max
                )

                # Process response
                if parse_json:
                    cleaned = self._extract_json(response)
                    try:
                        value = json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse {part_name} JSON: {e}")
                        self.logger.error(f"Problematic string: {response[:500]}")
                        raise ValidationError(f"Invalid {part_name} format: {e}")
                else:
                    value = response

                # Validate if needed
                if validate_func:
                    validate_func(value)

                # Set value in book
                self._set_nested_value(part_key, value)

                # Log result
                if parse_json:
                    self.logger.info(f"{part_name}:\n{self._format_value(value)}")
                else:
                    self.logger.info(f"# New {part_name}: {self._limit_text(value)}")

                # Emit completion event
                completion_data = {**event_data, part_key: value}
                if not parse_json:
                    completion_data[f'{part_key}_length'] = len(value)

                event_manager.emit(Event(event_completed, completion_data))
                yield self.book

            else:
                existing_value = self._get_nested_value(check_key)
                self.logger.info(f"# {part_name} already defined: {self._limit_text(str(existing_value))}")

                if update_history:
                    history_msg = f"{part_name}: {self._format_value(existing_value)}"
                    self.history.append({"role": "system", "content": history_msg})

            self.logger.info("")

        except ContentGenerationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to generate {part_name}: {e}")
            event_manager.emit(Event(EventType.GENERATION_FAILED, {
                'stage': part_name.lower().replace(' ', '_'),
                'error': str(e)
            }))
            raise ContentGenerationError(f"{part_name} generation failed: {e}")

    def _get_nested_value(self, key: str) -> Any:
        """Get nested value from book using dot notation"""
        parts = key.split('.')
        value = self.book
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def _set_nested_value(self, key: str, value: Any):
        """Set nested value in book using dot notation"""
        parts = key.split('.')
        target = self.book
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    def _format_value(self, value: Any) -> str:
        """Format value for display"""
        if isinstance(value, dict):
            if 'chapters' in value:  # TOC
                return self._toc_to_text(value)
            return json.dumps(value, indent=2)
        elif isinstance(value, list):
            return '\n'.join(str(item) for item in value)
        return str(value)

    def generate_title(self, original_title: str) -> Generator:
        """Generate book title"""
        yield from self._generate_part(
            part_name="title",
            part_key="title",
            prompt_func=lambda: prompts.title(original_title),
            event_started=EventType.TITLE_STARTED,
            event_completed=EventType.TITLE_COMPLETED,
            event_data={'original_title': original_title},
            wait_short=True
        )

        # Update history with title
        self.history[0]["content"] += f' The title of the book is "{self.book["title"]}"'

    def generate_toc(self, instructions: str) -> Generator:
        """Generate table of contents"""
        def validate_toc(toc):
            if not isinstance(toc, dict) or "chapters" not in toc:
                raise ValidationError("TOC must contain 'chapters' key")

        yield from self._generate_part(
            part_name="table of contents",
            part_key="toc",
            prompt_func=lambda: prompts.table_of_contents(instructions),
            event_started=EventType.TOC_STARTED,
            event_completed=EventType.TOC_COMPLETED,
            event_data={'instructions': instructions},
            parse_json=True,
            validate_func=validate_toc
        )

    def generate_summary(self, instructions: str) -> Generator:
        """Generate book summary"""
        yield from self._generate_part(
            part_name="summary",
            part_key="summary",
            prompt_func=lambda: prompts.summary(self.book, instructions),
            event_started=EventType.SUMMARY_STARTED,
            event_completed=EventType.SUMMARY_COMPLETED,
            event_data={'title': self.book.get('title')},
            wait_short=True,
            force_max=True
        )

        # Update history with summary
        self.history[0]["content"] += f' The summary of the book is: "{self.book["summary"]}"'

    def generate_chapter(self, chapter: Dict[str, Any]) -> Generator:
        """Generate chapter content"""
        chapter_desc = f'"{chapter["number"]}. {chapter["title"]}"'
        chapter_toc = self._toc_to_text(self.book["toc"], chapter)

        # Store reference to current chapter
        self.current_chapter = chapter

        # Generate topics
        if not chapter.get('topics'):
            self.logger.info(f"# Generating topics for chapter {chapter_desc}...")
            chapter["topics"] = callLLM(
                prompts.chapter_topics(self.book, chapter, chapter_toc),
                self.history,
                waitingShortAnswer=True
            )
            self.logger.info(f"Topics: {chapter['topics']}")

        # Generate content
        yield from self._generate_part(
            part_name=f"chapter {chapter_desc}",
            part_key=f"chapter_{chapter['number']}_content",
            prompt_func=lambda: prompts.chapter(self.book, chapter, chapter_toc),
            event_started=EventType.CHAPTER_STARTED,
            event_completed=EventType.CHAPTER_COMPLETED,
            event_data={
                'chapter_number': chapter["number"],
                'chapter_title': chapter["title"]
            },
            existing_check_key=f"chapter_{chapter['number']}_content",
            force_max=True
        )

        # Update chapter content in the actual location
        chapter["content"] = self._get_nested_value(f"chapter_{chapter['number']}_content")

    def generate_section(self, chapter: Dict[str, Any], section: Dict[str, Any]) -> Generator:
        """Generate section content"""
        section_desc = f'"{chapter["number"]}.{section["number"]}. {section["title"]}"'
        chapter_toc = self._toc_to_text(self.book["toc"], chapter)

        # Generate topics
        if not section.get('topics'):
            self.logger.info(f"# Generating topics for section {section_desc}...")
            section["topics"] = callLLM(
                prompts.section_topics(self.book, chapter, section, chapter_toc),
                self.history,
                waitingShortAnswer=True
            )
            self.logger.info(f"Topics: {section['topics']}")

        # Generate content
        yield from self._generate_part(
            part_name=f"section {section_desc}",
            part_key=f"section_{chapter['number']}_{section['number']}_content",
            prompt_func=lambda: prompts.section(self.book, chapter, section),
            event_started=EventType.SECTION_STARTED,
            event_completed=EventType.SECTION_COMPLETED,
            event_data={
                'chapter_number': chapter["number"],
                'section_number': section["number"],
                'section_title': section["title"]
            },
            existing_check_key=f"section_{chapter['number']}_{section['number']}_content",
            force_max=True
        )

        # Update section content in the actual location
        section["content"] = self._get_nested_value(f"section_{chapter['number']}_{section['number']}_content")


def write_book(book: Dict[str, Any], instructions: str,
               title: str, language: str = "English") -> Generator[Dict[str, Any], None, None]:
    """
    Main function to generate book using BookGenerator class
    
    Args:
        book: Book data dictionary
        instructions: Generation instructions
        title: Book title
        language: Book language
        
    Yields:
        Updated book after each generation step
    """
    history = [{
        "role": "system",
        "content": f"You are writing a book in {language}."
    }]

    # Create generator instance
    generator = BookGenerator(book, history)

    # Generate title
    yield from generator.generate_title(title)

    # Generate TOC
    yield from generator.generate_toc(instructions)

    # Generate summary
    yield from generator.generate_summary(instructions)

    # Generate chapters and sections
    for chapter in book["toc"]["chapters"]:
        yield from generator.generate_chapter(chapter)

        for section in chapter["sections"]:
            yield from generator.generate_section(chapter, section)

    # Final yield
    yield book


# Keep old functions for backward compatibility but mark as deprecated
def _write_title(book, original_title, history):
    """DEPRECATED: Use BookGenerator.generate_title() instead"""
    generator = BookGenerator(book, history)
    yield from generator.generate_title(original_title)

def _write_toc(book, instructions, history):
    """DEPRECATED: Use BookGenerator.generate_toc() instead"""
    generator = BookGenerator(book, history)
    yield from generator.generate_toc(instructions)

def _write_summary(book, instructions, history):
    """DEPRECATED: Use BookGenerator.generate_summary() instead"""
    generator = BookGenerator(book, history)
    yield from generator.generate_summary(instructions)

def _write_chapter(book, chapter, history):
    """DEPRECATED: Use BookGenerator.generate_chapter() instead"""
    generator = BookGenerator(book, history)
    yield from generator.generate_chapter(chapter)

def _write_section(book, chapter, section, history):
    """DEPRECATED: Use BookGenerator.generate_section() instead"""
    generator = BookGenerator(book, history)
    yield from generator.generate_section(chapter, section)
