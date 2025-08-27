"""
Book generation orchestration module.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, List

from app_config import settings
from checkpoint_manager import CheckpointManager
from cli_handler import CLIHandler
from events import Event, EventType, event_manager
from exceptions import ContentGenerationError
from file_operations import FileOperations
from services.generation_service import GenerationService

# New quality enhancement systems
from narrative_consistency import NarrativeConsistencyEngine
from character_tracker import CharacterDatabase, Character, CharacterRole
from chapter_validator import ChapterValidator, ChapterQuality
from dialogue_enhancer import DialogueEnhancer
from plot_originality import PlotOriginalityValidator


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
        
        # Initialize quality enhancement systems
        self.narrative_engine: Optional[NarrativeConsistencyEngine] = None
        self.character_db: Optional[CharacterDatabase] = None
        self.chapter_validator = ChapterValidator()
        self.dialogue_enhancer = DialogueEnhancer()
        self.plot_validator = PlotOriginalityValidator()

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
        
        # Initialize quality enhancement systems with project-specific paths
        self._initialize_quality_systems(book_base_dir, title)

        # Generate title if needed
        if not book.get('title'):
            book = self._generate_title(book, title, language, book_json_path, book_base_dir)

        # Generate table of contents if needed
        if not book.get('toc'):
            book = self._generate_toc(book, instructions, book_json_path, book_base_dir)

        # Generate chapters
        book = self._generate_chapters(book, book_json_path, book_base_dir)

        # Save final character and plot data
        self._save_quality_data()
        
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
        Generate a single chapter with retry logic and quality validation.

        Args:
            book: Current book data
            chapter_index: Index of the chapter (0-based)
            chapter: Chapter data
            total_chapters: Total number of chapters
            book_json_path: Path to save book.json
            book_base_dir: Base directory for the book
        """
        chapter_num = chapter_index + 1
        max_generation_attempts = 3
        generation_attempt = 0
        
        while generation_attempt < max_generation_attempts:
            generation_attempt += 1
            try:
                self.logger.info(
                    f"Generating chapter {chapter_num}/{total_chapters}: "
                    f"{chapter.get('title', 'Untitled')} (attempt {generation_attempt})"
                )
                
                # Build context and requirements for generation
                continuity_context = self._build_continuity_context(chapter_num)
                quality_requirements = self.chapter_validator.generate_quality_enforcement_prompt()
                originality_requirements = self._build_originality_requirements()
                
                # Generate chapter with enhanced context
                raw_content = self.generation_service.generate_book_chapter(
                    settings.LLM_PROVIDER,
                    book,
                    chapter_index,
                    book_dir=book_base_dir,
                    continuity_context=continuity_context,
                    quality_requirements=quality_requirements,
                    originality_requirements=originality_requirements
                )
                
                # Process and validate the generated content
                processed_content = self._process_chapter_content(
                    raw_content, chapter_num, chapter
                )
                
                # Validate chapter quality and length
                validation_result = self.chapter_validator.validate_chapter(
                    processed_content, chapter_num
                )
                
                # If chapter is too short or poor quality, try expansion or regeneration
                if validation_result.expansion_needed or validation_result.quality == ChapterQuality.POOR:
                    if generation_attempt < max_generation_attempts:
                        self.logger.warning(
                            f"Chapter {chapter_num} needs improvement: "
                            f"{validation_result.issues}. Regenerating..."
                        )
                        continue  # Retry generation
                    else:
                        # Last attempt - try to expand existing content
                        processed_content = self._expand_chapter(
                            processed_content, 
                            validation_result.expansion_amount,
                            chapter_num,
                            book_base_dir
                        )
                
                # Apply final enhancements
                final_content = self._apply_final_enhancements(
                    processed_content, chapter_num
                )
                
                # Update tracking systems
                self._update_tracking_systems(final_content, chapter_num, chapter)
                
                chapter['content'] = final_content
                
                # Save after successful chapter generation
                self.file_ops.save_json_atomically(book, book_json_path)
                
                # Save quality tracking data
                self._save_quality_data()
                
                # Create periodic checkpoint
                self.checkpoint_manager.create_periodic_checkpoint(
                    book, book_base_dir, chapter_index, total_chapters
                )
                
                # Emit progress event
                event_manager.emit(Event(EventType.CHAPTER_COMPLETED, {
                    'chapter_number': chapter_num,
                    'chapter_title': chapter.get('title', 'Untitled'),
                    'total_chapters': total_chapters,
                    'word_count': len(final_content.split())
                }))
                
                break  # Success!
                
            except Exception as e:
                if generation_attempt >= max_generation_attempts:
                    # All attempts failed - use fallback handling
                    self._handle_generation_failure(e, book, chapter_num, chapter_index, 
                                                  chapter, book_json_path, book_base_dir)
                    break
                self.logger.warning(
                    f"Generation attempt {generation_attempt} failed: {e}. Retrying..."
                )

    def _handle_generation_failure(
        self,
        error: Exception,
        book: Dict[str, Any],
        chapter_num: int,
        chapter_index: int,
        chapter: Dict[str, Any],
        book_json_path: str,
        book_base_dir: str
    ) -> None:
        """Handle chapter generation failure with fallback strategies."""
        self.logger.error(
            f"Failed to generate chapter {chapter_num} after all attempts: {error}"
        )
        
        # Add to dead letter queue for later retry
        from dead_letter_queue import add_to_dlq, OperationType
        operation_id = add_to_dlq(
            OperationType.CHAPTER_GENERATION,
            {
                "chapter_number": chapter_num,
                "chapter_title": chapter.get('title', 'Untitled'),
                "book": book,
                "chapter_index": chapter_index,
                "book_base_dir": book_base_dir
            },
            error
        )
        self.logger.info(f"Added failed chapter to DLQ with ID: {operation_id}")
        
        # Try fallback strategies
        from fallback_strategies import FallbackContext, fallback_manager
        fallback_context = FallbackContext(
            original_prompt=f"Generate chapter {chapter_num}: {chapter.get('title', 'Untitled')}",
            original_provider=settings.LLM_PROVIDER,
            original_error=str(error),
            attempt_number=1,
            chapter_number=chapter_num,
            book_title=book.get('title', '')
        )
        
        fallback_content = fallback_manager.execute_fallback(fallback_context)
        
        if fallback_content:
            # Use fallback content
            chapter['content'] = fallback_content
            self.logger.warning(f"Using fallback content for chapter {chapter_num}")
            self.file_ops.save_json_atomically(book, book_json_path)
        else:
            # Complete failure - handle with placeholder
            self._handle_chapter_failure(
                book, chapter, chapter_num, chapter_index,
                book_json_path, book_base_dir, error
            )

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
    
    def _initialize_quality_systems(self, book_base_dir: str, title: str) -> None:
        """
        Initialize quality enhancement systems with project-specific paths.
        
        Args:
            book_base_dir: Base directory for the book
            title: Book title
        """
        from pathlib import Path
        
        quality_dir = Path(book_base_dir) / "quality_data"
        quality_dir.mkdir(exist_ok=True)
        
        # Initialize systems with project-specific paths
        self.narrative_engine = NarrativeConsistencyEngine(
            book_title=title,
            save_path=quality_dir / "narrative_data.json"
        )
        
        self.character_db = CharacterDatabase(
            db_path=quality_dir / "characters.db"
        )
        
        self.dialogue_enhancer = DialogueEnhancer(
            save_path=quality_dir / "dialogue_patterns.json"
        )
        
        self.plot_validator = PlotOriginalityValidator(
            save_path=quality_dir / "plot_tracker.json"
        )
        
        self.logger.info("Quality enhancement systems initialized")
    
    def _build_continuity_context(self, chapter_num: int) -> str:
        """
        Build continuity context for chapter generation.
        
        Args:
            chapter_num: Current chapter number
            
        Returns:
            Continuity context string
        """
        if not self.narrative_engine:
            return ""
        
        return self.narrative_engine.generate_continuity_prompt(chapter_num)
    
    def _build_originality_requirements(self) -> str:
        """
        Build originality requirements based on tracked plot elements.
        
        Returns:
            Originality requirements string
        """
        if not self.plot_validator:
            return ""
        
        report = self.plot_validator.generate_originality_report()
        
        requirements = []
        if report.overused_devices:
            requirements.append(
                f"AVOID these overused plot devices: {', '.join([d.value for d in report.overused_devices])}"
            )
        
        if report.suggestions:
            requirements.append(
                f"Consider these alternatives: {'; '.join(report.suggestions[:3])}"
            )
        
        return "\n".join(requirements) if requirements else ""
    
    def _process_chapter_content(
        self, 
        raw_content: str, 
        chapter_num: int,
        chapter_data: Dict[str, Any]
    ) -> str:
        """
        Process raw chapter content through quality systems.
        
        Args:
            raw_content: Raw generated chapter content
            chapter_num: Chapter number
            chapter_data: Chapter metadata
            
        Returns:
            Processed chapter content
        """
        # Remove AI artifacts
        if self.narrative_engine:
            content = self.narrative_engine.remove_ai_artifacts(raw_content)
        else:
            content = raw_content
        
        # Validate narrative consistency
        if self.narrative_engine:
            is_valid, issues = self.narrative_engine.validate_chapter_start(
                chapter_num, content
            )
            if not is_valid:
                self.logger.warning(
                    f"Chapter {chapter_num} has consistency issues: {issues}"
                )
        
        # Track plot elements
        if self.plot_validator:
            self.plot_validator.track_chapter(chapter_num, content)
        
        # Track characters
        if self.character_db:
            self._extract_and_track_characters(content, chapter_num)
        
        # Track chapter summary
        if self.narrative_engine:
            summary = self._generate_chapter_summary(content, chapter_data)
            self.narrative_engine.update_chapter_summary(chapter_num, summary)
        
        return content
    
    def _apply_final_enhancements(
        self,
        content: str,
        chapter_num: int
    ) -> str:
        """
        Apply final dialogue and quality enhancements.
        
        Args:
            content: Chapter content
            chapter_num: Chapter number
            
        Returns:
            Enhanced chapter content
        """
        # Enhance dialogue
        if self.dialogue_enhancer:
            content = self.dialogue_enhancer.enhance_chapter_dialogue(
                content, chapter_num
            )
        
        # Final artifact removal pass
        if self.narrative_engine:
            content = self.narrative_engine.remove_ai_artifacts(content)
        
        return content
    
    def _update_tracking_systems(
        self,
        content: str,
        chapter_num: int,
        chapter_data: Dict[str, Any]
    ) -> None:
        """
        Update all tracking systems with chapter data.
        
        Args:
            content: Final chapter content
            chapter_num: Chapter number
            chapter_data: Chapter metadata
        """
        # Update continuity tracking
        if self.narrative_engine:
            self.narrative_engine.check_continuity(chapter_num, content)
        
        # Update character appearances
        if self.character_db:
            characters = self._extract_character_names(content)
            for char_name in characters:
                self.character_db.update_chapter_appearance(char_name, chapter_num)
        
        # Track plot complexity
        if self.plot_validator:
            self.plot_validator.validate_plot_diversity(chapter_num)
    
    def _expand_chapter(
        self,
        content: str,
        words_needed: int,
        chapter_num: int,
        book_base_dir: str
    ) -> str:
        """
        Expand a chapter to meet minimum length requirements.
        
        Args:
            content: Current chapter content
            words_needed: Number of additional words needed
            chapter_num: Chapter number
            book_base_dir: Base directory for the book
            
        Returns:
            Expanded chapter content
        """
        self.logger.info(
            f"Expanding chapter {chapter_num} by {words_needed} words"
        )
        
        # Generate expansion prompt
        expansion_prompt = self.chapter_validator.generate_expansion_prompt(
            content, words_needed
        )
        
        # Add current content as context
        full_prompt = f"""
        Current chapter content:
        {content}
        
        {expansion_prompt}
        
        Requirements:
        - Seamlessly integrate new content
        - Maintain character voice and style
        - Add substantive scenes, not filler
        - Focus on sensory details and emotional depth
        """
        
        try:
            # Generate expansion
            expansion = self.generation_service.generate_text(
                settings.LLM_PROVIDER,
                full_prompt
            )
            
            # Merge expansion with original content
            expanded_content = self._merge_expansion(content, expansion)
            
            return expanded_content
            
        except Exception as e:
            self.logger.error(f"Failed to expand chapter {chapter_num}: {e}")
            return content  # Return original if expansion fails
    
    def _merge_expansion(self, original: str, expansion: str) -> str:
        """
        Intelligently merge expansion with original content.
        
        Args:
            original: Original chapter content
            expansion: Expansion content
            
        Returns:
            Merged content
        """
        # Simple merge - in practice, this would be more sophisticated
        # to identify optimal insertion points
        return f"{original}\n\n{expansion}"
    
    def _extract_character_names(self, text: str) -> List[str]:
        """
        Extract character names from text.
        
        Args:
            text: Text to extract names from
            
        Returns:
            List of character names
        """
        import re
        from collections import Counter
        
        # Find capitalized words that might be names
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Count occurrences
        word_counts = Counter(words)
        
        # Filter common words and return likely names
        common_words = {
            'The', 'He', 'She', 'It', 'They', 'We', 'You', 'I',
            'This', 'That', 'These', 'Those', 'There', 'Here',
            'When', 'Where', 'What', 'Who', 'Why', 'How',
            'And', 'But', 'Or', 'If', 'Then', 'Because', 'After', 'Before'
        }
        
        names = [
            word for word, count in word_counts.items()
            if count >= 3 and word not in common_words
        ]
        
        return names
    
    def _extract_and_track_characters(
        self,
        content: str,
        chapter_num: int
    ) -> None:
        """
        Extract and track characters from chapter content.
        
        Args:
            content: Chapter content
            chapter_num: Chapter number
        """
        if not self.character_db:
            return
        
        names = self._extract_character_names(content)
        
        for name in names:
            # Check if character exists
            character = self.character_db.get_character(name)
            
            if not character:
                # Create new character entry
                character = Character(
                    name=name,
                    first_appearance=chapter_num,
                    role=CharacterRole.MINOR  # Default role
                )
                self.character_db.add_character(character)
            
            # Update appearance
            self.character_db.update_chapter_appearance(name, chapter_num)
    
    def _generate_chapter_summary(
        self,
        content: str,
        chapter_data: Dict[str, Any]
    ) -> str:
        """
        Generate a summary of the chapter for continuity tracking.
        
        Args:
            content: Chapter content
            chapter_data: Chapter metadata
            
        Returns:
            Chapter summary
        """
        # Extract first and last paragraphs for context
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return f"Chapter: {chapter_data.get('title', 'Untitled')}"
        
        # Simple summary based on chapter data and content snippets
        summary_parts = [
            f"Chapter: {chapter_data.get('title', 'Untitled')}"
        ]
        
        # Add key topics if available
        if 'topics' in chapter_data:
            summary_parts.append(f"Topics: {chapter_data['topics']}")
        
        # Add character mentions
        names = self._extract_character_names(content)
        if names:
            summary_parts.append(f"Characters: {', '.join(names[:5])}")
        
        # Add brief content indicator
        word_count = len(content.split())
        summary_parts.append(f"Length: {word_count} words")
        
        return " | ".join(summary_parts)
    
    def _save_quality_data(self) -> None:
        """Save all quality tracking data."""
        try:
            if self.narrative_engine:
                self.narrative_engine._save_data()
            
            if self.dialogue_enhancer:
                self.dialogue_enhancer._save_patterns()
            
            if self.plot_validator:
                self.plot_validator._save_data()
            
            self.logger.debug("Quality tracking data saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to save quality data: {e}")
    
    async def generate_async(
        self,
        book_base_dir: str = None,
        title: str = None,
        instructions: str = None,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Async wrapper for generate_book method to support Gradio's async operations.
        
        This method runs the synchronous generate_book in a thread pool to prevent
        blocking the event loop.
        
        Args:
            book_base_dir: Base directory for book files
            title: Book title
            instructions: Generation instructions  
            language: Target language
            
        Returns:
            Dict with success status and generated book data or error message
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        try:
            # Get default values if not provided
            if not all([book_base_dir, title, instructions, language]):
                # Try to get from current context/project if available
                from containers import get_container
                container = get_container()
                project_manager = container.resolve('ProjectManager')
                
                if project_manager and project_manager.current_project:
                    project = project_manager.get_project(project_manager.current_project)
                    book_base_dir = book_base_dir or str(project_manager.get_project_dir(project.id) / "content")
                    title = title or project.title
                    instructions = instructions or project.get('instructions', '')
                    language = language or project.get('language', 'English')
            
            # Validate required parameters
            if not all([book_base_dir, title, instructions, language]):
                return {
                    "success": False,
                    "error": "Missing required parameters for book generation"
                }
            
            # Run the synchronous method in a thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                book = await loop.run_in_executor(
                    executor,
                    self.generate_book,
                    book_base_dir,
                    title,
                    instructions,
                    language
                )
            
            return {
                "success": True,
                "book": book,
                "message": f"Successfully generated book: {title}"
            }
            
        except Exception as e:
            self.logger.error(f"Async book generation failed: {e}")
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }