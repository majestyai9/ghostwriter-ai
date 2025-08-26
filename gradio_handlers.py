"""
Gradio Handlers: Business logic for Gradio interface.
Separates UI components from actual functionality.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import traceback

from containers import get_container
from project_manager import ProjectManager
from style_templates import StyleManager
from character_tracker import CharacterDatabase
from events import event_manager, Event, EventType
from book_generator import BookGenerator
from export_formats import BookExporter

logger = logging.getLogger(__name__)


class GradioHandlers:
    """
    Handles business logic for Gradio interface.
    Separates UI concerns from application logic.
    """
    
    def __init__(self):
        """Initialize handlers with dependencies from DI container."""
        self.container = get_container()
        self.project_manager = self.container.project_manager()
        self.style_manager = StyleManager()
        self.character_db = None  # Initialized per project
        self.book_exporter = BookExporter()
        self.generation_active = False
        self.current_generation = None
        self.event_logs = []
        
        # Subscribe to events for progress tracking
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for real-time updates"""
        from events import event_manager, EventType
        
        def log_event(event):
            """Log generation events"""
            self.event_logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "data": event.data
            })
            logger.info(f"Event: {event.type} - {event.data}")
        
        # Subscribe to relevant events
        event_manager.subscribe(EventType.CHAPTER_COMPLETED, log_event)
        event_manager.subscribe(EventType.GENERATION_COMPLETED, log_event)
        event_manager.subscribe(EventType.GENERATION_FAILED, log_event)
    
    # ===== Project Management =====
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """Get list of all projects with metadata."""
        try:
            projects = self.project_manager.list_projects()
            formatted_projects = []
            for p in projects:
                # Convert ProjectMetadata to dict
                if hasattr(p, 'to_dict'):
                    project_dict = p.to_dict()
                else:
                    project_dict = p
                formatted_projects.append(self._format_project_info(project_dict))
            return formatted_projects
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return []
    
    def create_project(
        self,
        title: str,
        language: str,
        style: str,
        instructions: str,
        chapters: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Create a new book project.
        
        Returns:
            Tuple of (success, message, project_info)
        """
        try:
            # Validate inputs
            if not title or not title.strip():
                return False, "Title is required", {}
            
            if chapters < 1 or chapters > 100:
                return False, "Chapters must be between 1 and 100", {}
            
            # Create project
            project_id = self.project_manager.create_project(
                title=title.strip(),
                language=language or "English",
                style=style or "general",
                metadata={
                    "instructions": instructions,
                    "chapters": chapters,
                    "created_at": datetime.now().isoformat(),
                    "status": "draft"
                }
            )
            
            # Get project info from list
            projects = self.project_manager.list_projects()
            project_info = None
            for p in projects:
                p_id = p.project_id if hasattr(p, 'project_id') else p.get('id', '')
                if p_id == project_id:
                    project_info = p.to_dict() if hasattr(p, 'to_dict') else p
                    break
            
            if not project_info:
                # If not found in list, create minimal info
                project_info = {
                    'id': project_id,
                    'title': title,
                    'language': language,
                    'style': style,
                    'metadata': metadata
                }
            
            return True, f"Project '{title}' created successfully", self._format_project_info(project_info)
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return False, f"Error creating project: {str(e)}", {}
    
    def delete_project(self, project_id: str) -> Tuple[bool, str]:
        """Delete a project."""
        try:
            self.project_manager.delete_project(project_id)
            return True, "Project deleted successfully"
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            return False, f"Error deleting project: {str(e)}"
    
    def archive_project(self, project_id: str) -> Tuple[bool, str]:
        """Archive a project."""
        try:
            archive_path = self.project_manager.archive_project(project_id)
            return True, f"Project archived to {archive_path}"
        except Exception as e:
            logger.error(f"Error archiving project {project_id}: {e}")
            return False, f"Error archiving project: {str(e)}"
    
    def get_project_details(self, project_id: str) -> Dict[str, Any]:
        """Get detailed information about a project."""
        try:
            projects = self.project_manager.list_projects()
            for p in projects:
                p_id = p.project_id if hasattr(p, 'project_id') else p.get('id', '')
                if p_id == project_id:
                    project_dict = p.to_dict() if hasattr(p, 'to_dict') else p
                    return self._format_project_info(project_dict)
            return {}
        except Exception as e:
            logger.error(f"Error getting project details: {e}")
            return {}
    
    # ===== Book Generation =====
    
    async def generate_book(
        self,
        project_id: str,
        provider: str,
        model: str,
        temperature: float,
        enable_rag: bool,
        enable_quality: bool,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Generate a book for the specified project.
        
        Returns:
            Tuple of (success, message)
        """
        if self.generation_active:
            return False, "Generation already in progress"
        
        try:
            self.generation_active = True
            self.event_logs = []
            
            # Get project details
            project = self.project_manager.get_project(project_id)
            if not project:
                return False, "Project not found"
            
            # Create provider configuration
            provider_config = {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "api_key": self._get_api_key(provider)
            }
            
            # Create generation configuration
            generation_config = {
                "title": project["title"],
                "language": project.get("language", "English"),
                "style": project.get("style", "general"),
                "instructions": project.get("metadata", {}).get("instructions", ""),
                "chapters": project.get("metadata", {}).get("chapters", 10),
                "enable_rag": enable_rag,
                "enable_quality_checks": enable_quality,
                "project_id": project_id
            }
            
            # Initialize book generator
            generator = BookGenerator(
                provider_config=provider_config,
                generation_config=generation_config,
                project_manager=self.project_manager
            )
            
            # Set current generation for tracking
            self.current_generation = generator
            
            # Generate the book
            result = await generator.generate_async()
            
            if result["success"]:
                # Update project status
                self.project_manager.update_project(project_id, {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat()
                })
                return True, f"Book generated successfully: {result['book_path']}"
            else:
                return False, f"Generation failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error generating book: {e}\n{traceback.format_exc()}")
            return False, f"Error generating book: {str(e)}"
        finally:
            self.generation_active = False
            self.current_generation = None
    
    def stop_generation(self) -> Tuple[bool, str]:
        """Stop the current book generation."""
        if not self.generation_active or not self.current_generation:
            return False, "No generation in progress"
        
        try:
            self.current_generation.stop()
            self.generation_active = False
            self.current_generation = None
            return True, "Generation stopped"
        except Exception as e:
            logger.error(f"Error stopping generation: {e}")
            return False, f"Error stopping generation: {str(e)}"
    
    def get_generation_progress(self) -> Dict[str, Any]:
        """Get current generation progress."""
        if not self.generation_active or not self.current_generation:
            return {
                "active": False,
                "progress": 0,
                "message": "No generation in progress",
                "logs": self.event_logs[-20:]  # Last 20 log entries
            }
        
        return {
            "active": True,
            "progress": self.current_generation.get_progress(),
            "message": self.current_generation.get_status(),
            "logs": self.event_logs[-20:]
        }
    
    # ===== Character Management =====
    
    def list_characters(self, project_id: str) -> List[Dict[str, Any]]:
        """Get list of characters for a project."""
        try:
            db_path = self._get_character_db_path(project_id)
            if not db_path.exists():
                return []
            
            db = CharacterDatabase(db_path)
            characters = db.get_all_characters()
            return characters
        except Exception as e:
            logger.error(f"Error listing characters: {e}")
            return []
    
    def create_character(
        self,
        project_id: str,
        name: str,
        role: str,
        traits: Dict[str, float],
        description: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Create a new character."""
        try:
            from character_tracker import Character, CharacterRole, PersonalityTraits
            
            db_path = self._get_character_db_path(project_id)
            db = CharacterDatabase(db_path)
            
            # Convert role string to CharacterRole enum
            role_map = {
                "protagonist": CharacterRole.PROTAGONIST,
                "antagonist": CharacterRole.ANTAGONIST,
                "supporting": CharacterRole.SUPPORTING,
                "minor": CharacterRole.MINOR
            }
            char_role = role_map.get(role.lower(), CharacterRole.MINOR)
            
            # Create PersonalityTraits from OCEAN scores
            # Convert OCEAN scores to trait descriptions
            trait_descriptions = []
            if traits.get('openness', 50) > 70:
                trait_descriptions.append("Creative and open-minded")
            if traits.get('conscientiousness', 50) > 70:
                trait_descriptions.append("Organized and dependable")
            if traits.get('extraversion', 50) > 70:
                trait_descriptions.append("Outgoing and energetic")
            if traits.get('agreeableness', 50) > 70:
                trait_descriptions.append("Cooperative and trusting")
            if traits.get('neuroticism', 50) > 70:
                trait_descriptions.append("Emotionally reactive")
            
            personality = PersonalityTraits(
                traits=trait_descriptions if trait_descriptions else ["Balanced personality"]
            )
            
            # Create Character object
            character = Character(
                name=name,
                role=char_role,
                personality=personality,
                background=description
            )
            
            db.add_character(character)
            
            # Return character as dict for UI
            return True, f"Character '{name}' created", {
                "name": character.name,
                "role": character.role.value,
                "description": character.background,
                "personality_traits": traits
            }
            
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            return False, f"Error creating character: {str(e)}", {}
    
    def update_character(
        self,
        project_id: str,
        character_id: int,
        updates: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Update a character."""
        try:
            db_path = self._get_character_db_path(project_id)
            db = CharacterDatabase(db_path)
            
            db.update_character(character_id, updates)
            return True, "Character updated successfully"
            
        except Exception as e:
            logger.error(f"Error updating character: {e}")
            return False, f"Error updating character: {str(e)}"
    
    def delete_character(self, project_id: str, character_id: int) -> Tuple[bool, str]:
        """Delete a character."""
        try:
            db_path = self._get_character_db_path(project_id)
            db = CharacterDatabase(db_path)
            
            db.delete_character(character_id)
            return True, "Character deleted successfully"
            
        except Exception as e:
            logger.error(f"Error deleting character: {e}")
            return False, f"Error deleting character: {str(e)}"
    
    # ===== Style Management =====
    
    def list_styles(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available styles."""
        try:
            style_names = self.style_manager.list_styles(category)
            styles = []
            for name in style_names:
                style = self.style_manager.get_style(name)
                if style:
                    styles.append({
                        'name': style.name,
                        'category': style.category,
                        'description': f"{style.tone} writing with {style.vocabulary_level} vocabulary"
                    })
            return styles
        except Exception as e:
            logger.error(f"Error listing styles: {e}")
            return []
    
    def get_style_details(self, style_name: str) -> Dict[str, Any]:
        """Get details about a specific style."""
        try:
            style = self.style_manager.get_style(style_name)
            if style:
                return {
                    'name': style.name,
                    'category': style.category,
                    'description': f"{style.tone} writing with {style.vocabulary_level} vocabulary",
                    'tone': style.tone,
                    'vocabulary_level': style.vocabulary_level,
                    'pacing': style.paragraph_length,  # Use paragraph_length as pacing
                    'example': style.features[0] if style.features else "No example available"
                }
            return self.style_manager.get_style_info(style_name)
        except Exception as e:
            logger.error(f"Error getting style details: {e}")
            return {}
    
    def create_custom_style(
        self,
        name: str,
        base_style: str,
        modifications: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Create a custom style."""
        try:
            self.style_manager.create_custom_style(name, base_style, modifications)
            return True, f"Custom style '{name}' created successfully"
        except Exception as e:
            logger.error(f"Error creating custom style: {e}")
            return False, f"Error creating style: {str(e)}"
    
    # ===== Export Functions =====
    
    def export_book(
        self,
        project_id: str,
        format: str,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export a book to specified format.
        
        Returns:
            Tuple of (success, message, file_path)
        """
        try:
            # Get book data
            book_path = Path(f"projects/{project_id}/content/book.json")
            if not book_path.exists():
                return False, "Book not found. Generate the book first.", None
            
            with open(book_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Export to specified format
            export_path = self.book_exporter.export(book_data, format, metadata)
            
            return True, f"Book exported successfully", str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting book: {e}")
            return False, f"Error exporting: {str(e)}", None
    
    # ===== Analytics =====
    
    def get_book_statistics(self, project_id: str) -> Dict[str, Any]:
        """Get statistics about a generated book."""
        try:
            book_path = Path(f"projects/{project_id}/content/book.json")
            if not book_path.exists():
                return {"error": "Book not found"}
            
            with open(book_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Calculate statistics
            total_words = 0
            total_chapters = len(book_data.get("chapters", []))
            chapter_lengths = []
            
            for chapter in book_data.get("chapters", []):
                words = len(chapter.get("content", "").split())
                total_words += words
                chapter_lengths.append(words)
            
            return {
                "title": book_data.get("title", "Unknown"),
                "total_chapters": total_chapters,
                "total_words": total_words,
                "average_chapter_length": total_words // total_chapters if total_chapters > 0 else 0,
                "shortest_chapter": min(chapter_lengths) if chapter_lengths else 0,
                "longest_chapter": max(chapter_lengths) if chapter_lengths else 0,
                "language": book_data.get("language", "Unknown"),
                "style": book_data.get("style", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting book statistics: {e}")
            return {"error": str(e)}
    
    # ===== Helper Methods =====
    
    def _format_project_info(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Format project information for display."""
        # Handle both dict and object forms
        if isinstance(project, dict):
            return {
                "id": project.get("project_id", project.get("id", "")),
                "title": project.get("title", "Unknown"),
                "language": project.get("language", "English"),
                "style": project.get("style", "general"),
                "status": project.get("status", project.get("metadata", {}).get("status", "draft")),
                "created_at": project.get("created_at", project.get("metadata", {}).get("created_at", "")),
                "chapters": project.get("chapter_count", project.get("metadata", {}).get("chapters", 0)),
                "instructions": project.get("metadata", {}).get("instructions", "") if "metadata" in project else ""
            }
        else:
            # Handle object directly
            return {
                "id": getattr(project, "project_id", ""),
                "title": getattr(project, "title", "Unknown"),
                "language": getattr(project, "language", "English"),
                "style": getattr(project, "style", "general"),
                "status": getattr(project, "status", "draft"),
                "created_at": getattr(project, "created_at", ""),
                "chapters": getattr(project, "chapter_count", 0),
                "instructions": ""
            }
    
    def _get_character_db_path(self, project_id: str) -> Path:
        """Get path to character database for a project."""
        project_dir = self.project_manager.get_project_dir(project_id)
        return project_dir / "characters.db"
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        from app_config import settings
        
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"
        }
        
        key_name = key_mapping.get(provider.lower())
        if key_name:
            return getattr(settings, key_name, None)
        return None


# Global instance for easy access
handlers = GradioHandlers()