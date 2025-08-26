"""
Gradio State Management: Manages application state for Gradio interface.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ProjectState:
    """State for current project."""
    project_id: Optional[str] = None
    project_info: Dict[str, Any] = field(default_factory=dict)
    is_loaded: bool = False
    last_updated: Optional[datetime] = None


@dataclass
class GenerationState:
    """State for book generation process."""
    is_active: bool = False
    progress: float = 0.0
    current_chapter: int = 0
    total_chapters: int = 0
    status_message: str = ""
    logs: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


@dataclass
class UIState:
    """State for UI components."""
    active_tab: str = "Projects"
    show_new_project_form: bool = False
    show_character_form: bool = False
    show_style_form: bool = False
    selected_project_id: Optional[str] = None
    selected_character_id: Optional[int] = None
    selected_style: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SettingsState:
    """State for application settings."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    enable_rag: bool = True
    enable_quality_checks: bool = True
    api_keys: Dict[str, Optional[str]] = field(default_factory=dict)
    cache_backend: str = "memory"
    

class GradioSessionState:
    """
    Manages session state for Gradio interface.
    Provides centralized state management for the application.
    """
    
    def __init__(self):
        """Initialize session state."""
        self.project = ProjectState()
        self.generation = GenerationState()
        self.ui = UIState()
        self.settings = SettingsState()
        
        # Cache for frequently accessed data
        self._cache = {
            "projects_list": [],
            "styles_list": [],
            "characters_list": [],
            "last_refresh": None
        }
    
    # ===== Project State Management =====
    
    def set_current_project(self, project_id: str, project_info: Dict[str, Any]):
        """Set the current active project."""
        self.project.project_id = project_id
        self.project.project_info = project_info
        self.project.is_loaded = True
        self.project.last_updated = datetime.now()
        logger.info(f"Current project set to: {project_id}")
    
    def clear_current_project(self):
        """Clear the current project state."""
        self.project = ProjectState()
        logger.info("Current project cleared")
    
    def get_current_project_id(self) -> Optional[str]:
        """Get the current project ID."""
        return self.project.project_id
    
    def is_project_loaded(self) -> bool:
        """Check if a project is currently loaded."""
        return self.project.is_loaded
    
    # ===== Generation State Management =====
    
    def start_generation(self, total_chapters: int):
        """Mark generation as started."""
        self.generation = GenerationState(
            is_active=True,
            total_chapters=total_chapters,
            status_message="Starting generation...",
            start_time=datetime.now()
        )
        logger.info(f"Generation started with {total_chapters} chapters")
    
    def update_generation_progress(
        self,
        progress: float,
        current_chapter: int,
        status_message: str
    ):
        """Update generation progress."""
        self.generation.progress = progress
        self.generation.current_chapter = current_chapter
        self.generation.status_message = status_message
        
        # Estimate completion time
        if progress > 0 and self.generation.start_time:
            elapsed = (datetime.now() - self.generation.start_time).total_seconds()
            total_estimated = elapsed / (progress / 100)
            remaining = total_estimated - elapsed
            self.generation.estimated_completion = datetime.now() + timedelta(seconds=remaining)
    
    def add_generation_log(self, message: str):
        """Add a log message to generation logs."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.generation.logs.append(log_entry)
        
        # Keep only last 100 logs
        if len(self.generation.logs) > 100:
            self.generation.logs = self.generation.logs[-100:]
    
    def stop_generation(self):
        """Mark generation as stopped."""
        self.generation.is_active = False
        self.generation.status_message = "Generation stopped"
        logger.info("Generation stopped")
    
    def complete_generation(self):
        """Mark generation as completed."""
        self.generation.is_active = False
        self.generation.progress = 100.0
        self.generation.status_message = "Generation completed successfully"
        logger.info("Generation completed")
    
    def is_generation_active(self) -> bool:
        """Check if generation is currently active."""
        return self.generation.is_active
    
    # ===== UI State Management =====
    
    def set_active_tab(self, tab_name: str):
        """Set the active tab in the UI."""
        self.ui.active_tab = tab_name
        logger.debug(f"Active tab set to: {tab_name}")
    
    def toggle_new_project_form(self):
        """Toggle the new project form visibility."""
        self.ui.show_new_project_form = not self.ui.show_new_project_form
    
    def toggle_character_form(self):
        """Toggle the character form visibility."""
        self.ui.show_character_form = not self.ui.show_character_form
    
    def toggle_style_form(self):
        """Toggle the style form visibility."""
        self.ui.show_style_form = not self.ui.show_style_form
    
    def select_project(self, project_id: str):
        """Select a project in the UI."""
        self.ui.selected_project_id = project_id
    
    def select_character(self, character_id: int):
        """Select a character in the UI."""
        self.ui.selected_character_id = character_id
    
    def select_style(self, style_name: str):
        """Select a style in the UI."""
        self.ui.selected_style = style_name
    
    def set_filter(self, filter_name: str, value: Any):
        """Set a UI filter."""
        self.ui.filters[filter_name] = value
    
    def clear_filters(self):
        """Clear all UI filters."""
        self.ui.filters = {}
    
    # ===== Settings State Management =====
    
    def update_provider_settings(
        self,
        provider: str,
        model: str,
        temperature: float
    ):
        """Update provider settings."""
        self.settings.provider = provider
        self.settings.model = model
        self.settings.temperature = temperature
        logger.info(f"Provider settings updated: {provider}/{model} @ {temperature}")
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider."""
        self.settings.api_keys[provider] = api_key
        logger.info(f"API key set for provider: {provider}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.settings.api_keys.get(provider)
    
    def enable_feature(self, feature: str, enabled: bool):
        """Enable or disable a feature."""
        if feature == "rag":
            self.settings.enable_rag = enabled
        elif feature == "quality_checks":
            self.settings.enable_quality_checks = enabled
        logger.info(f"Feature '{feature}' set to: {enabled}")
    
    # ===== Cache Management =====
    
    def cache_projects_list(self, projects: List[Dict[str, Any]]):
        """Cache the projects list."""
        self._cache["projects_list"] = projects
        self._cache["last_refresh"] = datetime.now()
    
    def get_cached_projects(self) -> List[Dict[str, Any]]:
        """Get cached projects list."""
        return self._cache.get("projects_list", [])
    
    def cache_styles_list(self, styles: List[Dict[str, Any]]):
        """Cache the styles list."""
        self._cache["styles_list"] = styles
    
    def get_cached_styles(self) -> List[Dict[str, Any]]:
        """Get cached styles list."""
        return self._cache.get("styles_list", [])
    
    def cache_characters_list(self, characters: List[Dict[str, Any]]):
        """Cache the characters list for current project."""
        self._cache["characters_list"] = characters
    
    def get_cached_characters(self) -> List[Dict[str, Any]]:
        """Get cached characters list."""
        return self._cache.get("characters_list", [])
    
    def should_refresh_cache(self, cache_timeout_minutes: int = 5) -> bool:
        """Check if cache should be refreshed."""
        last_refresh = self._cache.get("last_refresh")
        if not last_refresh:
            return True
        
        from datetime import timedelta
        return datetime.now() - last_refresh > timedelta(minutes=cache_timeout_minutes)
    
    # ===== State Persistence =====
    
    def save_state(self, filepath: str):
        """Save current state to file."""
        try:
            state_dict = {
                "project": {
                    "project_id": self.project.project_id,
                    "project_info": self.project.project_info,
                    "is_loaded": self.project.is_loaded
                },
                "settings": {
                    "provider": self.settings.provider,
                    "model": self.settings.model,
                    "temperature": self.settings.temperature,
                    "enable_rag": self.settings.enable_rag,
                    "enable_quality_checks": self.settings.enable_quality_checks,
                    "cache_backend": self.settings.cache_backend
                },
                "ui": {
                    "active_tab": self.ui.active_tab,
                    "selected_project_id": self.ui.selected_project_id,
                    "selected_style": self.ui.selected_style,
                    "filters": self.ui.filters
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, default=str)
            
            logger.info(f"State saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, filepath: str):
        """Load state from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            # Restore project state
            if "project" in state_dict:
                self.project.project_id = state_dict["project"].get("project_id")
                self.project.project_info = state_dict["project"].get("project_info", {})
                self.project.is_loaded = state_dict["project"].get("is_loaded", False)
            
            # Restore settings
            if "settings" in state_dict:
                settings = state_dict["settings"]
                self.settings.provider = settings.get("provider", "openai")
                self.settings.model = settings.get("model", "gpt-4")
                self.settings.temperature = settings.get("temperature", 0.7)
                self.settings.enable_rag = settings.get("enable_rag", True)
                self.settings.enable_quality_checks = settings.get("enable_quality_checks", True)
                self.settings.cache_backend = settings.get("cache_backend", "memory")
            
            # Restore UI state
            if "ui" in state_dict:
                ui = state_dict["ui"]
                self.ui.active_tab = ui.get("active_tab", "Projects")
                self.ui.selected_project_id = ui.get("selected_project_id")
                self.ui.selected_style = ui.get("selected_style")
                self.ui.filters = ui.get("filters", {})
            
            logger.info(f"State loaded from {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"State file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def reset_state(self):
        """Reset all state to defaults."""
        self.__init__()
        logger.info("All state reset to defaults")


# Global state instance for Gradio session
session_state = GradioSessionState()


from datetime import timedelta  # Import at the end to avoid circular imports