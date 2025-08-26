"""
Narrative Consistency Engine for maintaining story coherence across chapters.

This module ensures narrative consistency by tracking plot points, removing AI artifacts,
and validating story continuity throughout book generation.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlotPoint:
    """Represents a significant plot event in the story."""
    chapter: int
    description: str
    characters_involved: List[str]
    resolved: bool = False
    resolution_chapter: Optional[int] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class StoryContext:
    """Maintains the overall story context and continuity."""
    setting: str = ""
    time_period: str = ""
    main_conflict: str = ""
    themes: List[str] = field(default_factory=list)
    locations: Dict[str, str] = field(default_factory=dict)
    important_objects: Dict[str, str] = field(default_factory=dict)
    timeline: List[Tuple[int, str]] = field(default_factory=list)


class NarrativeConsistencyEngine:
    """
    Ensures narrative consistency throughout book generation.
    
    This engine tracks plot points, maintains story context, removes AI artifacts,
    and validates continuity between chapters.
    """
    
    # Common AI artifacts to remove
    AI_ARTIFACTS = [
        r"Here is Chapter \d+",
        r"Here's Chapter \d+",
        r"Chapter \d+:",
        r"I'll write",
        r"I will write",
        r"Let me create",
        r"I'll create",
        r"I've written",
        r"I've created",
        r"This chapter",
        r"In this chapter",
        r"The chapter begins",
        r"The chapter ends",
        r"\[Continue\]",
        r"\[End of Chapter\]",
        r"\[Chapter End\]",
        r"\*\*Chapter \d+",
        r"Word count:",
        r"Approximately \d+ words",
    ]
    
    def __init__(self, book_title: str, save_path: Optional[Path] = None):
        """
        Initialize the Narrative Consistency Engine.
        
        Args:
            book_title: Title of the book being generated
            save_path: Optional path to save consistency data
        """
        self.book_title = book_title
        self.save_path = save_path or Path(f"projects/{book_title.lower().replace(' ', '_')}/narrative_data.json")
        
        self.plot_points: List[PlotPoint] = []
        self.unresolved_plots: Set[int] = set()
        self.story_context = StoryContext()
        self.chapter_summaries: Dict[int, str] = {}
        self.character_mentions: Dict[str, List[int]] = defaultdict(list)
        self.location_mentions: Dict[str, List[int]] = defaultdict(list)
        self.consistency_warnings: List[str] = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load existing narrative data if available."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Restore plot points
                    self.plot_points = [
                        PlotPoint(**pp) for pp in data.get('plot_points', [])
                    ]
                    self.unresolved_plots = set(data.get('unresolved_plots', []))
                    # Restore story context
                    ctx_data = data.get('story_context', {})
                    self.story_context = StoryContext(**ctx_data)
                    # Restore other data
                    self.chapter_summaries = data.get('chapter_summaries', {})
                    self.character_mentions = defaultdict(list, data.get('character_mentions', {}))
                    self.location_mentions = defaultdict(list, data.get('location_mentions', {}))
                    
                logger.info(f"Loaded narrative data from {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not load narrative data: {e}")
    
    def _save_data(self) -> None:
        """Save narrative data for persistence."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'book_title': self.book_title,
                'plot_points': [
                    {
                        'chapter': pp.chapter,
                        'description': pp.description,
                        'characters_involved': pp.characters_involved,
                        'resolved': pp.resolved,
                        'resolution_chapter': pp.resolution_chapter,
                        'tags': pp.tags
                    }
                    for pp in self.plot_points
                ],
                'unresolved_plots': list(self.unresolved_plots),
                'story_context': {
                    'setting': self.story_context.setting,
                    'time_period': self.story_context.time_period,
                    'main_conflict': self.story_context.main_conflict,
                    'themes': self.story_context.themes,
                    'locations': self.story_context.locations,
                    'important_objects': self.story_context.important_objects,
                    'timeline': self.story_context.timeline
                },
                'chapter_summaries': self.chapter_summaries,
                'character_mentions': dict(self.character_mentions),
                'location_mentions': dict(self.location_mentions),
                'consistency_warnings': self.consistency_warnings
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved narrative data to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save narrative data: {e}")
    
    def remove_ai_artifacts(self, text: str) -> str:
        """
        Remove common AI-generated artifacts from text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text without AI artifacts
        """
        cleaned_text = text
        
        # Remove each artifact pattern
        for pattern in self.AI_ARTIFACTS:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove multiple consecutive newlines (leave max 2)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Remove leading/trailing whitespace from lines
        lines = cleaned_text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove empty lines at the beginning
        while cleaned_text.startswith('\n'):
            cleaned_text = cleaned_text[1:]
        
        return cleaned_text.strip()
    
    def validate_chapter_start(self, chapter_number: int, chapter_text: str) -> Tuple[bool, List[str]]:
        """
        Validate that a chapter starts appropriately without artifacts.
        
        Args:
            chapter_number: The chapter number
            chapter_text: The chapter content
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for AI artifacts at the beginning
        first_paragraph = chapter_text[:500] if len(chapter_text) > 500 else chapter_text
        
        for pattern in self.AI_ARTIFACTS:
            if re.search(pattern, first_paragraph, re.IGNORECASE):
                issues.append(f"Chapter {chapter_number} contains AI artifact: {pattern}")
        
        # Check if chapter starts with actual narrative content
        if not re.match(r'^[A-Z"]', chapter_text.strip()):
            issues.append(f"Chapter {chapter_number} doesn't start with narrative content")
        
        # Check for proper chapter flow from previous
        if chapter_number > 1 and chapter_number - 1 in self.chapter_summaries:
            # Ensure there's narrative continuity
            prev_summary = self.chapter_summaries[chapter_number - 1]
            if "cliffhanger" in prev_summary.lower() and "resolved" not in chapter_text[:1000].lower():
                issues.append(f"Chapter {chapter_number} doesn't address previous chapter's cliffhanger")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def track_plot_point(self, chapter: int, description: str, 
                        characters: List[str], tags: Optional[List[str]] = None) -> None:
        """
        Track a new plot point in the story.
        
        Args:
            chapter: Chapter number where plot point occurs
            description: Description of the plot point
            characters: Characters involved
            tags: Optional tags for categorization
        """
        plot_point = PlotPoint(
            chapter=chapter,
            description=description,
            characters_involved=characters,
            tags=tags or []
        )
        
        self.plot_points.append(plot_point)
        plot_index = len(self.plot_points) - 1
        self.unresolved_plots.add(plot_index)
        
        # Track character involvement
        for character in characters:
            self.character_mentions[character].append(chapter)
        
        logger.debug(f"Tracked plot point in chapter {chapter}: {description[:50]}...")
        self._save_data()
    
    def resolve_plot_point(self, plot_index: int, resolution_chapter: int) -> None:
        """
        Mark a plot point as resolved.
        
        Args:
            plot_index: Index of the plot point to resolve
            resolution_chapter: Chapter where resolution occurs
        """
        if 0 <= plot_index < len(self.plot_points):
            self.plot_points[plot_index].resolved = True
            self.plot_points[plot_index].resolution_chapter = resolution_chapter
            self.unresolved_plots.discard(plot_index)
            
            logger.debug(f"Resolved plot point {plot_index} in chapter {resolution_chapter}")
            self._save_data()
    
    def get_unresolved_plots(self) -> List[PlotPoint]:
        """Get all unresolved plot points."""
        return [self.plot_points[i] for i in self.unresolved_plots]
    
    def update_story_context(self, **kwargs) -> None:
        """
        Update the story context with new information.
        
        Args:
            **kwargs: Story context attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self.story_context, key):
                setattr(self.story_context, key, value)
        
        self._save_data()
    
    def add_chapter_summary(self, chapter: int, summary: str) -> None:
        """
        Add a summary for a chapter.
        
        Args:
            chapter: Chapter number
            summary: Summary of the chapter
        """
        self.chapter_summaries[chapter] = summary
        self._save_data()
    
    def check_continuity(self, chapter: int, content: str) -> List[str]:
        """
        Check for continuity issues in a chapter.
        
        Args:
            chapter: Chapter number
            content: Chapter content
            
        Returns:
            List of continuity warnings
        """
        warnings = []
        
        # Check for character consistency
        mentioned_chars = self._extract_character_names(content)
        for char in mentioned_chars:
            if char in self.character_mentions:
                last_mention = max(self.character_mentions[char])
                if chapter - last_mention > 5:
                    warnings.append(
                        f"Character '{char}' reappears after {chapter - last_mention} chapters absence"
                    )
        
        # Check for unresolved plots that should be addressed
        for plot_idx in self.unresolved_plots:
            plot = self.plot_points[plot_idx]
            if chapter - plot.chapter > 3:
                warnings.append(
                    f"Unresolved plot from chapter {plot.chapter}: {plot.description[:50]}..."
                )
        
        # Check timeline consistency
        time_markers = self._extract_time_markers(content)
        if time_markers and self.story_context.timeline:
            last_time = self.story_context.timeline[-1]
            for marker in time_markers:
                if self._is_time_inconsistent(last_time[1], marker):
                    warnings.append(f"Potential timeline inconsistency: '{marker}' vs previous '{last_time[1]}'")
        
        self.consistency_warnings.extend(warnings)
        return warnings
    
    def _extract_character_names(self, text: str) -> List[str]:
        """Extract character names from text using simple heuristics."""
        # Look for capitalized words that appear multiple times
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        # Return words that appear at least 3 times (likely character names)
        return [word for word, count in word_counts.items() if count >= 3]
    
    def _extract_time_markers(self, text: str) -> List[str]:
        """Extract time markers from text."""
        patterns = [
            r'\b(?:morning|afternoon|evening|night|dawn|dusk|midnight)\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
            r'\b(?:next day|following day|day after|week later|month later|year later)\b'
        ]
        
        markers = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            markers.extend(found)
        
        return markers
    
    def _is_time_inconsistent(self, prev_time: str, curr_time: str) -> bool:
        """Check if two time markers are potentially inconsistent."""
        # Simple heuristic: check if going backwards in time
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_order = ['dawn', 'morning', 'afternoon', 'evening', 'dusk', 'night', 'midnight']
        
        prev_lower = prev_time.lower()
        curr_lower = curr_time.lower()
        
        # Check day progression
        for i, day in enumerate(day_order):
            if day.lower() in prev_lower:
                for j, next_day in enumerate(day_order):
                    if next_day.lower() in curr_lower and j < i:
                        return True
        
        # Check time of day progression
        for i, time in enumerate(time_order):
            if time in prev_lower:
                for j, next_time in enumerate(time_order):
                    if next_time in curr_lower and j < i:
                        # Only inconsistent if no day change indicated
                        if 'next' not in curr_lower and 'following' not in curr_lower:
                            return True
        
        return False
    
    def generate_continuity_prompt(self, chapter: int) -> str:
        """
        Generate a prompt to maintain continuity for the next chapter.
        
        Args:
            chapter: Chapter number being generated
            
        Returns:
            Continuity prompt string
        """
        prompt_parts = []
        
        # Add story context
        if self.story_context.setting:
            prompt_parts.append(f"Setting: {self.story_context.setting}")
        if self.story_context.time_period:
            prompt_parts.append(f"Time Period: {self.story_context.time_period}")
        if self.story_context.main_conflict:
            prompt_parts.append(f"Main Conflict: {self.story_context.main_conflict}")
        
        # Add recent plot points
        recent_plots = [p for p in self.plot_points if chapter - p.chapter <= 3]
        if recent_plots:
            prompt_parts.append("\nRecent Plot Points:")
            for plot in recent_plots:
                status = "resolved" if plot.resolved else "unresolved"
                prompt_parts.append(f"- Chapter {plot.chapter} ({status}): {plot.description}")
        
        # Add unresolved plots that need attention
        urgent_plots = [self.plot_points[i] for i in self.unresolved_plots 
                       if chapter - self.plot_points[i].chapter >= 2]
        if urgent_plots:
            prompt_parts.append("\nUnresolved Plots Needing Attention:")
            for plot in urgent_plots:
                prompt_parts.append(f"- From Chapter {plot.chapter}: {plot.description}")
        
        # Add character tracking
        active_characters = [char for char, chapters in self.character_mentions.items() 
                           if chapters and max(chapters) >= chapter - 2]
        if active_characters:
            prompt_parts.append(f"\nActive Characters: {', '.join(active_characters)}")
        
        # Add previous chapter summary
        if chapter - 1 in self.chapter_summaries:
            prompt_parts.append(f"\nPrevious Chapter Summary:\n{self.chapter_summaries[chapter - 1]}")
        
        return "\n".join(prompt_parts)
    
    def update_chapter_summary(self, chapter: int, summary: str) -> None:
        """
        Update the summary for a specific chapter.
        
        Args:
            chapter: Chapter number
            summary: Chapter summary
        """
        self.chapter_summaries[chapter] = summary
        self._save_data()
        logger.debug(f"Updated summary for chapter {chapter}")
    
    def validate_book_consistency(self) -> Dict[str, any]:
        """
        Validate overall book consistency and return a report.
        
        Returns:
            Dictionary containing consistency metrics and issues
        """
        report = {
            'total_plot_points': len(self.plot_points),
            'unresolved_plots': len(self.unresolved_plots),
            'plot_resolution_rate': 0.0,
            'chapters_analyzed': len(self.chapter_summaries),
            'total_warnings': len(self.consistency_warnings),
            'character_count': len(self.character_mentions),
            'location_count': len(self.location_mentions),
            'issues': []
        }
        
        if self.plot_points:
            resolved = len([p for p in self.plot_points if p.resolved])
            report['plot_resolution_rate'] = (resolved / len(self.plot_points)) * 100
        
        # Check for unresolved major plots
        for idx in self.unresolved_plots:
            plot = self.plot_points[idx]
            if 'major' in plot.tags or 'main' in plot.tags:
                report['issues'].append(f"Major plot unresolved: {plot.description}")
        
        # Check for character disappearances
        for char, chapters in self.character_mentions.items():
            if chapters:
                first_appearance = min(chapters)
                last_appearance = max(chapters)
                total_chapters = max(self.chapter_summaries.keys()) if self.chapter_summaries else 0
                
                if last_appearance < total_chapters - 3:
                    report['issues'].append(
                        f"Character '{char}' disappeared after chapter {last_appearance}"
                    )
        
        # Add any consistency warnings
        report['issues'].extend(self.consistency_warnings)
        
        return report
    
    def integrate_with_character_manager(self, character_manager):
        """
        Integrate narrative consistency with advanced character manager.
        
        Args:
            character_manager: Instance of CharacterManager with evolution tracking
        """
        # Track character evolution in narrative
        for name, character in character_manager.characters.items():
            # Track all chapter appearances
            for chapter in character.chapter_appearances:
                if chapter not in self.character_mentions[name]:
                    self.character_mentions[name].append(chapter)
            
            # Track character relationships in story context
            for other_name, rel_data in character.relationships.items():
                if isinstance(rel_data, dict) and 'description' in rel_data:
                    self.story_context.important_objects[f"{name}-{other_name} relationship"] = rel_data['description']
            
            # Track character evolution events
            if hasattr(character, 'evolution_history'):
                for evolution in character.evolution_history:
                    # Add to timeline
                    growth_desc = evolution.growth_description[:100] if evolution.growth_description else "Character development"
                    self.story_context.timeline.append(
                        (evolution.chapter, f"{name}: {growth_desc}")
                    )
                    
                    # Track trauma events as plot points
                    if hasattr(evolution, 'trauma_events'):
                        for trauma in evolution.trauma_events:
                            self.track_plot_point(
                                chapter=evolution.chapter,
                                description=f"{name} experiences: {trauma}",
                                characters=[name],
                                tags=['trauma', 'character_development']
                            )
        
        self._save_data()
        logger.info(f"Integrated {len(character_manager.characters)} characters into narrative consistency")