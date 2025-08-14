"""
Character Development System for Fiction Writing
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CharacterRole(Enum):
    """Character role types"""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    SUPPORTING = "supporting"
    MINOR = "minor"
    MENTOR = "mentor"
    LOVE_INTEREST = "love_interest"
    SIDEKICK = "sidekick"
    RIVAL = "rival"

class RelationshipType(Enum):
    """Relationship types between characters"""
    FAMILY = "family"
    ROMANTIC = "romantic"
    FRIENDSHIP = "friendship"
    PROFESSIONAL = "professional"
    RIVALRY = "rivalry"
    MENTORSHIP = "mentorship"
    ENEMY = "enemy"

@dataclass
class CharacterProfile:
    """Complete character profile"""
    # Basic Information
    name: str
    role: CharacterRole
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None

    # Physical Description
    appearance: Dict[str, str] = field(default_factory=dict)
    distinguishing_features: List[str] = field(default_factory=list)

    # Personality
    personality_traits: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)

    # Background
    backstory: str = ""
    family_background: str = ""
    education: str = ""
    key_life_events: List[Dict[str, str]] = field(default_factory=list)

    # Speech and Behavior
    speech_pattern: str = ""
    vocabulary_level: str = "moderate"  # simple, moderate, advanced
    catchphrases: List[str] = field(default_factory=list)
    mannerisms: List[str] = field(default_factory=list)
    accent_dialect: str = ""

    # Story Arc
    character_arc: str = ""
    goals: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    growth_points: List[Dict[str, Any]] = field(default_factory=list)

    # Relationships
    relationships: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    chapter_appearances: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['role'] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterProfile':
        """Create from dictionary"""
        data['role'] = CharacterRole(data['role'])
        return cls(**data)

@dataclass
class DialogueEntry:
    """Track dialogue for consistency"""
    character: str
    chapter: int
    text: str
    context: str
    emotion: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class PlotPoint:
    """Track plot points and events"""
    chapter: int
    event: str
    characters_involved: List[str]
    impact: str  # major, minor, critical
    resolved: bool = False
    resolution_chapter: Optional[int] = None
    foreshadowing_refs: List[int] = field(default_factory=list)

class CharacterManager:
    """Manages character development and consistency"""

    def __init__(self, project_dir: Path = None):
        """
        Initialize character manager
        
        Args:
            project_dir: Project directory for character data
        """
        self.project_dir = project_dir or Path(".")
        self.characters_dir = self.project_dir / "characters"
        self.characters_dir.mkdir(exist_ok=True)

        self.characters: Dict[str, CharacterProfile] = {}
        self.dialogue_history: List[DialogueEntry] = []
        self.plot_tracker: List[PlotPoint] = []
        self.relationship_matrix: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)
        self._load_characters()

    def _load_characters(self):
        """Load existing characters from disk"""
        for char_file in self.characters_dir.glob("*.json"):
            try:
                with open(char_file, encoding='utf-8') as f:
                    data = json.load(f)
                    char = CharacterProfile.from_dict(data)
                    self.characters[char.name] = char
            except Exception as e:
                self.logger.error(f"Failed to load character {char_file}: {e}")

    def create_character(self,
                        name: str,
                        role: CharacterRole,
                        **kwargs) -> CharacterProfile:
        """
        Create a new character
        
        Args:
            name: Character name
            role: Character role
            **kwargs: Additional character attributes
            
        Returns:
            CharacterProfile object
        """
        character = CharacterProfile(name=name, role=role, **kwargs)
        self.characters[name] = character
        self._save_character(character)

        self.logger.info(f"Created character: {name} ({role.value})")
        return character

    def _save_character(self, character: CharacterProfile):
        """Save character to disk"""
        char_file = self.characters_dir / f"{character.name.lower().replace(' ', '_')}.json"
        with open(char_file, 'w', encoding='utf-8') as f:
            json.dump(character.to_dict(), f, indent=2, ensure_ascii=False)

    def update_character(self, name: str, **updates):
        """Update character attributes"""
        if name not in self.characters:
            raise ValueError(f"Character {name} not found")

        character = self.characters[name]
        for key, value in updates.items():
            if hasattr(character, key):
                setattr(character, key, value)

        character.modified_at = datetime.now().isoformat()
        self._save_character(character)

    def add_relationship(self,
                        char1: str,
                        char2: str,
                        relationship_type: RelationshipType,
                        description: str = "",
                        dynamics: str = ""):
        """
        Add or update relationship between characters
        
        Args:
            char1: First character name
            char2: Second character name
            relationship_type: Type of relationship
            description: Relationship description
            dynamics: How they interact
        """
        if char1 not in self.characters or char2 not in self.characters:
            raise ValueError("Both characters must exist")

        # Update both characters' relationship data
        self.characters[char1].relationships[char2] = {
            'type': relationship_type.value,
            'description': description,
            'dynamics': dynamics
        }

        self.characters[char2].relationships[char1] = {
            'type': relationship_type.value,
            'description': description,
            'dynamics': dynamics
        }

        # Update relationship matrix
        key = tuple(sorted([char1, char2]))
        self.relationship_matrix[key] = {
            'type': relationship_type,
            'description': description,
            'dynamics': dynamics,
            'interactions': []
        }

        self._save_character(self.characters[char1])
        self._save_character(self.characters[char2])

    def generate_dialogue(self,
                         character_name: str,
                         context: str,
                         emotion: str = "neutral",
                         target_character: str = None) -> str:
        """
        Generate character-appropriate dialogue
        
        Args:
            character_name: Speaking character
            context: Scene context
            emotion: Character's emotional state
            target_character: Who they're speaking to
            
        Returns:
            Dialogue prompt for AI generation
        """
        if character_name not in self.characters:
            raise ValueError(f"Character {character_name} not found")

        character = self.characters[character_name]

        # Build dialogue prompt based on character profile
        prompt = f"Generate dialogue for {character_name}:\n"
        prompt += f"Role: {character.role.value}\n"
        prompt += f"Personality: {', '.join(character.personality_traits)}\n"
        prompt += f"Speech pattern: {character.speech_pattern}\n"
        prompt += f"Vocabulary level: {character.vocabulary_level}\n"

        if character.accent_dialect:
            prompt += f"Accent/Dialect: {character.accent_dialect}\n"

        if character.catchphrases:
            prompt += f"Catchphrases: {', '.join(character.catchphrases)}\n"

        prompt += f"\nContext: {context}\n"
        prompt += f"Emotion: {emotion}\n"

        if target_character and target_character in character.relationships:
            rel = character.relationships[target_character]
            prompt += f"Speaking to: {target_character} ({rel['type']})\n"
            prompt += f"Relationship dynamics: {rel.get('dynamics', '')}\n"

        prompt += "\nGenerate appropriate dialogue that fits this character's voice and current emotional state."

        return prompt

    def track_dialogue(self,
                      character: str,
                      chapter: int,
                      text: str,
                      context: str = "",
                      emotion: str = "neutral"):
        """Track dialogue for consistency checking"""
        entry = DialogueEntry(
            character=character,
            chapter=chapter,
            text=text,
            context=context,
            emotion=emotion
        )
        self.dialogue_history.append(entry)

        # Update character's chapter appearances
        if character in self.characters:
            if chapter not in self.characters[character].chapter_appearances:
                self.characters[character].chapter_appearances.append(chapter)
                self.characters[character].chapter_appearances.sort()
                self._save_character(self.characters[character])

    def add_plot_point(self,
                      chapter: int,
                      event: str,
                      characters: List[str],
                      impact: str = "minor",
                      foreshadowing: List[int] = None):
        """
        Add a plot point to track
        
        Args:
            chapter: Chapter number
            event: Event description
            characters: Characters involved
            impact: Impact level (minor, major, critical)
            foreshadowing: Chapters that foreshadowed this
        """
        plot_point = PlotPoint(
            chapter=chapter,
            event=event,
            characters_involved=characters,
            impact=impact,
            foreshadowing_refs=foreshadowing or []
        )
        self.plot_tracker.append(plot_point)

    def resolve_plot_point(self, event: str, resolution_chapter: int):
        """Mark a plot point as resolved"""
        for point in self.plot_tracker:
            if point.event == event:
                point.resolved = True
                point.resolution_chapter = resolution_chapter
                break

    def check_consistency(self, chapter: int) -> Dict[str, List[str]]:
        """
        Check for consistency issues
        
        Args:
            chapter: Current chapter number
            
        Returns:
            Dictionary of potential issues
        """
        issues = {
            'character_gaps': [],
            'unresolved_plots': [],
            'relationship_conflicts': [],
            'timeline_issues': []
        }

        # Check for character appearance gaps
        for name, char in self.characters.items():
            if char.chapter_appearances:
                last_appearance = char.chapter_appearances[-1]
                if chapter - last_appearance > 5 and char.role in [CharacterRole.PROTAGONIST, CharacterRole.ANTAGONIST]:
                    issues['character_gaps'].append(
                        f"{name} hasn't appeared since chapter {last_appearance}"
                    )

        # Check for unresolved plot points
        for point in self.plot_tracker:
            if not point.resolved and point.impact in ["major", "critical"]:
                chapters_since = chapter - point.chapter
                if chapters_since > 10:
                    issues['unresolved_plots'].append(
                        f"Plot point from chapter {point.chapter}: {point.event}"
                    )

        return issues

    def get_character_context(self,
                             character_name: str,
                             chapter: int) -> Dict[str, Any]:
        """
        Get character context for current chapter
        
        Args:
            character_name: Character name
            chapter: Current chapter
            
        Returns:
            Character context dictionary
        """
        if character_name not in self.characters:
            return {}

        character = self.characters[character_name]

        # Find character's current state in arc
        current_growth = None
        for growth in character.growth_points:
            if growth.get('chapter', 0) <= chapter:
                current_growth = growth

        # Get recent dialogue
        recent_dialogue = [
            d for d in self.dialogue_history
            if d.character == character_name and abs(d.chapter - chapter) <= 2
        ]

        # Get active relationships
        active_relationships = {}
        for other_name, rel in character.relationships.items():
            other_char = self.characters.get(other_name)
            if other_char and chapter in other_char.chapter_appearances:
                active_relationships[other_name] = rel

        return {
            'character': character.to_dict(),
            'current_arc_point': current_growth,
            'recent_dialogue': [asdict(d) for d in recent_dialogue[-5:]],
            'active_relationships': active_relationships,
            'last_appearance': character.chapter_appearances[-1] if character.chapter_appearances else None
        }

    def generate_character_sheet(self, character_name: str) -> str:
        """Generate a formatted character sheet"""
        if character_name not in self.characters:
            return f"Character {character_name} not found"

        char = self.characters[character_name]

        sheet = f"""
CHARACTER SHEET: {char.name}
{'=' * 50}

BASIC INFORMATION
-----------------
Role: {char.role.value}
Age: {char.age or 'Unknown'}
Gender: {char.gender or 'Unknown'}
Occupation: {char.occupation or 'Unknown'}

APPEARANCE
----------
{json.dumps(char.appearance, indent=2) if char.appearance else 'Not specified'}
Distinguishing Features: {', '.join(char.distinguishing_features) if char.distinguishing_features else 'None'}

PERSONALITY
-----------
Traits: {', '.join(char.personality_traits) if char.personality_traits else 'Not defined'}
Strengths: {', '.join(char.strengths) if char.strengths else 'Not defined'}
Weaknesses: {', '.join(char.weaknesses) if char.weaknesses else 'Not defined'}
Fears: {', '.join(char.fears) if char.fears else 'Not defined'}
Motivations: {', '.join(char.motivations) if char.motivations else 'Not defined'}

BACKGROUND
----------
{char.backstory or 'No backstory provided'}

SPEECH CHARACTERISTICS
----------------------
Pattern: {char.speech_pattern or 'Standard'}
Vocabulary: {char.vocabulary_level}
Catchphrases: {', '.join(char.catchphrases) if char.catchphrases else 'None'}
Accent/Dialect: {char.accent_dialect or 'None'}

CHARACTER ARC
-------------
{char.character_arc or 'No arc defined'}
Goals: {', '.join(char.goals) if char.goals else 'None'}
Conflicts: {', '.join(char.conflicts) if char.conflicts else 'None'}

RELATIONSHIPS
-------------
"""
        for other, rel in char.relationships.items():
            sheet += f"- {other}: {rel['type']} - {rel.get('description', '')}\n"

        sheet += f"""
STORY PRESENCE
--------------
Chapter Appearances: {', '.join(map(str, char.chapter_appearances)) if char.chapter_appearances else 'None yet'}
Total Appearances: {len(char.chapter_appearances)}
"""

        return sheet

    def export_all_characters(self, filepath: str = "characters_export.json"):
        """Export all characters to a file"""
        data = {
            'exported_at': datetime.now().isoformat(),
            'characters': {
                name: char.to_dict()
                for name, char in self.characters.items()
            },
            'dialogue_history': [asdict(d) for d in self.dialogue_history],
            'plot_points': [asdict(p) for p in self.plot_tracker],
            'relationships': {
                f"{k[0]}-{k[1]}": v
                for k, v in self.relationship_matrix.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported character data to {filepath}")

# Note: Character manager should be obtained from ProjectManager
# to ensure proper project isolation. Use:
# from project_manager import get_project_manager
# pm = get_project_manager()
# character_manager = pm.get_character_manager()
