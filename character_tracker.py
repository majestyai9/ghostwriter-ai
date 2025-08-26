"""
Character Tracking System for maintaining character consistency throughout book generation.

This module provides a comprehensive character database that tracks character traits,
relationships, appearances, and development arcs across chapters.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import re

logger = logging.getLogger(__name__)


class Gender(Enum):
    """Character gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNSPECIFIED = "unspecified"


class CharacterRole(Enum):
    """Character role in the story."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    DEUTERAGONIST = "deuteragonist"
    SUPPORTING = "supporting"
    MINOR = "minor"
    BACKGROUND = "background"


class RelationshipType(Enum):
    """Types of relationships between characters."""
    FAMILY = "family"
    ROMANTIC = "romantic"
    FRIEND = "friend"
    ENEMY = "enemy"
    RIVAL = "rival"
    MENTOR = "mentor"
    STUDENT = "student"
    COLLEAGUE = "colleague"
    ALLY = "ally"
    NEUTRAL = "neutral"


@dataclass
class PhysicalDescription:
    """Physical attributes of a character."""
    height: Optional[str] = None
    build: Optional[str] = None
    hair_color: Optional[str] = None
    hair_style: Optional[str] = None
    eye_color: Optional[str] = None
    skin_tone: Optional[str] = None
    age: Optional[int] = None
    distinguishing_features: List[str] = field(default_factory=list)
    clothing_style: Optional[str] = None


@dataclass
class PersonalityTraits:
    """Personality attributes of a character."""
    traits: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    quirks: List[str] = field(default_factory=list)
    moral_alignment: Optional[str] = None


@dataclass
class CharacterArc:
    """Character development arc information."""
    starting_state: str = ""
    ending_state: str = ""
    key_events: List[Tuple[int, str]] = field(default_factory=list)  # (chapter, event)
    growth_points: List[str] = field(default_factory=list)
    internal_conflicts: List[str] = field(default_factory=list)
    resolutions: List[str] = field(default_factory=list)


@dataclass
class DialogueProfile:
    """Character's dialogue characteristics."""
    speech_patterns: List[str] = field(default_factory=list)
    vocabulary_level: str = "average"  # simple, average, advanced, technical
    catchphrases: List[str] = field(default_factory=list)
    accent: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: ["English"])
    formality_level: str = "neutral"  # casual, neutral, formal
    typical_greetings: List[str] = field(default_factory=list)
    typical_farewells: List[str] = field(default_factory=list)
    emotional_expressions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Character:
    """Complete character profile."""
    name: str
    full_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    gender: Gender = Gender.UNSPECIFIED
    role: CharacterRole = CharacterRole.MINOR
    physical: PhysicalDescription = field(default_factory=PhysicalDescription)
    personality: PersonalityTraits = field(default_factory=PersonalityTraits)
    arc: CharacterArc = field(default_factory=CharacterArc)
    dialogue: DialogueProfile = field(default_factory=DialogueProfile)
    background: str = ""
    occupation: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    first_appearance: Optional[int] = None
    last_appearance: Optional[int] = None
    chapter_appearances: Set[int] = field(default_factory=set)
    relationships: Dict[str, RelationshipType] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class CharacterDatabase:
    """
    SQLite-based character database for tracking character information.
    
    This database maintains comprehensive character profiles and ensures
    consistency across the entire book generation process.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the character database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or Path("projects/character_database.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
        self._cached_characters: Dict[str, Character] = {}
        
    def _create_tables(self) -> None:
        """Create necessary database tables."""
        cursor = self.conn.cursor()
        
        # Main character table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                name TEXT PRIMARY KEY,
                full_name TEXT,
                aliases TEXT,
                gender TEXT,
                role TEXT,
                physical_desc TEXT,
                personality TEXT,
                arc TEXT,
                dialogue TEXT,
                background TEXT,
                occupation TEXT,
                skills TEXT,
                first_appearance INTEGER,
                last_appearance INTEGER,
                chapter_appearances TEXT,
                relationships TEXT,
                notes TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        # Character interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character1 TEXT,
                character2 TEXT,
                chapter INTEGER,
                interaction_type TEXT,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (character1) REFERENCES characters (name),
                FOREIGN KEY (character2) REFERENCES characters (name)
            )
        """)
        
        # Character name variations table (for tracking consistency)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS name_variations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT,
                variation TEXT,
                context TEXT,
                chapter INTEGER,
                FOREIGN KEY (canonical_name) REFERENCES characters (name)
            )
        """)
        
        self.conn.commit()
    
    def register_character(self, character: Character) -> None:
        """
        Register a new character in the database.
        
        Args:
            character: Character object to register
        """
        cursor = self.conn.cursor()
        
        # Serialize complex fields to JSON
        character_data = {
            'name': character.name,
            'full_name': character.full_name,
            'aliases': json.dumps(character.aliases),
            'gender': character.gender.value,
            'role': character.role.value,
            'physical_desc': json.dumps(asdict(character.physical)),
            'personality': json.dumps(asdict(character.personality)),
            'arc': json.dumps(asdict(character.arc)),
            'dialogue': json.dumps(asdict(character.dialogue)),
            'background': character.background,
            'occupation': character.occupation,
            'skills': json.dumps(character.skills),
            'first_appearance': character.first_appearance,
            'last_appearance': character.last_appearance,
            'chapter_appearances': json.dumps(list(character.chapter_appearances)),
            'relationships': json.dumps({k: v.value for k, v in character.relationships.items()}),
            'notes': json.dumps(character.notes),
            'created_at': character.created_at.isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Insert or replace character
        cursor.execute("""
            INSERT OR REPLACE INTO characters 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(character_data.values()))
        
        self.conn.commit()
        self._cached_characters[character.name] = character
        
        logger.info(f"Registered character: {character.name}")
    
    def add_character(self, character: Character) -> None:
        """
        Add a new character to the database (alias for register_character).
        
        Args:
            character: Character object to add
        """
        self.register_character(character)
    
    def update_character(self, name: str, character: Optional[Character] = None, updates: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update character information.
        
        Args:
            name: Character name to update
            character: Updated Character object (if provided, overrides updates)
            updates: Dictionary of fields to update (if character not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if character:
            # Direct update with Character object
            self.register_character(character)
            return True
        
        # Fetch existing character for dictionary updates
        existing_character = self.get_character(name)
        if not existing_character:
            logger.warning(f"Character '{name}' not found for update")
            return False
        
        if not updates:
            logger.warning("No updates provided")
            return False
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(existing_character, key):
                if key == 'physical' and isinstance(value, dict):
                    for p_key, p_value in value.items():
                        setattr(existing_character.physical, p_key, p_value)
                elif key == 'personality' and isinstance(value, dict):
                    for p_key, p_value in value.items():
                        setattr(existing_character.personality, p_key, p_value)
                elif key == 'arc' and isinstance(value, dict):
                    for a_key, a_value in value.items():
                        setattr(existing_character.arc, a_key, a_value)
                elif key == 'dialogue' and isinstance(value, dict):
                    for d_key, d_value in value.items():
                        setattr(existing_character.dialogue, d_key, d_value)
                elif key == 'chapter_appearances' and isinstance(value, (list, set)):
                    existing_character.chapter_appearances.update(value)
                elif key == 'relationships' and isinstance(value, dict):
                    for rel_name, rel_type in value.items():
                        existing_character.relationships[rel_name] = RelationshipType(rel_type)
                else:
                    setattr(existing_character, key, value)
        
        existing_character.updated_at = datetime.now()
        self.register_character(existing_character)
        
        return True
    
    def get_character(self, name: str) -> Optional[Character]:
        """
        Retrieve a character by name.
        
        Args:
            name: Character name
            
        Returns:
            Character object if found, None otherwise
        """
        # Check cache first
        if name in self._cached_characters:
            return self._cached_characters[name]
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM characters WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Deserialize character from database
        character = self._deserialize_character(dict(row))
        self._cached_characters[name] = character
        
        return character
    
    def _deserialize_character(self, row: Dict) -> Character:
        """Deserialize character from database row."""
        physical = PhysicalDescription(**json.loads(row['physical_desc']))
        personality = PersonalityTraits(**json.loads(row['personality']))
        arc_data = json.loads(row['arc'])
        arc_data['key_events'] = [(e[0], e[1]) for e in arc_data.get('key_events', [])]
        arc = CharacterArc(**arc_data)
        dialogue = DialogueProfile(**json.loads(row['dialogue']))
        
        relationships = {}
        for k, v in json.loads(row['relationships']).items():
            relationships[k] = RelationshipType(v)
        
        character = Character(
            name=row['name'],
            full_name=row['full_name'],
            aliases=json.loads(row['aliases']),
            gender=Gender(row['gender']),
            role=CharacterRole(row['role']),
            physical=physical,
            personality=personality,
            arc=arc,
            dialogue=dialogue,
            background=row['background'],
            occupation=row['occupation'],
            skills=json.loads(row['skills']),
            first_appearance=row['first_appearance'],
            last_appearance=row['last_appearance'],
            chapter_appearances=set(json.loads(row['chapter_appearances'])),
            relationships=relationships,
            notes=json.loads(row['notes']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
        
        return character
    
    def validate_consistency(self, name: str, chapter: int, 
                           mentioned_attributes: Dict[str, Any]) -> List[str]:
        """
        Validate character consistency against stored profile.
        
        Args:
            name: Character name
            chapter: Chapter number
            mentioned_attributes: Attributes mentioned in the chapter
            
        Returns:
            List of consistency issues found
        """
        issues = []
        character = self.get_character(name)
        
        if not character:
            # Check for similar names (possible typo or variation)
            similar = self.find_similar_names(name)
            if similar:
                issues.append(f"Unknown character '{name}', did you mean: {', '.join(similar)}?")
            else:
                logger.info(f"New character detected: {name}")
            return issues
        
        # Check gender consistency
        if 'gender' in mentioned_attributes:
            mentioned_gender = mentioned_attributes['gender'].lower()
            if character.gender != Gender.UNSPECIFIED:
                expected_pronouns = {
                    Gender.MALE: ['he', 'him', 'his'],
                    Gender.FEMALE: ['she', 'her', 'hers'],
                    Gender.NON_BINARY: ['they', 'them', 'their']
                }
                
                if character.gender in expected_pronouns:
                    valid_pronouns = expected_pronouns[character.gender]
                    if mentioned_gender not in valid_pronouns:
                        issues.append(
                            f"Gender inconsistency for '{name}': "
                            f"expected {character.gender.value}, found '{mentioned_gender}'"
                        )
        
        # Check physical description consistency
        if 'physical' in mentioned_attributes:
            for attr, value in mentioned_attributes['physical'].items():
                stored_value = getattr(character.physical, attr, None)
                if stored_value and stored_value != value:
                    issues.append(
                        f"Physical inconsistency for '{name}': "
                        f"{attr} was '{stored_value}', now '{value}'"
                    )
        
        # Check name variations
        if 'aliases' in mentioned_attributes:
            for alias in mentioned_attributes['aliases']:
                if alias not in character.aliases:
                    self.add_name_variation(character.name, alias, chapter)
        
        # Update appearance tracking
        character.chapter_appearances.add(chapter)
        if not character.first_appearance or chapter < character.first_appearance:
            character.first_appearance = chapter
        if not character.last_appearance or chapter > character.last_appearance:
            character.last_appearance = chapter
        
        self.update_character(name, {
            'chapter_appearances': character.chapter_appearances,
            'first_appearance': character.first_appearance,
            'last_appearance': character.last_appearance
        })
        
        return issues
    
    def find_similar_names(self, name: str, threshold: float = 0.8) -> List[str]:
        """
        Find similar character names (for typo detection).
        
        Args:
            name: Name to search for
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar character names
        """
        similar = []
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM characters")
        
        for row in cursor.fetchall():
            existing_name = row[0]
            similarity = self._calculate_similarity(name.lower(), existing_name.lower())
            if similarity >= threshold:
                similar.append(existing_name)
        
        return similar
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein distance."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return 0.0
        
        # Calculate Levenshtein distance
        distances = range(len(s2) + 1)
        for i1, c1 in enumerate(s1):
            new_distances = [i1 + 1]
            for i2, c2 in enumerate(s2):
                if c1 == c2:
                    new_distances.append(distances[i2])
                else:
                    new_distances.append(1 + min((distances[i2],
                                                 distances[i2 + 1],
                                                 new_distances[-1])))
            distances = new_distances
        
        # Convert to similarity score
        max_len = max(len(s1), len(s2))
        return 1 - (distances[-1] / max_len)
    
    def add_name_variation(self, canonical_name: str, variation: str, chapter: int) -> None:
        """
        Record a name variation for consistency tracking.
        
        Args:
            canonical_name: The character's canonical name
            variation: The variation found
            chapter: Chapter where variation appeared
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO name_variations (canonical_name, variation, chapter)
            VALUES (?, ?, ?)
        """, (canonical_name, variation, chapter))
        self.conn.commit()
    
    def record_interaction(self, char1: str, char2: str, chapter: int,
                          interaction_type: str, description: str) -> None:
        """
        Record an interaction between characters.
        
        Args:
            char1: First character name
            char2: Second character name
            chapter: Chapter number
            interaction_type: Type of interaction
            description: Description of the interaction
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (character1, character2, chapter, interaction_type, description)
            VALUES (?, ?, ?, ?, ?)
        """, (char1, char2, chapter, interaction_type, description))
        self.conn.commit()
    
    def get_character_relationships(self, name: str) -> Dict[str, RelationshipType]:
        """
        Get all relationships for a character.
        
        Args:
            name: Character name
            
        Returns:
            Dictionary of relationships
        """
        character = self.get_character(name)
        return character.relationships if character else {}
    
    def get_all_characters(self, role_filter: Optional[CharacterRole] = None) -> List[Character]:
        """
        Get all characters, optionally filtered by role.
        
        Args:
            role_filter: Optional role to filter by
            
        Returns:
            List of characters
        """
        cursor = self.conn.cursor()
        
        if role_filter:
            cursor.execute("SELECT * FROM characters WHERE role = ?", (role_filter.value,))
        else:
            cursor.execute("SELECT * FROM characters")
        
        characters = []
        for row in cursor.fetchall():
            character = self._deserialize_character(dict(row))
            characters.append(character)
        
        return characters
    
    def generate_character_summary(self, name: str) -> str:
        """
        Generate a comprehensive summary of a character.
        
        Args:
            name: Character name
            
        Returns:
            Character summary string
        """
        character = self.get_character(name)
        if not character:
            return f"Character '{name}' not found."
        
        summary_parts = [
            f"=== {character.name} ===",
            f"Full Name: {character.full_name or 'N/A'}",
            f"Role: {character.role.value.title()}",
            f"Gender: {character.gender.value.title()}",
        ]
        
        if character.occupation:
            summary_parts.append(f"Occupation: {character.occupation}")
        
        if character.physical.age:
            summary_parts.append(f"Age: {character.physical.age}")
        
        if character.physical.distinguishing_features:
            summary_parts.append(
                f"Distinguishing Features: {', '.join(character.physical.distinguishing_features)}"
            )
        
        if character.personality.traits:
            summary_parts.append(f"Personality: {', '.join(character.personality.traits[:5])}")
        
        if character.personality.motivations:
            summary_parts.append(f"Motivations: {', '.join(character.personality.motivations[:3])}")
        
        if character.dialogue.speech_patterns:
            summary_parts.append(f"Speech Patterns: {', '.join(character.dialogue.speech_patterns[:3])}")
        
        if character.relationships:
            rel_summary = []
            for other, rel_type in list(character.relationships.items())[:5]:
                rel_summary.append(f"{other} ({rel_type.value})")
            summary_parts.append(f"Key Relationships: {', '.join(rel_summary)}")
        
        if character.chapter_appearances:
            summary_parts.append(
                f"Appearances: Chapters {min(character.chapter_appearances)}-"
                f"{max(character.chapter_appearances)} "
                f"({len(character.chapter_appearances)} total)"
            )
        
        if character.arc.key_events:
            summary_parts.append("\nCharacter Arc:")
            for chapter, event in character.arc.key_events[:3]:
                summary_parts.append(f"  - Chapter {chapter}: {event}")
        
        return "\n".join(summary_parts)
    
    def export_character_bible(self, output_path: Path) -> None:
        """
        Export a complete character bible as JSON.
        
        Args:
            output_path: Path to save the character bible
        """
        all_characters = self.get_all_characters()
        
        bible = {
            'generated_at': datetime.now().isoformat(),
            'total_characters': len(all_characters),
            'characters': {}
        }
        
        for character in all_characters:
            bible['characters'][character.name] = {
                'full_name': character.full_name,
                'role': character.role.value,
                'gender': character.gender.value,
                'physical': asdict(character.physical),
                'personality': asdict(character.personality),
                'arc': asdict(character.arc),
                'dialogue': asdict(character.dialogue),
                'background': character.background,
                'occupation': character.occupation,
                'skills': character.skills,
                'appearances': list(character.chapter_appearances),
                'relationships': {k: v.value for k, v in character.relationships.items()},
                'notes': character.notes
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bible, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported character bible to {output_path}")
    
    def update_chapter_appearance(self, character_name: str, chapter_num: int) -> None:
        """
        Update a character's chapter appearance.
        
        Args:
            character_name: Name of the character
            chapter_num: Chapter number where character appears
        """
        character = self.get_character(character_name)
        if character:
            character.chapter_appearances.add(chapter_num)
            character.last_appearance = chapter_num
            self.update_character(character_name, character=character)
            logger.debug(f"Updated {character_name} appearance in chapter {chapter_num}")
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()