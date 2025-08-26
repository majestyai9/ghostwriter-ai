"""
Advanced Character Development and Evolution Tracking System.

This module provides comprehensive character management with:
- OCEAN personality modeling
- Dialogue consistency checking with embeddings
- Character evolution tracking through chapters
- Relationship interaction matrices
- Voice synthesis patterns
- Per-character knowledge bases
"""
import json
import logging
import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
import sqlite3
import hashlib
from character_tracker import (
    CharacterDatabase, 
    Character as TrackedCharacter,
    Gender,
    CharacterRole as TrackedRole,
    RelationshipType,
    PhysicalDescription,
    PersonalityTraits,
    CharacterArc,
    DialogueProfile
)


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
class OCEANPersonality:
    """OCEAN (Big Five) personality model for deep character modeling."""
    openness: float = 0.5  # 0-1: Creativity, curiosity, open to new experiences
    conscientiousness: float = 0.5  # 0-1: Organization, dependability, self-discipline
    extraversion: float = 0.5  # 0-1: Sociability, assertiveness, emotional expression
    agreeableness: float = 0.5  # 0-1: Cooperation, trust, empathy
    neuroticism: float = 0.5  # 0-1: Emotional instability, anxiety, moodiness
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector for similarity calculations."""
        return np.array([self.openness, self.conscientiousness, 
                        self.extraversion, self.agreeableness, self.neuroticism])
    
    def similarity(self, other: 'OCEANPersonality') -> float:
        """Calculate similarity with another personality (0-1)."""
        v1, v2 = self.to_vector(), other.to_vector()
        return 1 - (np.linalg.norm(v1 - v2) / np.sqrt(5))

@dataclass
class CharacterEvolution:
    """Track character evolution through the story."""
    chapter: int
    personality_changes: Dict[str, float] = field(default_factory=dict)
    emotional_state: str = "neutral"
    relationships_changed: Dict[str, str] = field(default_factory=dict)
    skills_gained: List[str] = field(default_factory=list)
    beliefs_changed: List[str] = field(default_factory=list)
    trauma_events: List[str] = field(default_factory=list)
    growth_description: str = ""

@dataclass
class VoiceSynthesisPattern:
    """Unique voice synthesis patterns for character speech."""
    pitch_variation: float = 0.5  # 0-1: Monotone to highly varied
    speaking_pace: float = 0.5  # 0-1: Very slow to very fast
    pause_frequency: float = 0.5  # 0-1: No pauses to frequent pauses
    emphasis_patterns: List[str] = field(default_factory=list)
    vocal_tics: List[str] = field(default_factory=list)
    emotional_inflections: Dict[str, float] = field(default_factory=dict)

@dataclass
class CharacterKnowledge:
    """Per-character knowledge base."""
    known_facts: Set[str] = field(default_factory=set)
    known_characters: Dict[str, str] = field(default_factory=dict)  # name -> relationship
    known_locations: Set[str] = field(default_factory=set)
    known_events: List[Tuple[int, str]] = field(default_factory=list)  # chapter, event
    secrets: List[str] = field(default_factory=list)
    false_beliefs: List[str] = field(default_factory=list)

@dataclass
class DialogueEmbedding:
    """Store dialogue with embeddings for consistency checking."""
    text: str
    chapter: int
    context: str
    emotion: str
    embedding: Optional[np.ndarray] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class CharacterProfile:
    """Advanced character profile with evolution tracking."""
    # Basic Information
    name: str
    role: CharacterRole
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None

    # Physical Description
    appearance: Dict[str, str] = field(default_factory=dict)
    distinguishing_features: List[str] = field(default_factory=list)

    # Advanced Personality (OCEAN model)
    ocean_personality: OCEANPersonality = field(default_factory=OCEANPersonality)
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

    # Voice Synthesis
    voice_pattern: VoiceSynthesisPattern = field(default_factory=VoiceSynthesisPattern)
    speech_pattern: str = ""
    vocabulary_level: str = "moderate"
    catchphrases: List[str] = field(default_factory=list)
    mannerisms: List[str] = field(default_factory=list)
    accent_dialect: str = ""

    # Story Arc & Evolution
    character_arc: str = ""
    goals: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    evolution_history: List[CharacterEvolution] = field(default_factory=list)
    
    # Knowledge Base
    knowledge: CharacterKnowledge = field(default_factory=CharacterKnowledge)
    
    # Relationships with interaction matrix
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    interaction_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Dialogue History with Embeddings
    dialogue_embeddings: List[DialogueEmbedding] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    chapter_appearances: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['role'] = self.role.value
        # Convert numpy arrays to lists for JSON serialization
        for embedding in data.get('dialogue_embeddings', []):
            if isinstance(embedding.get('embedding'), np.ndarray):
                embedding['embedding'] = embedding['embedding'].tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterProfile':
        """Create from dictionary."""
        data['role'] = CharacterRole(data['role'])
        # Reconstruct complex objects
        if 'ocean_personality' in data:
            data['ocean_personality'] = OCEANPersonality(**data['ocean_personality'])
        if 'voice_pattern' in data:
            data['voice_pattern'] = VoiceSynthesisPattern(**data['voice_pattern'])
        if 'knowledge' in data:
            data['knowledge'] = CharacterKnowledge(**data['knowledge'])
        if 'evolution_history' in data:
            data['evolution_history'] = [CharacterEvolution(**e) for e in data['evolution_history']]
        if 'dialogue_embeddings' in data:
            embeddings = []
            for e in data['dialogue_embeddings']:
                if 'embedding' in e and e['embedding']:
                    e['embedding'] = np.array(e['embedding'])
                embeddings.append(DialogueEmbedding(**e))
            data['dialogue_embeddings'] = embeddings
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
    """Advanced character manager with evolution tracking and embeddings."""

    def __init__(self, project_dir: Path = None):
        """
        Initialize advanced character manager.
        
        Args:
            project_dir: Project directory for character data
        """
        self.project_dir = project_dir or Path(".")
        self.characters_dir = self.project_dir / "characters"
        self.characters_dir.mkdir(exist_ok=True)

        # Character storage
        self.characters: Dict[str, CharacterProfile] = {}
        self.dialogue_history: List[DialogueEntry] = []
        self.plot_tracker: List[PlotPoint] = []
        
        # Advanced features
        self.relationship_matrix: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.dialogue_embeddings_cache: Dict[str, np.ndarray] = {}
        self.character_evolution_tracker: Dict[str, List[CharacterEvolution]] = {}
        
        # Initialize database connection
        self.character_db = CharacterDatabase(self.project_dir / "character_database.db")
        
        self.logger = logging.getLogger(__name__)
        self._load_characters()
        self._sync_with_database()

    def _load_characters(self):
        """Load existing characters from disk."""
        for char_file in self.characters_dir.glob("*.json"):
            try:
                with open(char_file, encoding='utf-8') as f:
                    data = json.load(f)
                    char = CharacterProfile.from_dict(data)
                    self.characters[char.name] = char
                    # Load evolution history
                    if char.evolution_history:
                        self.character_evolution_tracker[char.name] = char.evolution_history
            except Exception as e:
                self.logger.error(f"Failed to load character {char_file}: {e}")
    
    def _sync_with_database(self):
        """Sync with SQLite character database."""
        # Get all characters from database
        db_characters = self.character_db.get_all_characters()
        
        for db_char in db_characters:
            if db_char.name not in self.characters:
                # Convert database character to our profile
                profile = self._convert_from_db_character(db_char)
                self.characters[db_char.name] = profile
            else:
                # Update existing character with database info
                self._update_from_db_character(self.characters[db_char.name], db_char)
    
    def _convert_from_db_character(self, db_char: TrackedCharacter) -> CharacterProfile:
        """Convert database character to advanced profile."""
        # Map database role to our role
        role_mapping = {
            TrackedRole.PROTAGONIST: CharacterRole.PROTAGONIST,
            TrackedRole.ANTAGONIST: CharacterRole.ANTAGONIST,
            TrackedRole.SUPPORTING: CharacterRole.SUPPORTING,
            TrackedRole.MINOR: CharacterRole.MINOR
        }
        
        profile = CharacterProfile(
            name=db_char.name,
            role=role_mapping.get(db_char.role, CharacterRole.SUPPORTING),
            age=db_char.physical.age,
            gender=db_char.gender.value if db_char.gender else None,
            occupation=db_char.occupation
        )
        
        # Convert personality traits to OCEAN model
        profile.ocean_personality = self._analyze_ocean_from_traits(db_char.personality.traits)
        profile.personality_traits = db_char.personality.traits
        profile.strengths = db_char.personality.strengths
        profile.weaknesses = db_char.personality.weaknesses
        profile.fears = db_char.personality.fears
        profile.motivations = db_char.personality.motivations
        
        # Convert dialogue profile to voice pattern
        profile.voice_pattern = self._analyze_voice_pattern(db_char.dialogue)
        profile.speech_pattern = ' '.join(db_char.dialogue.speech_patterns)
        profile.vocabulary_level = db_char.dialogue.vocabulary_level
        profile.catchphrases = db_char.dialogue.catchphrases
        profile.accent_dialect = db_char.dialogue.accent or ""
        
        # Set relationships
        for rel_name, rel_type in db_char.relationships.items():
            profile.relationships[rel_name] = {
                'type': rel_type.value,
                'description': '',
                'dynamics': ''
            }
        
        profile.chapter_appearances = list(db_char.chapter_appearances)
        
        return profile
    
    def _update_from_db_character(self, profile: CharacterProfile, db_char: TrackedCharacter):
        """Update existing profile with database info."""
        # Update appearances
        profile.chapter_appearances = list(set(profile.chapter_appearances + list(db_char.chapter_appearances)))
        
        # Update relationships
        for rel_name, rel_type in db_char.relationships.items():
            if rel_name not in profile.relationships:
                profile.relationships[rel_name] = {
                    'type': rel_type.value,
                    'description': '',
                    'dynamics': ''
                }
    
    def _analyze_ocean_from_traits(self, traits: List[str]) -> OCEANPersonality:
        """Analyze OCEAN personality from trait list."""
        ocean = OCEANPersonality()
        
        # Keywords for each OCEAN dimension
        openness_keywords = ['creative', 'curious', 'imaginative', 'artistic', 'inventive']
        conscientiousness_keywords = ['organized', 'responsible', 'reliable', 'disciplined', 'careful']
        extraversion_keywords = ['outgoing', 'social', 'talkative', 'energetic', 'assertive']
        agreeableness_keywords = ['kind', 'cooperative', 'trusting', 'helpful', 'compassionate']
        neuroticism_keywords = ['anxious', 'moody', 'tense', 'nervous', 'sensitive']
        
        trait_lower = [t.lower() for t in traits]
        
        # Calculate scores based on keyword matches
        ocean.openness = min(1.0, sum(1 for k in openness_keywords if any(k in t for t in trait_lower)) / 3)
        ocean.conscientiousness = min(1.0, sum(1 for k in conscientiousness_keywords if any(k in t for t in trait_lower)) / 3)
        ocean.extraversion = min(1.0, sum(1 for k in extraversion_keywords if any(k in t for t in trait_lower)) / 3)
        ocean.agreeableness = min(1.0, sum(1 for k in agreeableness_keywords if any(k in t for t in trait_lower)) / 3)
        ocean.neuroticism = min(1.0, sum(1 for k in neuroticism_keywords if any(k in t for t in trait_lower)) / 3)
        
        return ocean
    
    def _analyze_voice_pattern(self, dialogue: DialogueProfile) -> VoiceSynthesisPattern:
        """Analyze voice pattern from dialogue profile."""
        voice = VoiceSynthesisPattern()
        
        # Analyze formality for pitch variation
        if dialogue.formality_level == 'formal':
            voice.pitch_variation = 0.3
            voice.speaking_pace = 0.4
        elif dialogue.formality_level == 'casual':
            voice.pitch_variation = 0.7
            voice.speaking_pace = 0.6
        
        # Set vocal tics from catchphrases
        voice.vocal_tics = dialogue.catchphrases[:3] if dialogue.catchphrases else []
        
        # Set emphasis patterns
        voice.emphasis_patterns = dialogue.speech_patterns[:3] if dialogue.speech_patterns else []
        
        return voice

    def create_character(self,
                        name: str,
                        role: CharacterRole,
                        ocean_scores: Optional[Dict[str, float]] = None,
                        **kwargs) -> CharacterProfile:
        """
        Create a new character with OCEAN personality.
        
        Args:
            name: Character name
            role: Character role
            ocean_scores: Optional OCEAN personality scores
            **kwargs: Additional character attributes
            
        Returns:
            CharacterProfile object
        """
        character = CharacterProfile(name=name, role=role, **kwargs)
        
        # Set OCEAN personality if provided
        if ocean_scores:
            character.ocean_personality = OCEANPersonality(**ocean_scores)
        
        # Initialize knowledge base
        character.knowledge = CharacterKnowledge()
        
        # Initialize voice pattern
        character.voice_pattern = VoiceSynthesisPattern()
        
        self.characters[name] = character
        self._save_character(character)
        
        # Also save to database
        self._sync_to_database(character)

        self.logger.info(f"Created character: {name} ({role.value})")
        return character
    
    def _sync_to_database(self, character: CharacterProfile):
        """Sync character profile to database."""
        # Convert to database character format
        db_char = TrackedCharacter(
            name=character.name,
            gender=Gender(character.gender) if character.gender else Gender.UNSPECIFIED,
            role=self._convert_role_to_db(character.role)
        )
        
        # Set personality traits
        db_char.personality.traits = character.personality_traits
        db_char.personality.strengths = character.strengths
        db_char.personality.weaknesses = character.weaknesses
        db_char.personality.fears = character.fears
        db_char.personality.motivations = character.motivations
        
        # Set dialogue profile
        db_char.dialogue.speech_patterns = [character.speech_pattern] if character.speech_pattern else []
        db_char.dialogue.vocabulary_level = character.vocabulary_level
        db_char.dialogue.catchphrases = character.catchphrases
        db_char.dialogue.accent = character.accent_dialect
        
        # Save to database
        self.character_db.register_character(db_char)
    
    def _convert_role_to_db(self, role: CharacterRole) -> TrackedRole:
        """Convert our role to database role."""
        role_mapping = {
            CharacterRole.PROTAGONIST: TrackedRole.PROTAGONIST,
            CharacterRole.ANTAGONIST: TrackedRole.ANTAGONIST,
            CharacterRole.SUPPORTING: TrackedRole.SUPPORTING,
            CharacterRole.MINOR: TrackedRole.MINOR,
            CharacterRole.MENTOR: TrackedRole.SUPPORTING,
            CharacterRole.LOVE_INTEREST: TrackedRole.SUPPORTING,
            CharacterRole.SIDEKICK: TrackedRole.SUPPORTING,
            CharacterRole.RIVAL: TrackedRole.SUPPORTING
        }
        return role_mapping.get(role, TrackedRole.MINOR)

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
                        dynamics: str = "",
                        interaction_strength: float = 0.5):
        """
        Add or update relationship with interaction matrix.
        
        Args:
            char1: First character name
            char2: Second character name
            relationship_type: Type of relationship
            description: Relationship description
            dynamics: How they interact
            interaction_strength: Strength of interaction (0-1)
        """
        if char1 not in self.characters or char2 not in self.characters:
            raise ValueError("Both characters must exist")

        # Update both characters' relationship data
        self.characters[char1].relationships[char2] = {
            'type': relationship_type.value,
            'description': description,
            'dynamics': dynamics,
            'strength': interaction_strength
        }

        self.characters[char2].relationships[char1] = {
            'type': relationship_type.value,
            'description': description,
            'dynamics': dynamics,
            'strength': interaction_strength
        }
        
        # Update interaction matrix
        key = tuple(sorted([char1, char2]))
        self.characters[char1].interaction_matrix[key] = interaction_strength
        self.characters[char2].interaction_matrix[key] = interaction_strength

        # Update relationship matrix with personality compatibility
        compatibility = self._calculate_personality_compatibility(char1, char2)
        self.relationship_matrix[key] = {
            'type': relationship_type,
            'description': description,
            'dynamics': dynamics,
            'strength': interaction_strength,
            'compatibility': compatibility,
            'interactions': []
        }

        self._save_character(self.characters[char1])
        self._save_character(self.characters[char2])
    
    def _calculate_personality_compatibility(self, char1: str, char2: str) -> float:
        """Calculate personality compatibility using OCEAN model."""
        if char1 not in self.characters or char2 not in self.characters:
            return 0.5
        
        ocean1 = self.characters[char1].ocean_personality
        ocean2 = self.characters[char2].ocean_personality
        
        return ocean1.similarity(ocean2)
    
    def get_interaction_matrix(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Get complete interaction matrix with all relationships."""
        matrix = {}
        
        for (char1, char2), rel_data in self.relationship_matrix.items():
            matrix[(char1, char2)] = {
                'type': rel_data['type'].value if hasattr(rel_data['type'], 'value') else rel_data['type'],
                'strength': rel_data.get('strength', 0.5),
                'compatibility': rel_data.get('compatibility', 0.5),
                'interactions_count': len(rel_data.get('interactions', [])),
                'dynamics': rel_data.get('dynamics', '')
            }
        
        return matrix

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
        """Track dialogue with embedding for consistency checking."""
        entry = DialogueEntry(
            character=character,
            chapter=chapter,
            text=text,
            context=context,
            emotion=emotion
        )
        self.dialogue_history.append(entry)
        
        # Create and store dialogue embedding
        if character in self.characters:
            embedding = self._create_dialogue_embedding(text, context, emotion)
            dialogue_emb = DialogueEmbedding(
                text=text,
                chapter=chapter,
                context=context,
                emotion=emotion,
                embedding=embedding
            )
            self.characters[character].dialogue_embeddings.append(dialogue_emb)
            
            # Cache the embedding
            cache_key = f"{character}_{chapter}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
            self.dialogue_embeddings_cache[cache_key] = embedding
            
            # Update character's chapter appearances
            if chapter not in self.characters[character].chapter_appearances:
                self.characters[character].chapter_appearances.append(chapter)
                self.characters[character].chapter_appearances.sort()
                
            # Update character's knowledge based on dialogue
            self._update_character_knowledge(character, text, context, chapter)
            
            self._save_character(self.characters[character])
    
    def _create_dialogue_embedding(self, text: str, context: str, emotion: str) -> np.ndarray:
        """Create embedding for dialogue (simplified version)."""
        # In production, use a proper embedding model like sentence-transformers
        # This is a simplified version for demonstration
        
        # Create feature vector based on text characteristics
        features = []
        
        # Length features
        features.append(len(text) / 100)  # Normalized length
        features.append(len(text.split()) / 50)  # Word count
        
        # Emotional features
        emotion_map = {
            'neutral': 0, 'happy': 1, 'sad': -1, 'angry': -0.5,
            'fearful': -0.8, 'surprised': 0.5, 'disgusted': -0.3
        }
        features.append(emotion_map.get(emotion.lower(), 0))
        
        # Punctuation features
        features.append(text.count('!') / max(1, len(text.split())))
        features.append(text.count('?') / max(1, len(text.split())))
        features.append(text.count('...') / max(1, len(text.split())))
        
        # Context similarity (simplified)
        features.append(len(context) / 100)
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0)
        
        return np.array(features[:10])
    
    def check_dialogue_consistency(self, character: str, new_dialogue: str, 
                                  context: str, emotion: str) -> float:
        """Check dialogue consistency using embeddings."""
        if character not in self.characters:
            return 1.0  # No history, assume consistent
        
        char_profile = self.characters[character]
        if not char_profile.dialogue_embeddings:
            return 1.0  # No history, assume consistent
        
        # Create embedding for new dialogue
        new_embedding = self._create_dialogue_embedding(new_dialogue, context, emotion)
        
        # Compare with recent dialogue embeddings
        recent_embeddings = char_profile.dialogue_embeddings[-10:]  # Last 10 dialogues
        
        similarities = []
        for prev_dialogue in recent_embeddings:
            if prev_dialogue.embedding is not None:
                # Cosine similarity
                similarity = np.dot(new_embedding, prev_dialogue.embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(prev_dialogue.embedding) + 1e-8
                )
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Return average similarity (higher is more consistent)
        return np.mean(similarities)

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
        Advanced consistency checking with character evolution.
        
        Args:
            chapter: Current chapter number
            
        Returns:
            Dictionary of potential issues
        """
        issues = {
            'character_gaps': [],
            'unresolved_plots': [],
            'relationship_conflicts': [],
            'timeline_issues': [],
            'evolution_issues': [],
            'dialogue_consistency': [],
            'knowledge_conflicts': []
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
        
        # Check character evolution consistency
        for name, char in self.characters.items():
            if char.evolution_history:
                # Check for sudden personality changes
                for i in range(1, len(char.evolution_history)):
                    prev = char.evolution_history[i-1]
                    curr = char.evolution_history[i]
                    
                    # Check for dramatic personality shifts
                    for trait, change in curr.personality_changes.items():
                        if abs(change) > 0.5:  # Major change threshold
                            if not curr.trauma_events and not curr.growth_description:
                                issues['evolution_issues'].append(
                                    f"{name}: Large personality change in {trait} without justification (chapter {curr.chapter})"
                                )
        
        # Check dialogue consistency
        for name, char in self.characters.items():
            if char.dialogue_embeddings and len(char.dialogue_embeddings) > 5:
                # Check last 5 dialogues for consistency
                recent = char.dialogue_embeddings[-5:]
                for i in range(1, len(recent)):
                    if recent[i].embedding is not None and recent[i-1].embedding is not None:
                        similarity = np.dot(recent[i].embedding, recent[i-1].embedding) / (
                            np.linalg.norm(recent[i].embedding) * np.linalg.norm(recent[i-1].embedding) + 1e-8
                        )
                        if similarity < 0.3:  # Too different
                            issues['dialogue_consistency'].append(
                                f"{name}: Inconsistent dialogue style in chapter {recent[i].chapter}"
                            )
        
        # Check knowledge conflicts
        for name, char in self.characters.items():
            if char.knowledge:
                # Check for contradictory beliefs
                for false_belief in char.knowledge.false_beliefs:
                    for known_fact in char.knowledge.known_facts:
                        if self._contradicts(false_belief, known_fact):
                            issues['knowledge_conflicts'].append(
                                f"{name}: Holds contradictory beliefs: '{false_belief}' vs '{known_fact}'"
                            )

        return issues
    
    def _contradicts(self, belief1: str, belief2: str) -> bool:
        """Simple contradiction checker (can be enhanced with NLP)."""
        # Simple keyword-based contradiction detection
        negations = ['not', 'never', 'no', "n't", 'false', 'wrong']
        
        b1_words = belief1.lower().split()
        b2_words = belief2.lower().split()
        
        # Check if one negates the other
        b1_has_neg = any(neg in b1_words for neg in negations)
        b2_has_neg = any(neg in b2_words for neg in negations)
        
        # Check for shared key terms
        shared_terms = set(b1_words) & set(b2_words)
        shared_terms -= set(['the', 'a', 'an', 'is', 'are', 'was', 'were'])
        
        # If they share key terms but one is negative, likely contradiction
        if len(shared_terms) > 2 and (b1_has_neg != b2_has_neg):
            return True
        
        return False

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

    def track_character_evolution(self, character_name: str, chapter: int, 
                                 evolution: CharacterEvolution):
        """Track character evolution through chapters."""
        if character_name not in self.characters:
            raise ValueError(f"Character {character_name} not found")
        
        char = self.characters[character_name]
        
        # Apply personality changes
        for trait, change in evolution.personality_changes.items():
            if trait == 'openness':
                char.ocean_personality.openness = max(0, min(1, char.ocean_personality.openness + change))
            elif trait == 'conscientiousness':
                char.ocean_personality.conscientiousness = max(0, min(1, char.ocean_personality.conscientiousness + change))
            elif trait == 'extraversion':
                char.ocean_personality.extraversion = max(0, min(1, char.ocean_personality.extraversion + change))
            elif trait == 'agreeableness':
                char.ocean_personality.agreeableness = max(0, min(1, char.ocean_personality.agreeableness + change))
            elif trait == 'neuroticism':
                char.ocean_personality.neuroticism = max(0, min(1, char.ocean_personality.neuroticism + change))
        
        # Update knowledge base
        for skill in evolution.skills_gained:
            char.knowledge.known_facts.add(f"Has skill: {skill}")
        
        # Store evolution history
        char.evolution_history.append(evolution)
        if character_name not in self.character_evolution_tracker:
            self.character_evolution_tracker[character_name] = []
        self.character_evolution_tracker[character_name].append(evolution)
        
        self._save_character(char)
        self.logger.info(f"Tracked evolution for {character_name} in chapter {chapter}")
    
    def _update_character_knowledge(self, character_name: str, dialogue: str, 
                                   context: str, chapter: int):
        """Update character's knowledge base from dialogue."""
        if character_name not in self.characters:
            return
        
        char = self.characters[character_name]
        
        # Extract potential facts from dialogue (simplified)
        # In production, use NER and information extraction
        
        # Look for character mentions
        for other_char in self.characters:
            if other_char != character_name and other_char in dialogue:
                char.knowledge.known_characters[other_char] = context
        
        # Store the event
        if context:
            char.knowledge.known_events.append((chapter, context[:100]))
    
    def generate_voice_synthesis_prompt(self, character_name: str, text: str, 
                                       emotion: str = "neutral") -> str:
        """Generate voice synthesis parameters for unique speech."""
        if character_name not in self.characters:
            return text
        
        char = self.characters[character_name]
        voice = char.voice_pattern
        
        prompt = f"""Voice Synthesis Parameters for {character_name}:
        
        Base Text: {text}
        Emotion: {emotion}
        
        Voice Characteristics:
        - Pitch Variation: {voice.pitch_variation:.1%} (0=monotone, 100%=highly varied)
        - Speaking Pace: {voice.speaking_pace:.1%} (0=very slow, 100%=very fast)
        - Pause Frequency: {voice.pause_frequency:.1%}
        
        Vocal Tics: {', '.join(voice.vocal_tics) if voice.vocal_tics else 'None'}
        Emphasis Patterns: {', '.join(voice.emphasis_patterns) if voice.emphasis_patterns else 'Standard'}
        
        Personality Influence:
        - Extraversion: {char.ocean_personality.extraversion:.1%} (affects volume and energy)
        - Neuroticism: {char.ocean_personality.neuroticism:.1%} (affects tremor and hesitation)
        - Openness: {char.ocean_personality.openness:.1%} (affects intonation variety)
        
        Apply these characteristics to make the speech unique to this character.
        """
        
        return prompt
    
    def get_character_knowledge_base(self, character_name: str) -> CharacterKnowledge:
        """Get a character's knowledge base."""
        if character_name not in self.characters:
            return CharacterKnowledge()
        
        return self.characters[character_name].knowledge
    
    def export_all_characters(self, filepath: str = "characters_export.json"):
        """Export all characters with advanced features."""
        data = {
            'exported_at': datetime.now().isoformat(),
            'characters': {
                name: char.to_dict()
                for name, char in self.characters.items()
            },
            'dialogue_history': [asdict(d) for d in self.dialogue_history],
            'plot_points': [asdict(p) for p in self.plot_tracker],
            'relationships': {
                f"{k[0]}-{k[1]}": {
                    'type': v.get('type').value if hasattr(v.get('type'), 'value') else v.get('type'),
                    'description': v.get('description', ''),
                    'dynamics': v.get('dynamics', ''),
                    'strength': v.get('strength', 0.5),
                    'compatibility': v.get('compatibility', 0.5)
                }
                for k, v in self.relationship_matrix.items()
            },
            'interaction_matrix': {
                f"{k[0]}-{k[1]}": v
                for k, v in self.get_interaction_matrix().items()
            },
            'evolution_tracking': {
                name: [asdict(e) for e in evolutions]
                for name, evolutions in self.character_evolution_tracker.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Exported character data to {filepath}")

# Note: Character manager should be obtained from ProjectManager
# to ensure proper project isolation. Use:
# from project_manager import get_project_manager
# pm = get_project_manager()
# character_manager = pm.get_character_manager()
