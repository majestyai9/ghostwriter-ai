"""
Dialogue Enhancement System for improving character dialogue quality and uniqueness.

This module ensures dialogue is character-specific, eliminates clichés, and maintains
unique speech patterns for each character.
"""

import re
import random
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DialoguePattern:
    """Represents a character's unique dialogue pattern."""
    greetings: List[str] = field(default_factory=list)
    farewells: List[str] = field(default_factory=list)
    exclamations: List[str] = field(default_factory=list)
    filler_words: List[str] = field(default_factory=list)
    unique_phrases: List[str] = field(default_factory=list)
    speech_quirks: List[str] = field(default_factory=list)
    formality_level: str = "neutral"  # casual, neutral, formal
    contractions_usage: str = "normal"  # never, rare, normal, frequent
    sentence_starters: List[str] = field(default_factory=list)
    
    
class DialogueEnhancer:
    """
    Enhances dialogue quality by personalizing speech patterns and removing clichés.
    
    This system ensures each character has unique, consistent speech patterns
    and eliminates overused phrases.
    """
    
    # Common spy thriller clichés to avoid
    SPY_CLICHES = [
        "The name's Bond",
        "I work alone",
        "Trust no one",
        "This is above your pay grade",
        "Need to know basis",
        "Off the grid",
        "Dark web",
        "Go dark",
        "Eyes only",
        "Burn after reading",
        "We have a situation",
        "Copy that",
        "Roger that",
        "Stand down",
        "Abort mission",
        "Extract the package",
        "The eagle has landed",
        "Radio silence",
        "Going in hot",
        "Locked and loaded"
    ]
    
    # Generic dialogue clichés
    GENERIC_CLICHES = [
        "To be honest",
        "At the end of the day",
        "It is what it is",
        "Think outside the box",
        "Give 110%",
        "Take it to the next level",
        "Circle back",
        "Touch base",
        "Low-hanging fruit",
        "Win-win situation",
        "Game changer",
        "Move the needle",
        "Paradigm shift",
        "Synergy",
        "Leverage"
    ]
    
    # Character archetypes with default patterns
    ARCHETYPE_PATTERNS = {
        'military': {
            'greetings': ['Sir', 'Ma\'am', 'Officer'],
            'farewells': ['Dismissed', 'Carry on', 'That is all'],
            'exclamations': ['Affirmative', 'Negative', 'Copy'],
            'formality': 'formal',
            'contractions': 'rare'
        },
        'scientist': {
            'greetings': ['Greetings', 'Hello there', 'Good to see you'],
            'farewells': ['Until next time', 'I must return to my work', 'Farewell'],
            'exclamations': ['Fascinating', 'Remarkable', 'Extraordinary', 'Curious'],
            'formality': 'formal',
            'contractions': 'normal'
        },
        'street_smart': {
            'greetings': ['Yo', 'Hey', 'What\'s up', '\'Sup'],
            'farewells': ['Later', 'Catch you later', 'Peace out', 'I\'m out'],
            'exclamations': ['Damn', 'Hell yeah', 'No way', 'For real'],
            'formality': 'casual',
            'contractions': 'frequent'
        },
        'aristocrat': {
            'greetings': ['Good evening', 'How delightful', 'Charmed'],
            'farewells': ['Until we meet again', 'Farewell', 'Good day'],
            'exclamations': ['Indeed', 'Quite', 'Rather', 'Precisely'],
            'formality': 'formal',
            'contractions': 'never'
        },
        'tech_expert': {
            'greetings': ['Hey', 'What\'s the status', 'Talk to me'],
            'farewells': ['Signing off', 'Going offline', 'Catch you on the flip'],
            'exclamations': ['Boom', 'Bingo', 'Gotcha', 'Sweet'],
            'formality': 'casual',
            'contractions': 'frequent'
        }
    }
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize the Dialogue Enhancer.
        
        Args:
            save_path: Optional path to save dialogue patterns
        """
        self.save_path = save_path or Path("projects/dialogue_patterns.json")
        self.character_patterns: Dict[str, DialoguePattern] = {}
        self.used_phrases: Dict[str, Set[str]] = defaultdict(set)
        self.dialogue_history: List[Tuple[str, str]] = []  # (character, dialogue)
        
        self._load_patterns()
    
    def _load_patterns(self) -> None:
        """Load saved dialogue patterns if available."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for char_name, pattern_data in data.get('patterns', {}).items():
                        self.character_patterns[char_name] = DialoguePattern(**pattern_data)
                    self.used_phrases = defaultdict(set, data.get('used_phrases', {}))
                    
                logger.info(f"Loaded dialogue patterns from {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not load dialogue patterns: {e}")
    
    def _save_patterns(self) -> None:
        """Save dialogue patterns for persistence."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            patterns_data = {}
            for char_name, pattern in self.character_patterns.items():
                patterns_data[char_name] = {
                    'greetings': pattern.greetings,
                    'farewells': pattern.farewells,
                    'exclamations': pattern.exclamations,
                    'filler_words': pattern.filler_words,
                    'unique_phrases': pattern.unique_phrases,
                    'speech_quirks': pattern.speech_quirks,
                    'formality_level': pattern.formality_level,
                    'contractions_usage': pattern.contractions_usage,
                    'sentence_starters': pattern.sentence_starters
                }
            
            data = {
                'patterns': patterns_data,
                'used_phrases': {k: list(v) for k, v in self.used_phrases.items()}
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved dialogue patterns to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save dialogue patterns: {e}")
    
    def personalize_speech(self, character_name: str, dialogue: str, 
                          character_type: Optional[str] = None) -> str:
        """
        Personalize dialogue for a specific character.
        
        Args:
            character_name: Name of the character
            dialogue: The dialogue to personalize
            character_type: Optional character archetype
            
        Returns:
            Personalized dialogue
        """
        # Get or create character pattern
        if character_name not in self.character_patterns:
            self._create_character_pattern(character_name, character_type)
        
        pattern = self.character_patterns[character_name]
        personalized = dialogue
        
        # Apply formality level
        personalized = self._apply_formality(personalized, pattern.formality_level)
        
        # Apply contractions usage
        personalized = self._apply_contractions(personalized, pattern.contractions_usage)
        
        # Add speech quirks
        personalized = self._add_speech_quirks(personalized, pattern)
        
        # Add filler words occasionally
        if random.random() < 0.3 and pattern.filler_words:
            filler = random.choice(pattern.filler_words)
            personalized = self._insert_filler(personalized, filler)
        
        # Replace generic greetings/farewells
        personalized = self._replace_greetings_farewells(personalized, pattern)
        
        # Track usage
        self.dialogue_history.append((character_name, personalized))
        self.used_phrases[character_name].add(personalized[:50])  # Track beginning
        
        self._save_patterns()
        return personalized
    
    def _create_character_pattern(self, name: str, archetype: Optional[str] = None) -> None:
        """Create a unique speech pattern for a character."""
        pattern = DialoguePattern()
        
        # Use archetype as base if provided
        if archetype and archetype in self.ARCHETYPE_PATTERNS:
            base = self.ARCHETYPE_PATTERNS[archetype]
            pattern.greetings = base.get('greetings', []).copy()
            pattern.farewells = base.get('farewells', []).copy()
            pattern.exclamations = base.get('exclamations', []).copy()
            pattern.formality_level = base.get('formality', 'neutral')
            pattern.contractions_usage = base.get('contractions', 'normal')
        
        # Add unique variations
        pattern.unique_phrases = self._generate_unique_phrases(name)
        pattern.speech_quirks = self._generate_speech_quirks()
        pattern.filler_words = self._generate_filler_words(pattern.formality_level)
        pattern.sentence_starters = self._generate_sentence_starters(pattern.formality_level)
        
        self.character_patterns[name] = pattern
        logger.debug(f"Created dialogue pattern for {name}")
    
    def _generate_unique_phrases(self, character_name: str) -> List[str]:
        """Generate unique catchphrases for a character."""
        # These would be more sophisticated in production
        templates = [
            f"As I always say",
            f"You know what they say",
            f"In my experience",
            f"If you ask me",
            f"The way I see it",
            f"Here's the thing",
            f"Let me tell you something",
            f"Between you and me"
        ]
        
        # Select 2-3 random templates
        return random.sample(templates, min(3, len(templates)))
    
    def _generate_speech_quirks(self) -> List[str]:
        """Generate random speech quirks."""
        quirks = [
            "repeats_last_word",  # "Yes, yes"
            "starts_with_so",  # "So, here's what..."
            "ends_with_right",  # "...right?"
            "uses_metaphors",
            "asks_rhetorical_questions",
            "trails_off",  # "..."
            "emphasizes_words",  # "VERY important"
            "self_corrects"  # "I mean..."
        ]
        
        # Select 1-2 quirks
        num_quirks = random.randint(1, 2)
        return random.sample(quirks, num_quirks)
    
    def _generate_filler_words(self, formality: str) -> List[str]:
        """Generate appropriate filler words based on formality."""
        if formality == 'formal':
            return ['indeed', 'certainly', 'of course', 'naturally']
        elif formality == 'casual':
            return ['like', 'you know', 'I mean', 'basically', 'literally']
        else:
            return ['well', 'actually', 'honestly', 'really']
    
    def _generate_sentence_starters(self, formality: str) -> List[str]:
        """Generate sentence starters based on formality."""
        if formality == 'formal':
            return ['I must say', 'One might argue', 'It appears that', 
                   'I would suggest', 'Perhaps we should']
        elif formality == 'casual':
            return ['Look', 'Listen', 'Okay so', 'Thing is', 'Honestly']
        else:
            return ['I think', 'Maybe', 'We should', 'Let\'s', 'How about']
    
    def _apply_formality(self, dialogue: str, formality: str) -> str:
        """Apply formality level to dialogue."""
        if formality == 'formal':
            # Remove casual contractions and slang
            dialogue = dialogue.replace("gonna", "going to")
            dialogue = dialogue.replace("wanna", "want to")
            dialogue = dialogue.replace("gotta", "have to")
            dialogue = dialogue.replace("ain't", "is not")
            dialogue = dialogue.replace("yeah", "yes")
            dialogue = dialogue.replace("nope", "no")
        elif formality == 'casual':
            # Add casual elements
            dialogue = dialogue.replace("going to", "gonna")
            dialogue = dialogue.replace("want to", "wanna")
            dialogue = dialogue.replace("yes", "yeah")
            dialogue = dialogue.replace("no", "nah")
        
        return dialogue
    
    def _apply_contractions(self, dialogue: str, usage: str) -> str:
        """Apply contraction usage pattern."""
        if usage == 'never':
            # Expand all contractions
            contractions = {
                "don't": "do not", "won't": "will not", "can't": "cannot",
                "isn't": "is not", "aren't": "are not", "wasn't": "was not",
                "weren't": "were not", "haven't": "have not", "hasn't": "has not",
                "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
                "couldn't": "could not", "mightn't": "might not", "mustn't": "must not",
                "I'm": "I am", "you're": "you are", "he's": "he is",
                "she's": "she is", "it's": "it is", "we're": "we are",
                "they're": "they are", "I've": "I have", "you've": "you have",
                "we've": "we have", "they've": "they have", "I'd": "I would",
                "you'd": "you would", "he'd": "he would", "she'd": "she would",
                "we'd": "we would", "they'd": "they would", "I'll": "I will",
                "you'll": "you will", "he'll": "he will", "she'll": "she will",
                "we'll": "we will", "they'll": "they will"
            }
            for contraction, expanded in contractions.items():
                dialogue = dialogue.replace(contraction, expanded)
        elif usage == 'frequent':
            # Contract where possible (simplified)
            expansions = {
                "do not": "don't", "will not": "won't", "cannot": "can't",
                "is not": "isn't", "are not": "aren't", "was not": "wasn't",
                "were not": "weren't", "have not": "haven't", "has not": "hasn't",
                "I am": "I'm", "you are": "you're", "he is": "he's",
                "she is": "she's", "it is": "it's", "we are": "we're",
                "they are": "they're"
            }
            for expanded, contraction in expansions.items():
                dialogue = dialogue.replace(expanded, contraction)
        
        return dialogue
    
    def _add_speech_quirks(self, dialogue: str, pattern: DialoguePattern) -> str:
        """Add speech quirks to dialogue."""
        for quirk in pattern.speech_quirks:
            if quirk == "repeats_last_word":
                # Repeat last word occasionally
                if random.random() < 0.2:
                    words = dialogue.split()
                    if words:
                        last_word = words[-1].rstrip('.,!?')
                        dialogue = f"{dialogue.rstrip('.,!?')}, {last_word}."
            
            elif quirk == "starts_with_so" and random.random() < 0.3:
                if not dialogue.startswith(("So,", "So ")):
                    dialogue = f"So, {dialogue[0].lower()}{dialogue[1:]}"
            
            elif quirk == "ends_with_right" and random.random() < 0.2:
                if not dialogue.endswith("?"):
                    dialogue = f"{dialogue.rstrip('.')}, right?"
            
            elif quirk == "trails_off" and random.random() < 0.15:
                dialogue = f"{dialogue.rstrip('.')}..."
        
        return dialogue
    
    def _insert_filler(self, dialogue: str, filler: str) -> str:
        """Insert filler word into dialogue naturally."""
        sentences = dialogue.split('. ')
        if len(sentences) > 1:
            # Insert filler at beginning of a middle sentence
            insert_pos = random.randint(1, len(sentences) - 1)
            sentences[insert_pos] = f"{filler.capitalize()}, {sentences[insert_pos][0].lower()}{sentences[insert_pos][1:]}"
            return '. '.join(sentences)
        return dialogue
    
    def _replace_greetings_farewells(self, dialogue: str, pattern: DialoguePattern) -> str:
        """Replace generic greetings and farewells with character-specific ones."""
        # Generic greetings to replace
        generic_greetings = ['Hello', 'Hi', 'Hey', 'Greetings']
        generic_farewells = ['Goodbye', 'Bye', 'See you', 'Farewell']
        
        # Check and replace greetings at the beginning
        for greeting in generic_greetings:
            if dialogue.startswith(greeting):
                if pattern.greetings:
                    replacement = random.choice(pattern.greetings)
                    dialogue = dialogue.replace(greeting, replacement, 1)
                break
        
        # Check and replace farewells at the end
        for farewell in generic_farewells:
            if any(dialogue.endswith(f"{farewell}{end}") for end in ['.', '!', '']):
                if pattern.farewells:
                    replacement = random.choice(pattern.farewells)
                    for end in ['.', '!', '']:
                        dialogue = dialogue.replace(f"{farewell}{end}", f"{replacement}{end}")
                break
        
        return dialogue
    
    def remove_cliches(self, dialogue: str, genre: str = 'thriller') -> str:
        """
        Remove clichéd phrases from dialogue.
        
        Args:
            dialogue: The dialogue to clean
            genre: The genre (affects which clichés to check)
            
        Returns:
            Dialogue with clichés removed or replaced
        """
        cleaned = dialogue
        
        # Check for genre-specific clichés
        cliches_to_check = self.GENERIC_CLICHES.copy()
        if genre == 'thriller' or genre == 'spy':
            cliches_to_check.extend(self.SPY_CLICHES)
        
        # Replace clichés with alternatives
        for cliche in cliches_to_check:
            if cliche.lower() in cleaned.lower():
                alternative = self._get_cliche_alternative(cliche)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(cliche), re.IGNORECASE)
                cleaned = pattern.sub(alternative, cleaned)
        
        return cleaned
    
    def _get_cliche_alternative(self, cliche: str) -> str:
        """Get an alternative for a clichéd phrase."""
        alternatives = {
            "I work alone": ["I prefer to operate independently", "Solo is how I operate", 
                           "Partners complicate things"],
            "Trust no one": ["Keep your guard up", "Everyone has an agenda", 
                           "Verify everything"],
            "Need to know basis": ["Classified information", "Restricted access", 
                                  "Not for general distribution"],
            "We have a situation": ["There's been a development", "Something's come up", 
                                   "We have a problem"],
            "Copy that": ["Understood", "Acknowledged", "Got it"],
            "It is what it is": ["That's the reality", "We deal with what we have", 
                               "No point dwelling on it"],
            "At the end of the day": ["Ultimately", "When all is said and done", 
                                     "The bottom line is"]
        }
        
        if cliche in alternatives:
            return random.choice(alternatives[cliche])
        
        # If no specific alternative, just remove it
        return ""
    
    def generate_unique_phrases(self, character_name: str, count: int = 5) -> List[str]:
        """
        Generate unique phrases for a character to use.
        
        Args:
            character_name: Character name
            count: Number of phrases to generate
            
        Returns:
            List of unique phrases
        """
        if character_name not in self.character_patterns:
            self._create_character_pattern(character_name)
        
        pattern = self.character_patterns[character_name]
        phrases = []
        
        # Generate based on character's style
        if pattern.formality_level == 'formal':
            phrase_templates = [
                "I must insist that {}",
                "One cannot simply {}",
                "It would be prudent to {}",
                "I dare say {}",
                "Rest assured, {}"
            ]
        elif pattern.formality_level == 'casual':
            phrase_templates = [
                "No way I'm {}",
                "You gotta be kidding me with {}",
                "Seriously though, {}",
                "Not gonna lie, {}",
                "For real, {}"
            ]
        else:
            phrase_templates = [
                "I have to say {}",
                "The thing is {}",
                "You need to understand {}",
                "Look at it this way: {}",
                "Here's what I think: {}"
            ]
        
        # Generate unique completions
        completions = [
            "we need to reconsider",
            "this changes everything",
            "we're running out of time",
            "there's more to this",
            "we can handle this"
        ]
        
        for template in random.sample(phrase_templates, min(count, len(phrase_templates))):
            completion = random.choice(completions)
            phrases.append(template.format(completion))
        
        # Add to character's unique phrases
        pattern.unique_phrases.extend(phrases)
        self._save_patterns()
        
        return phrases
    
    def check_dialogue_uniqueness(self, dialogue: str, character_name: str) -> bool:
        """
        Check if dialogue is unique for this character.
        
        Args:
            dialogue: Dialogue to check
            character_name: Character name
            
        Returns:
            True if unique, False if too similar to previous dialogue
        """
        # Check if very similar dialogue was already used
        dialogue_start = dialogue[:50].lower()
        
        for used_phrase in self.used_phrases[character_name]:
            if self._calculate_similarity(dialogue_start, used_phrase.lower()) > 0.8:
                return False
        
        return True
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (simple version)."""
        if not s1 or not s2:
            return 0.0
        
        # Count matching characters
        matches = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        return matches / max(len(s1), len(s2))
    
    def enhance_dialogue_prompt(self, character_name: str, 
                              context: str, emotion: str = "neutral") -> str:
        """
        Generate a prompt for creating enhanced dialogue.
        
        Args:
            character_name: Character name
            context: Scene context
            emotion: Character's emotional state
            
        Returns:
            Enhanced dialogue generation prompt
        """
        if character_name not in self.character_patterns:
            self._create_character_pattern(character_name)
        
        pattern = self.character_patterns[character_name]
        
        prompt = f"""
        Generate dialogue for {character_name} with these characteristics:
        
        Speech Style:
        - Formality: {pattern.formality_level}
        - Contractions: {pattern.contractions_usage}
        - Unique phrases: {', '.join(pattern.unique_phrases[:3]) if pattern.unique_phrases else 'None yet'}
        
        Current emotion: {emotion}
        Context: {context}
        
        Requirements:
        1. Make dialogue unique and character-specific
        2. Avoid these clichés: {', '.join(random.sample(self.SPY_CLICHES + self.GENERIC_CLICHES, 5))}
        3. Include character's speech patterns and quirks
        4. Sound natural and conversational
        5. Express the character's personality through word choice
        
        Speech quirks to include:
        {', '.join(pattern.speech_quirks) if pattern.speech_quirks else 'None'}
        
        Preferred greetings: {', '.join(pattern.greetings) if pattern.greetings else 'Standard'}
        Preferred exclamations: {', '.join(pattern.exclamations) if pattern.exclamations else 'Standard'}
        """
        
        return prompt
    
    def enhance_chapter_dialogue(self, content: str, chapter_num: int) -> str:
        """
        Enhance all dialogue in a chapter.
        
        Args:
            content: Chapter content
            chapter_num: Chapter number
            
        Returns:
            Enhanced chapter content
        """
        import re
        
        # Find all dialogue in the chapter
        dialogue_pattern = r'"([^"]*)"'
        dialogues = re.findall(dialogue_pattern, content)
        
        enhanced_content = content
        
        for dialogue in dialogues:
            # Skip very short dialogue
            if len(dialogue) < 10:
                continue
            
            # Remove clichés from dialogue
            enhanced_dialogue = self.remove_cliches(dialogue)
            
            # Only replace if we made changes
            if enhanced_dialogue != dialogue:
                enhanced_content = enhanced_content.replace(
                    f'"{dialogue}"',
                    f'"{enhanced_dialogue}"',
                    1  # Replace only first occurrence
                )
        
        return enhanced_content
    
    def analyze_dialogue_diversity(self, chapter_text: str) -> Dict[str, any]:
        """
        Analyze dialogue diversity in a chapter.
        
        Args:
            chapter_text: Chapter text to analyze
            
        Returns:
            Analysis results
        """
        dialogue_pieces = re.findall(r'"([^"]+)"', chapter_text)
        
        analysis = {
            'total_dialogue_pieces': len(dialogue_pieces),
            'unique_starters': len(set(d.split()[0].lower() for d in dialogue_pieces if d.split())),
            'average_length': sum(len(d.split()) for d in dialogue_pieces) / len(dialogue_pieces) if dialogue_pieces else 0,
            'cliche_count': 0,
            'repetitive_phrases': []
        }
        
        # Check for clichés
        all_cliches = self.SPY_CLICHES + self.GENERIC_CLICHES
        for dialogue in dialogue_pieces:
            for cliche in all_cliches:
                if cliche.lower() in dialogue.lower():
                    analysis['cliche_count'] += 1
        
        # Check for repetitive phrases
        phrase_count = defaultdict(int)
        for dialogue in dialogue_pieces:
            # Check first 5 words
            words = dialogue.split()[:5]
            if len(words) >= 3:
                phrase = ' '.join(words[:3])
                phrase_count[phrase] += 1
        
        analysis['repetitive_phrases'] = [
            phrase for phrase, count in phrase_count.items() if count > 2
        ]
        
        return analysis