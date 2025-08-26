"""
Chapter Validator for enforcing quality and length requirements in book generation.

This module ensures chapters meet minimum length requirements, maintains quality standards,
and automatically expands content when necessary.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ChapterQuality(Enum):
    """Chapter quality rating levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class ChapterMetrics:
    """Metrics for evaluating chapter quality."""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    dialogue_percentage: float = 0.0
    average_sentence_length: float = 0.0
    average_paragraph_length: float = 0.0
    unique_words: int = 0
    vocabulary_richness: float = 0.0
    scene_count: int = 0
    pacing_score: float = 0.0
    emotional_depth_score: float = 0.0
    sensory_detail_score: float = 0.0
    
    
@dataclass
class ValidationResult:
    """Result of chapter validation."""
    is_valid: bool
    quality: ChapterQuality
    metrics: ChapterMetrics
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    expansion_needed: bool = False
    expansion_amount: int = 0  # Words needed


class ChapterValidator:
    """
    Validates and enhances chapter quality and length.
    
    This validator ensures chapters meet minimum requirements for length,
    quality, and narrative depth.
    """
    
    # Minimum requirements
    MIN_WORDS_PER_CHAPTER = 6000
    MIN_SENTENCES_PER_PARAGRAPH = 3
    MIN_PARAGRAPHS_PER_CHAPTER = 20
    MIN_SCENES_PER_CHAPTER = 3
    MIN_DIALOGUE_PERCENTAGE = 0.2
    MAX_DIALOGUE_PERCENTAGE = 0.7
    
    # Quality thresholds
    VOCABULARY_RICHNESS_THRESHOLD = 0.4  # Unique words / total words
    MIN_SENSORY_DETAILS_PER_SCENE = 3
    MIN_EMOTIONAL_BEATS_PER_SCENE = 2
    
    # Sensory and emotional keywords for scoring
    SENSORY_WORDS = {
        'sight': ['saw', 'looked', 'glanced', 'gazed', 'peered', 'watched', 'observed',
                 'bright', 'dark', 'colorful', 'shadowy', 'gleaming', 'dim', 'vivid'],
        'sound': ['heard', 'listened', 'whispered', 'shouted', 'murmured', 'echoed',
                 'silent', 'loud', 'quiet', 'noisy', 'rustling', 'buzzing', 'clicking'],
        'touch': ['felt', 'touched', 'grabbed', 'smooth', 'rough', 'soft', 'hard',
                 'cold', 'warm', 'hot', 'wet', 'dry', 'sticky', 'slippery'],
        'smell': ['smelled', 'sniffed', 'aroma', 'fragrance', 'stench', 'odor',
                 'perfume', 'scent', 'musty', 'fresh', 'acrid', 'sweet'],
        'taste': ['tasted', 'savored', 'bitter', 'sweet', 'sour', 'salty', 'spicy',
                 'delicious', 'bland', 'flavored', 'tangy']
    }
    
    EMOTION_WORDS = [
        'happy', 'sad', 'angry', 'afraid', 'surprised', 'disgusted', 'anxious',
        'excited', 'nervous', 'confident', 'embarrassed', 'proud', 'ashamed',
        'guilty', 'jealous', 'lonely', 'loved', 'hatred', 'joy', 'sorrow',
        'fear', 'rage', 'terror', 'delight', 'despair', 'hope', 'dread'
    ]
    
    def __init__(self, min_words: Optional[int] = None):
        """
        Initialize the Chapter Validator.
        
        Args:
            min_words: Optional custom minimum word count
        """
        if min_words:
            self.MIN_WORDS_PER_CHAPTER = min_words
    
    def validate_chapter(self, chapter_text: str, chapter_number: int) -> ValidationResult:
        """
        Validate a chapter for quality and length.
        
        Args:
            chapter_text: The chapter content
            chapter_number: The chapter number
            
        Returns:
            ValidationResult with metrics and issues
        """
        metrics = self._calculate_metrics(chapter_text)
        issues = []
        suggestions = []
        
        # Check word count
        if metrics.word_count < self.MIN_WORDS_PER_CHAPTER:
            word_deficit = self.MIN_WORDS_PER_CHAPTER - metrics.word_count
            issues.append(
                f"Chapter {chapter_number} is too short: {metrics.word_count} words "
                f"(need {word_deficit} more)"
            )
            suggestions.append(f"Expand chapter by adding {word_deficit} words of content")
        
        # Check paragraph count
        if metrics.paragraph_count < self.MIN_PARAGRAPHS_PER_CHAPTER:
            issues.append(
                f"Insufficient paragraphs: {metrics.paragraph_count} "
                f"(minimum {self.MIN_PARAGRAPHS_PER_CHAPTER})"
            )
            suggestions.append("Add more paragraph breaks and develop scenes further")
        
        # Check scene count
        if metrics.scene_count < self.MIN_SCENES_PER_CHAPTER:
            issues.append(
                f"Too few scenes: {metrics.scene_count} "
                f"(minimum {self.MIN_SCENES_PER_CHAPTER})"
            )
            suggestions.append("Add more scene transitions and location changes")
        
        # Check dialogue balance
        if metrics.dialogue_percentage < self.MIN_DIALOGUE_PERCENTAGE:
            issues.append(
                f"Insufficient dialogue: {metrics.dialogue_percentage:.1%} "
                f"(minimum {self.MIN_DIALOGUE_PERCENTAGE:.1%})"
            )
            suggestions.append("Add more character dialogue and conversations")
        elif metrics.dialogue_percentage > self.MAX_DIALOGUE_PERCENTAGE:
            issues.append(
                f"Too much dialogue: {metrics.dialogue_percentage:.1%} "
                f"(maximum {self.MAX_DIALOGUE_PERCENTAGE:.1%})"
            )
            suggestions.append("Add more narrative description and action")
        
        # Check vocabulary richness
        if metrics.vocabulary_richness < self.VOCABULARY_RICHNESS_THRESHOLD:
            issues.append(
                f"Limited vocabulary: {metrics.vocabulary_richness:.1%} unique words"
            )
            suggestions.append("Use more varied vocabulary and avoid word repetition")
        
        # Check sensory details
        if metrics.sensory_detail_score < 0.5:
            issues.append("Insufficient sensory details")
            suggestions.append("Add more descriptions using sight, sound, touch, smell, and taste")
        
        # Check emotional depth
        if metrics.emotional_depth_score < 0.5:
            issues.append("Lacking emotional depth")
            suggestions.append("Add more emotional reactions and internal character thoughts")
        
        # Determine quality rating
        quality = self._determine_quality(metrics, len(issues))
        
        # Determine if expansion is needed
        expansion_needed = metrics.word_count < self.MIN_WORDS_PER_CHAPTER
        expansion_amount = max(0, self.MIN_WORDS_PER_CHAPTER - metrics.word_count)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality=quality,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions,
            expansion_needed=expansion_needed,
            expansion_amount=expansion_amount
        )
    
    def _calculate_metrics(self, text: str) -> ChapterMetrics:
        """Calculate detailed metrics for a chapter."""
        metrics = ChapterMetrics()
        
        # Basic counts
        words = text.split()
        metrics.word_count = len(words)
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        metrics.sentence_count = len(sentences)
        
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        metrics.paragraph_count = len(paragraphs)
        
        # Dialogue analysis
        dialogue_matches = re.findall(r'"[^"]*"', text)
        dialogue_words = sum(len(d.split()) for d in dialogue_matches)
        metrics.dialogue_percentage = dialogue_words / metrics.word_count if metrics.word_count > 0 else 0
        
        # Average lengths
        if metrics.sentence_count > 0:
            metrics.average_sentence_length = metrics.word_count / metrics.sentence_count
        
        if metrics.paragraph_count > 0:
            metrics.average_paragraph_length = metrics.word_count / metrics.paragraph_count
        
        # Vocabulary richness
        unique_words = set(word.lower() for word in words)
        metrics.unique_words = len(unique_words)
        metrics.vocabulary_richness = metrics.unique_words / metrics.word_count if metrics.word_count > 0 else 0
        
        # Scene detection (simple heuristic based on scene breaks)
        scene_markers = ['* * *', '***', '---', '• • •']
        metrics.scene_count = 1  # Start with 1 for the initial scene
        for marker in scene_markers:
            metrics.scene_count += text.count(marker)
        
        # Detect major time/location shifts as scene changes
        time_shifts = len(re.findall(
            r'\b(later|afterward|next day|following morning|that evening|hours later)\b',
            text, re.IGNORECASE
        ))
        metrics.scene_count += time_shifts // 2  # Conservative estimate
        
        # Pacing score (variation in sentence lengths)
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            if len(sentence_lengths) > 1:
                metrics.pacing_score = min(1.0, statistics.stdev(sentence_lengths) / 10)
            else:
                metrics.pacing_score = 0.5
        
        # Sensory detail score
        metrics.sensory_detail_score = self._calculate_sensory_score(text)
        
        # Emotional depth score
        metrics.emotional_depth_score = self._calculate_emotional_score(text)
        
        return metrics
    
    def _calculate_sensory_score(self, text: str) -> float:
        """Calculate sensory detail score based on sensory word usage."""
        text_lower = text.lower()
        total_sensory_words = 0
        
        for sense, words in self.SENSORY_WORDS.items():
            for word in words:
                total_sensory_words += text_lower.count(word)
        
        # Normalize by text length (per 1000 words)
        word_count = len(text.split())
        if word_count > 0:
            sensory_density = (total_sensory_words / word_count) * 1000
            # Score from 0 to 1, where 20 sensory words per 1000 is considered good
            return min(1.0, sensory_density / 20)
        
        return 0.0
    
    def _calculate_emotional_score(self, text: str) -> float:
        """Calculate emotional depth score based on emotion word usage."""
        text_lower = text.lower()
        total_emotion_words = 0
        
        for word in self.EMOTION_WORDS:
            total_emotion_words += text_lower.count(word)
        
        # Also check for emotional indicators
        emotional_indicators = [
            r'felt\s+\w+',
            r'heart\s+(pounded|raced|sank|soared)',
            r'tears\s+(fell|streamed|welled)',
            r'smile[d]?\s+(widened|faded|appeared)',
            r'frown[ed]?',
            r'sigh[ed]?',
            r'trembl[ed|ing]'
        ]
        
        for pattern in emotional_indicators:
            total_emotion_words += len(re.findall(pattern, text_lower))
        
        # Normalize by text length
        word_count = len(text.split())
        if word_count > 0:
            emotional_density = (total_emotion_words / word_count) * 1000
            # Score from 0 to 1, where 15 emotion words per 1000 is considered good
            return min(1.0, emotional_density / 15)
        
        return 0.0
    
    def _determine_quality(self, metrics: ChapterMetrics, issue_count: int) -> ChapterQuality:
        """Determine overall chapter quality based on metrics."""
        score = 0
        max_score = 10
        
        # Word count score (3 points)
        if metrics.word_count >= self.MIN_WORDS_PER_CHAPTER:
            score += 3
        elif metrics.word_count >= self.MIN_WORDS_PER_CHAPTER * 0.8:
            score += 2
        elif metrics.word_count >= self.MIN_WORDS_PER_CHAPTER * 0.6:
            score += 1
        
        # Structure score (2 points)
        if metrics.paragraph_count >= self.MIN_PARAGRAPHS_PER_CHAPTER:
            score += 1
        if metrics.scene_count >= self.MIN_SCENES_PER_CHAPTER:
            score += 1
        
        # Dialogue balance (1 point)
        if self.MIN_DIALOGUE_PERCENTAGE <= metrics.dialogue_percentage <= self.MAX_DIALOGUE_PERCENTAGE:
            score += 1
        
        # Vocabulary richness (1 point)
        if metrics.vocabulary_richness >= self.VOCABULARY_RICHNESS_THRESHOLD:
            score += 1
        
        # Sensory and emotional depth (2 points)
        if metrics.sensory_detail_score >= 0.5:
            score += 1
        if metrics.emotional_depth_score >= 0.5:
            score += 1
        
        # Pacing (1 point)
        if metrics.pacing_score >= 0.5:
            score += 1
        
        # Determine quality level
        percentage = (score / max_score) * 100
        
        if percentage >= 90 and issue_count == 0:
            return ChapterQuality.EXCELLENT
        elif percentage >= 75 and issue_count <= 2:
            return ChapterQuality.GOOD
        elif percentage >= 60 and issue_count <= 4:
            return ChapterQuality.ACCEPTABLE
        elif percentage >= 40:
            return ChapterQuality.NEEDS_IMPROVEMENT
        else:
            return ChapterQuality.POOR
    
    def enforce_minimum_length(self, chapter_text: str, target_words: int) -> str:
        """
        Enforce minimum chapter length by identifying expansion points.
        
        Args:
            chapter_text: The current chapter text
            target_words: Target word count
            
        Returns:
            Expanded chapter text (placeholder - actual expansion done by LLM)
        """
        current_words = len(chapter_text.split())
        words_needed = target_words - current_words
        
        if words_needed <= 0:
            return chapter_text
        
        # This is a placeholder - actual expansion would be done by the LLM
        # with specific prompts to expand scenes
        expansion_prompt = self.generate_expansion_prompt(chapter_text, words_needed)
        
        logger.info(f"Chapter needs {words_needed} more words. Expansion prompt generated.")
        
        # Return original text with expansion instructions
        # (actual expansion will be handled by the generation service)
        return chapter_text
    
    def generate_expansion_prompt(self, chapter_text: str, words_needed: int) -> str:
        """
        Generate a prompt to guide chapter expansion.
        
        Args:
            chapter_text: Current chapter text
            words_needed: Number of words to add
            
        Returns:
            Expansion prompt for the LLM
        """
        metrics = self._calculate_metrics(chapter_text)
        expansion_areas = []
        
        # Identify areas for expansion
        if metrics.sensory_detail_score < 0.7:
            expansion_areas.append(
                "Add more sensory details: describe what characters see, hear, "
                "smell, taste, and feel. Paint vivid pictures of the environment."
            )
        
        if metrics.emotional_depth_score < 0.7:
            expansion_areas.append(
                "Deepen emotional content: explore characters' internal thoughts, "
                "feelings, reactions, and emotional responses to events."
            )
        
        if metrics.dialogue_percentage < 0.3:
            expansion_areas.append(
                "Add more dialogue: include conversations between characters, "
                "revealing personality and advancing the plot through speech."
            )
        
        if metrics.scene_count < 5:
            expansion_areas.append(
                "Develop existing scenes further: add more action, description, "
                "and character interaction within each scene."
            )
        
        # Always suggest these expansions
        expansion_areas.extend([
            "Elaborate on character actions and motivations",
            "Add descriptive passages about settings and atmospheres",
            "Include more specific details about objects, clothing, and technology",
            "Develop subplot elements and secondary character interactions"
        ])
        
        prompt = f"""
        EXPANSION REQUIRED: This chapter needs {words_needed} additional words.
        
        Current metrics:
        - Word count: {metrics.word_count}
        - Scenes: {metrics.scene_count}
        - Dialogue: {metrics.dialogue_percentage:.1%}
        - Sensory details: {metrics.sensory_detail_score:.1%}
        - Emotional depth: {metrics.emotional_depth_score:.1%}
        
        Areas to expand:
        {chr(10).join(f'- {area}' for area in expansion_areas)}
        
        Requirements:
        1. Maintain story continuity and character consistency
        2. Add substantive content, not filler
        3. Enhance dramatic tension and pacing
        4. Deepen character development
        5. Create immersive, cinematic scenes
        
        Target: Expand to at least {metrics.word_count + words_needed} words.
        """
        
        return prompt
    
    def validate_quality(self, chapter_text: str) -> Tuple[bool, List[str]]:
        """
        Quick quality validation check.
        
        Args:
            chapter_text: Chapter text to validate
            
        Returns:
            Tuple of (is_acceptable, list_of_quality_issues)
        """
        quality_issues = []
        
        # Check for common quality problems
        if chapter_text.count('.') < 50:
            quality_issues.append("Too few sentences - needs more development")
        
        if not re.search(r'"[^"]{10,}"', chapter_text):
            quality_issues.append("No substantial dialogue found")
        
        # Check for repetitive words (excluding common words)
        words = chapter_text.lower().split()
        word_freq = {}
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                       'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                       'he', 'she', 'it', 'they', 'them', 'their', 'his', 'her'}
        
        for word in words:
            if word not in common_words and len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repetitive_words = [word for word, count in word_freq.items() if count > 10]
        if repetitive_words:
            quality_issues.append(
                f"Repetitive vocabulary: {', '.join(repetitive_words[:5])}"
            )
        
        # Check for very short paragraphs
        paragraphs = [p for p in chapter_text.split('\n\n') if p.strip()]
        short_paragraphs = [p for p in paragraphs if len(p.split()) < 20]
        if len(short_paragraphs) > len(paragraphs) * 0.5:
            quality_issues.append("Too many short paragraphs - needs more development")
        
        # Check for scene transitions
        if not re.search(
            r'(later|afterward|meanwhile|then|suddenly|finally)', 
            chapter_text, re.IGNORECASE
        ):
            quality_issues.append("Lacks clear scene transitions")
        
        is_acceptable = len(quality_issues) == 0
        return is_acceptable, quality_issues
    
    def generate_quality_enforcement_prompt(self) -> str:
        """
        Generate a prompt that enforces quality standards.
        
        Returns:
            Quality enforcement prompt for chapter generation
        """
        return f"""
        QUALITY REQUIREMENTS FOR THIS CHAPTER:
        
        Length Requirements:
        - MINIMUM {self.MIN_WORDS_PER_CHAPTER} words per chapter (currently generating much less)
        - At least {self.MIN_PARAGRAPHS_PER_CHAPTER} paragraphs
        - At least {self.MIN_SCENES_PER_CHAPTER} distinct scenes or scene segments
        
        Content Requirements:
        - {self.MIN_DIALOGUE_PERCENTAGE:.0%}-{self.MAX_DIALOGUE_PERCENTAGE:.0%} of content should be dialogue
        - Include rich sensory details (sight, sound, touch, smell, taste)
        - Deep emotional content and character thoughts
        - Varied sentence structure for good pacing
        - Use diverse vocabulary (avoid repetition)
        
        Scene Development:
        - Each scene must have: setting, action, dialogue, emotional beat
        - Include specific, concrete details (not generic descriptions)
        - Show character emotions through actions and body language
        - Create immersive atmospheres with environmental details
        
        Narrative Techniques:
        - Use "show, don't tell" for emotions and character traits
        - Include internal monologue and character reflections
        - Build tension and suspense within scenes
        - End with a hook or cliffhanger to drive the story forward
        
        IMPORTANT: Generate complete, fully-developed chapters. Short chapters are unacceptable.
        Each chapter should feel like a complete mini-story within the larger narrative.
        """
    
    def expand_chapter(self, chapter_text: str, expansion_instructions: str) -> str:
        """
        Placeholder for chapter expansion logic.
        
        In practice, this would interface with the LLM to expand the chapter
        based on the provided instructions.
        
        Args:
            chapter_text: Original chapter text
            expansion_instructions: Instructions for expansion
            
        Returns:
            Expanded chapter text (placeholder)
        """
        # This is a placeholder - actual implementation would call the LLM
        logger.info("Chapter expansion requested with specific instructions")
        return chapter_text