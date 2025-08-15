"""
Configuration and customization for the PromptService.

This module provides configuration classes and utilities for customizing
prompt generation behavior, including style profiles, language settings,
and domain-specific configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WritingTone(str, Enum):
    """Available writing tones."""
    
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"
    SERIOUS = "serious"


class ContentDepth(str, Enum):
    """Content depth levels."""
    
    SURFACE = "surface"
    MODERATE = "moderate"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"


class AudienceLevel(str, Enum):
    """Target audience expertise levels."""
    
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    GENERAL = "general"
    CHILDREN = "children"
    YOUNG_ADULT = "young_adult"
    ACADEMIC = "academic"
    PROFESSIONAL = "professional"


@dataclass
class StyleProfile:
    """
    Defines a writing style profile for consistent prompt generation.
    """
    
    name: str
    tone: WritingTone = WritingTone.PROFESSIONAL
    depth: ContentDepth = ContentDepth.MODERATE
    audience: AudienceLevel = AudienceLevel.GENERAL
    formality: float = 0.5  # 0.0 (very casual) to 1.0 (very formal)
    creativity: float = 0.5  # 0.0 (factual) to 1.0 (creative)
    technicality: float = 0.5  # 0.0 (simple) to 1.0 (technical)
    word_choices: List[str] = field(default_factory=list)
    avoid_words: List[str] = field(default_factory=list)
    example_count: int = 2
    use_metaphors: bool = False
    use_analogies: bool = True
    use_statistics: bool = False
    use_quotes: bool = False
    
    def to_prompt_context(self) -> str:
        """Convert style profile to prompt context string."""
        context_parts = [
            f"Writing tone: {self.tone.value}",
            f"Content depth: {self.depth.value}",
            f"Target audience: {self.audience.value}",
        ]
        
        if self.formality > 0.7:
            context_parts.append("Use formal language and structure")
        elif self.formality < 0.3:
            context_parts.append("Use casual, conversational language")
        
        if self.creativity > 0.7:
            context_parts.append("Be creative and imaginative")
        elif self.creativity < 0.3:
            context_parts.append("Focus on facts and practical information")
        
        if self.technicality > 0.7:
            context_parts.append("Include technical details and terminology")
        elif self.technicality < 0.3:
            context_parts.append("Use simple, accessible language")
        
        if self.use_metaphors:
            context_parts.append("Include relevant metaphors")
        if self.use_analogies:
            context_parts.append("Use analogies to explain concepts")
        if self.use_statistics:
            context_parts.append("Include supporting statistics")
        if self.use_quotes:
            context_parts.append("Include relevant quotes")
        
        if self.word_choices:
            context_parts.append(f"Preferred vocabulary: {', '.join(self.word_choices)}")
        if self.avoid_words:
            context_parts.append(f"Avoid using: {', '.join(self.avoid_words)}")
        
        return "\n".join(context_parts)


# Predefined style profiles
STYLE_PROFILES = {
    "academic": StyleProfile(
        name="academic",
        tone=WritingTone.ACADEMIC,
        depth=ContentDepth.COMPREHENSIVE,
        audience=AudienceLevel.ACADEMIC,
        formality=0.9,
        creativity=0.2,
        technicality=0.8,
        use_statistics=True,
        use_quotes=True,
        example_count=3
    ),
    "business": StyleProfile(
        name="business",
        tone=WritingTone.PROFESSIONAL,
        depth=ContentDepth.DETAILED,
        audience=AudienceLevel.PROFESSIONAL,
        formality=0.7,
        creativity=0.3,
        technicality=0.5,
        use_statistics=True,
        example_count=2
    ),
    "creative": StyleProfile(
        name="creative",
        tone=WritingTone.CREATIVE,
        depth=ContentDepth.MODERATE,
        audience=AudienceLevel.GENERAL,
        formality=0.3,
        creativity=0.9,
        technicality=0.2,
        use_metaphors=True,
        use_analogies=True,
        example_count=3
    ),
    "technical": StyleProfile(
        name="technical",
        tone=WritingTone.TECHNICAL,
        depth=ContentDepth.EXPERT,
        audience=AudienceLevel.ADVANCED,
        formality=0.6,
        creativity=0.2,
        technicality=0.9,
        use_statistics=True,
        example_count=4
    ),
    "casual": StyleProfile(
        name="casual",
        tone=WritingTone.CONVERSATIONAL,
        depth=ContentDepth.MODERATE,
        audience=AudienceLevel.GENERAL,
        formality=0.2,
        creativity=0.6,
        technicality=0.2,
        use_analogies=True,
        example_count=2
    ),
    "children": StyleProfile(
        name="children",
        tone=WritingTone.CASUAL,
        depth=ContentDepth.SURFACE,
        audience=AudienceLevel.CHILDREN,
        formality=0.1,
        creativity=0.8,
        technicality=0.1,
        use_metaphors=True,
        use_analogies=True,
        avoid_words=["complex", "difficult", "complicated"],
        example_count=3
    )
}


class PromptConfig(BaseModel):
    """
    Configuration for prompt generation behavior.
    """
    
    # Style settings
    style_profile: Optional[str] = Field(
        default="business",
        description="Name of the style profile to use"
    )
    custom_style: Optional[StyleProfile] = Field(
        default=None,
        description="Custom style profile (overrides style_profile)"
    )
    
    # Content settings
    min_word_count: int = Field(default=100, ge=10)
    max_word_count: int = Field(default=5000, le=10000)
    default_chapter_word_count: int = Field(default=2500)
    default_section_word_count: int = Field(default=1000)
    
    # Structure settings
    default_chapter_count: int = Field(default=15, ge=5, le=50)
    default_sections_per_chapter: int = Field(default=4, ge=2, le=10)
    
    # Generation settings
    include_examples: bool = Field(default=True)
    include_summaries: bool = Field(default=True)
    include_transitions: bool = Field(default=True)
    include_key_takeaways: bool = Field(default=True)
    
    # Advanced settings
    use_chapter_continuity: bool = Field(
        default=True,
        description="Maintain continuity between chapters"
    )
    use_progressive_depth: bool = Field(
        default=True,
        description="Gradually increase complexity through the book"
    )
    use_recap_sections: bool = Field(
        default=False,
        description="Include recap sections at chapter starts"
    )
    
    # Customization
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Additional instructions for all prompts"
    )
    domain_specific_terms: List[str] = Field(
        default_factory=list,
        description="Domain-specific terminology to use"
    )
    references_style: Optional[str] = Field(
        default=None,
        description="Citation style (APA, MLA, Chicago, etc.)"
    )
    
    def get_style_profile(self) -> StyleProfile:
        """Get the active style profile."""
        if self.custom_style:
            return self.custom_style
        
        return STYLE_PROFILES.get(
            self.style_profile,
            STYLE_PROFILES["business"]
        )
    
    def apply_to_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration to prompt variables.
        
        Args:
            variables: Original prompt variables
            
        Returns:
            Updated variables with configuration applied
        """
        updated = variables.copy()
        
        # Apply style profile
        style = self.get_style_profile()
        updated["style"] = style.tone.value
        updated["audience"] = style.audience.value
        updated["tone"] = style.tone.value
        
        # Apply word counts
        if "word_count" not in updated:
            if "chapter" in variables.get("template_name", ""):
                updated["word_count"] = f"{self.default_chapter_word_count}"
            elif "section" in variables.get("template_name", ""):
                updated["word_count"] = f"{self.default_section_word_count}"
        
        # Apply structure settings
        updated.setdefault("chapter_count", self.default_chapter_count)
        updated.setdefault("sections_per_chapter", self.default_sections_per_chapter)
        
        # Apply generation settings
        if self.include_examples:
            updated.setdefault("examples_count", style.example_count)
            updated.setdefault("example_type", "practical")
        
        if self.include_key_takeaways:
            updated.setdefault("callout_count", "2-3")
        
        # Apply custom instructions
        if self.custom_instructions:
            current_instructions = updated.get("instructions", "")
            updated["instructions"] = f"{current_instructions}\n\n{self.custom_instructions}"
        
        # Apply domain terms
        if self.domain_specific_terms:
            updated["domain_terms"] = ", ".join(self.domain_specific_terms)
        
        return updated


class DomainConfig(BaseModel):
    """
    Domain-specific configuration for specialized content generation.
    """
    
    domain: str = Field(description="Domain name (e.g., 'fiction', 'technical', 'academic')")
    
    # Domain-specific templates
    template_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Template overrides for this domain"
    )
    
    # Vocabulary
    required_terms: List[str] = Field(
        default_factory=list,
        description="Terms that must be included"
    )
    forbidden_terms: List[str] = Field(
        default_factory=list,
        description="Terms to avoid"
    )
    
    # Structure
    chapter_structure: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom chapter structure"
    )
    
    # Formatting
    formatting_rules: Dict[str, str] = Field(
        default_factory=dict,
        description="Domain-specific formatting rules"
    )
    
    # Validation
    validation_rules: List[str] = Field(
        default_factory=list,
        description="Rules for validating generated content"
    )


# Predefined domain configurations
DOMAIN_CONFIGS = {
    "fiction": DomainConfig(
        domain="fiction",
        template_overrides={
            "chapter": "Write a compelling narrative chapter...",
            "section": "Continue the story with focus on..."
        },
        required_terms=["protagonist", "conflict", "resolution"],
        chapter_structure={
            "include_scene_breaks": True,
            "include_dialogue": True,
            "narrative_perspective": "third_person"
        }
    ),
    "technical": DomainConfig(
        domain="technical",
        template_overrides={
            "chapter": "Provide technical documentation for...",
            "section": "Detail the implementation of..."
        },
        required_terms=["implementation", "architecture", "specification"],
        formatting_rules={
            "code_style": "syntax_highlighted",
            "diagrams": "mermaid"
        }
    ),
    "academic": DomainConfig(
        domain="academic",
        template_overrides={
            "chapter": "Present scholarly analysis of...",
            "section": "Examine the theoretical framework..."
        },
        required_terms=["hypothesis", "methodology", "conclusion"],
        formatting_rules={
            "citations": "APA",
            "footnotes": "enabled"
        },
        validation_rules=[
            "Must include citations",
            "Must have clear thesis statement"
        ]
    )
}


class PromptOptimizer:
    """
    Optimizes prompts for better results and token efficiency.
    """
    
    @staticmethod
    def optimize_for_tokens(prompt: str, max_tokens: int = 4000) -> str:
        """
        Optimize a prompt to fit within token limits.
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum token count
            
        Returns:
            Optimized prompt
        """
        # Simple optimization - in practice, use tokenizer
        if len(prompt) > max_tokens * 4:  # Rough estimate
            # Trim less important parts
            lines = prompt.split("\n")
            essential_lines = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) < max_tokens * 4:
                    essential_lines.append(line)
                    current_length += len(line)
                else:
                    break
            
            return "\n".join(essential_lines)
        
        return prompt
    
    @staticmethod
    def add_context_window(
        prompt: str,
        previous_context: str,
        max_context_tokens: int = 1000
    ) -> str:
        """
        Add previous context to a prompt.
        
        Args:
            prompt: Current prompt
            previous_context: Previous generation context
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Prompt with context
        """
        if not previous_context:
            return prompt
        
        # Truncate context if needed
        if len(previous_context) > max_context_tokens * 4:
            previous_context = previous_context[-(max_context_tokens * 4):]
        
        return f"Previous context:\n{previous_context}\n\n{prompt}"
    
    @staticmethod
    def add_constraints(prompt: str, constraints: List[str]) -> str:
        """
        Add constraints to a prompt.
        
        Args:
            prompt: Original prompt
            constraints: List of constraints
            
        Returns:
            Prompt with constraints
        """
        if not constraints:
            return prompt
        
        constraint_text = "\n".join(f"- {c}" for c in constraints)
        return f"{prompt}\n\nConstraints:\n{constraint_text}"