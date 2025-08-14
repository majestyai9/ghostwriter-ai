"""
Style Templates System for Different Writing Genres
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class StyleTemplate:
    """Writing style template configuration"""
    name: str
    category: str  # fiction, non-fiction, academic, etc.
    tone: str
    vocabulary_level: str  # simple, moderate, advanced, technical
    sentence_structure: str  # simple, complex, varied
    paragraph_length: str  # short, medium, long
    features: List[str]
    prompt_modifiers: Dict[str, str]
    content_warnings: List[str] = None
    age_rating: str = "general"
    
    def to_dict(self):
        return {
            'name': self.name,
            'category': self.category,
            'tone': self.tone,
            'vocabulary_level': self.vocabulary_level,
            'sentence_structure': self.sentence_structure,
            'paragraph_length': self.paragraph_length,
            'features': self.features,
            'prompt_modifiers': self.prompt_modifiers,
            'content_warnings': self.content_warnings or [],
            'age_rating': self.age_rating
        }

class StyleManager:
    """Manages writing style templates"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
        self.custom_templates = {}
        self.current_style = None
        
    def _load_default_templates(self) -> Dict[str, StyleTemplate]:
        """Load default style templates"""
        return {
            # Academic Styles
            'academic': StyleTemplate(
                name='academic',
                category='non-fiction',
                tone='formal, objective, analytical',
                vocabulary_level='advanced',
                sentence_structure='complex',
                paragraph_length='long',
                features=['citations', 'footnotes', 'bibliography', 'abstract'],
                prompt_modifiers={
                    'base': "Write in a formal academic style with scholarly language. Use passive voice where appropriate. Include evidence-based arguments.",
                    'title': "Create an academic title that clearly states the research focus",
                    'chapter': "Write with clear thesis statements, supporting evidence, and analytical depth",
                    'section': "Include topic sentences, evidence, analysis, and transitions"
                }
            ),
            
            'technical': StyleTemplate(
                name='technical',
                category='non-fiction',
                tone='precise, instructional, clear',
                vocabulary_level='technical',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['code examples', 'diagrams', 'step-by-step', 'specifications'],
                prompt_modifiers={
                    'base': "Write in a clear technical style. Be precise and unambiguous. Use technical terminology correctly.",
                    'title': "Create a descriptive technical title",
                    'chapter': "Structure content logically with clear explanations and examples",
                    'section': "Break down complex concepts into understandable steps"
                }
            ),
            
            # Fiction Styles
            'literary_fiction': StyleTemplate(
                name='literary_fiction',
                category='fiction',
                tone='sophisticated, nuanced, reflective',
                vocabulary_level='advanced',
                sentence_structure='varied',
                paragraph_length='varied',
                features=['metaphors', 'symbolism', 'character_depth', 'themes'],
                prompt_modifiers={
                    'base': "Write literary fiction with rich prose, deep characterization, and thematic depth. Use vivid imagery and metaphors.",
                    'title': "Create an evocative, meaningful title",
                    'chapter': "Focus on character development, internal conflict, and symbolic meaning",
                    'section': "Blend action with introspection, use sensory details"
                }
            ),
            
            'thriller': StyleTemplate(
                name='thriller',
                category='fiction',
                tone='tense, fast-paced, suspenseful',
                vocabulary_level='moderate',
                sentence_structure='simple',
                paragraph_length='short',
                features=['cliffhangers', 'action', 'suspense', 'plot_twists'],
                prompt_modifiers={
                    'base': "Write in a fast-paced thriller style. Short, punchy sentences. Build tension and suspense.",
                    'title': "Create an intriguing, suspenseful title",
                    'chapter': "End with cliffhangers. Keep action moving. Build tension.",
                    'section': "Use short paragraphs, active voice, immediate action"
                }
            ),
            
            'romance': StyleTemplate(
                name='romance',
                category='fiction',
                tone='emotional, passionate, intimate',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['emotions', 'relationships', 'dialogue', 'chemistry'],
                prompt_modifiers={
                    'base': "Write romantic fiction focusing on emotional connections, chemistry, and relationship development.",
                    'title': "Create an emotionally evocative romantic title",
                    'chapter': "Focus on emotional beats, romantic tension, character chemistry",
                    'section': "Develop intimate moments, emotional dialogue, sensory details"
                }
            ),
            
            'erotic_romance': StyleTemplate(
                name='erotic_romance',
                category='fiction',
                tone='sensual, passionate, explicit',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='varied',
                features=['intimacy', 'sensuality', 'consent', 'chemistry'],
                prompt_modifiers={
                    'base': "Write adult romantic fiction with explicit sensual content. Focus on consent, chemistry, and emotional connection alongside physical intimacy.",
                    'title': "Create a provocative yet tasteful title",
                    'chapter': "Balance emotional development with sensual scenes. Ensure consent is clear.",
                    'section': "Use sensual language appropriately. Focus on both emotional and physical connection."
                },
                content_warnings=['sexual content', 'adult themes'],
                age_rating='18+'
            ),
            
            'fantasy': StyleTemplate(
                name='fantasy',
                category='fiction',
                tone='imaginative, epic, adventurous',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['worldbuilding', 'magic_systems', 'quests', 'mythology'],
                prompt_modifiers={
                    'base': "Write fantasy with rich worldbuilding, magical elements, and epic storytelling.",
                    'title': "Create an epic, imaginative fantasy title",
                    'chapter': "Include worldbuilding details, magical elements, adventure",
                    'section': "Balance action with world description, develop magic system"
                }
            ),
            
            'scifi': StyleTemplate(
                name='scifi',
                category='fiction',
                tone='speculative, technological, futuristic',
                vocabulary_level='technical',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['technology', 'speculation', 'worldbuilding', 'concepts'],
                prompt_modifiers={
                    'base': "Write science fiction with plausible technology, speculative concepts, and futuristic settings.",
                    'title': "Create an intriguing sci-fi title",
                    'chapter': "Explore technological and societal implications",
                    'section': "Balance technical exposition with narrative flow"
                }
            ),
            
            'mystery': StyleTemplate(
                name='mystery',
                category='fiction',
                tone='intriguing, analytical, suspenseful',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['clues', 'red_herrings', 'investigation', 'revelation'],
                prompt_modifiers={
                    'base': "Write mystery fiction with careful clue placement, red herrings, and logical deduction.",
                    'title': "Create an intriguing mystery title",
                    'chapter': "Plant clues carefully, develop suspects, maintain mystery",
                    'section': "Balance revelation with misdirection"
                }
            ),
            
            'horror': StyleTemplate(
                name='horror',
                category='fiction',
                tone='dark, atmospheric, unsettling',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='varied',
                features=['atmosphere', 'fear', 'suspense', 'supernatural'],
                prompt_modifiers={
                    'base': "Write horror with atmospheric dread, building fear, and unsettling imagery.",
                    'title': "Create a chilling, ominous title",
                    'chapter': "Build atmosphere, escalate dread, include unsettling imagery",
                    'section': "Use sensory details for fear, vary pacing for effect"
                },
                content_warnings=['violence', 'disturbing content'],
                age_rating='16+'
            ),
            
            # Children's Literature
            'childrens': StyleTemplate(
                name='childrens',
                category='fiction',
                tone='friendly, educational, imaginative',
                vocabulary_level='simple',
                sentence_structure='simple',
                paragraph_length='short',
                features=['moral_lessons', 'illustrations', 'repetition', 'rhymes'],
                prompt_modifiers={
                    'base': "Write for children aged 6-10. Use simple language, positive themes, and educational elements.",
                    'title': "Create a fun, engaging children's book title",
                    'chapter': "Include moral lessons, keep language simple, be imaginative",
                    'section': "Use repetition, simple sentences, positive messages"
                },
                age_rating='6+'
            ),
            
            'young_adult': StyleTemplate(
                name='young_adult',
                category='fiction',
                tone='relatable, emotional, coming-of-age',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['identity', 'relationships', 'growth', 'challenges'],
                prompt_modifiers={
                    'base': "Write for teenagers and young adults. Address coming-of-age themes, identity, and relationships.",
                    'title': "Create a compelling YA title",
                    'chapter': "Focus on relatable conflicts, emotional growth, identity",
                    'section': "Address teen experiences authentically"
                },
                age_rating='13+'
            ),
            
            # Non-Fiction Styles
            'self_help': StyleTemplate(
                name='self_help',
                category='non-fiction',
                tone='motivational, practical, empowering',
                vocabulary_level='moderate',
                sentence_structure='simple',
                paragraph_length='short',
                features=['exercises', 'examples', 'action_steps', 'summaries'],
                prompt_modifiers={
                    'base': "Write practical self-help content with actionable advice and motivational tone.",
                    'title': "Create an inspiring, benefit-focused title",
                    'chapter': "Provide practical steps, real examples, exercises",
                    'section': "Include action items, motivational language"
                }
            ),
            
            'biography': StyleTemplate(
                name='biography',
                category='non-fiction',
                tone='narrative, factual, engaging',
                vocabulary_level='moderate',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['chronology', 'anecdotes', 'context', 'quotes'],
                prompt_modifiers={
                    'base': "Write engaging biographical content with factual accuracy and narrative flow.",
                    'title': "Create a compelling biographical title",
                    'chapter': "Balance facts with storytelling, include context",
                    'section': "Use anecdotes, maintain chronological flow"
                }
            ),
            
            'business': StyleTemplate(
                name='business',
                category='non-fiction',
                tone='professional, practical, authoritative',
                vocabulary_level='technical',
                sentence_structure='varied',
                paragraph_length='medium',
                features=['case_studies', 'frameworks', 'data', 'strategies'],
                prompt_modifiers={
                    'base': "Write professional business content with practical strategies and real-world applications.",
                    'title': "Create a professional, results-oriented title",
                    'chapter': "Include case studies, data, actionable strategies",
                    'section': "Provide frameworks, examples, implementation steps"
                }
            )
        }
        
    def get_style(self, style_name: str) -> Optional[StyleTemplate]:
        """Get a style template by name"""
        # Check custom templates first
        if style_name in self.custom_templates:
            return self.custom_templates[style_name]
        # Then check default templates
        return self.templates.get(style_name)
        
    def list_styles(self, category: str = None) -> List[str]:
        """List available styles"""
        all_styles = {**self.templates, **self.custom_templates}
        
        if category:
            return [
                name for name, style in all_styles.items()
                if style.category == category
            ]
        return list(all_styles.keys())
        
    def create_custom_style(self, 
                          name: str,
                          base_style: str = None,
                          **kwargs) -> StyleTemplate:
        """Create a custom style template"""
        if base_style and base_style in self.templates:
            # Start from existing template
            base = self.templates[base_style]
            style_dict = base.to_dict()
            style_dict.update(kwargs)
            style_dict['name'] = name
        else:
            # Create from scratch
            style_dict = {
                'name': name,
                'category': kwargs.get('category', 'custom'),
                'tone': kwargs.get('tone', 'neutral'),
                'vocabulary_level': kwargs.get('vocabulary_level', 'moderate'),
                'sentence_structure': kwargs.get('sentence_structure', 'varied'),
                'paragraph_length': kwargs.get('paragraph_length', 'medium'),
                'features': kwargs.get('features', []),
                'prompt_modifiers': kwargs.get('prompt_modifiers', {}),
                'content_warnings': kwargs.get('content_warnings', []),
                'age_rating': kwargs.get('age_rating', 'general')
            }
            
        custom_style = StyleTemplate(**style_dict)
        self.custom_templates[name] = custom_style
        
        return custom_style
        
    def apply_style_to_prompt(self, 
                             prompt: str,
                             style_name: str,
                             prompt_type: str = 'base') -> str:
        """
        Apply style modifications to a prompt
        
        Args:
            prompt: Original prompt
            style_name: Style to apply
            prompt_type: Type of prompt (base, title, chapter, section)
            
        Returns:
            Modified prompt with style applied
        """
        style = self.get_style(style_name)
        if not style:
            return prompt
            
        # Get style modifier for this prompt type
        modifier = style.prompt_modifiers.get(
            prompt_type, 
            style.prompt_modifiers.get('base', '')
        )
        
        # Apply style context
        styled_prompt = f"{modifier}\n\n{prompt}"
        
        # Add additional context based on style features
        if 'dialogue' in style.features:
            styled_prompt += "\nEnsure dialogue is natural and character-appropriate."
        if 'worldbuilding' in style.features:
            styled_prompt += "\nInclude rich worldbuilding details."
        if 'citations' in style.features:
            styled_prompt += "\nInclude appropriate citations and references."
            
        # Add content warnings if applicable
        if style.content_warnings:
            warnings = ', '.join(style.content_warnings)
            styled_prompt = f"[Content includes: {warnings}]\n\n{styled_prompt}"
            
        return styled_prompt
        
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get detailed information about a style"""
        style = self.get_style(style_name)
        if not style:
            return {}
            
        return {
            'name': style.name,
            'category': style.category,
            'description': f"{style.tone} writing with {style.vocabulary_level} vocabulary",
            'features': style.features,
            'age_rating': style.age_rating,
            'content_warnings': style.content_warnings or [],
            'suitable_for': self._get_suitable_uses(style)
        }
        
    def _get_suitable_uses(self, style: StyleTemplate) -> List[str]:
        """Determine suitable uses for a style"""
        uses = []
        
        if style.category == 'academic':
            uses.extend(['research', 'thesis', 'papers'])
        elif style.category == 'fiction':
            uses.extend(['novels', 'short stories', 'creative writing'])
        elif style.category == 'non-fiction':
            uses.extend(['guides', 'manuals', 'informational'])
            
        if style.age_rating == '18+':
            uses.append('adult content')
        elif style.age_rating == '6+':
            uses.append('children\'s books')
            
        return uses
        
    def save_custom_styles(self, filepath: str = "custom_styles.json"):
        """Save custom styles to file"""
        data = {
            name: style.to_dict() 
            for name, style in self.custom_templates.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def load_custom_styles(self, filepath: str = "custom_styles.json"):
        """Load custom styles from file"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for name, style_dict in data.items():
                self.custom_templates[name] = StyleTemplate(**style_dict)

# Global style manager
_style_manager = None

def get_style_manager() -> StyleManager:
    """Get global style manager instance"""
    global _style_manager
    if _style_manager is None:
        _style_manager = StyleManager()
    return _style_manager

def apply_style(prompt: str, style: str, prompt_type: str = 'base') -> str:
    """Convenience function to apply style to prompt"""
    manager = get_style_manager()
    return manager.apply_style_to_prompt(prompt, style, prompt_type)