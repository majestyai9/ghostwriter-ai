"""
Services package for Ghostwriter AI.

This package contains various service modules for the application,
including generation services and the comprehensive prompt management system.
"""

from .generation_service import GenerationService
from .prompt_config import (
    AudienceLevel,
    ContentDepth,
    DomainConfig,
    PromptConfig,
    PromptOptimizer,
    StyleProfile,
    WritingTone,
    DOMAIN_CONFIGS,
    STYLE_PROFILES,
)
from .prompt_service import (
    PromptCache,
    PromptLanguage,
    PromptMetrics,
    PromptService,
    PromptStyle,
    PromptTemplate,
    PromptType,
    get_prompt_service,
)
from .prompt_wrapper import (
    chapter,
    chapter_topics,
    clear_prompt_cache,
    compose_prompts,
    get_prompt_metrics,
    register_custom_prompt,
    reload_prompts,
    render_prompt,
    section,
    section_topics,
    summary,
    table_of_contents,
    title,
)

__all__ = [
    # Generation Service
    "GenerationService",
    
    # Prompt Service
    "PromptService",
    "PromptTemplate",
    "PromptLanguage",
    "PromptStyle",
    "PromptType",
    "PromptMetrics",
    "PromptCache",
    "get_prompt_service",
    
    # Prompt Configuration
    "PromptConfig",
    "StyleProfile",
    "WritingTone",
    "ContentDepth",
    "AudienceLevel",
    "DomainConfig",
    "PromptOptimizer",
    "STYLE_PROFILES",
    "DOMAIN_CONFIGS",
    
    # Prompt Wrapper Functions
    "title",
    "table_of_contents",
    "summary",
    "chapter_topics",
    "chapter",
    "section_topics",
    "section",
    "render_prompt",
    "compose_prompts",
    "register_custom_prompt",
    "get_prompt_metrics",
    "reload_prompts",
    "clear_prompt_cache",
]