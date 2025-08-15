"""
Backward compatibility wrapper for the PromptService.

This module provides wrapper functions that maintain compatibility
with the existing codebase while using the new PromptService internally.
"""

from typing import Any, Dict, Optional

from .prompt_service import PromptLanguage, PromptStyle, PromptType, get_prompt_service


def title(original_title: str) -> str:
    """
    Generate title prompt.
    
    Args:
        original_title: Initial title idea
        
    Returns:
        Rendered title prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.TITLE,
        original_title=original_title
    )


def table_of_contents(instructions: str) -> str:
    """
    Generate table of contents prompt.
    
    Args:
        instructions: Book generation instructions
        
    Returns:
        Rendered TOC prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.TABLE_OF_CONTENTS,
        instructions=instructions
    )


def summary(book: Dict[str, Any], instructions: str) -> str:
    """
    Generate summary prompt.
    
    Args:
        book: Book data dictionary
        instructions: Generation instructions
        
    Returns:
        Rendered summary prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.SUMMARY,
        title=book.get("title", "Untitled"),
        instructions=instructions
    )


def chapter_topics(
    book: Dict[str, Any],
    chapter: Dict[str, Any],
    toc_context: str
) -> str:
    """
    Generate chapter topics prompt.
    
    Args:
        book: Book data dictionary
        chapter: Chapter data dictionary
        toc_context: Table of contents context
        
    Returns:
        Rendered chapter topics prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.CHAPTER_TOPICS,
        title=book.get("title", "Untitled"),
        chapter_number=chapter["number"],
        chapter_title=chapter["title"],
        toc_context=toc_context
    )


def chapter(
    book: Dict[str, Any],
    chapter: Dict[str, Any],
    toc_context: str
) -> str:
    """
    Generate chapter content prompt.
    
    Args:
        book: Book data dictionary
        chapter: Chapter data dictionary
        toc_context: Table of contents context
        
    Returns:
        Rendered chapter prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.CHAPTER,
        title=book.get("title", "Untitled"),
        summary=book.get("summary", ""),
        chapter_number=chapter["number"],
        chapter_title=chapter["title"],
        topics=chapter.get("topics", ""),
        toc_context=toc_context
    )


def section_topics(
    book: Dict[str, Any],
    chapter: Dict[str, Any],
    section: Dict[str, Any],
    toc_context: str
) -> str:
    """
    Generate section topics prompt.
    
    Args:
        book: Book data dictionary
        chapter: Chapter data dictionary
        section: Section data dictionary
        toc_context: Table of contents context
        
    Returns:
        Rendered section topics prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.SECTION_TOPICS,
        chapter_number=chapter["number"],
        chapter_title=chapter["title"],
        section_number=section["number"],
        section_title=section["title"],
        chapter_topics=chapter.get("topics", ""),
        toc_context=toc_context
    )


def section(
    book: Dict[str, Any],
    chapter: Dict[str, Any],
    section: Dict[str, Any]
) -> str:
    """
    Generate section content prompt.
    
    Args:
        book: Book data dictionary
        chapter: Chapter data dictionary
        section: Section data dictionary
        
    Returns:
        Rendered section prompt
    """
    service = get_prompt_service()
    return service.render(
        PromptType.SECTION,
        title=book.get("title", "Untitled"),
        chapter_number=chapter["number"],
        chapter_title=chapter["title"],
        section_number=section["number"],
        section_title=section["title"],
        topics=section.get("topics", ""),
        chapter_topics=chapter.get("topics", "")
    )


# Advanced functions for customization
def render_prompt(
    template_name: str,
    language: Optional[PromptLanguage] = None,
    style: Optional[PromptStyle] = None,
    **variables: Any
) -> str:
    """
    Render a custom prompt with specific language and style.
    
    Args:
        template_name: Name of the template
        language: Target language
        style: Writing style
        **variables: Template variables
        
    Returns:
        Rendered prompt
    """
    service = get_prompt_service()
    return service.render(
        template_name,
        language=language,
        style=style,
        **variables
    )


def compose_prompts(*template_names: str, separator: str = "\n\n", **variables: Any) -> str:
    """
    Compose multiple prompts into one.
    
    Args:
        *template_names: Names of templates to compose
        separator: String to join prompts
        **variables: Variables for all templates
        
    Returns:
        Composed prompt string
    """
    service = get_prompt_service()
    return service.compose(*template_names, separator=separator, **variables)


def register_custom_prompt(
    name: str,
    template: str,
    description: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Register a custom prompt template.
    
    Args:
        name: Template name
        template: Template string
        description: Template description
        **kwargs: Additional template parameters
    """
    service = get_prompt_service()
    service.register_template(name, template, description, **kwargs)


def get_prompt_metrics(template_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get usage metrics for prompts.
    
    Args:
        template_name: Specific template or None for all
        
    Returns:
        Dictionary of metrics
    """
    service = get_prompt_service()
    return service.get_metrics(template_name)


def reload_prompts() -> None:
    """Reload all prompt templates from disk."""
    service = get_prompt_service()
    service.reload_templates()


def clear_prompt_cache() -> None:
    """Clear the prompt cache."""
    service = get_prompt_service()
    service.clear_cache()