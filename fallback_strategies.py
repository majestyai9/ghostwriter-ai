"""
Fallback strategies for content generation failures.
Provides multiple recovery mechanisms when primary generation fails.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import time
import json

from tracing import trace_span, record_event

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Types of fallback strategies."""
    RETRY_WITH_SIMPLIFIED = "retry_with_simplified"
    SWITCH_PROVIDER = "switch_provider"
    USE_CACHED_SIMILAR = "use_cached_similar"
    GENERATE_PLACEHOLDER = "generate_placeholder"
    PARTIAL_CONTENT = "partial_content"
    TEMPLATE_BASED = "template_based"


@dataclass
class FallbackContext:
    """Context for fallback operations."""
    original_prompt: str
    original_provider: str
    original_error: str
    attempt_number: int
    chapter_number: Optional[int] = None
    section_number: Optional[int] = None
    book_title: Optional[str] = None
    language: str = "English"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FallbackManager:
    """Manages fallback strategies for content generation."""
    
    def __init__(self):
        """Initialize fallback manager."""
        self.strategies: Dict[FallbackStrategy, Callable] = {
            FallbackStrategy.RETRY_WITH_SIMPLIFIED: self._retry_with_simplified_prompt,
            FallbackStrategy.SWITCH_PROVIDER: self._switch_to_alternative_provider,
            FallbackStrategy.USE_CACHED_SIMILAR: self._use_cached_similar_content,
            FallbackStrategy.GENERATE_PLACEHOLDER: self._generate_placeholder_content,
            FallbackStrategy.PARTIAL_CONTENT: self._generate_partial_content,
            FallbackStrategy.TEMPLATE_BASED: self._use_template_based_generation
        }
        
        # Provider fallback chain
        self.provider_chain = [
            "openai",
            "anthropic",
            "gemini",
            "cohere",
            "openrouter"
        ]
    
    def execute_fallback(
        self,
        context: FallbackContext,
        strategies: Optional[List[FallbackStrategy]] = None
    ) -> Optional[str]:
        """
        Execute fallback strategies in order until one succeeds.
        
        Args:
            context: Fallback context
            strategies: List of strategies to try (default: all)
            
        Returns:
            Generated content or None if all strategies fail
        """
        if strategies is None:
            strategies = [
                FallbackStrategy.RETRY_WITH_SIMPLIFIED,
                FallbackStrategy.SWITCH_PROVIDER,
                FallbackStrategy.USE_CACHED_SIMILAR,
                FallbackStrategy.PARTIAL_CONTENT,
                FallbackStrategy.TEMPLATE_BASED,
                FallbackStrategy.GENERATE_PLACEHOLDER
            ]
        
        with trace_span("fallback.execution", {
            "original_provider": context.original_provider,
            "strategies_count": len(strategies)
        }):
            for strategy in strategies:
                try:
                    logger.info(f"Attempting fallback strategy: {strategy.value}")
                    record_event("fallback.strategy.attempt", {
                        "strategy": strategy.value,
                        "attempt": context.attempt_number
                    })
                    
                    if strategy in self.strategies:
                        result = self.strategies[strategy](context)
                        
                        if result:
                            logger.info(f"Fallback strategy {strategy.value} succeeded")
                            record_event("fallback.strategy.success", {
                                "strategy": strategy.value
                            })
                            return result
                    
                except Exception as e:
                    logger.error(f"Fallback strategy {strategy.value} failed: {e}")
                    record_event("fallback.strategy.failed", {
                        "strategy": strategy.value,
                        "error": str(e)
                    })
            
            logger.error("All fallback strategies exhausted")
            return None
    
    def _retry_with_simplified_prompt(self, context: FallbackContext) -> Optional[str]:
        """
        Retry with a simplified version of the prompt.
        
        Args:
            context: Fallback context
            
        Returns:
            Generated content or None
        """
        logger.info("Retrying with simplified prompt")
        
        try:
            from providers.factory import ProviderFactory
            from app_config import settings
            
            # Simplify the prompt
            simplified_prompt = self._simplify_prompt(context.original_prompt)
            
            # Get the original provider
            provider = ProviderFactory.create_provider(
                context.original_provider,
                {"api_key": getattr(settings, f"{context.original_provider.upper()}_API_KEY", None)}
            )
            
            # Try with reduced token limit and temperature
            result = provider.generate(
                simplified_prompt,
                max_tokens=1000,  # Reduced token limit
                temperature=0.3  # Lower temperature for more predictable output
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Simplified prompt retry failed: {e}")
            return None
    
    def _switch_to_alternative_provider(self, context: FallbackContext) -> Optional[str]:
        """
        Switch to an alternative provider.
        
        Args:
            context: Fallback context
            
        Returns:
            Generated content or None
        """
        logger.info("Switching to alternative provider")
        
        try:
            from providers.factory import ProviderFactory
            from app_config import settings
            
            # Find next provider in chain
            current_index = self.provider_chain.index(context.original_provider)
            
            for i in range(current_index + 1, len(self.provider_chain)):
                alternative_provider = self.provider_chain[i]
                
                # Check if API key exists for this provider
                api_key = getattr(settings, f"{alternative_provider.upper()}_API_KEY", None)
                if not api_key:
                    continue
                
                try:
                    logger.info(f"Trying provider: {alternative_provider}")
                    
                    provider = ProviderFactory.create_provider(
                        alternative_provider,
                        {"api_key": api_key}
                    )
                    
                    result = provider.generate(context.original_prompt)
                    
                    if result:
                        logger.info(f"Successfully generated with {alternative_provider}")
                        return result
                        
                except Exception as e:
                    logger.warning(f"Provider {alternative_provider} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Provider switch failed: {e}")
            return None
    
    def _use_cached_similar_content(self, context: FallbackContext) -> Optional[str]:
        """
        Use cached content from similar chapters/sections.
        
        Args:
            context: Fallback context
            
        Returns:
            Generated content or None
        """
        logger.info("Searching for cached similar content")
        
        try:
            from cache_manager import CacheManager
            
            cache = CacheManager(backend="memory")
            
            # Build cache key patterns for similar content
            if context.chapter_number:
                # Try adjacent chapters
                for offset in [-1, 1, -2, 2]:
                    similar_chapter = context.chapter_number + offset
                    if similar_chapter > 0:
                        cache_key = f"chapter_{similar_chapter}_content"
                        cached_content = cache.get(cache_key)
                        
                        if cached_content:
                            logger.info(f"Found similar content from chapter {similar_chapter}")
                            # Modify the content to indicate it's adapted
                            return self._adapt_content(cached_content, context)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None
    
    def _generate_partial_content(self, context: FallbackContext) -> Optional[str]:
        """
        Generate partial content with reduced scope.
        
        Args:
            context: Fallback context
            
        Returns:
            Generated content or None
        """
        logger.info("Generating partial content")
        
        try:
            from providers.factory import ProviderFactory
            from app_config import settings
            
            # Create a prompt for partial content
            partial_prompt = f"""
            Generate a brief summary or outline for the following request.
            This is a simplified version due to technical limitations.
            
            Original request: {context.original_prompt[:500]}...
            
            Provide key points and main ideas in a concise format.
            """
            
            # Try with the original provider first
            provider = ProviderFactory.create_provider(
                context.original_provider,
                {"api_key": getattr(settings, f"{context.original_provider.upper()}_API_KEY", None)}
            )
            
            result = provider.generate(
                partial_prompt,
                max_tokens=500,
                temperature=0.5
            )
            
            if result:
                return f"[Partial Content - Full generation unavailable]\n\n{result}"
            
            return None
            
        except Exception as e:
            logger.error(f"Partial content generation failed: {e}")
            return None
    
    def _use_template_based_generation(self, context: FallbackContext) -> Optional[str]:
        """
        Use template-based generation for structured content.
        
        Args:
            context: Fallback context
            
        Returns:
            Generated content or None
        """
        logger.info("Using template-based generation")
        
        try:
            # Define templates for different content types
            templates = {
                "chapter": """
Chapter {chapter_number}: [Title]

Introduction:
[A brief introduction to the chapter's main themes and objectives]

Main Content:
1. [First main point or event]
   - [Supporting detail]
   - [Supporting detail]

2. [Second main point or event]
   - [Supporting detail]
   - [Supporting detail]

3. [Third main point or event]
   - [Supporting detail]
   - [Supporting detail]

Conclusion:
[Summary of key points and transition to next chapter]

[Note: This is a template-generated structure. Full content generation was unavailable.]
                """,
                "section": """
Section {section_number}: [Title]

[Opening paragraph introducing the section's topic]

Key Points:
• [First key point]
• [Second key point]
• [Third key point]

[Concluding paragraph with transition]

[Note: This is a template-generated structure.]
                """
            }
            
            # Select appropriate template
            if context.chapter_number is not None:
                template = templates.get("chapter", "")
                content = template.format(chapter_number=context.chapter_number)
            elif context.section_number is not None:
                template = templates.get("section", "")
                content = template.format(section_number=context.section_number)
            else:
                content = "[Template-based content generation]"
            
            return content
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return None
    
    def _generate_placeholder_content(self, context: FallbackContext) -> Optional[str]:
        """
        Generate placeholder content as last resort.
        
        Args:
            context: Fallback context
            
        Returns:
            Placeholder content
        """
        logger.info("Generating placeholder content")
        
        placeholder = f"""
[PLACEHOLDER CONTENT]

Due to technical difficulties, the full content for this section could not be generated.

Original Request Summary:
- Provider: {context.original_provider}
- Error: {context.original_error}
- Attempt: {context.attempt_number}

The content for this section will need to be regenerated or manually created.

Chapter: {context.chapter_number or 'N/A'}
Section: {context.section_number or 'N/A'}
Book: {context.book_title or 'N/A'}

Please try again later or use an alternative generation method.

[END PLACEHOLDER]
        """
        
        return placeholder
    
    def _simplify_prompt(self, prompt: str) -> str:
        """
        Simplify a prompt to reduce complexity.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Simplified prompt
        """
        # Remove complex instructions
        simplified = prompt
        
        # Truncate if too long
        if len(simplified) > 1000:
            simplified = simplified[:1000] + "\n\n[Simplified for fallback generation]"
        
        # Remove special formatting
        simplified = simplified.replace("```", "")
        simplified = simplified.replace("**", "")
        simplified = simplified.replace("__", "")
        
        # Add simplification notice
        simplified = "Generate a simplified version:\n\n" + simplified
        
        return simplified
    
    def _adapt_content(self, content: str, context: FallbackContext) -> str:
        """
        Adapt cached content for current context.
        
        Args:
            content: Cached content
            context: Current context
            
        Returns:
            Adapted content
        """
        adapted = f"""
[Adapted Content - Based on Similar Chapter]

Chapter {context.chapter_number or 'N/A'}

{content}

[Note: This content has been adapted from a similar section due to generation issues.]
        """
        
        return adapted


# Global fallback manager instance
fallback_manager = FallbackManager()


def execute_with_fallback(
    primary_function: Callable,
    context: FallbackContext,
    strategies: Optional[List[FallbackStrategy]] = None
) -> Any:
    """
    Execute a function with fallback strategies.
    
    Args:
        primary_function: Primary function to execute
        context: Fallback context
        strategies: Fallback strategies to use
        
    Returns:
        Result from primary function or fallback
    """
    try:
        # Try primary function
        result = primary_function()
        if result:
            return result
    except Exception as e:
        logger.error(f"Primary function failed: {e}")
        context.original_error = str(e)
    
    # Execute fallback strategies
    return fallback_manager.execute_fallback(context, strategies)