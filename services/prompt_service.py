"""
Comprehensive prompt management service with advanced features.

This module provides a thread-safe, high-performance prompt management system
with support for multiple languages, styles, versioning, and analytics.
"""

import hashlib
import json
import logging
import threading
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..exceptions import PromptServiceError


class PromptLanguage(str, Enum):
    """Supported prompt languages."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


class PromptStyle(str, Enum):
    """Supported prompt styles."""
    
    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    BUSINESS = "business"
    NARRATIVE = "narrative"


class PromptType(str, Enum):
    """Types of prompts in the system."""
    
    TITLE = "title"
    TABLE_OF_CONTENTS = "table_of_contents"
    SUMMARY = "summary"
    CHAPTER_TOPICS = "chapter_topics"
    CHAPTER = "chapter"
    SECTION_TOPICS = "section_topics"
    SECTION = "section"
    REVIEW = "review"
    EDIT = "edit"
    EXPAND = "expand"


class PromptTemplate(BaseModel):
    """Model for a prompt template."""
    
    name: str
    template: str
    description: Optional[str] = None
    version: str = "1.0.0"
    language: PromptLanguage = PromptLanguage.ENGLISH
    style: PromptStyle = PromptStyle.FORMAL
    variables: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    @field_validator("variables", mode="before")
    @classmethod
    def extract_variables(cls, v: Any, info: Any) -> List[str]:
        """Extract variables from template if not provided."""
        if v:
            return v
        
        template = info.data.get("template", "")
        if not template:
            return []
        
        # Extract variables from {var} and $var patterns
        import re
        vars_braces = re.findall(r"\{(\w+)\}", template)
        vars_dollar = re.findall(r"\$(\w+)", template)
        return list(set(vars_braces + vars_dollar))
    
    def render(self, **kwargs: Any) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Rendered prompt string
            
        Raises:
            ValueError: If required variables are missing
        """
        # Check for missing required variables
        provided = set(kwargs.keys())
        required = set(self.variables)
        missing = required - provided
        
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Try format-style first, then Template-style
        try:
            return self.template.format(**kwargs)
        except KeyError:
            # Fallback to Template for $var syntax
            tmpl = Template(self.template)
            return tmpl.safe_substitute(**kwargs)


class PromptMetrics(BaseModel):
    """Metrics for prompt usage and performance."""
    
    prompt_name: str
    usage_count: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    error_count: int = 0
    
    def update_usage(
        self,
        tokens: int,
        response_time: float,
        success: bool = True
    ) -> None:
        """Update metrics with new usage data."""
        self.usage_count += 1
        self.total_tokens += tokens
        
        # Update average response time
        if self.usage_count == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.usage_count - 1) + response_time)
                / self.usage_count
            )
        
        # Update success rate
        if not success:
            self.error_count += 1
        self.success_rate = (self.usage_count - self.error_count) / self.usage_count
        
        self.last_used = datetime.now(UTC)


class PromptCache:
    """Thread-safe prompt cache with TTL support."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[str, datetime]] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[str]:
        """Get a cached prompt if not expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                age = (datetime.now(UTC) - timestamp).total_seconds()
                if age < self.ttl_seconds:
                    return value
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, value: str) -> None:
        """Store a prompt in the cache."""
        with self._lock:
            self._cache[key] = (value, datetime.now(UTC))
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        with self._lock:
            now = datetime.now(UTC)
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if (now - timestamp).total_seconds() >= self.ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]


class PromptService:
    """
    Comprehensive prompt management service.
    
    This service provides:
    - Template loading from YAML/JSON files
    - Dynamic prompt composition
    - Variable substitution with validation
    - Multi-language and style support
    - Caching for performance
    - Versioning support
    - Metrics and analytics
    - Thread safety
    """
    
    def __init__(
        self,
        template_dir: str = "templates",
        cache_ttl: int = 3600,
        enable_metrics: bool = True,
        default_language: PromptLanguage = PromptLanguage.ENGLISH,
        default_style: PromptStyle = PromptStyle.FORMAL
    ):
        """
        Initialize the PromptService.
        
        Args:
            template_dir: Directory containing template files
            cache_ttl: Cache time-to-live in seconds
            enable_metrics: Whether to track usage metrics
            default_language: Default language for prompts
            default_style: Default style for prompts
        """
        self.template_dir = Path(template_dir)
        self.cache = PromptCache(ttl_seconds=cache_ttl)
        self.enable_metrics = enable_metrics
        self.default_language = default_language
        self.default_style = default_style
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Template storage
        self._templates: Dict[str, PromptTemplate] = {}
        self._template_versions: Dict[str, List[PromptTemplate]] = defaultdict(list)
        
        # Metrics storage
        self._metrics: Dict[str, PromptMetrics] = {}
        
        # Custom validators
        self._validators: Dict[str, callable] = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Load initial templates
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from the template directory."""
        if not self.template_dir.exists():
            self.logger.warning(f"Template directory {self.template_dir} not found")
            self._load_default_templates()
            return
        
        # Load YAML files
        for yaml_file in self.template_dir.glob("*.yaml"):
            self._load_yaml_file(yaml_file)
        
        # Load JSON files
        for json_file in self.template_dir.glob("*.json"):
            self._load_json_file(json_file)
        
        self.logger.info(f"Loaded {len(self._templates)} templates")
    
    def _load_yaml_file(self, file_path: Path) -> None:
        """Load templates from a YAML file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self._process_template_data(data, file_path.stem)
        except Exception as e:
            self.logger.error(f"Failed to load YAML file {file_path}: {e}")
    
    def _load_json_file(self, file_path: Path) -> None:
        """Load templates from a JSON file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                self._process_template_data(data, file_path.stem)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {file_path}: {e}")
    
    def _process_template_data(self, data: Dict[str, Any], source: str) -> None:
        """Process template data from file."""
        for name, template_data in data.items():
            try:
                if isinstance(template_data, str):
                    # Simple string template
                    template = PromptTemplate(
                        name=name,
                        template=template_data,
                        metadata={"source": source}
                    )
                elif isinstance(template_data, dict):
                    # Complex template with metadata
                    template = PromptTemplate(
                        name=name,
                        **template_data,
                        metadata={**template_data.get("metadata", {}), "source": source}
                    )
                else:
                    continue
                
                self._register_template(template)
                
            except ValidationError as e:
                self.logger.error(f"Invalid template '{name}': {e}")
    
    def _register_template(self, template: PromptTemplate) -> None:
        """Register a template internally."""
        with self._lock:
            # Store current version
            self._templates[template.name] = template
            
            # Store in version history
            self._template_versions[template.name].append(template)
            
            # Initialize metrics if enabled
            if self.enable_metrics and template.name not in self._metrics:
                self._metrics[template.name] = PromptMetrics(
                    prompt_name=template.name
                )
    
    def _load_default_templates(self) -> None:
        """Load default built-in templates."""
        defaults = {
            "title": {
                "template": "Generate a creative title for: {original_title}",
                "description": "Generate book title",
                "variables": ["original_title"]
            },
            "table_of_contents": {
                "template": "Create a table of contents for: {instructions}",
                "description": "Generate table of contents",
                "variables": ["instructions"]
            },
            "summary": {
                "template": "Write a summary for '{title}': {instructions}",
                "description": "Generate book summary",
                "variables": ["title", "instructions"]
            },
            "chapter": {
                "template": "Write Chapter {chapter_number}: {chapter_title}",
                "description": "Generate chapter content",
                "variables": ["chapter_number", "chapter_title"]
            },
            "section": {
                "template": "Write Section {section_number}: {section_title}",
                "description": "Generate section content",
                "variables": ["section_number", "section_title"]
            }
        }
        
        for name, data in defaults.items():
            template = PromptTemplate(name=name, **data)
            self._register_template(template)
    
    @lru_cache(maxsize=128)
    def _generate_cache_key(self, **kwargs: Any) -> str:
        """Generate a cache key from parameters."""
        # Sort items for consistent hashing
        items = sorted(kwargs.items())
        key_str = json.dumps(items, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def render(
        self,
        template_name: str,
        language: Optional[PromptLanguage] = None,
        style: Optional[PromptStyle] = None,
        version: Optional[str] = None,
        use_cache: bool = True,
        validate: bool = True,
        **variables: Any
    ) -> str:
        """
        Render a prompt template with variables.
        
        Args:
            template_name: Name of the template to render
            language: Language for the prompt
            style: Style for the prompt
            version: Specific version to use
            use_cache: Whether to use caching
            validate: Whether to validate variables
            **variables: Variables to substitute
            
        Returns:
            Rendered prompt string
            
        Raises:
            PromptServiceError: If template not found or validation fails
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(
                template_name=template_name,
                language=language,
                style=style,
                version=version,
                **variables
            )
            cached = self.cache.get(cache_key)
            if cached:
                self._update_metrics(template_name, 0, 0.0, True)
                return cached
        
        # Get template
        template = self._get_template(template_name, language, style, version)
        if not template:
            raise PromptServiceError(f"Template '{template_name}' not found")
        
        # Validate variables if requested
        if validate and template_name in self._validators:
            validator = self._validators[template_name]
            try:
                validator(variables)
            except Exception as e:
                raise PromptServiceError(f"Validation failed: {e}")
        
        # Render template
        try:
            import time
            start_time = time.time()
            
            rendered = template.render(**variables)
            
            response_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(
                template_name,
                len(rendered),  # Approximate token count
                response_time,
                True
            )
            
            # Cache result
            if use_cache:
                self.cache.set(cache_key, rendered)
            
            return rendered
            
        except Exception as e:
            self._update_metrics(template_name, 0, 0.0, False)
            raise PromptServiceError(f"Failed to render template: {e}")
    
    def _get_template(
        self,
        name: str,
        language: Optional[PromptLanguage] = None,
        style: Optional[PromptStyle] = None,
        version: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Get a specific template."""
        with self._lock:
            # Try to find exact match first
            key = f"{name}_{language or self.default_language}_{style or self.default_style}"
            if key in self._templates:
                return self._templates[key]
            
            # Fall back to base template
            if name in self._templates:
                return self._templates[name]
            
            # Try version-specific
            if version and name in self._template_versions:
                for tmpl in self._template_versions[name]:
                    if tmpl.version == version:
                        return tmpl
            
            return None
    
    def _update_metrics(
        self,
        template_name: str,
        tokens: int,
        response_time: float,
        success: bool
    ) -> None:
        """Update metrics for a template."""
        if not self.enable_metrics:
            return
        
        with self._lock:
            if template_name not in self._metrics:
                self._metrics[template_name] = PromptMetrics(
                    prompt_name=template_name
                )
            
            self._metrics[template_name].update_usage(
                tokens, response_time, success
            )
    
    def compose(
        self,
        *template_names: str,
        separator: str = "\n\n",
        **variables: Any
    ) -> str:
        """
        Compose multiple templates into a single prompt.
        
        Args:
            *template_names: Names of templates to compose
            separator: String to join templates
            **variables: Variables for all templates
            
        Returns:
            Composed prompt string
        """
        parts = []
        for name in template_names:
            part = self.render(name, **variables)
            parts.append(part)
        
        return separator.join(parts)
    
    def register_template(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        language: PromptLanguage = PromptLanguage.ENGLISH,
        style: PromptStyle = PromptStyle.FORMAL,
        version: str = "1.0.0",
        **metadata: Any
    ) -> None:
        """
        Register a new template or update existing one.
        
        Args:
            name: Template name
            template: Template string
            description: Template description
            language: Template language
            style: Template style
            version: Template version
            **metadata: Additional metadata
        """
        prompt_template = PromptTemplate(
            name=name,
            template=template,
            description=description,
            language=language,
            style=style,
            version=version,
            metadata=metadata,
            updated_at=datetime.now(UTC)
        )
        
        self._register_template(prompt_template)
        self.logger.info(f"Registered template '{name}' v{version}")
    
    def register_validator(
        self,
        template_name: str,
        validator: callable
    ) -> None:
        """
        Register a custom validator for a template.
        
        Args:
            template_name: Name of template to validate
            validator: Validation function
        """
        with self._lock:
            self._validators[template_name] = validator
    
    def get_metrics(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for templates.
        
        Args:
            template_name: Specific template or None for all
            
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            if template_name:
                if template_name in self._metrics:
                    return self._metrics[template_name].model_dump()
                return {}
            
            return {
                name: metrics.model_dump()
                for name, metrics in self._metrics.items()
            }
    
    def export_templates(
        self,
        output_path: str,
        format: str = "yaml"
    ) -> None:
        """
        Export all templates to a file.
        
        Args:
            output_path: Path to output file
            format: Export format (yaml or json)
        """
        with self._lock:
            data = {
                name: {
                    "template": tmpl.template,
                    "description": tmpl.description,
                    "version": tmpl.version,
                    "language": tmpl.language.value,
                    "style": tmpl.style.value,
                    "variables": tmpl.variables,
                    "metadata": tmpl.metadata
                }
                for name, tmpl in self._templates.items()
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "yaml":
                with open(output_file, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif format == "json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(data)} templates to {output_path}")
    
    def import_templates(
        self,
        input_path: str,
        overwrite: bool = False
    ) -> None:
        """
        Import templates from a file.
        
        Args:
            input_path: Path to input file
            overwrite: Whether to overwrite existing templates
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Determine format from extension
        if input_file.suffix in [".yaml", ".yml"]:
            with open(input_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif input_file.suffix == ".json":
            with open(input_file, encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
        # Process templates
        imported = 0
        for name, template_data in data.items():
            if not overwrite and name in self._templates:
                self.logger.debug(f"Skipping existing template: {name}")
                continue
            
            try:
                self.register_template(name=name, **template_data)
                imported += 1
            except Exception as e:
                self.logger.error(f"Failed to import template '{name}': {e}")
        
        self.logger.info(f"Imported {imported} templates from {input_path}")
    
    def list_templates(
        self,
        language: Optional[PromptLanguage] = None,
        style: Optional[PromptStyle] = None
    ) -> List[Dict[str, Any]]:
        """
        List available templates.
        
        Args:
            language: Filter by language
            style: Filter by style
            
        Returns:
            List of template information
        """
        with self._lock:
            templates = []
            for name, tmpl in self._templates.items():
                if language and tmpl.language != language:
                    continue
                if style and tmpl.style != style:
                    continue
                
                templates.append({
                    "name": name,
                    "description": tmpl.description,
                    "version": tmpl.version,
                    "language": tmpl.language.value,
                    "style": tmpl.style.value,
                    "variables": tmpl.variables,
                    "updated_at": tmpl.updated_at.isoformat()
                })
            
            return templates
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self.cache.clear()
        self.logger.info("Cleared prompt cache")
    
    def reload_templates(self) -> None:
        """Reload all templates from disk."""
        with self._lock:
            self._templates.clear()
            self._template_versions.clear()
            self._load_templates()
        
        self.logger.info("Reloaded all templates")


# Singleton instance
_prompt_service: Optional[PromptService] = None
_lock = threading.Lock()


def get_prompt_service(**kwargs: Any) -> PromptService:
    """
    Get or create the global PromptService instance.
    
    Args:
        **kwargs: Arguments for PromptService initialization
        
    Returns:
        PromptService instance
    """
    global _prompt_service
    
    with _lock:
        if _prompt_service is None:
            _prompt_service = PromptService(**kwargs)
    
    return _prompt_service