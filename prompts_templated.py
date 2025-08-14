"""
Prompt management using templates
"""
import logging
from pathlib import Path
from string import Template
from typing import Any, Dict

import yaml


class PromptManager:
    """Manages prompts using external templates"""

    def __init__(self, template_file: str = "templates/prompts.yaml"):
        """
        Initialize PromptManager
        
        Args:
            template_file: Path to YAML file with prompt templates
        """
        self.template_file = Path(template_file)
        self.templates = {}
        self.logger = logging.getLogger(__name__)
        self._load_templates()

    def _load_templates(self):
        """Load templates from YAML file"""
        try:
            if self.template_file.exists():
                with open(self.template_file, encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    for key, value in data.items():
                        if isinstance(value, dict) and 'template' in value:
                            self.templates[key] = value['template']
                        else:
                            self.templates[key] = value
                self.logger.info(f"Loaded {len(self.templates)} prompt templates")
            else:
                self.logger.warning(f"Template file {self.template_file} not found")
                self._use_fallback_templates()
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            self._use_fallback_templates()

    def _use_fallback_templates(self):
        """Use built-in templates as fallback"""
        self.templates = {
            'title': 'Generate a title for: {original_title}',
            'table_of_contents': 'Create a table of contents for: {instructions}',
            'summary': 'Write a summary for "{title}": {instructions}',
            'chapter_topics': 'Topics for Chapter {chapter_number}: {chapter_title}',
            'chapter': 'Write Chapter {chapter_number}: {chapter_title}',
            'section_topics': 'Topics for Section {section_number}: {section_title}',
            'section': 'Write Section {section_number}: {section_title}'
        }

    def render(self, template_name: str, **kwargs) -> str:
        """
        Render a prompt template with given variables
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to substitute in template
            
        Returns:
            Rendered prompt string
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]

        # Use safe substitution to avoid KeyError for missing variables
        try:
            # Try format first (supports {var} syntax)
            return template.format(**kwargs)
        except KeyError as e:
            # Fall back to Template (supports $var syntax)
            self.logger.warning(f"Missing variable in template: {e}")
            tmpl = Template(template)
            return tmpl.safe_substitute(**kwargs)

    def get_template(self, template_name: str) -> str:
        """Get raw template string"""
        return self.templates.get(template_name, "")

    def update_template(self, template_name: str, template: str):
        """Update a template at runtime"""
        self.templates[template_name] = template

    def save_templates(self):
        """Save current templates back to YAML file"""
        data = {
            name: {'template': template}
            for name, template in self.templates.items()
        }

        self.template_file.parent.mkdir(exist_ok=True)
        with open(self.template_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# Global instance for backward compatibility
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


# Wrapper functions for backward compatibility with existing code
def title(original_title: str) -> str:
    """Generate title prompt"""
    pm = get_prompt_manager()
    return pm.render('title', original_title=original_title)

def table_of_contents(instructions: str) -> str:
    """Generate table of contents prompt"""
    pm = get_prompt_manager()
    return pm.render('table_of_contents', instructions=instructions)

def summary(book: Dict[str, Any], instructions: str) -> str:
    """Generate summary prompt"""
    pm = get_prompt_manager()
    return pm.render('summary',
                    title=book.get('title', 'Untitled'),
                    instructions=instructions)

def chapter_topics(book: Dict[str, Any], chapter: Dict[str, Any],
                  toc_context: str) -> str:
    """Generate chapter topics prompt"""
    pm = get_prompt_manager()
    return pm.render('chapter_topics',
                    title=book.get('title', 'Untitled'),
                    chapter_number=chapter['number'],
                    chapter_title=chapter['title'],
                    toc_context=toc_context)

def chapter(book: Dict[str, Any], chapter: Dict[str, Any],
           toc_context: str) -> str:
    """Generate chapter content prompt"""
    pm = get_prompt_manager()
    return pm.render('chapter',
                    title=book.get('title', 'Untitled'),
                    summary=book.get('summary', ''),
                    chapter_number=chapter['number'],
                    chapter_title=chapter['title'],
                    topics=chapter.get('topics', ''),
                    toc_context=toc_context)

def section_topics(book: Dict[str, Any], chapter: Dict[str, Any],
                  section: Dict[str, Any], toc_context: str) -> str:
    """Generate section topics prompt"""
    pm = get_prompt_manager()
    return pm.render('section_topics',
                    chapter_number=chapter['number'],
                    chapter_title=chapter['title'],
                    section_number=section['number'],
                    section_title=section['title'],
                    chapter_topics=chapter.get('topics', ''),
                    toc_context=toc_context)

def section(book: Dict[str, Any], chapter: Dict[str, Any],
           section: Dict[str, Any]) -> str:
    """Generate section content prompt"""
    pm = get_prompt_manager()
    return pm.render('section',
                    title=book.get('title', 'Untitled'),
                    chapter_number=chapter['number'],
                    chapter_title=chapter['title'],
                    section_number=section['number'],
                    section_title=section['title'],
                    topics=section.get('topics', ''),
                    chapter_topics=chapter.get('topics', ''))


# Advanced template management functions
def load_custom_templates(file_path: str):
    """Load custom templates from a different file"""
    pm = get_prompt_manager()
    pm.template_file = Path(file_path)
    pm._load_templates()

def update_prompt(name: str, template: str):
    """Update a specific prompt template"""
    pm = get_prompt_manager()
    pm.update_template(name, template)

def save_current_templates():
    """Save current templates to file"""
    pm = get_prompt_manager()
    pm.save_templates()
