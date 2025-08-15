# PromptService Documentation

## Overview

The PromptService is a comprehensive, thread-safe prompt management system that provides advanced features for template management, dynamic composition, multi-language support, and performance optimization.

## Features

- **Template Management**: Load, store, and manage prompt templates from YAML/JSON files
- **Dynamic Composition**: Combine multiple templates and apply variable substitution
- **Multi-Language Support**: Generate prompts in different languages
- **Style Profiles**: Apply consistent writing styles across prompts
- **Caching**: High-performance caching with TTL support
- **Metrics & Analytics**: Track usage, performance, and success rates
- **Thread Safety**: Safe for concurrent use in multi-threaded applications
- **Versioning**: Support multiple versions of templates
- **Validation**: Custom validators for template variables
- **Optimization**: Token-aware prompt optimization

## Quick Start

### Basic Usage

```python
from services.prompt_service import get_prompt_service

# Get the service instance
service = get_prompt_service()

# Render a simple prompt
prompt = service.render(
    "title",
    original_title="My Amazing Book Idea"
)
```

### Using the Wrapper Functions

For backward compatibility and convenience, use the wrapper functions:

```python
from services.prompt_wrapper import (
    title,
    table_of_contents,
    summary,
    chapter,
    section
)

# Generate a title
book_title = title("Initial book concept")

# Generate table of contents
toc = table_of_contents("A comprehensive guide to Python programming")

# Generate chapter content
chapter_content = chapter(
    book={"title": "Python Mastery", "summary": "..."},
    chapter={"number": 1, "title": "Introduction", "topics": "..."},
    toc_context="..."
)
```

## Advanced Usage

### Custom Style Profiles

```python
from services.prompt_config import StyleProfile, WritingTone, ContentDepth

# Create a custom style profile
custom_style = StyleProfile(
    name="technical_blog",
    tone=WritingTone.TECHNICAL,
    depth=ContentDepth.DETAILED,
    formality=0.6,
    creativity=0.3,
    use_examples=True,
    example_count=3
)

# Apply to prompt generation
from services.prompt_service import get_prompt_service

service = get_prompt_service()
service.render(
    "chapter",
    style=custom_style.tone.value,
    chapter_number=1,
    chapter_title="Getting Started"
)
```

### Registering Custom Templates

```python
service = get_prompt_service()

# Register a new template
service.register_template(
    name="code_review",
    template="""
    Review the following {language} code:
    
    {code}
    
    Focus on:
    - {focus_areas}
    
    Provide suggestions for improvement.
    """,
    description="Code review prompt",
    version="1.0.0"
)

# Use the custom template
review_prompt = service.render(
    "code_review",
    language="Python",
    code="def hello(): print('world')",
    focus_areas="performance and readability"
)
```

### Composing Multiple Templates

```python
# Compose multiple templates into one
combined_prompt = service.compose(
    "chapter_intro",
    "chapter_topics",
    "chapter_conclusion",
    separator="\n\n---\n\n",
    chapter_number=1,
    chapter_title="Introduction"
)
```

### Using Domain Configurations

```python
from services.prompt_config import DOMAIN_CONFIGS

# Get a predefined domain configuration
fiction_config = DOMAIN_CONFIGS["fiction"]

# Apply domain-specific rules
service.register_template(
    name="fiction_chapter",
    template=fiction_config.template_overrides["chapter"],
    metadata={"domain": "fiction"}
)
```

## Configuration

### PromptConfig Options

```python
from services.prompt_config import PromptConfig

config = PromptConfig(
    # Style settings
    style_profile="academic",
    
    # Content settings
    min_word_count=500,
    max_word_count=3000,
    default_chapter_word_count=2500,
    
    # Structure settings
    default_chapter_count=12,
    default_sections_per_chapter=5,
    
    # Generation settings
    include_examples=True,
    include_summaries=True,
    include_transitions=True,
    
    # Advanced settings
    use_chapter_continuity=True,
    use_progressive_depth=True,
    
    # Custom instructions
    custom_instructions="Always include practical examples"
)
```

### Template File Format

Templates can be defined in YAML or JSON format:

```yaml
# templates/custom_prompts.yaml
research_paper:
  template: |
    Write a research paper section on {topic}.
    
    Requirements:
    - Academic tone
    - Include {citation_count} citations
    - Length: {word_count} words
    - Focus: {research_focus}
    
  description: "Academic research paper generation"
  version: "1.0.0"
  variables:
    - topic
    - citation_count
    - word_count
    - research_focus
  metadata:
    category: academic
    difficulty: advanced
```

## Metrics and Analytics

Track prompt usage and performance:

```python
# Get metrics for a specific template
metrics = service.get_metrics("chapter")
print(f"Usage count: {metrics['usage_count']}")
print(f"Average response time: {metrics['avg_response_time']}s")
print(f"Success rate: {metrics['success_rate']}")

# Get metrics for all templates
all_metrics = service.get_metrics()
for template_name, template_metrics in all_metrics.items():
    print(f"{template_name}: {template_metrics['usage_count']} uses")
```

## Migration Guide

### From Old System

1. **Update Imports**:
```python
# Old
from prompts_templated import title, chapter

# New
from services.prompt_wrapper import title, chapter
```

2. **Run Migration Script**:
```bash
python migrate_to_prompt_service.py --root-dir . --execute
```

3. **Update Custom Code**:
```python
# Old
prompt_manager = PromptManager()
prompt = prompt_manager.render("title", original_title="Book")

# New
service = get_prompt_service()
prompt = service.render("title", original_title="Book")
```

## Performance Optimization

### Caching

```python
# Enable caching (default)
prompt = service.render("template_name", use_cache=True, **variables)

# Clear cache when needed
service.clear_cache()

# Configure cache TTL
service = PromptService(cache_ttl=7200)  # 2 hours
```

### Token Optimization

```python
from services.prompt_config import PromptOptimizer

# Optimize prompt for token limits
optimizer = PromptOptimizer()
optimized = optimizer.optimize_for_tokens(prompt, max_tokens=4000)

# Add context window
with_context = optimizer.add_context_window(
    prompt,
    previous_context="Previous chapter summary...",
    max_context_tokens=1000
)
```

## Thread Safety

The PromptService is designed to be thread-safe:

```python
import threading

def worker(thread_id):
    service = get_prompt_service()  # Safe to share
    result = service.render(
        "template",
        thread_id=thread_id
    )
    return result

# Create multiple threads
threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## Best Practices

1. **Use Style Profiles**: Define consistent style profiles for your use case
2. **Cache Templates**: Enable caching for frequently used templates
3. **Version Templates**: Use versioning when updating templates
4. **Monitor Metrics**: Track performance and optimize slow templates
5. **Validate Input**: Use validators for critical templates
6. **Organize Templates**: Group related templates in separate files
7. **Document Variables**: Clearly document required variables in templates

## API Reference

### PromptService

- `render(template_name, **variables)`: Render a template with variables
- `compose(*template_names, separator, **variables)`: Compose multiple templates
- `register_template(name, template, **metadata)`: Register a new template
- `register_validator(template_name, validator)`: Add a custom validator
- `get_metrics(template_name=None)`: Get usage metrics
- `export_templates(output_path, format)`: Export templates to file
- `import_templates(input_path, overwrite)`: Import templates from file
- `list_templates(language, style)`: List available templates
- `clear_cache()`: Clear the prompt cache
- `reload_templates()`: Reload templates from disk

### Configuration Classes

- `PromptConfig`: Main configuration class
- `StyleProfile`: Writing style configuration
- `DomainConfig`: Domain-specific configuration
- `PromptOptimizer`: Prompt optimization utilities

## Testing

Run the comprehensive test suite:

```bash
# Run all prompt service tests
pytest tests/unit/test_prompt_service.py -v

# Run specific test
pytest tests/unit/test_prompt_service.py::TestPromptService::test_render_template -v

# Run with coverage
pytest tests/unit/test_prompt_service.py --cov=services.prompt_service
```

## Troubleshooting

### Common Issues

1. **Templates Not Loading**:
   - Check template directory path
   - Verify YAML/JSON syntax
   - Check file permissions

2. **Variable Substitution Errors**:
   - Ensure all required variables are provided
   - Check variable names match template placeholders
   - Use validators to catch errors early

3. **Performance Issues**:
   - Enable caching for frequently used templates
   - Optimize template complexity
   - Monitor metrics to identify slow templates

4. **Thread Safety Concerns**:
   - Always use `get_prompt_service()` to get the singleton
   - Don't modify templates during runtime in production
   - Use the service's thread-safe methods

## Contributing

When adding new features to the PromptService:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update this documentation
4. Ensure thread safety
5. Consider backward compatibility
6. Add type hints and docstrings

## License

See the main project LICENSE file.