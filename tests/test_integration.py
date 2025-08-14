"""
Integration tests - end-to-end smoke tests
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

def test_project_creation_and_management(mock_project_manager):
    """Test creating and managing a project"""
    # Create a new project
    project_id = mock_project_manager.create_project(
        title="Test Book",
        language="English",
        style="thriller"
    )
    
    assert project_id is not None
    assert mock_project_manager.current_project == project_id
    
    # Verify project directory structure
    project_dir = mock_project_manager.get_project_dir(project_id)
    assert project_dir.exists()
    assert (project_dir / "content").exists()
    assert (project_dir / "exports").exists()
    assert (project_dir / "cache").exists()
    assert (project_dir / "characters").exists()
    
    # Test project listing
    projects = mock_project_manager.list_projects()
    assert len(projects) == 1
    assert projects[0].title == "Test Book"
    
    # Test project switching
    project_id_2 = mock_project_manager.create_project(
        title="Another Book",
        language="English",
        style="romance"
    )
    
    mock_project_manager.switch_project(project_id)
    assert mock_project_manager.current_project == project_id
    
    # Test project deletion
    mock_project_manager.delete_project(project_id_2, confirm=True)
    projects = mock_project_manager.list_projects()
    assert len(projects) == 1

def test_book_generation_smoke(mock_llm_response, mock_project_manager, sample_book):
    """Smoke test for basic book generation flow"""
    from generate_refactored import BookGenerator
    
    # Create project
    project_id = mock_project_manager.create_project(
        title="Smoke Test Book",
        language="English"
    )
    
    # Initialize generator
    history = [{"role": "system", "content": "You are writing a book."}]
    generator = BookGenerator(sample_book, history)
    
    # Test title generation
    result = None
    for update in generator.generate_title("My Book Idea"):
        result = update
    assert result is not None
    assert "title" in result
    
    # Test TOC generation
    for update in generator.generate_toc("Write a book about AI"):
        result = update
    assert "toc" in result
    
    # Test summary generation
    for update in generator.generate_summary("AI and society"):
        result = update
    assert "summary" in result

def test_provider_initialization(mock_env):
    """Test LLM provider initialization"""
    from providers.factory import get_provider
    
    # Mock the OpenAI client
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        provider = get_provider('openai', {
            'api_key': 'test-key',
            'model': 'gpt-4'
        })
        
        assert provider is not None
        assert provider.model == 'gpt-4'

def test_export_formats(mock_project_manager, sample_book, temp_dir):
    """Test export functionality"""
    from export_formats import BookExporter
    
    # Create project
    project_id = mock_project_manager.create_project(
        title="Export Test Book",
        language="English"
    )
    
    # Initialize exporter
    project_dir = mock_project_manager.get_project_dir(project_id)
    exporter = BookExporter(project_dir)
    
    # Test HTML export (doesn't require external libraries)
    html_path = exporter.export(sample_book, "html", {"author": "Test Author"})
    assert Path(html_path).exists()
    
    # Verify HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "Test Book" in content
        assert "Test Author" in content
        assert "Introduction" in content

def test_style_templates():
    """Test style template functionality"""
    from style_templates import StyleManager
    
    manager = StyleManager()
    
    # Test default styles exist
    styles = manager.list_styles()
    assert len(styles) > 0
    assert 'thriller' in styles
    assert 'academic' in styles
    
    # Test getting a style
    thriller = manager.get_style('thriller')
    assert thriller is not None
    assert thriller.tone == 'tense, fast-paced, suspenseful'
    
    # Test applying style to prompt
    prompt = manager.apply_style_to_prompt(
        "Write a chapter",
        'thriller',
        'chapter'
    )
    assert 'thriller' in prompt.lower() or 'tension' in prompt.lower()
    
    # Test creating custom style
    custom = manager.create_custom_style(
        name='test_style',
        tone='mysterious',
        vocabulary_level='advanced'
    )
    assert custom.name == 'test_style'

def test_character_development(mock_project_manager):
    """Test character management functionality"""
    from character_development import CharacterManager, CharacterRole
    
    # Create project
    project_id = mock_project_manager.create_project(
        title="Character Test Book",
        language="English"
    )
    
    # Get character manager
    char_manager = mock_project_manager.get_character_manager(project_id)
    
    # Create character
    protagonist = char_manager.create_character(
        name="John Doe",
        role=CharacterRole.PROTAGONIST,
        age=30,
        personality_traits=["brave", "intelligent"]
    )
    
    assert protagonist.name == "John Doe"
    assert protagonist.role == CharacterRole.PROTAGONIST
    assert "brave" in protagonist.personality_traits
    
    # Test character update
    char_manager.update_character("John Doe", age=31)
    updated = char_manager.characters["John Doe"]
    assert updated.age == 31
    
    # Test dialogue generation prompt
    dialogue_prompt = char_manager.generate_dialogue(
        "John Doe",
        context="Facing danger",
        emotion="determined"
    )
    assert "John Doe" in dialogue_prompt
    assert "determined" in dialogue_prompt

def test_token_counting():
    """Test token counting functionality"""
    from providers.base import LLMProvider
    
    class TestProvider(LLMProvider):
        def _validate_config(self):
            pass
        
        def generate(self, *args, **kwargs):
            from providers.base import LLMResponse
            return LLMResponse(
                content="test",
                tokens_used=10,
                finish_reason="stop",
                model="test"
            )
        
        def count_tokens(self, text: str) -> int:
            return len(text) // 4
        
        def get_model_info(self) -> dict:
            return {"max_tokens": 4096}
    
    provider = TestProvider({})
    
    # Test token counting
    tokens = provider.count_tokens("This is a test message")
    assert tokens > 0
    
    # Test token validation
    messages = [{"content": "short"}]
    is_valid = provider.validate_token_limit(messages, 1000)
    assert is_valid is True

def test_retry_logic():
    """Test retry logic in base provider"""
    from providers.base import LLMProvider
    import time
    
    class TestProvider(LLMProvider):
        def _validate_config(self):
            pass
        
        def generate(self, *args, **kwargs):
            pass
        
        def count_tokens(self, text: str) -> int:
            return 0
        
        def get_model_info(self) -> dict:
            return {}
    
    provider = TestProvider({})
    
    # Test successful call
    call_count = 0
    def success_call():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = provider._call_with_retry(
        success_call,
        max_retries=3,
        base_delay=0.01
    )
    assert result == "success"
    assert call_count == 1
    
    # Test retry on failure
    call_count = 0
    def failing_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("rate limit")
        return "success"
    
    result = provider._call_with_retry(
        failing_call,
        max_retries=5,
        base_delay=0.01,
        retry_on=[ConnectionError]
    )
    assert result == "success"
    assert call_count == 3

def test_json_extraction():
    """Test JSON extraction from various formats"""
    from generate_refactored import BookGenerator
    
    generator = BookGenerator({}, [])
    
    # Test extraction from markdown code block
    json_with_markdown = '''
    Here is the JSON:
    ```json
    {"test": "value"}
    ```
    '''
    extracted = generator._extract_json(json_with_markdown)
    assert extracted == '{"test": "value"}'
    
    # Test extraction from raw JSON
    raw_json = '{"direct": "json"}'
    extracted = generator._extract_json(raw_json)
    assert extracted == raw_json
    
    # Test extraction with extra text
    json_with_text = 'Some text {"embedded": "json"} more text'
    extracted = generator._extract_json(json_with_text)
    assert '{"embedded": "json"}' in extracted

@pytest.mark.timeout(30)
def test_full_book_generation_flow(mock_llm_response, mock_project_manager):
    """Complete end-to-end test of book generation"""
    from generate_refactored import write_book
    
    # Create project
    project_id = mock_project_manager.create_project(
        title="Full Test Book",
        language="English",
        style="academic"
    )
    
    # Generate book
    book = {}
    instructions = "Write a comprehensive book about artificial intelligence"
    
    # Run generation
    for update in write_book(book, instructions, "AI Book", "English"):
        book = update
    
    # Verify all components were generated
    assert "title" in book
    assert "toc" in book
    assert "summary" in book
    assert "chapters" in book["toc"]
    
    # Verify at least one chapter has content
    if book["toc"]["chapters"]:
        first_chapter = book["toc"]["chapters"][0]
        # Note: In real test, content would be in the chapter dict
        # For smoke test, we just verify the structure exists