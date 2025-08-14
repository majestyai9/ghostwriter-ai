"""
Pytest configuration and fixtures
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import json
import os

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing"""
    # Set test environment variables
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("BOOK_LANGUAGE", "English")
    monkeypatch.setenv("TEMPERATURE", "0.2")
    monkeypatch.setenv("TOKEN_LIMIT", "4096")
    
@pytest.fixture
def sample_book():
    """Sample book data for testing"""
    return {
        "title": "Test Book",
        "summary": "This is a test book summary",
        "toc": {
            "chapters": [
                {
                    "number": 1,
                    "title": "Introduction",
                    "sections": [
                        {"number": 1, "title": "Getting Started"},
                        {"number": 2, "title": "Basic Concepts"}
                    ]
                },
                {
                    "number": 2,
                    "title": "Advanced Topics",
                    "sections": [
                        {"number": 1, "title": "Deep Dive"},
                        {"number": 2, "title": "Best Practices"}
                    ]
                }
            ]
        }
    }

@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM API responses"""
    def mock_call(*args, **kwargs):
        # Return different responses based on the prompt
        prompt = args[0] if args else ""
        
        if "title" in prompt.lower():
            return "Amazing Test Book"
        elif "table of contents" in prompt.lower():
            return json.dumps({
                "chapters": [
                    {
                        "number": 1,
                        "title": "Chapter One",
                        "sections": [
                            {"number": 1, "title": "Section 1.1"}
                        ]
                    }
                ]
            })
        elif "summary" in prompt.lower():
            return "This is a test book about testing."
        elif "chapter" in prompt.lower():
            return "This is the content of a test chapter. It contains interesting information."
        elif "section" in prompt.lower():
            return "This is test section content."
        else:
            return "Generic test response"
    
    # Mock the callLLM function
    monkeypatch.setattr("ai.callLLM", mock_call)
    monkeypatch.setattr("generate.callLLM", mock_call)
    monkeypatch.setattr("generate_refactored.callLLM", mock_call)

@pytest.fixture
def mock_project_manager(temp_dir):
    """Create a mock project manager with temp directory"""
    from project_manager import ProjectManager
    
    pm = ProjectManager(base_dir=str(temp_dir / "projects"))
    return pm

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset global singletons before each test"""
    # Reset any global state
    import project_manager
    project_manager._project_manager = None
    
    yield
    
    # Cleanup after test
    project_manager._project_manager = None