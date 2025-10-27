"""
Pytest configuration and fixtures.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Hello, world! This is a test."


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock = Mock()
    mock.choices = [Mock()]
    mock.choices[0].message.content = "Привет, мир! Это тест."
    mock.usage.total_tokens = 100
    mock.usage.prompt_tokens = 50
    mock.usage.completion_tokens = 50
    return mock


@pytest.fixture
def mock_deepl_response():
    """Mock DeepL API response."""
    return {
        "translations": [
            {"text": "Hallo, Welt! Das ist ein Test."}
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
