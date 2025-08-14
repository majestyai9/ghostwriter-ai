import unittest
from unittest.mock import MagicMock, patch

from services.generation_service import GenerationService
from providers.base import LLMResponse

class TestGenerationService(unittest.TestCase):

    def setUp(self):
        self.provider_factory = MagicMock()
        self.cache_manager = MagicMock()
        self.token_optimizer = MagicMock()
        self.generation_service = GenerationService(
            self.provider_factory,
            self.cache_manager,
            self.token_optimizer
        )

    def test_generate_text_with_cache(self):
        self.cache_manager.get.return_value = "cached result"
        result = self.generation_service.generate_text("openai", "test prompt")
        self.assertEqual(result, "cached result")
        self.cache_manager.get.assert_called_once()
        self.provider_factory.create_provider.assert_not_called()

    def test_generate_text_without_cache(self):
        self.cache_manager.get.return_value = None
        mock_provider = MagicMock()
        mock_provider.generate.return_value = LLMResponse(
            content="newly generated text",
            tokens_used=10,
            finish_reason="stop",
            model="test_model"
        )
        self.provider_factory.create_provider.return_value = mock_provider

        result = self.generation_service.generate_text("openai", "test prompt")

        self.assertEqual(result, "newly generated text")
        self.cache_manager.get.assert_called_once()
        self.provider_factory.create_provider.assert_called_once()
        self.cache_manager.set.assert_called_once()

    def test_generate_text_stream(self):
        mock_provider = MagicMock()
        mock_provider.generate_stream.return_value = iter(["chunk1", "chunk2"])
        self.provider_factory.create_provider.return_value = mock_provider

        result = list(self.generation_service.generate_text_stream("openai", "test prompt"))

        self.assertEqual(result, ["chunk1", "chunk2"])
        self.provider_factory.create_provider.assert_called_once()

if __name__ == '__main__':
    unittest.main()
