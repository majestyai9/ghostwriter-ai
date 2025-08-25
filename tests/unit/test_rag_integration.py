"""
Comprehensive tests for RAG integration in Ghostwriter AI.

Tests cover:
- Smart summarization functionality
- Semantic search and retrieval
- Hybrid context management
- Backward compatibility
- Performance benchmarks
"""

import os

# Add parent directory to path for imports
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cache_manager import CacheManager
from providers.base import LLMProvider, LLMResponse


class TestSmartSummarizerRemoved:
    """SmartSummarizer class was removed from codebase"""
    # Tests commented out as SmartSummarizer no longer exists
    pass


class TestSmartSummarizerOld(unittest.TestCase):
    """Test the SmartSummarizer class - DISABLED"""

    def setUp(self):
        """Set up test fixtures"""
        self.cache_manager = CacheManager(backend="memory")
        self.mock_provider = Mock(spec=LLMProvider)

        # Mock provider response
        self.mock_provider.generate.return_value = LLMResponse(
            content="Test summary: Chapter introduces main character and sets up conflict.",
            tokens_used=50,
            finish_reason="stop",
            model="test-model"
        )

    @unittest.skip("SmartSummarizer removed from codebase")
    def test_summarize_chapter_with_llm(self):
        """Test chapter summarization using LLM"""
        return  # SmartSummarizer no longer exists

        summarizer = SmartSummarizer(
            provider=self.mock_provider,
            cache_manager=self.cache_manager
        )

        chapter = {
            "number": 1,
            "title": "The Beginning",
            "content": "Once upon a time in a land far away, there lived a brave knight. " * 50
        }

        summary = summarizer.summarize_chapter(chapter)

        # Verify LLM was called
        self.mock_provider.generate.assert_called_once()
        self.assertIsNotNone(summary)
        self.assertIn("Test summary", summary)

    @unittest.skip("SmartSummarizer removed from codebase")
    def test_summarize_chapter_fallback(self):
        """Test fallback summarization without LLM"""
        return  # SmartSummarizer no longer exists

        summarizer = SmartSummarizer(
            provider=None,  # No provider
            cache_manager=self.cache_manager
        )

        chapter = {
            "number": 1,
            "title": "The Beginning",
            "content": "Once upon a time in a land far away, there lived a brave knight.",
            "topics": "Introduction, Knight, Quest"
        }

        summary = summarizer.summarize_chapter(chapter)

        self.assertIsNotNone(summary)
        self.assertIn("Topics:", summary)
        self.assertIn("Once upon a time", summary)

    @unittest.skip("SmartSummarizer removed from codebase")
    def test_summary_caching(self):
        """Test that summaries are cached"""
        return  # SmartSummarizer no longer exists

        summarizer = SmartSummarizer(
            provider=self.mock_provider,
            cache_manager=self.cache_manager
        )

        chapter = {
            "number": 1,
            "title": "Test Chapter",
            "content": "Test content for caching."
        }

        # First call - should generate
        summary1 = summarizer.summarize_chapter(chapter)
        call_count1 = self.mock_provider.generate.call_count

        # Second call - should use cache
        summary2 = summarizer.summarize_chapter(chapter)
        call_count2 = self.mock_provider.generate.call_count

        self.assertEqual(summary1, summary2)
        self.assertEqual(call_count1, call_count2)  # No additional calls


class TestSemanticContextRetriever(unittest.TestCase):
    """Test the SemanticContextRetriever class - DISABLED"""
    # SemanticContextRetriever no longer exists in codebase

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(backend="memory")

        from token_optimizer_rag import RAGConfig
        self.config = RAGConfig(
            vector_store_dir=".test_rag",
            chunk_size=100,
            chunk_overlap=20,
            top_k=5
        )

    def tearDown(self):
        """Clean up temp files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('token_optimizer_rag.FAISS_AVAILABLE', True)
    @patch('token_optimizer_rag.SentenceTransformer')
    @patch('token_optimizer_rag.faiss')
    def test_initialization(self, mock_faiss, mock_st):
        """Test retriever initialization"""
        self.skipTest("SemanticContextRetriever removed from codebase")
        return

        # Mock sentence transformer
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_encoder

        # Mock FAISS index
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        retriever = SemanticContextRetriever(self.config, self.cache_manager)

        self.assertIsNotNone(retriever.encoder)
        self.assertIsNotNone(retriever.index)
        self.assertTrue(retriever._initialized)

    @patch('token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    @patch('token_optimizer_rag.FAISS_AVAILABLE', False)
    def test_initialization_without_dependencies(self):
        """Test retriever handles missing dependencies gracefully"""
        self.skipTest("SemanticContextRetriever removed from codebase")
        return

        retriever = SemanticContextRetriever(self.config, self.cache_manager)

        self.assertFalse(retriever._initialized)
        self.assertIsNone(retriever.encoder)
        self.assertIsNone(retriever.index)

    def test_text_chunking(self):
        """Test text splitting into chunks"""
        self.skipTest("SemanticContextRetriever removed from codebase")
        return

        retriever = SemanticContextRetriever(self.config, self.cache_manager)

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
        chunks = retriever._split_text(text)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.config.chunk_size + 50)  # Allow some overflow

    @patch('token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('token_optimizer_rag.FAISS_AVAILABLE', True)
    def test_book_indexing(self):
        """Test indexing book content"""
        self.skipTest("SemanticContextRetriever removed from codebase")
        return

        with patch('token_optimizer_rag.SentenceTransformer') as mock_st, \
             patch('token_optimizer_rag.faiss') as mock_faiss:

            # Setup mocks
            mock_encoder = Mock()
            mock_encoder.encode.return_value = np.random.rand(10, 384).astype(np.float32)
            mock_st.return_value = mock_encoder

            mock_index = Mock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index

            retriever = SemanticContextRetriever(self.config, self.cache_manager)

            # Create test book
            book = {
                "title": "Test Book",
                "toc": {
                    "chapters": [
                        {
                            "number": 1,
                            "title": "Chapter 1",
                            "content": "This is the content of chapter 1. " * 20
                        },
                        {
                            "number": 2,
                            "title": "Chapter 2",
                            "content": "This is the content of chapter 2. " * 20
                        }
                    ]
                }
            }

            success = retriever.index_book(book, self.temp_dir)

            # Verify encoding was called
            mock_encoder.encode.assert_called()
            # Verify index was populated
            mock_index.add.assert_called()


class TestHybridContextManager(unittest.TestCase):
    """Test the HybridContextManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.cache_manager = CacheManager(backend="memory")
        self.mock_provider = Mock(spec=LLMProvider)

        # Mock token counting
        self.mock_provider.count_tokens.side_effect = lambda text: len(text) // 4

        from token_optimizer_rag import RAGConfig, RAGMode
        self.config = RAGConfig(mode=RAGMode.HYBRID)

    def test_rag_basic_mode(self):
        """Test basic RAG mode functionality"""
        from token_optimizer_rag import HybridContextManager, RAGConfig, RAGMode

        config = RAGConfig(mode=RAGMode.BASIC)
        manager = HybridContextManager(
            config=config,
            provider=self.mock_provider,
            cache_manager=self.cache_manager
        )

        book = {
            "title": "Test Book",
            "toc": {
                "chapters": [
                    {"number": 1, "title": "Chapter 1", "content": "Content 1"},
                    {"number": 2, "title": "Chapter 2", "content": "Content 2"}
                ]
            }
        }

        context = manager.prepare_context(book, current_chapter=1)

        self.assertIsInstance(context, list)
        self.assertGreater(len(context), 0)
        self.assertEqual(context[0]["role"], "system")

    def test_hybrid_context_preparation(self):
        """Test hybrid context preparation with all components"""
        from token_optimizer_rag import HybridContextManager

        manager = HybridContextManager(
            config=self.config,
            provider=self.mock_provider,
            cache_manager=self.cache_manager,
            max_tokens=10000
        )

        book = {
            "title": "Epic Fantasy Novel",
            "summary": "A tale of heroes and dragons",
            "toc": {
                "chapters": [
                    {
                        "number": 1,
                        "title": "The Beginning",
                        "content": "Once upon a time..." * 50,
                        "topics": "Introduction, Setting"
                    },
                    {
                        "number": 2,
                        "title": "The Journey",
                        "content": "The hero set forth..." * 50,
                        "topics": "Adventure, Travel"
                    },
                    {
                        "number": 3,
                        "title": "The Challenge",
                        "content": "",  # Empty - to be generated
                        "topics": "Conflict, Battle"
                    }
                ]
            }
        }

        context = manager.prepare_context(
            book=book,
            current_chapter=2,
            query="The Challenge battle conflict"
        )

        self.assertIsInstance(context, list)
        self.assertGreater(len(context), 0)

        # Check for different context types
        context_str = " ".join(msg["content"] for msg in context)
        self.assertIn("Epic Fantasy Novel", context_str)  # Title
        self.assertIn("Chapter", context_str)  # Chapter info

    def test_token_budget_allocation(self):
        """Test that token budgets are properly allocated"""
        from token_optimizer_rag import HybridContextManager

        manager = HybridContextManager(
            config=self.config,
            provider=self.mock_provider,
            cache_manager=self.cache_manager,
            max_tokens=10000
        )

        available = 10000 - 4096  # Reserved for response
        core_budget = int(available * self.config.core_context_ratio)
        rag_budget = int(available * self.config.rag_context_ratio)
        summary_budget = int(available * self.config.summary_context_ratio)

        total_budget = core_budget + rag_budget + summary_budget

        # Verify budgets sum to approximately available tokens
        self.assertAlmostEqual(total_budget, available, delta=10)

        # Verify ratios are correct
        self.assertAlmostEqual(core_budget / available, 0.4, places=1)
        self.assertAlmostEqual(rag_budget / available, 0.4, places=1)
        self.assertAlmostEqual(summary_budget / available, 0.2, places=1)


class TestRAGIntegration(unittest.TestCase):
    """Integration tests for the complete RAG system"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.book_dir = os.path.join(self.temp_dir, "test_book")
        os.makedirs(self.book_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temp files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_generation(self):
        """Test end-to-end book generation with RAG"""
        from cache_manager import CacheManager
        from token_optimizer_rag import RAGConfig, RAGMode, create_hybrid_manager

        # Create mock provider
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.count_tokens.side_effect = lambda text: len(text) // 4
        mock_provider.generate.return_value = LLMResponse(
            content="Generated chapter content here.",
            tokens_used=100,
            finish_reason="stop",
            model="test-model"
        )

        # Create hybrid manager
        config = RAGConfig(mode=RAGMode.BASIC)  # Use basic mode for testing
        cache_manager = CacheManager(backend="memory")

        manager = create_hybrid_manager(
            provider=mock_provider,
            cache_manager=cache_manager,
            config=config
        )

        # Create test book
        book = {
            "title": "Test Book",
            "toc": {
                "chapters": [
                    {
                        "number": 1,
                        "title": "Introduction",
                        "content": "This is the introduction chapter."
                    },
                    {
                        "number": 2,
                        "title": "Development",
                        "content": ""  # To be generated
                    }
                ]
            }
        }

        # Prepare context for chapter 2
        context = manager.prepare_context(
            book=book,
            current_chapter=1,
            book_dir=self.book_dir
        )

        self.assertIsInstance(context, list)
        self.assertGreater(len(context), 0)

        # Verify context contains relevant information
        context_str = " ".join(msg["content"] for msg in context)
        self.assertIn("Test Book", context_str)
        self.assertIn("Introduction", context_str)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for RAG system"""

    def test_context_preparation_performance(self):
        """Benchmark context preparation time"""
        from cache_manager import CacheManager
        from token_optimizer_rag import HybridContextManager, RAGConfig, RAGMode

        # Create manager with basic mode (no vector search)
        config = RAGConfig(mode=RAGMode.BASIC)
        manager = HybridContextManager(
            config=config,
            cache_manager=CacheManager(backend="memory")
        )

        # Create large book for testing
        book = {
            "title": "Large Book",
            "toc": {
                "chapters": [
                    {
                        "number": i,
                        "title": f"Chapter {i}",
                        "content": f"Content for chapter {i}. " * 500
                    }
                    for i in range(1, 21)  # 20 chapters
                ]
            }
        }

        # Measure context preparation time
        start_time = time.time()
        context = manager.prepare_context(book, current_chapter=10)
        end_time = time.time()

        preparation_time = end_time - start_time

        # Context preparation should be fast (< 1 second for basic mode)
        self.assertLess(preparation_time, 1.0)
        self.assertIsInstance(context, list)
        self.assertGreater(len(context), 0)

    def test_caching_performance(self):
        """Test that caching improves performance"""
        self.skipTest("SmartSummarizer removed from codebase")
        return

        cache_manager = CacheManager(backend="memory")
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = LLMResponse(
            content="Summary",
            tokens_used=50,
            finish_reason="stop",
            model="test"
        )

        summarizer = SmartSummarizer(
            provider=mock_provider,
            cache_manager=cache_manager
        )

        chapter = {
            "number": 1,
            "title": "Test",
            "content": "Long content " * 1000
        }

        # First call (no cache)
        start1 = time.time()
        summary1 = summarizer.summarize_chapter(chapter)
        time1 = time.time() - start1

        # Second call (with cache)
        start2 = time.time()
        summary2 = summarizer.summarize_chapter(chapter)
        time2 = time.time() - start2

        # Cached call should be significantly faster
        self.assertLess(time2, time1 * 0.1)  # At least 10x faster
        self.assertEqual(summary1, summary2)


if __name__ == "__main__":
    unittest.main()
