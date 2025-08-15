"""
Optimized text chunking strategies for RAG systems.

Provides intelligent text splitting with semantic awareness.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    strategy: str = "semantic"  # "semantic", "fixed", "sliding"
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    sentence_split: bool = True
    paragraph_aware: bool = True
    preserve_sentences: bool = True


class OptimizedTextChunker:
    """
    Advanced text chunking with multiple strategies.
    
    Features:
    - Semantic-aware chunking
    - Sentence boundary preservation
    - Paragraph-aware splitting
    - Configurable overlap strategies
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Compile regex patterns for efficiency
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+|\n\n+'
        )
        self.paragraph_pattern = re.compile(r'\n\n+')
        self.word_pattern = re.compile(r'\S+')
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using configured strategy.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Choose strategy
        if self.config.strategy == "semantic":
            return self._semantic_chunking(text)
        elif self.config.strategy == "sliding":
            return self._sliding_window_chunking(text)
        else:
            return self._fixed_chunking(text)
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Semantic-aware chunking that preserves meaning.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        
        # Split into paragraphs first if enabled
        if self.config.paragraph_aware:
            paragraphs = self.paragraph_pattern.split(text)
        else:
            paragraphs = [text]
        
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Split paragraph into sentences if needed
            if self.config.sentence_split and len(paragraph) > self.config.chunk_size:
                sentences = self._split_sentences(paragraph)
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    # Check if adding sentence exceeds chunk size
                    if current_size + sentence_size > self.config.chunk_size:
                        # Save current chunk
                        if current_chunk and current_size >= self.config.min_chunk_size:
                            chunks.append(current_chunk.strip())
                            
                            # Start new chunk with overlap
                            if self.config.chunk_overlap > 0:
                                overlap = self._get_overlap(current_chunk)
                                current_chunk = overlap + " " + sentence
                                current_size = len(current_chunk)
                            else:
                                current_chunk = sentence
                                current_size = sentence_size
                        else:
                            # Current chunk too small, extend it
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                            current_size = len(current_chunk)
                    else:
                        # Add sentence to current chunk
                        if current_chunk:
                            current_chunk += " " + sentence
                            current_size += sentence_size + 1
                        else:
                            current_chunk = sentence
                            current_size = sentence_size
            else:
                # Add entire paragraph if it fits
                para_size = len(paragraph)
                
                if current_size + para_size > self.config.chunk_size:
                    # Save current chunk and start new one
                    if current_chunk and current_size >= self.config.min_chunk_size:
                        chunks.append(current_chunk.strip())
                        
                        # Start new chunk with overlap
                        if self.config.chunk_overlap > 0:
                            overlap = self._get_overlap(current_chunk)
                            current_chunk = overlap + "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                        current_size = len(current_chunk)
                    else:
                        # Extend current chunk
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                        current_size = len(current_chunk)
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                        current_size += para_size + 2
                    else:
                        current_chunk = paragraph
                        current_size = para_size
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sliding_window_chunking(self, text: str) -> List[str]:
        """
        Sliding window chunking with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of overlapping chunks
        """
        chunks = []
        text_length = len(text)
        
        start = 0
        while start < text_length:
            # Calculate end position
            end = min(start + self.config.chunk_size, text_length)
            
            # Try to find sentence boundary if preserving sentences
            if self.config.preserve_sentences and end < text_length:
                # Look for sentence end near chunk boundary
                search_start = max(start, end - 50)
                search_text = text[search_start:end + 50]
                
                match = re.search(r'[.!?]\s', search_text)
                if match:
                    end = search_start + match.end()
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Move window
            start += self.config.chunk_size - self.config.chunk_overlap
            
            # Prevent infinite loop
            if start <= 0:
                break
        
        return chunks
    
    def _fixed_chunking(self, text: str) -> List[str]:
        """
        Simple fixed-size chunking.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of fixed-size chunks
        """
        chunks = []
        
        for i in range(0, len(text), self.config.chunk_size):
            chunk = text[i:i + self.config.chunk_size].strip()
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, chunk: str) -> str:
        """
        Get overlap text from end of chunk.
        
        Args:
            chunk: Chunk to get overlap from
            
        Returns:
            Overlap text
        """
        if len(chunk) <= self.config.chunk_overlap:
            return chunk
        
        # Try to find sentence boundary in overlap region
        overlap_start = len(chunk) - self.config.chunk_overlap
        overlap_text = chunk[overlap_start:]
        
        if self.config.preserve_sentences:
            # Find sentence start in overlap
            match = re.search(r'[.!?]\s+', overlap_text)
            if match:
                overlap_text = overlap_text[match.end():]
        
        return overlap_text
    
    def analyze_chunks(self, chunks: List[str]) -> dict:
        """
        Analyze chunk statistics.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "total_size": 0
            }
        
        sizes = [len(chunk) for chunk in chunks]
        
        return {
            "num_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_size": sum(sizes),
            "strategy": self.config.strategy,
            "overlap": self.config.chunk_overlap
        }


def create_optimized_chunker(
    strategy: str = "semantic",
    chunk_size: int = 512,
    overlap: int = 128
) -> OptimizedTextChunker:
    """
    Factory function to create configured chunker.
    
    Args:
        strategy: Chunking strategy
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        Configured chunker instance
    """
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return OptimizedTextChunker(config)