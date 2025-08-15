"""
Data models for RAG system.

Provides shared data structures used across RAG modules.
"""

import time
from typing import Any, Dict, Optional


class ChunkMetadata:
    """Metadata for indexed text chunks."""
    
    def __init__(self,
                 chapter_number: int,
                 section_number: Optional[int] = None,
                 chunk_index: int = 0,
                 chapter_title: str = "",
                 section_title: str = "",
                 timestamp: float = None,
                 tokens: int = 0):
        """
        Initialize chunk metadata.
        
        Args:
            chapter_number: Chapter number in book
            section_number: Optional section number within chapter
            chunk_index: Index of this chunk within chapter/section
            chapter_title: Title of the chapter
            section_title: Title of the section if applicable
            timestamp: Creation timestamp
            tokens: Estimated token count
        """
        self.chapter_number = chapter_number
        self.section_number = section_number
        self.chunk_index = chunk_index
        self.chapter_title = chapter_title
        self.section_title = section_title
        self.timestamp = timestamp or time.time()
        self.tokens = tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chapter_number": self.chapter_number,
            "section_number": self.section_number,
            "chunk_index": self.chunk_index,
            "chapter_title": self.chapter_title,
            "section_title": self.section_title,
            "timestamp": self.timestamp,
            "tokens": self.tokens
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChunkMetadata(chapter={self.chapter_number}, "
            f"section={self.section_number}, chunk={self.chunk_index})"
        )