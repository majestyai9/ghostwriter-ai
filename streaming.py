"""
Streaming support for real-time content generation
"""
import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class StreamChunk:
    """Represents a chunk of streamed content"""
    content: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None

class StreamingManager:
    """Manages streaming responses from LLM providers"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_streams = {}

    def create_stream(self, stream_id: str, provider_stream: Iterator) -> Generator[StreamChunk, None, None]:
        """
        Create a managed stream from provider stream
        
        Args:
            stream_id: Unique identifier for this stream
            provider_stream: Raw stream from provider
            
        Yields:
            StreamChunk objects
        """
        self.active_streams[stream_id] = {
            'status': 'active',
            'chunks_sent': 0,
            'total_content': ''
        }

        try:
            for chunk in provider_stream:
                # Extract content based on provider format
                if hasattr(chunk, 'choices'):
                    # OpenAI format
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                    else:
                        continue
                elif hasattr(chunk, 'text'):
                    # Anthropic/Cohere format
                    content = chunk.text
                elif isinstance(chunk, str):
                    # Simple string format
                    content = chunk
                else:
                    continue

                # Update stream state
                self.active_streams[stream_id]['chunks_sent'] += 1
                self.active_streams[stream_id]['total_content'] += content

                # Yield chunk
                yield StreamChunk(
                    content=content,
                    is_final=False,
                    metadata={
                        'stream_id': stream_id,
                        'chunk_index': self.active_streams[stream_id]['chunks_sent']
                    }
                )

            # Final chunk
            yield StreamChunk(
                content='',
                is_final=True,
                metadata={
                    'stream_id': stream_id,
                    'total_chunks': self.active_streams[stream_id]['chunks_sent'],
                    'total_length': len(self.active_streams[stream_id]['total_content'])
                }
            )

            self.active_streams[stream_id]['status'] = 'completed'

        except Exception as e:
            self.logger.error(f"Stream {stream_id} error: {e}")
            self.active_streams[stream_id]['status'] = 'error'
            raise

        finally:
            # Cleanup after delay
            if stream_id in self.active_streams:
                content = self.active_streams[stream_id]['total_content']
                del self.active_streams[stream_id]

    def cancel_stream(self, stream_id: str):
        """Cancel an active stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['status'] = 'cancelled'

    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a stream"""
        return self.active_streams.get(stream_id)

# Global streaming manager
streaming_manager = StreamingManager()

class StreamingMixin:
    """Mixin to add streaming capabilities to LLM providers"""

    def generate_stream(self,
                       prompt: str,
                       history: list = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """
        Generate text with streaming
        
        Args:
            prompt: Input prompt
            history: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Text chunks as they're generated
        """
        raise NotImplementedError("Provider must implement generate_stream")

    def _create_streaming_request(self, messages, max_tokens, temperature, **kwargs):
        """Create a streaming request based on provider type"""
        raise NotImplementedError("Provider must implement _create_streaming_request")
