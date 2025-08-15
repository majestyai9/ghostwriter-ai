"""
Cohere Provider Implementation
"""
from collections.abc import Generator
from typing import Any, Dict, List

from events import Event, EventType, event_manager
from exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)

from .base import LLMProvider, LLMResponse

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

class CohereProvider(LLMProvider):
    """Cohere API provider implementation"""

    def _validate_config(self):
        """Validate Cohere configuration"""
        if not COHERE_AVAILABLE:
            raise ProviderError("Cohere package not installed. Run: pip install cohere")

        if not self.config.get('api_key'):
            raise ProviderAuthError("Cohere API key is required")

        self.model = self.config.get('model', 'command-r-plus')
        self.max_tokens_limit = self.config.get('token_limit', 128000)

        # Initialize Cohere client
        self.client = cohere.Client(self.config['api_key'])

    def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """Generate text using Cohere API"""

        chat_history = self._convert_history(history) if history else []

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'cohere',
            'model': self.model,
            'max_tokens': max_tokens
        }))

        try:
            response = self._call_with_retry(
                self._create_chat,
                message=prompt,
                chat_history=chat_history,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            content = response.text
            tokens_used = response.meta.get('tokens', {}).get('output_tokens', 0)

            event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
                'provider': 'cohere',
                'model': self.model,
                'tokens_used': tokens_used,
                'finish_reason': 'complete'
            }))

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                finish_reason='complete',
                model=self.model,
                raw_response=response.__dict__ if hasattr(response, '__dict__') else None
            )

        except Exception as e:
            raise self._handle_error(e)

    def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming using Cohere API"""

        chat_history = self._convert_history(history) if history else []

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'cohere',
            'model': self.model,
            'max_tokens': max_tokens,
            'streaming': True
        }))

        try:
            stream = self.client.chat(
                message=prompt,
                chat_history=chat_history,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )

            for event in stream:
                if event.event_type == "text-generation":
                    yield event.text

        except Exception as e:
            raise self._handle_error(e)

    def _create_chat(self, **kwargs):
        return self.client.chat(**kwargs)

    def _convert_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert message history to Cohere chat format"""
        chat_history = []

        for msg in history:
            role = msg['role']
            if role == 'system':
                continue
            elif role == 'user':
                chat_history.append({
                    'role': 'USER',
                    'message': msg['content']
                })
            elif role == 'assistant':
                chat_history.append({
                    'role': 'CHATBOT',
                    'message': msg['content']
                })

        return chat_history

    def _handle_error(self, error: Exception) -> ProviderError:
        if isinstance(error, cohere.errors.AuthError):
            return ProviderAuthError(str(error))
        elif isinstance(error, cohere.errors.RateLimitError):
            return ProviderRateLimitError(str(error))
        elif isinstance(error, cohere.errors.CohereError):
            return ProviderError(str(error))
        else:
            return ProviderError(str(error))

    def get_model_info(self) -> Dict[str, Any]:
        """Get Cohere model information"""
        model_limits = {
            'command-r-plus': 128000,
            'command-r': 128000,
            'command': 4096,
            'command-light': 4096,
            'command-nightly': 4096,
            'embed-english-v3.0': 512,
            'embed-multilingual-v3.0': 512
        }

        return {
            'provider': 'cohere',
            'model': self.model,
            'max_tokens': model_limits.get(self.model, 4096),
            'supports_streaming': True,
            'supports_functions': False
        }
