"""
Anthropic Claude Provider Implementation
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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation"""

    def _validate_config(self):
        """Validate Anthropic configuration"""
        if not ANTHROPIC_AVAILABLE:
            raise ProviderError("Anthropic package not installed. Run: pip install anthropic")

        if not self.config.get('api_key'):
            raise ProviderAuthError("Anthropic API key is required")

        self.model = self.config.get('model', 'claude-3-opus-20240229')
        self.max_tokens_limit = self.config.get('token_limit', 200000)

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.config['api_key'])

    def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """Generate text using Anthropic API"""

        messages = self._convert_messages(prompt, history)

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'anthropic',
            'model': self.model,
            'max_tokens': max_tokens
        }))

        try:
            response = self._call_with_retry(
                self._create_message,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            finish_reason = response.stop_reason or 'stop'

            event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
                'provider': 'anthropic',
                'model': self.model,
                'tokens_used': tokens_used,
                'finish_reason': finish_reason
            }))

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model=self.model,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

        except Exception as e:
            raise self._handle_error(e)

    def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming using Anthropic API"""

        messages = self._convert_messages(prompt, history)

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'anthropic',
            'model': self.model,
            'max_tokens': max_tokens,
            'streaming': True
        }))

        try:
            with self.client.messages.stream(
                max_tokens=max_tokens,
                messages=messages,
                model=self.model,
                temperature=temperature,
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise self._handle_error(e)

    def _create_message(self, **kwargs):
        return self.client.messages.create(**kwargs)

    def _convert_messages(self, prompt: str, history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Convert messages to Anthropic format"""
        messages = []

        if history:
            for msg in history:
                role = msg['role']
                if role == 'system':
                    continue
                elif role == 'assistant':
                    role = 'assistant'
                else:
                    role = 'user'

                messages.append({
                    'role': role,
                    'content': msg['content']
                })

        messages.append({'role': 'user', 'content': prompt})
        return messages

    def _handle_error(self, error: Exception) -> ProviderError:
        if isinstance(error, anthropic.AuthenticationError):
            return ProviderAuthError(str(error))
        elif isinstance(error, anthropic.RateLimitError):
            return ProviderRateLimitError(str(error))
        elif isinstance(error, anthropic.APIError):
            return ProviderError(str(error))
        else:
            return ProviderError(str(error))

    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information"""
        model_limits = {
            'claude-3-opus-20240229': 200000,
            'claude-3-sonnet-20240229': 200000,
            'claude-3-haiku-20240307': 200000,
            'claude-2.1': 200000,
            'claude-2.0': 100000,
            'claude-instant-1.2': 100000
        }

        return {
            'provider': 'anthropic',
            'model': self.model,
            'max_tokens': model_limits.get(self.model, 100000),
            'supports_streaming': True,
            'supports_functions': False
        }
