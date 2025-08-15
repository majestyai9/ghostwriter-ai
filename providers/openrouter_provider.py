"""
OpenRouter Provider Implementation - Access to multiple models through one API
"""
import json
from collections.abc import Generator
from typing import Any, Dict, List

import requests

from events import Event, EventType, event_manager
from exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)

from .base import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider implementation"""

    def _validate_config(self):
        """Validate OpenRouter configuration"""
        if not self.config.get('api_key'):
            raise ProviderAuthError("OpenRouter API key is required")

        self.api_key = self.config['api_key']
        self.base_url = self.config.get('base_url', 'https://openrouter.ai/api/v1')
        self.model = self.config.get('model', 'anthropic/claude-3-opus-20240229')
        self.site_url = self.config.get('site_url', 'https://github.com/ghostwriter-ai')
        self.site_name = self.config.get('site_name', 'GhostWriter AI')

        self.model_limits = {
            'anthropic/claude-3-opus-20240229': 200000,
            'anthropic/claude-3-sonnet-20240229': 200000,
            'anthropic/claude-3-haiku-20240307': 200000,
            'openai/gpt-4o': 128000,
            'openai/gpt-4-turbo': 128000,
            'openai/gpt-4': 8192,
            'openai/gpt-3.5-turbo': 4096,
            'google/gemini-pro-1.5': 1048576,
            'google/gemini-pro': 32768,
            'meta-llama/llama-3-70b-instruct': 8192,
            'mistralai/mistral-large': 32768,
            'cohere/command-r-plus': 128000,
        }

        self.max_tokens_limit = self.model_limits.get(self.model, 4096)

    def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """Generate text using OpenRouter API"""

        messages = self.prepare_messages(prompt, history)

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'openrouter',
            'model': self.model,
            'max_tokens': max_tokens
        }))

        try:
            response = self._call_with_retry(
                self._create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            data = response.json()
            choice = data['choices'][0]
            content = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')
            usage = data.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)

            event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
                'provider': 'openrouter',
                'model': self.model,
                'tokens_used': tokens_used,
                'finish_reason': finish_reason
            }))

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model=self.model,
                raw_response=data
            )

        except Exception as e:
            raise self._handle_error(e)

    def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming using OpenRouter API"""

        messages = self.prepare_messages(prompt, history)

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'openrouter',
            'model': self.model,
            'max_tokens': max_tokens,
            'streaming': True
        }))

        try:
            response = self._create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True, **kwargs)

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[len("data: "):]
                        if json_str.strip() == "[DONE]":
                            break
                        data = json.loads(json_str)
                        delta = data['choices'][0]['delta']
                        if 'content' in delta:
                            yield delta['content']

        except Exception as e:
            raise self._handle_error(e)

    def _create_chat_completion(self, **kwargs):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': self.site_url,
            'X-Title': self.site_name,
        }

        payload = {
            'model': self.model,
            'messages': kwargs.get('messages'),
            'max_tokens': kwargs.get('max_tokens'),
            'temperature': kwargs.get('temperature'),
            'stream': kwargs.get('stream', False)
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=120,
            stream=kwargs.get('stream', False)
        )
        response.raise_for_status()
        return response

    def _handle_error(self, error: Exception) -> ProviderError:
        if isinstance(error, requests.exceptions.HTTPError):
            if error.response.status_code == 401:
                return ProviderAuthError(str(error))
            elif error.response.status_code == 429:
                return ProviderRateLimitError(str(error))
            else:
                return ProviderError(str(error))
        else:
            return ProviderError(str(error))

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model information"""
        return {
            'provider': 'openrouter',
            'model': self.model,
            'max_tokens': self.max_tokens_limit,
            'supports_streaming': True,
            'supports_functions': self.model.startswith('openai/'),
            'available_models': list(self.model_limits.keys())
        }
