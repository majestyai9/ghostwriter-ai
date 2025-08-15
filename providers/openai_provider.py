"""
OpenAI Provider Implementation
"""
from collections.abc import Generator
from typing import Any, Dict, List

import openai
import tiktoken

from events import Event, EventType, event_manager
from exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    TokenLimitError,
)

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""

    def _validate_config(self):
        """Validate OpenAI configuration"""
        if not self.config.get('api_key'):
            raise ProviderAuthError("OpenAI API key is required")

        if not self.config.get('model') and not self.config.get('engine'):
            raise ProviderError("Either 'model' or 'engine' must be specified")

        # Set OpenAI configuration
        openai.api_key = self.config['api_key']
        openai.api_base = self.config.get('api_base')
        openai.api_type = self.config.get('api_type')
        openai.api_version = self.config.get('api_version')

        # Initialize tokenizer
        model_name = self.config.get('model') or self.config.get('engine') or 'gpt-4'
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.model = model_name
        self.max_tokens_limit = self.config.get('token_limit', 4096)

    def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """Generate text using OpenAI API"""

        messages = self.prepare_messages(prompt, history)

        if not self.validate_token_limit(messages, max_tokens):
            tokens_used = sum(self.count_tokens(msg['content']) for msg in messages)
            raise TokenLimitError(
                f"Token limit exceeded: {tokens_used} + {max_tokens} > {self.max_tokens_limit}",
                tokens_used=tokens_used,
                token_limit=self.max_tokens_limit
            )

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'openai',
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

            choice = response['choices'][0]
            content = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')
            tokens_used = response.get('usage', {}).get('total_tokens', 0)

            event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
                'provider': 'openai',
                'model': self.model,
                'tokens_used': tokens_used,
                'finish_reason': finish_reason
            }))

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model=self.model,
                raw_response=response
            )

        except Exception as e:
            raise self._handle_error(e)

    def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming using OpenAI API"""

        messages = self.prepare_messages(prompt, history)

        if not self.validate_token_limit(messages, max_tokens):
            tokens_used = sum(self.count_tokens(msg['content']) for msg in messages)
            raise TokenLimitError(
                f"Token limit exceeded: {tokens_used} + {max_tokens} > {self.max_tokens_limit}",
                tokens_used=tokens_used,
                token_limit=self.max_tokens_limit
            )

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'openai',
            'model': self.model,
            'max_tokens': max_tokens,
            'streaming': True
        }))

        try:
            response = self._call_with_retry(
                self._create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )

            for chunk in response:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    yield delta['content']

        except Exception as e:
            raise self._handle_error(e)

    def _create_chat_completion(self, **kwargs):
        if self.config.get('engine'):
            return openai.ChatCompletion.create(engine=self.config['engine'], **kwargs)
        else:
            return openai.ChatCompletion.create(model=self.model, **kwargs)

    def _handle_error(self, error: Exception) -> ProviderError:
        if isinstance(error, openai.error.AuthenticationError):
            return ProviderAuthError(str(error))
        elif isinstance(error, openai.error.RateLimitError):
            return ProviderRateLimitError(str(error))
        elif isinstance(error, openai.error.InvalidRequestError):
            if "content_filter" in str(error):
                return ProviderContentFilterError(str(error))
            else:
                return ProviderError(str(error))
        else:
            return ProviderError(str(error))

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        model_limits = {
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'gpt-3.5-turbo': 16385,
            'gpt-3.5-turbo-16k': 16385,
            'gpt-3.5-turbo-instruct': 4096,
        }

        return {
            'provider': 'openai',
            'model': self.model,
            'max_tokens': model_limits.get(self.model, self.max_tokens_limit),
            'supports_streaming': True,
            'supports_functions': True,
            'supports_vision': self.model in ['gpt-4-turbo', 'gpt-4-turbo-preview'],
        }
