"""
OpenRouter Provider Implementation - Access to multiple models through one API
"""
import time
from typing import Any, Dict, List

import requests

from events import Event, EventType, event_manager
from exceptions import APIKeyError, LLMProviderError, RateLimitError

from .base import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider implementation"""

    def _validate_config(self):
        """Validate OpenRouter configuration"""
        if not self.config.get('api_key'):
            raise APIKeyError("OpenRouter API key is required")

        self.api_key = self.config['api_key']
        self.base_url = self.config.get('base_url', 'https://openrouter.ai/api/v1')

        # OpenRouter supports many models - default to Claude Opus 4.1
        self.model = self.config.get('model', 'anthropic/claude-opus-4.1')

        # Get site URL and name for OpenRouter headers (optional but recommended)
        self.site_url = self.config.get('site_url', 'https://github.com/ghostwriter-ai')
        self.site_name = self.config.get('site_name', 'GhostWriter AI')

        # Model-specific token limits (updated 2025)
        self.model_limits = {
            # Anthropic Claude 4 models (2025)
            'anthropic/claude-opus-4.1': 200000,
            'anthropic/claude-opus-4': 200000,
            'anthropic/claude-sonnet-4': 200000,
            'anthropic/claude-3.7-sonnet': 200000,
            # Legacy Anthropic models
            'anthropic/claude-3.5-sonnet': 200000,
            'anthropic/claude-3-opus': 200000,
            'anthropic/claude-3-sonnet': 200000,
            'anthropic/claude-3-haiku': 200000,
            'anthropic/claude-2.1': 200000,
            'anthropic/claude-2': 100000,

            # OpenAI GPT-5 models (2025)
            'openai/gpt-5': 256000,
            'openai/gpt-5-mini': 128000,
            'openai/gpt-5-nano': 64000,
            # Legacy OpenAI models
            'openai/gpt-4o': 128000,
            'openai/gpt-4o-mini': 128000,
            'openai/o3': 128000,
            'openai/o4-mini': 128000,
            'openai/gpt-4-turbo': 128000,
            'openai/gpt-4': 8192,
            'openai/gpt-3.5-turbo': 4096,

            # Google Gemini 2.5 models (2025)
            'google/gemini-2.5-pro': 2097152,
            'google/gemini-2.5-flash': 1048576,
            'google/gemini-2.5-flash-lite': 524288,
            # Legacy Google models
            'google/gemini-pro-1.5': 1048576,
            'google/gemini-flash-1.5': 1048576,
            'google/gemini-pro': 32768,

            # Meta Llama models
            'meta-llama/llama-3.1-405b-instruct': 32768,
            'meta-llama/llama-3.1-70b-instruct': 32768,
            'meta-llama/llama-3-70b-instruct': 8192,

            # Mistral models
            'mistralai/mistral-large': 32768,
            'mistralai/mixtral-8x22b-instruct': 65536,
            'mistralai/mixtral-8x7b-instruct': 32768,

            # Cohere models
            'cohere/command-r-plus': 128000,
            'cohere/command-r': 128000,

            # Other models
            'databricks/dbrx-instruct': 32768,
            'deepseek/deepseek-coder': 16384,
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

        # Emit API call started event
        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'openrouter',
            'model': self.model,
            'max_tokens': max_tokens
        }))

        try:
            response = self._call_with_retry(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Parse response
            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')

                if response.status_code == 429:
                    event_manager.emit(Event(EventType.RATE_LIMIT_HIT, {
                        'provider': 'openrouter',
                        'error': error_msg
                    }))
                    raise RateLimitError(error_msg)
                else:
                    raise LLMProviderError(f"OpenRouter API error: {error_msg}")

            data = response.json()

            # Extract response data
            choice = data['choices'][0]
            content = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')

            # Get token usage
            usage = data.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)

            # Emit API call completed event
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

        except (RateLimitError, LLMProviderError):
            raise
        except Exception as e:
            event_manager.emit(Event(EventType.API_CALL_FAILED, {
                'provider': 'openrouter',
                'error': str(e)
            }))
            raise LLMProviderError(f"OpenRouter API error: {e}")

    def _call_with_retry(self, messages, max_tokens, temperature, max_retries=3, **kwargs):
        """Make API call with retry logic"""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': self.site_url,
            'X-Title': self.site_name,
        }

        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': kwargs.get('top_p'),
            'frequency_penalty': kwargs.get('frequency_penalty', 0),
            'presence_penalty': kwargs.get('presence_penalty', 0),
            'stream': False
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 429 and attempt < max_retries - 1:
                    # Rate limit - check for retry-after header
                    retry_after = int(response.headers.get('Retry-After', 30))
                    event_manager.emit(Event(EventType.RETRY_ATTEMPTED, {
                        'provider': 'openrouter',
                        'attempt': attempt + 1,
                        'retry_after': retry_after
                    }))
                    self.logger.info(f"Rate limit hit, retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    return response

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request failed, retrying: {e}")
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

        return response

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # OpenRouter doesn't provide token counting, use estimation
        # Different models have different tokenizers, this is a rough average
        return len(text) // 4

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

    def list_available_models(self) -> List[str]:
        """List all available models on OpenRouter"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }

            response = requests.get(
                f'{self.base_url}/models',
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            else:
                self.logger.warning(f"Failed to fetch models list: {response.status_code}")
                return list(self.model_limits.keys())

        except Exception as e:
            self.logger.warning(f"Error fetching models: {e}")
            return list(self.model_limits.keys())
