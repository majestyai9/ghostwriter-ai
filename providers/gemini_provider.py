"""
Google Gemini Provider Implementation
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
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation"""

    def _validate_config(self):
        """Validate Gemini configuration"""
        if not GEMINI_AVAILABLE:
            raise ProviderError("Google Generative AI package not installed. Run: pip install google-generativeai")

        if not self.config.get('api_key'):
            raise ProviderAuthError("Gemini API key is required")

        self.model_name = self.config.get('model', 'gemini-1.5-pro-latest')
        self.max_tokens_limit = self.config.get('token_limit', 1048576)

        genai.configure(api_key=self.config['api_key'])

        generation_config = {
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.95),
            'top_k': self.config.get('top_k', 40),
        }

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )

    def generate(self,
                prompt: str,
                history: List[Dict[str, str]] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                **kwargs) -> LLMResponse:
        """Generate text using Gemini API"""

        messages = self._convert_messages(prompt, history)

        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
            'top_p': kwargs.get('top_p', 0.95),
            'top_k': kwargs.get('top_k', 40),
        }

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'gemini',
            'model': self.model_name,
            'max_tokens': max_tokens
        }))

        try:
            # Use the base class retry mechanism with circuit breaker
            response = self._call_with_retry(
                self._generate_content,
                prompt=messages,  # Gemini expects 'prompt' not 'messages'
                generation_config=generation_config,
                **kwargs
            )

            # Check if response has valid content
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    content = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                else:
                    # Try direct text access
                    content = response.text if hasattr(response, 'text') else ""
            elif hasattr(response, 'text'):
                content = response.text
            else:
                # Handle case where response has no content
                self.logger.warning("Gemini response has no valid content")
                content = ""
                
            tokens_used = self.count_tokens(prompt) + self.count_tokens(content)

            event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
                'provider': 'gemini',
                'model': self.model_name,
                'tokens_used': tokens_used,
                'finish_reason': 'stop'
            }))

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                finish_reason='stop',
                model=self.model_name,
                raw_response=None
            )

        except Exception as e:
            # _handle_error should raise an exception, not return it
            error = self._handle_error(e)
            raise error

    def generate_stream(self,
                       prompt: str,
                       history: List[Dict[str, str]] = None,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming using Gemini API"""

        messages = self._convert_messages(prompt, history)

        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
            'top_p': kwargs.get('top_p', 0.95),
            'top_k': kwargs.get('top_k', 40),
        }

        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'gemini',
            'model': self.model_name,
            'max_tokens': max_tokens,
            'streaming': True
        }))

        try:
            # Use retry mechanism for streaming as well
            response = self._call_with_retry(
                self.model.generate_content,
                messages,  # This is already a string prompt from _convert_messages
                generation_config=generation_config,
                stream=True,
                **kwargs
            )

            for chunk in response:
                if hasattr(chunk, 'text'):
                    yield chunk.text

        except Exception as e:
            error = self._handle_error(e)
            raise error

    def _generate_content(self, prompt: str, **kwargs):
        # Gemini expects the prompt as the first argument
        return self.model.generate_content(prompt, **kwargs)

    def _convert_messages(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Convert messages to Gemini format"""

        full_prompt = ""

        if history:
            for msg in history:
                role = msg.get('role', '')
                content = msg.get('content', '')

                if role == 'system':
                    full_prompt += f"Instructions: {content}\n\n"
                elif role == 'user':
                    full_prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    full_prompt += f"Assistant: {content}\n\n"

        full_prompt += f"User: {prompt}\n\nAssistant:"
        return full_prompt

    def _handle_error(self, error: Exception) -> ProviderError:
        """Handle Gemini-specific errors and return appropriate exception"""
        error_str = str(error)
        
        if "PermissionDenied" in error_str or "API key" in error_str:
            return ProviderAuthError(error_str)
        elif "ResourceExhausted" in error_str or "rate limit" in error_str.lower():
            return ProviderRateLimitError(error_str)
        elif "InvalidArgument" in error_str:
            return ProviderError(f"Invalid request: {error_str}")
        else:
            return ProviderError(error_str)

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's token counter"""
        if not text:
            return 0
            
        try:
            result = self.model.count_tokens(text)
            if hasattr(result, 'total_tokens'):
                return result.total_tokens
            return int(result)
        except Exception as e:
            self.logger.warning(f"Token counting failed with model.count_tokens: {e}")
            # Fallback to approximation
            return len(text) // 4

    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information"""
        model_limits = {
            'gemini-1.5-pro-latest': 1048576,
            'gemini-1.5-pro': 1048576,
            'gemini-1.5-flash': 1048576,
            'gemini-1.5-flash-latest': 1048576,
            'gemini-1.0-pro': 32768,
            'gemini-1.0-pro-latest': 32768,
        }

        return {
            'provider': 'gemini',
            'model': self.model_name,
            'max_tokens': model_limits.get(self.model_name, 32768),
            'supports_streaming': True,
            'supports_functions': True,
            'supports_vision': True,
        }
