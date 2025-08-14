"""
OpenAI Provider Implementation
"""
import time
import openai
import tiktoken
from typing import List, Dict, Any
from .base import LLMProvider, LLMResponse
from ..exceptions import APIKeyError, RateLimitError, TokenLimitError, LLMProviderError
from ..events import event_manager, Event, EventType

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def _validate_config(self):
        """Validate OpenAI configuration"""
        if not self.config.get('api_key'):
            raise APIKeyError("OpenAI API key is required")
            
        if not self.config.get('model') and not self.config.get('engine'):
            raise LLMProviderError("Either 'model' or 'engine' must be specified")
            
        # Set OpenAI configuration
        openai.api_key = self.config['api_key']
        openai.api_base = self.config.get('api_base')
        openai.api_type = self.config.get('api_type')
        openai.api_version = self.config.get('api_version')
        
        # Initialize tokenizer
        # Latest OpenAI models (as of 2025)
        # GPT-5 released August 2025
        model_name = self.config.get('model') or self.config.get('engine') or 'gpt-5'
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
        
        # Validate token limit
        if not self.validate_token_limit(messages, max_tokens):
            tokens_used = sum(self.count_tokens(msg['content']) for msg in messages)
            raise TokenLimitError(
                f"Token limit exceeded: {tokens_used} + {max_tokens} > {self.max_tokens_limit}",
                tokens_used=tokens_used,
                token_limit=self.max_tokens_limit
            )
        
        # Emit API call started event
        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'openai',
            'model': self.model,
            'max_tokens': max_tokens
        }))
        
        try:
            # Make API call with retry logic
            response = self._call_with_retry(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract response data
            choice = response['choices'][0]
            content = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')
            tokens_used = response.get('usage', {}).get('total_tokens', 0)
            
            # Emit API call completed event
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
            
        except openai.error.RateLimitError as e:
            event_manager.emit(Event(EventType.RATE_LIMIT_HIT, {
                'provider': 'openai',
                'error': str(e)
            }))
            raise RateLimitError(str(e), retry_after=self._get_retry_after(e))
            
        except openai.error.InvalidRequestError as e:
            event_manager.emit(Event(EventType.API_CALL_FAILED, {
                'provider': 'openai',
                'error': str(e)
            }))
            raise LLMProviderError(f"Invalid request: {e}")
            
        except Exception as e:
            event_manager.emit(Event(EventType.API_CALL_FAILED, {
                'provider': 'openai',
                'error': str(e)
            }))
            raise LLMProviderError(f"OpenAI API error: {e}")
            
    def _call_with_retry(self, messages, max_tokens, temperature, max_retries=3, **kwargs):
        """Make API call with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.config.get('engine'):
                    # Azure OpenAI
                    return openai.ChatCompletion.create(
                        engine=self.config['engine'],
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                        presence_penalty=kwargs.get('presence_penalty', 0.0),
                        **kwargs
                    )
                else:
                    # OpenAI
                    return openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                        presence_penalty=kwargs.get('presence_penalty', 0.0),
                        **kwargs
                    )
                    
            except openai.error.RateLimitError as e:
                if attempt < max_retries - 1:
                    retry_after = self._get_retry_after(e) + 5
                    event_manager.emit(Event(EventType.RETRY_ATTEMPTED, {
                        'provider': 'openai',
                        'attempt': attempt + 1,
                        'retry_after': retry_after
                    }))
                    self.logger.info(f"Rate limit hit, retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    raise
                    
    def _get_retry_after(self, error, default=30):
        """Extract retry-after time from error"""
        if hasattr(error, 'headers') and error.headers.get('Retry-After'):
            return int(error.headers['Retry-After'])
        
        # Try to parse from error message
        try:
            message = str(error)
            if 'retry after' in message:
                return int(message.split('retry after ')[1].split(' seconds')[0])
        except:
            pass
            
        return default
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        model_limits = {
            # GPT-5 series (2025)
            'gpt-5': 256000,  # Default GPT-5, 94.6% AIME, 74.9% SWE-bench
            'gpt-5-mini': 128000,  # Smaller, faster variant
            'gpt-5-nano': 64000,  # Lightweight variant
            # Legacy models (now deprecated)
            'gpt-4o': 128000,
            'gpt-4o-mini': 128000,
            'o3': 128000,  # Legacy reasoning model
            'o4-mini': 128000,  # Legacy reasoning model
            'gpt-4-turbo': 128000,
            'gpt-4-turbo-preview': 128000,
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
            'supports_vision': self.model in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-turbo-preview'],
            'supports_thinking': self.model.startswith('gpt-5')  # GPT-5 has built-in thinking
        }