"""
Anthropic Claude Provider Implementation
"""
import time
from typing import List, Dict, Any
from .base import LLMProvider, LLMResponse
from ..exceptions import APIKeyError, RateLimitError, TokenLimitError, LLMProviderError
from ..events import event_manager, Event, EventType

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
            raise LLMProviderError("Anthropic package not installed. Run: pip install anthropic")
            
        if not self.config.get('api_key'):
            raise APIKeyError("Anthropic API key is required")
            
        # Latest Anthropic models (as of 2025)
        # Claude 4 series released May 2025
        self.model = self.config.get('model', 'claude-opus-4.1-20250805')
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
        
        # Convert messages to Anthropic format
        messages = self._convert_messages(prompt, history)
        
        # Emit API call started event
        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'anthropic',
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
            
            # Extract response data
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            finish_reason = response.stop_reason or 'stop'
            
            # Emit API call completed event
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
            
        except anthropic.RateLimitError as e:
            event_manager.emit(Event(EventType.RATE_LIMIT_HIT, {
                'provider': 'anthropic',
                'error': str(e)
            }))
            raise RateLimitError(str(e))
            
        except anthropic.APIError as e:
            event_manager.emit(Event(EventType.API_CALL_FAILED, {
                'provider': 'anthropic',
                'error': str(e)
            }))
            raise LLMProviderError(f"Anthropic API error: {e}")
            
        except Exception as e:
            event_manager.emit(Event(EventType.API_CALL_FAILED, {
                'provider': 'anthropic',
                'error': str(e)
            }))
            raise LLMProviderError(f"Unexpected error: {e}")
            
    def _convert_messages(self, prompt: str, history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Convert messages to Anthropic format"""
        messages = []
        
        if history:
            for msg in history:
                role = msg['role']
                if role == 'system':
                    # Anthropic uses system messages differently
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
        
    def _call_with_retry(self, messages, max_tokens, temperature, max_retries=3, **kwargs):
        """Make API call with retry logic"""
        
        # Extract system message if present
        system_message = None
        for msg in messages:
            if msg.get('role') == 'system':
                system_message = msg['content']
                messages = [m for m in messages if m.get('role') != 'system']
                break
                
        for attempt in range(max_retries):
            try:
                params = {
                    'model': self.model,
                    'messages': messages,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }
                
                if system_message:
                    params['system'] = system_message
                    
                return self.client.messages.create(**params)
                
            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    retry_after = 30  # Anthropic doesn't provide retry-after header
                    event_manager.emit(Event(EventType.RETRY_ATTEMPTED, {
                        'provider': 'anthropic',
                        'attempt': attempt + 1,
                        'retry_after': retry_after
                    }))
                    self.logger.info(f"Rate limit hit, retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    raise
                    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's token counting API"""
        try:
            # Use the official token counting method
            response = self.client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=self.model,
                messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens
        except Exception as e:
            self.logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback to rough approximation
            return len(text) // 4
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information"""
        model_limits = {
            # Claude 4 series (2025)
            'claude-opus-4.1-20250805': 200000,  # Best coding model, 72.5% SWE-bench
            'claude-opus-4-20250522': 200000,  # Original Claude 4 Opus
            'claude-sonnet-4-20250522': 200000,  # Claude 4 Sonnet
            'claude-3.7-sonnet-20250224': 200000,  # Hybrid reasoning model
            # Legacy Claude 3 series
            'claude-3-5-sonnet-20241022': 200000,
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