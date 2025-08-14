"""
Google Gemini Provider Implementation
"""
import time
from typing import List, Dict, Any
from .base import LLMProvider, LLMResponse
from ..exceptions import APIKeyError, RateLimitError, TokenLimitError, LLMProviderError
from ..events import event_manager, Event, EventType

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
            raise LLMProviderError("Google Generative AI package not installed. Run: pip install google-generativeai")
            
        if not self.config.get('api_key'):
            raise APIKeyError("Gemini API key is required")
            
        # Latest Gemini models (as of 2025)
        # Gemini 2.5 released March 2025
        self.model_name = self.config.get('model', 'gemini-2.5-pro')
        self.max_tokens_limit = self.config.get('token_limit', 2097152)  # 2M tokens for Gemini 2.5
        
        # Configure Gemini
        genai.configure(api_key=self.config['api_key'])
        
        # Initialize model
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
        
        # Convert messages to Gemini format
        messages = self._convert_messages(prompt, history)
        
        # Update generation config for this request
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
            'top_p': kwargs.get('top_p', 0.95),
            'top_k': kwargs.get('top_k', 40),
        }
        
        # Emit API call started event
        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'gemini',
            'model': self.model_name,
            'max_tokens': max_tokens
        }))
        
        try:
            response = self._call_with_retry(
                messages=messages,
                generation_config=generation_config,
                **kwargs
            )
            
            # Extract response data
            content = response.text
            
            # Estimate token usage (Gemini doesn't provide exact counts in response)
            tokens_used = self.count_tokens(prompt) + self.count_tokens(content)
            
            # Emit API call completed event
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
            if 'quota' in str(e).lower() or 'rate' in str(e).lower():
                event_manager.emit(Event(EventType.RATE_LIMIT_HIT, {
                    'provider': 'gemini',
                    'error': str(e)
                }))
                raise RateLimitError(str(e))
            else:
                event_manager.emit(Event(EventType.API_CALL_FAILED, {
                    'provider': 'gemini',
                    'error': str(e)
                }))
                raise LLMProviderError(f"Gemini API error: {e}")
                
    def _convert_messages(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Convert messages to Gemini format"""
        
        # Build conversation context
        full_prompt = ""
        
        if history:
            for msg in history:
                role = msg['role']
                content = msg['content']
                
                if role == 'system':
                    full_prompt += f"Instructions: {content}\n\n"
                elif role == 'user':
                    full_prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    full_prompt += f"Assistant: {content}\n\n"
                    
        full_prompt += f"User: {prompt}\n\nAssistant:"
        return full_prompt
        
    def _call_with_retry(self, messages, generation_config, max_retries=3, **kwargs):
        """Make API call with retry logic"""
        for attempt in range(max_retries):
            try:
                # Create a new model instance with updated config for this call
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
                
                # Generate content
                response = model.generate_content(messages)
                return response
                
            except Exception as e:
                if ('quota' in str(e).lower() or 'rate' in str(e).lower()) and attempt < max_retries - 1:
                    retry_after = 30
                    event_manager.emit(Event(EventType.RETRY_ATTEMPTED, {
                        'provider': 'gemini',
                        'attempt': attempt + 1,
                        'retry_after': retry_after
                    }))
                    self.logger.info(f"Rate limit hit, retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    raise
                    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's token counter"""
        try:
            # Use Gemini's built-in token counting
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            self.logger.warning(f"Token counting failed with model.count_tokens: {e}")
            # Fallback to estimation (approximately 1 token per 4 characters)
            return len(text) // 4
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information"""
        model_limits = {
            # Gemini 2.5 series (2025)
            'gemini-2.5-pro': 2097152,  # 2M context, 63.8% SWE-bench
            'gemini-2.5-pro-experimental': 2097152,  # Experimental version
            'gemini-2.5-flash': 1048576,  # 1M context, best price/performance
            'gemini-2.5-flash-lite': 524288,  # 512K context, lightweight
            # Legacy Gemini 1.5 series
            'gemini-1.5-pro-latest': 1048576,  # 1M context window
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
            'supports_thinking': self.model_name.startswith('gemini-2.5'),  # Gemini 2.5 has thinking capabilities
            'supports_deep_think': self.model_name.startswith('gemini-2.5-pro')  # Pro has Deep Think mode
        }