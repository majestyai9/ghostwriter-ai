"""
Cohere Provider Implementation
"""
import time
from typing import List, Dict, Any
from .base import LLMProvider, LLMResponse
from ..exceptions import APIKeyError, RateLimitError, TokenLimitError, LLMProviderError
from ..events import event_manager, Event, EventType

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
            raise LLMProviderError("Cohere package not installed. Run: pip install cohere")
            
        if not self.config.get('api_key'):
            raise APIKeyError("Cohere API key is required")
            
        # Latest Cohere models (as of 2024)
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
        
        # Convert history to chat format
        chat_history = self._convert_history(history) if history else []
        
        # Emit API call started event
        event_manager.emit(Event(EventType.API_CALL_STARTED, {
            'provider': 'cohere',
            'model': self.model,
            'max_tokens': max_tokens
        }))
        
        try:
            response = self._call_with_retry(
                message=prompt,
                chat_history=chat_history,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract response data
            content = response.text
            tokens_used = response.meta.get('tokens', {}).get('output_tokens', 0)
            
            # Emit API call completed event
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
            
        except cohere.error.CohereAPIError as e:
            if 'rate limit' in str(e).lower():
                event_manager.emit(Event(EventType.RATE_LIMIT_HIT, {
                    'provider': 'cohere',
                    'error': str(e)
                }))
                raise RateLimitError(str(e))
            else:
                event_manager.emit(Event(EventType.API_CALL_FAILED, {
                    'provider': 'cohere',
                    'error': str(e)
                }))
                raise LLMProviderError(f"Cohere API error: {e}")
                
        except Exception as e:
            event_manager.emit(Event(EventType.API_CALL_FAILED, {
                'provider': 'cohere',
                'error': str(e)
            }))
            raise LLMProviderError(f"Unexpected error: {e}")
            
    def _convert_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert message history to Cohere chat format"""
        chat_history = []
        
        for msg in history:
            role = msg['role']
            if role == 'system':
                # Cohere doesn't have system messages, prepend to first user message
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
        
    def _call_with_retry(self, message, chat_history, max_tokens, temperature, max_retries=3, **kwargs):
        """Make API call with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.model in ['command-r-plus', 'command-r', 'command', 'command-light', 'command-nightly']:
                    # Use chat endpoint for command models
                    return self.client.chat(
                        message=message,
                        chat_history=chat_history,
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                else:
                    # Use generate endpoint for other models
                    full_prompt = self._format_prompt_with_history(message, chat_history)
                    return self.client.generate(
                        prompt=full_prompt,
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                    
            except Exception as e:
                if 'rate limit' in str(e).lower() and attempt < max_retries - 1:
                    retry_after = 30
                    event_manager.emit(Event(EventType.RETRY_ATTEMPTED, {
                        'provider': 'cohere',
                        'attempt': attempt + 1,
                        'retry_after': retry_after
                    }))
                    self.logger.info(f"Rate limit hit, retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    raise
                    
    def _format_prompt_with_history(self, message: str, chat_history: List[Dict[str, str]]) -> str:
        """Format prompt with chat history for generate endpoint"""
        formatted = ""
        for entry in chat_history:
            role = entry['role']
            content = entry['message']
            formatted += f"{role}: {content}\n"
        formatted += f"USER: {message}\nCHATBOT:"
        return formatted
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using Cohere's tokenize method"""
        try:
            # Use the official tokenize method
            response = self.client.tokenize(
                text=text,
                model=self.model,
                offline=False  # Use API-based tokenizer for accuracy
            )
            return len(response.tokens)
        except Exception as e:
            self.logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback to rough approximation
            return len(text) // 4
        
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