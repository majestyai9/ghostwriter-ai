"""
Enhanced AI module with provider abstraction and error handling
"""
import logging
import os
from typing import Any, Dict, List

import config
from events import Event, EventType, event_manager
from exceptions import ContentGenerationError, LLMProviderError, RateLimitError, TokenLimitError
from providers import LLMProvider, get_provider

# Initialize provider
provider: LLMProvider = None

def initialize_provider():
    """Initialize the LLM provider"""
    global provider
    try:
        provider = get_provider(config=config.PROVIDER_CONFIG)
        logging.info(f"Initialized {config.LLM_PROVIDER} provider")
    except Exception as e:
        logging.error(f"Failed to initialize provider: {e}")
        raise

# Initialize on module load (skip during tests)
if not os.getenv('PYTEST_CURRENT_TEST'):
    initialize_provider()

def callLLM(prompt: str,
           history: List[Dict[str, str]] = None,
           waitingShortAnswer: bool = False,
           forceMaximum: bool = False,
           appendResponse: bool = True,
           max_retries: int = 3) -> str:
    """
    Enhanced LLM call with error handling and event emission
    
    Args:
        prompt: The prompt to send
        history: Conversation history
        waitingShortAnswer: Whether expecting a short answer
        forceMaximum: Whether to force maximum token usage
        appendResponse: Whether to append response to history
        max_retries: Maximum number of retries on failure
        
    Returns:
        Generated text
        
    Raises:
        ContentGenerationError: If generation fails after retries
    """
    if provider is None:
        raise ContentGenerationError("Provider not initialized")

    max_tokens = config.MAX_TOKENS_SHORT if waitingShortAnswer else config.MAX_TOKENS

    if forceMaximum:
        prompt += f"\nTry to use the maximum {max_tokens} tokens available."

    prompt += f"\nLimit the output to {max_tokens} tokens."

    # Clean history to manage token limits
    if history:
        history = _manage_history_tokens(history, max_tokens)

    for attempt in range(max_retries):
        try:
            # Generate response
            response = provider.generate(
                prompt=prompt,
                history=history,
                max_tokens=max_tokens,
                temperature=config.TEMPERATURE
            )

            if response.finish_reason == 'length':
                event_manager.emit(Event(EventType.TOKEN_LIMIT_REACHED, {
                    'tokens_used': response.tokens_used,
                    'max_tokens': max_tokens
                }))
                logging.warning("Response truncated due to token limit")

            # Append to history if requested
            if appendResponse and history is not None:
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": response.content})

            return response.content

        except RateLimitError as e:
            if attempt < max_retries - 1:
                logging.info(f"Rate limit hit, retry {attempt + 1}/{max_retries}")
                # RateLimitError may contain retry_after
                if hasattr(e, 'retry_after') and e.retry_after:
                    import time
                    time.sleep(e.retry_after)
                continue
            else:
                raise ContentGenerationError(f"Rate limit exceeded after {max_retries} retries")

        except TokenLimitError as e:
            # Try to reduce history and retry
            if history and len(history) > 2:
                logging.info("Token limit exceeded, reducing history")
                history = _reduce_history(history)
                continue
            else:
                raise ContentGenerationError(f"Token limit exceeded: {e}")

        except LLMProviderError as e:
            logging.error(f"Provider error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise ContentGenerationError(f"Provider error after {max_retries} retries: {e}")

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise ContentGenerationError(f"Unexpected error during generation: {e}")

    raise ContentGenerationError(f"Failed to generate content after {max_retries} attempts")

def _manage_history_tokens(history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """
    Manage history to fit within token limits
    
    Args:
        history: Conversation history
        max_tokens: Maximum tokens for response
        
    Returns:
        Managed history
    """
    if not history:
        return history

    # Always keep the first (system) message
    managed = [history[0]] if history else []

    # Calculate available tokens
    available_tokens = config.TOKEN_LIMIT - max_tokens
    current_tokens = provider.count_tokens(managed[0]['content']) if managed else 0

    # Collect messages that fit, starting from most recent
    temp_messages = []
    for msg in reversed(history[1:]):
        msg_tokens = provider.count_tokens(msg['content'])
        if current_tokens + msg_tokens < available_tokens:
            temp_messages.append(msg)
            current_tokens += msg_tokens
        else:
            break

    # Reverse to maintain chronological order and add to managed list
    temp_messages.reverse()
    managed.extend(temp_messages)

    if len(managed) < len(history):
        removed = len(history) - len(managed)
        logging.info(f"Removed {removed} messages from history to fit token limit")
        event_manager.emit(Event(EventType.TOKEN_LIMIT_REACHED, {
            'messages_removed': removed,
            'messages_kept': len(managed)
        }))

    return managed

def _reduce_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Reduce history by removing oldest messages
    
    Args:
        history: Conversation history
        
    Returns:
        Reduced history
    """
    if len(history) <= 2:
        return history

    # Keep system message and most recent messages
    return [history[0]] + history[-(len(history) // 2):]

def get_provider_info() -> Dict[str, Any]:
    """Get information about the current provider"""
    if provider:
        return provider.get_model_info()
    return {}

def switch_provider(provider_name: str, config: Dict[str, Any]):
    """
    Switch to a different provider at runtime
    
    Args:
        provider_name: Name of the new provider
        config: Configuration for the new provider
    """
    global provider
    try:
        provider = get_provider(provider_name, config)
        logging.info(f"Switched to {provider_name} provider")
        event_manager.emit(Event(EventType.GENERATION_STARTED, {
            'provider': provider_name,
            'model': config.get('model')
        }))
    except Exception as e:
        logging.error(f"Failed to switch provider: {e}")
        raise ConfigurationError(f"Failed to switch to {provider_name}: {e}")
