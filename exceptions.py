"""
Custom exception classes for Ghostwriter AI
"""

class GhostwriterException(Exception):
    """Base exception class for Ghostwriter AI"""
    pass

class ConfigurationError(GhostwriterException):
    """Raised when there's a configuration issue"""
    pass

class ProviderError(GhostwriterException):
    """Base class for all provider-related errors."""
    pass

class ProviderAuthError(ProviderError):
    """Raised for authentication failures (e.g., invalid API key)."""
    pass

class ProviderRateLimitError(ProviderError):
    """Raised when an API rate limit is exceeded."""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

class ProviderContentFilterError(ProviderError):
    """Raised when a request is blocked by a content filter."""
    pass

class TokenLimitError(ProviderError):
    """Raised when token limit is exceeded"""
    def __init__(self, message, tokens_used=None, token_limit=None):
        super().__init__(message)
        self.tokens_used = tokens_used
        self.token_limit = token_limit

class ContentGenerationError(GhostwriterException):
    """Raised when content generation fails"""
    pass

class FileOperationError(GhostwriterException):
    """Raised when file operations fail"""
    pass

class ValidationError(GhostwriterException):
    """Raised when input validation fails"""
    pass