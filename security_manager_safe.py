"""
Security Manager for GhostWriter AI with graceful fallbacks
Handles API key encryption, path validation, and rate limiting
Works even when cryptography module is not available
"""

import os
import re
import time
import hashlib
import logging
import json
import base64
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Try to import cryptography, fallback to simple obfuscation if not available
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography module not available. Using fallback obfuscation (NOT SECURE FOR PRODUCTION)")


class SecurityError(Exception):
    """Base exception for security-related errors"""
    pass


class PathTraversalError(SecurityError):
    """Exception raised when path traversal is detected"""
    pass


class RateLimitError(SecurityError):
    """Exception raised when rate limit is exceeded"""
    pass


class SimpleObfuscator:
    """Simple obfuscation when cryptography is not available (NOT SECURE)"""
    
    def __init__(self, key: str = "ghostwriter"):
        """Initialize with a simple key"""
        self.key = key
        
    def encrypt(self, data: bytes) -> bytes:
        """Simple XOR obfuscation"""
        key_bytes = self.key.encode()
        result = bytearray()
        
        for i, byte in enumerate(data):
            result.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return base64.b64encode(bytes(result))
    
    def decrypt(self, data: bytes) -> bytes:
        """Reverse the XOR obfuscation"""
        data = base64.b64decode(data)
        key_bytes = self.key.encode()
        result = bytearray()
        
        for i, byte in enumerate(data):
            result.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return bytes(result)


class SecureKeyStorage:
    """
    Secure storage for API keys using encryption when available
    Falls back to obfuscation if cryptography module not installed
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize secure key storage
        
        Args:
            master_key: Optional master key for encryption
        """
        self.storage_path = Path("secure_keys.enc")
        self.cipher = self._initialize_cipher(master_key)
        self._keys_cache: Dict[str, str] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_timestamps: Dict[str, float] = {}
        
        if not CRYPTO_AVAILABLE:
            logger.warning("Using simple obfuscation for API keys. Install 'cryptography' for proper encryption.")
        
    def _initialize_cipher(self, master_key: Optional[str] = None):
        """
        Initialize the cipher with master key
        
        Args:
            master_key: Optional master key
            
        Returns:
            Cipher instance (Fernet or SimpleObfuscator)
        """
        if CRYPTO_AVAILABLE:
            if master_key:
                key = self._derive_key_from_password(master_key)
            else:
                key = self._get_or_create_master_key()
            return Fernet(key)
        else:
            # Fallback to simple obfuscation
            return SimpleObfuscator(master_key or "ghostwriter_default")
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """
        Derive an encryption key from a password
        
        Args:
            password: Password to derive key from
            
        Returns:
            Derived encryption key
        """
        if CRYPTO_AVAILABLE:
            salt = b'ghostwriter_salt_v1'
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key
        else:
            # Simple hash for fallback
            return hashlib.sha256(password.encode()).digest()
    
    def _get_or_create_master_key(self) -> bytes:
        """
        Get existing master key or create a new one
        
        Returns:
            Master encryption key
        """
        key_file = Path("master.key")
        env_key = os.environ.get("GHOSTWRITER_MASTER_KEY")
        
        if CRYPTO_AVAILABLE:
            if env_key:
                return base64.urlsafe_b64decode(env_key.encode())
            elif key_file.exists():
                with open(key_file, 'rb') as f:
                    return f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                try:
                    os.chmod(key_file, 0o600)
                except:
                    pass  # Windows may not support chmod
                logger.warning(
                    "Generated new master key. For production, set GHOSTWRITER_MASTER_KEY "
                    "environment variable and delete master.key file"
                )
                return key
        else:
            # Fallback: use environment or generate simple key
            if env_key:
                return env_key.encode()
            else:
                return b"default_master_key"
    
    def store_api_key(self, provider: str, api_key: str) -> None:
        """
        Store an API key for a provider
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            api_key: The API key to store
        """
        # Load existing keys
        keys = self._load_keys()
        
        # Encrypt and store the new key
        encrypted_key = self.cipher.encrypt(api_key.encode()).decode('utf-8')
        keys[provider] = {
            'encrypted_key': encrypted_key,
            'stored_at': datetime.now().isoformat(),
            'last_accessed': None,
            'encrypted_with': 'Fernet' if CRYPTO_AVAILABLE else 'SimpleObfuscator'
        }
        
        # Save back to file
        self._save_keys(keys)
        
        # Clear cache for this provider
        if provider in self._keys_cache:
            del self._keys_cache[provider]
            del self._cache_timestamps[provider]
        
        logger.info(f"Stored API key for provider: {provider}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Retrieve and decrypt an API key for a provider
        
        Args:
            provider: Provider name
            
        Returns:
            Decrypted API key or None if not found
        """
        # Check cache first
        if provider in self._keys_cache:
            if time.time() - self._cache_timestamps[provider] < self._cache_ttl:
                return self._keys_cache[provider]
            else:
                del self._keys_cache[provider]
                del self._cache_timestamps[provider]
        
        # Load from encrypted storage
        keys = self._load_keys()
        
        if provider not in keys:
            return None
        
        try:
            encrypted_key = keys[provider]['encrypted_key']
            decrypted_key = self.cipher.decrypt(encrypted_key.encode()).decode('utf-8')
            
            # Update last accessed time
            keys[provider]['last_accessed'] = datetime.now().isoformat()
            self._save_keys(keys)
            
            # Cache the decrypted key
            self._keys_cache[provider] = decrypted_key
            self._cache_timestamps[provider] = time.time()
            
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Failed to decrypt API key for {provider}: {e}")
            return None
    
    def remove_api_key(self, provider: str) -> bool:
        """
        Remove a stored API key
        
        Args:
            provider: Provider name
            
        Returns:
            True if removed, False if not found
        """
        keys = self._load_keys()
        
        if provider in keys:
            del keys[provider]
            self._save_keys(keys)
            
            # Clear cache
            if provider in self._keys_cache:
                del self._keys_cache[provider]
                del self._cache_timestamps[provider]
            
            logger.info(f"Removed API key for provider: {provider}")
            return True
        
        return False
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """
        List all providers with stored keys (without exposing keys)
        
        Returns:
            List of provider information
        """
        keys = self._load_keys()
        providers = []
        
        for provider, info in keys.items():
            providers.append({
                'provider': provider,
                'stored_at': info.get('stored_at'),
                'last_accessed': info.get('last_accessed'),
                'encryption_method': info.get('encrypted_with', 'Unknown')
            })
        
        return providers
    
    def _load_keys(self) -> Dict[str, Any]:
        """Load encrypted keys from storage"""
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load encrypted keys: {e}")
            return {}
    
    def _save_keys(self, keys: Dict[str, Any]) -> None:
        """Save encrypted keys to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(keys, f, indent=2)
            try:
                os.chmod(self.storage_path, 0o600)
            except:
                pass  # Windows compatibility
        except Exception as e:
            logger.error(f"Failed to save encrypted keys: {e}")
            raise SecurityError(f"Failed to save encrypted keys: {e}")


class PathValidator:
    """
    Validates paths to prevent path traversal attacks
    """
    
    # Patterns that indicate potential path traversal
    DANGEROUS_PATTERNS = [
        r'\.\.',  # Parent directory reference
        r'^/',     # Absolute path (Unix)
        r'^[A-Za-z]:',  # Absolute path (Windows)
        r'^~',     # Home directory reference
        r'\x00',   # Null bytes
        r'[<>"|?*]',  # Invalid filename characters
    ]
    
    # Maximum allowed path depth
    MAX_PATH_DEPTH = 3
    
    @classmethod
    def validate_project_id(cls, project_id: str) -> bool:
        """
        Validate a project ID to ensure it's safe
        
        Args:
            project_id: Project ID to validate
            
        Returns:
            True if valid
            
        Raises:
            PathTraversalError: If validation fails
        """
        if not project_id:
            raise PathTraversalError("Project ID cannot be empty")
        
        # Check length limits
        if len(project_id) > 100:
            raise PathTraversalError("Project ID too long")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, project_id):
                raise PathTraversalError(
                    f"Invalid project ID: contains dangerous pattern"
                )
        
        # Check for directory separators
        if '/' in project_id or '\\' in project_id:
            raise PathTraversalError("Project ID cannot contain path separators")
        
        # Ensure it's alphanumeric with limited special characters
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', project_id):
            raise PathTraversalError(
                "Project ID can only contain letters, numbers, underscore, dash, and dot"
            )
        
        return True
    
    @classmethod
    def validate_file_path(cls, file_path: str, base_dir: Optional[str] = None) -> bool:
        """
        Validate a file path to ensure it's within allowed boundaries
        
        Args:
            file_path: File path to validate
            base_dir: Optional base directory to restrict to
            
        Returns:
            True if valid
            
        Raises:
            PathTraversalError: If validation fails
        """
        try:
            path = Path(file_path)
            
            # Resolve to absolute path
            absolute_path = path.resolve()
            
            # If base_dir is provided, ensure path is within it
            if base_dir:
                base = Path(base_dir).resolve()
                try:
                    absolute_path.relative_to(base)
                except ValueError:
                    raise PathTraversalError(
                        f"Path '{file_path}' is outside allowed base directory"
                    )
            
            # Check if path exists and is a file (not a directory)
            if absolute_path.exists() and absolute_path.is_dir():
                raise PathTraversalError(f"Path '{file_path}' is a directory, not a file")
            
            return True
            
        except PathTraversalError:
            raise
        except Exception as e:
            raise PathTraversalError(f"Invalid file path: {e}")
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize a filename to make it safe
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename


class RateLimiter:
    """
    Rate limiter for API endpoints and resource-intensive operations
    Uses token bucket algorithm for flexible rate limiting
    """
    
    def __init__(self):
        """Initialize rate limiter with default configurations"""
        self.limiters: Dict[str, 'TokenBucket'] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Default rate limits (can be overridden)
        self.default_limits = {
            'generation': (10, 60),  # 10 requests per 60 seconds
            'export': (20, 60),      # 20 exports per 60 seconds
            'api_call': (100, 60),   # 100 API calls per 60 seconds
            'character_creation': (50, 60),  # 50 character creations per 60 seconds
        }
    
    def check_rate_limit(
        self,
        resource: str,
        identifier: str,
        max_requests: Optional[int] = None,
        time_window: Optional[int] = None
    ) -> bool:
        """
        Check if a request is within rate limits
        
        Args:
            resource: Resource type being accessed
            identifier: Unique identifier (e.g., session ID, IP address)
            max_requests: Maximum requests allowed (uses default if None)
            time_window: Time window in seconds (uses default if None)
            
        Returns:
            True if within limits
            
        Raises:
            RateLimitError: If rate limit exceeded
        """
        # Use default limits if not specified
        if max_requests is None or time_window is None:
            if resource in self.default_limits:
                default_max, default_window = self.default_limits[resource]
                max_requests = max_requests or default_max
                time_window = time_window or default_window
            else:
                max_requests = max_requests or 60
                time_window = time_window or 60
        
        # Create limiter key
        limiter_key = f"{resource}:{identifier}"
        
        # Get or create token bucket for this limiter
        if limiter_key not in self.limiters:
            self.limiters[limiter_key] = TokenBucket(max_requests, time_window)
        
        bucket = self.limiters[limiter_key]
        
        # Try to consume a token
        if bucket.consume():
            # Record successful request
            self.request_history[limiter_key].append(time.time())
            return True
        else:
            # Calculate wait time
            wait_time = bucket.time_until_token()
            raise RateLimitError(
                f"Rate limit exceeded for {resource}. "
                f"Maximum {max_requests} requests per {time_window} seconds. "
                f"Try again in {wait_time:.1f} seconds."
            )
    
    def get_remaining_requests(self, resource: str, identifier: str) -> int:
        """
        Get the number of remaining requests for a resource
        
        Args:
            resource: Resource type
            identifier: Unique identifier
            
        Returns:
            Number of remaining requests
        """
        limiter_key = f"{resource}:{identifier}"
        
        if limiter_key in self.limiters:
            return int(self.limiters[limiter_key].tokens)
        
        # If no limiter exists, return default max
        if resource in self.default_limits:
            return self.default_limits[resource][0]
        
        return 60
    
    def reset_limit(self, resource: str, identifier: str) -> None:
        """
        Reset rate limit for a specific resource/identifier
        
        Args:
            resource: Resource type
            identifier: Unique identifier
        """
        limiter_key = f"{resource}:{identifier}"
        
        if limiter_key in self.limiters:
            del self.limiters[limiter_key]
        
        if limiter_key in self.request_history:
            del self.request_history[limiter_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'active_limiters': len(self.limiters),
            'limiters': {},
            'total_requests': 0
        }
        
        for key, bucket in self.limiters.items():
            resource, identifier = key.split(':', 1)
            stats['limiters'][key] = {
                'resource': resource,
                'identifier': identifier,
                'tokens_remaining': int(bucket.tokens),
                'max_tokens': bucket.capacity,
                'refill_rate': bucket.refill_rate,
                'last_refill': bucket.last_refill
            }
        
        for history in self.request_history.values():
            stats['total_requests'] += len(history)
        
        return stats


class TokenBucket:
    """
    Token bucket implementation for rate limiting
    """
    
    def __init__(self, capacity: int, time_window: int):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            time_window: Time window in seconds for refill
        """
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = capacity / time_window
        self.time_window = time_window
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        current_time = time.time()
        elapsed = current_time - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        
        # Add tokens up to capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time
    
    def time_until_token(self) -> float:
        """
        Calculate time until next token is available
        
        Returns:
            Time in seconds until a token is available
        """
        self._refill()
        
        if self.tokens >= 1:
            return 0.0
        
        tokens_needed = 1 - self.tokens
        time_needed = tokens_needed / self.refill_rate
        
        return time_needed


# Global instances
_secure_storage: Optional[SecureKeyStorage] = None
_rate_limiter: Optional[RateLimiter] = None


def get_secure_storage() -> SecureKeyStorage:
    """Get or create the global secure storage instance"""
    global _secure_storage
    if _secure_storage is None:
        _secure_storage = SecureKeyStorage()
    return _secure_storage


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# Decorator for rate limiting
def rate_limit(resource: str, max_requests: Optional[int] = None, time_window: Optional[int] = None):
    """
    Decorator for applying rate limiting to functions
    
    Args:
        resource: Resource type for rate limiting
        max_requests: Maximum requests allowed
        time_window: Time window in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to extract identifier from function arguments
            identifier = kwargs.get('session_id', kwargs.get('identifier', 'default'))
            
            limiter = get_rate_limiter()
            limiter.check_rate_limit(resource, identifier, max_requests, time_window)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator