"""
Test file to verify Gemini provider fixes
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gemini_provider():
    """Test that Gemini provider has proper retry and error handling"""
    from providers.gemini_provider import GeminiProvider
    from providers.base import LLMProvider
    
    # Test 1: Verify _call_with_retry exists in base class
    assert hasattr(LLMProvider, '_call_with_retry'), "LLMProvider missing _call_with_retry method"
    print("[OK] LLMProvider has _call_with_retry method")
    
    # Test 2: Verify GeminiProvider error handling
    provider = GeminiProvider.__new__(GeminiProvider)
    error = provider._handle_error(Exception("Test error"))
    assert error is not None, "Error handler returned None"
    assert isinstance(error, Exception), "Error handler should return an Exception"
    print("[OK] GeminiProvider._handle_error returns exceptions properly")
    
    return True

def test_tokenizer():
    """Test that tokenizer accepts API key parameter"""
    from tokenizer import GeminiTokenizer
    
    # Test that GeminiTokenizer accepts api_key parameter
    try:
        tokenizer = GeminiTokenizer(model_name="gemini-pro", api_key="test_key")
        print("[OK] GeminiTokenizer accepts api_key parameter")
    except TypeError as e:
        if "api_key" in str(e):
            print("[FAIL] GeminiTokenizer doesn't accept api_key parameter")
            return False
    
    # Test token counting with no text
    count = tokenizer.count_tokens("")
    assert count == 0, "Empty text should return 0 tokens"
    print("[OK] Token counting handles empty text")
    
    # Test token counting with text (fallback)
    count = tokenizer.count_tokens("Test text")
    assert count > 0, "Text should return positive token count"
    print("[OK] Token counting fallback works")
    
    return True

def test_file_encoding():
    """Test that file operations use UTF-8 encoding"""
    import inspect
    import cache_manager
    import bookprinter
    
    # Check cache_manager
    source = inspect.getsource(cache_manager.FileCache._load_index)
    assert "encoding='utf-8'" in source, "cache_manager missing UTF-8 encoding"
    print("[OK] cache_manager uses UTF-8 encoding")
    
    # Check bookprinter
    source = inspect.getsource(bookprinter.create_book_frontpage)
    assert "encoding='utf-8'" in source, "bookprinter missing UTF-8 encoding"
    print("[OK] bookprinter uses UTF-8 encoding")
    
    return True

def test_cache_expire():
    """Test that MemoryCache accepts expire parameter"""
    from cache_manager import MemoryCache
    
    # Test that MemoryCache accepts expire parameter
    try:
        cache = MemoryCache(max_size=100, expire=3600)
        print("[OK] MemoryCache accepts expire parameter")
    except TypeError as e:
        if "expire" in str(e):
            print("[FAIL] MemoryCache doesn't accept expire parameter")
            return False
    
    # Test set with expire
    cache.set("test_key", "test_value", expire=60)
    value = cache.get("test_key")
    assert value == "test_value", "Cache should return stored value"
    print("[OK] Cache set/get with expire works")
    
    return True

def main():
    print("=" * 60)
    print("TESTING GEMINI PROVIDER FIXES")
    print("=" * 60)
    
    results = []
    
    print("\n1. Testing Gemini Provider...")
    results.append(test_gemini_provider())
    
    print("\n2. Testing Tokenizer...")
    results.append(test_tokenizer())
    
    print("\n3. Testing File Encoding...")
    results.append(test_file_encoding())
    
    print("\n4. Testing Cache Expire...")
    results.append(test_cache_expire())
    
    print("\n" + "=" * 60)
    if all(results):
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[FAILED] SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()