"""
OPTIMIZATION: Cached text normalization to avoid repeated processing.
"""
import unicodedata
import re
from functools import lru_cache
from typing import Dict


class CachedTextNormalizer:
    """
    Text normalization with LRU caching for performance.
    
    Optimization: Avoids re-normalizing the same text multiple times.
    Cache size: 10,000 entries (sufficient for most documents).
    """
    
    # Character replacement map
    CHAR_MAP = {
        '\u2013': '-', '\u2014': '-',  # En dash, em dash
        '\u201c': '"', '\u201d': '"',  # Smart quotes
        '\u2018': "'", '\u2019': "'",  # Smart apostrophes
        '\u00A0': ' ', '\u200B': '',   # Non-breaking space, zero-width space
    }
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def normalize(text: str) -> str:
        """
        CACHED: Normalize text for consistent caching and comparison.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        
        Note:
            Results are cached for performance. Cache size: 10,000 entries.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        
        # Unicode normalization (NFC = canonical composition)
        text = unicodedata.normalize("NFC", text)
        
        # Character replacements
        for old, new in CachedTextNormalizer.CHAR_MAP.items():
            text = text.replace(old, new)
        
        # Whitespace normalization
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def norm_key(text: str, case_sensitive: bool) -> str:
        """
        CACHED: Generate normalized lookup key.
        
        Args:
            text: Input text
            case_sensitive: Whether to preserve case
        
        Returns:
            Normalized key for lookups
        """
        normalized = CachedTextNormalizer.normalize(text)
        return normalized if case_sensitive else normalized.casefold()
    
    @staticmethod
    def get_cache_info() -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache hits, misses, size
        """
        normalize_info = CachedTextNormalizer.normalize.cache_info()
        norm_key_info = CachedTextNormalizer.norm_key.cache_info()
        
        return {
            'normalize_hits': normalize_info.hits,
            'normalize_misses': normalize_info.misses,
            'normalize_size': normalize_info.currsize,
            'norm_key_hits': norm_key_info.hits,
            'norm_key_misses': norm_key_info.misses,
            'norm_key_size': norm_key_info.currsize,
            'total_hit_rate': (
                (normalize_info.hits + norm_key_info.hits) / 
                (normalize_info.hits + normalize_info.misses + 
                 norm_key_info.hits + norm_key_info.misses) * 100
                if (normalize_info.hits + normalize_info.misses + 
                    norm_key_info.hits + norm_key_info.misses) > 0 
                else 0
            )
        }
    
    @staticmethod
    def clear_cache():
        """Clear normalization cache."""
        CachedTextNormalizer.normalize.cache_clear()
        CachedTextNormalizer.norm_key.cache_clear()


# Convenience functions for backward compatibility
def normalize(text: str) -> str:
    """Normalize text (cached)."""
    return CachedTextNormalizer.normalize(text)


def norm_key(text: str, case_sensitive: bool = False) -> str:
    """Generate normalized key (cached)."""
    return CachedTextNormalizer.norm_key(text, case_sensitive)


if __name__ == "__main__":
    # Test caching
    normalizer = CachedTextNormalizer()
    
    # First calls - cache misses
    text1 = "Hello   World!"
    norm1 = normalizer.normalize(text1)
    print(f"Normalized: '{text1}' -> '{norm1}'")
    
    # Second calls - cache hits
    norm2 = normalizer.normalize(text1)
    assert norm1 == norm2
    
    # Check cache stats
    stats = normalizer.get_cache_info()
    print(f"\nCache stats:")
    print(f"  Hits: {stats['normalize_hits']}")
    print(f"  Misses: {stats['normalize_misses']}")
    print(f"  Hit rate: {stats['total_hit_rate']:.1f}%")
