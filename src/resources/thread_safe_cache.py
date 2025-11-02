"""
Thread-safe data structures to prevent race conditions.
FIXED: Race condition in OnceExecutor
"""
import threading
import time
from typing import Optional, Dict, Any, Callable, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheItem(Generic[T]):
    """Cache item with metadata."""
    value: T
    created_at: float
    access_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at


class ThreadSafeLRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.
    Prevents race conditions in cache operations.
    """
    
    def __init__(
        self, 
        maxsize: int = 1000, 
        ttl_seconds: Optional[float] = None
    ):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheItem[T]] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.debug(f"Initialized LRU cache: maxsize={maxsize}, ttl={ttl_seconds}")
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            item = self._cache.get(key)
            
            if item is None:
                self._misses += 1
                return None
            
            # Check expiration
            if self._is_expired(item):
                del self._cache[key]
                self._evictions += 1
                self._misses += 1
                return None
            
            # Update access metadata
            item.access_count += 1
            item.last_accessed = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return item.value
    
    def set(self, key: str, value: T) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            now = time.time()
            
            # Update existing item
            if key in self._cache:
                item = self._cache[key]
                item.value = value
                item.created_at = now
                item.last_accessed = now
                self._cache.move_to_end(key)
                return
            
            # Add new item
            item = CacheItem(
                value=value,
                created_at=now,
                last_accessed=now
            )
            
            self._cache[key] = item
            
            # Evict LRU item if over limit
            if len(self._cache) > self.maxsize:
                self._evict_lru()
    
    def setdefault(self, key: str, default: T) -> T:
        """
        Get value or set default if not exists.
        Atomic operation - prevents race condition.
        
        Args:
            key: Cache key
            default: Default value
            
        Returns:
            Existing or default value
        """
        with self._lock:
            value = self.get(key)
            if value is None:
                self.set(key, default)
                return default
            return value
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item existed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all items."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug(f"Cleared {count} items from cache")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired items.
        
        Returns:
            Number of items removed
        """
        if self.ttl_seconds is None:
            return 0
        
        with self._lock:
            expired_keys = [
                key for key, item in self._cache.items()
                if self._is_expired(item)
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired items")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'maxsize': self.maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 2),
                'evictions': self._evictions,
                'ttl_seconds': self.ttl_seconds
            }
    
    def _is_expired(self, item: CacheItem) -> bool:
        """Check if item is expired."""
        if self.ttl_seconds is None:
            return False
        
        age = time.time() - item.created_at
        return age > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # OrderedDict maintains insertion order
        # Items are moved to end on access
        # So first item is LRU
        lru_key = next(iter(self._cache))
        del self._cache[lru_key]
        self._evictions += 1
        
        logger.debug(f"Evicted LRU item: {lru_key}")
    
    def __len__(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists (not expired)."""
        return self.get(key) is not None


class ThreadSafeCounter:
    """Thread-safe counter with atomic operations."""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, delta: int = 1) -> int:
        """
        Increment counter atomically.
        
        Args:
            delta: Amount to increment
            
        Returns:
            New value
        """
        with self._lock:
            self._value += delta
            return self._value
    
    def decrement(self, delta: int = 1) -> int:
        """
        Decrement counter atomically.
        
        Args:
            delta: Amount to decrement
            
        Returns:
            New value
        """
        with self._lock:
            self._value -= delta
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def set(self, value: int):
        """Set value."""
        with self._lock:
            self._value = value
    
    def reset(self):
        """Reset to zero."""
        self.set(0)
    
    def __int__(self) -> int:
        return self.get()
    
    def __str__(self) -> str:
        return str(self.get())


class ThreadSafeDict(Generic[T]):
    """Thread-safe dictionary wrapper."""
    
    def __init__(self):
        self._dict: Dict[str, T] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get value."""
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key: str, value: T):
        """Set value."""
        with self._lock:
            self._dict[key] = value
    
    def setdefault(self, key: str, default: T) -> T:
        """Atomic setdefault."""
        with self._lock:
            return self._dict.setdefault(key, default)
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        with self._lock:
            if key in self._dict:
                del self._dict[key]
                return True
            return False
    
    def clear(self):
        """Clear all items."""
        with self._lock:
            self._dict.clear()
    
    def keys(self):
        """Get keys (snapshot)."""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self):
        """Get values (snapshot)."""
        with self._lock:
            return list(self._dict.values())
    
    def items(self):
        """Get items (snapshot)."""
        with self._lock:
            return list(self._dict.items())
    
    def update(self, other: Dict[str, T]):
        """Update from dict."""
        with self._lock:
            self._dict.update(other)
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._dict)
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._dict
    
    def __getitem__(self, key: str) -> T:
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key: str, value: T):
        with self._lock:
            self._dict[key] = value


class OnceExecutor:
    """
    Ensure a function is executed only once, even with concurrent access.
    Similar to sync.Once in Go.
    
    FIXED: Race condition in double-checked locking pattern.
    """
    
    def __init__(self):
        self._executed = False
        self._lock = threading.Lock()
        self._result = None
        self._exception = None
    
    def execute(self, func: Callable, *args, **kwargs):
        """
        Execute function once with correct double-checked locking.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception if function raised exception
        
        FIXED: Proper memory ordering with lock
        """
        # Fast path: Check if already executed
        # IMPORTANT: This is safe only for reading the flag
        if self._executed:
            # MUST acquire lock to ensure memory visibility of result/exception
            # This prevents seeing _executed=True but stale _result/_exception
            with self._lock:
                # Double-check after lock (defensive)
                if self._exception:
                    raise self._exception
                return self._result
        
        # Slow path: Need to execute or wait for execution
        with self._lock:
            # Double-check after acquiring lock
            # Another thread might have executed between our check and lock acquisition
            if self._executed:
                if self._exception:
                    raise self._exception
                return self._result
            
            # Execute the function
            try:
                self._result = func(*args, **kwargs)
                # CRITICAL: Set _executed LAST to ensure proper memory ordering
                # All writes above must be visible before _executed becomes True
                # This is guaranteed by the lock release
                self._executed = True
                return self._result
            except Exception as e:
                self._exception = e
                self._executed = True
                raise


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum wait time (None = wait forever)
            
        Returns:
            True if acquired, False if timeout
        """
        deadline = time.time() + timeout if timeout else None
        
        while True:
            with self._lock:
                self._refill()
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            
            # Check timeout
            if deadline and time.time() >= deadline:
                return False
            
            # Wait a bit before retry
            time.sleep(0.01)
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        
        # Add tokens based on rate
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_update = now
