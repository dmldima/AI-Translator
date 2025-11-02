"""
Production-ready cache manager with thread-safety and batch operations.
COMPLETE FILE with all methods.
"""
import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging


logger = logging.getLogger(__name__)


# ===== Configuration =====

@dataclass
class CacheConfig:
    """Cache configuration."""
    max_age_days: int = 180
    max_size_mb: int = 500
    log_level: str = "INFO"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    source: str
    target: str
    source_lang: str
    target_lang: str
    model: str
    glossary_version: str
    domain: str
    confidence: float
    timestamp: str


# ===== Storage Layer =====

class SQLiteStorage:
    """Thread-safe SQLite storage for cache."""
    
    def __init__(self, db_path: Path, logger: logging.Logger):
        self.db_path = Path(db_path)
        self.logger = logger
        self._local = threading.local()
        
        # Create database
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            self._init_schema(conn)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                isolation_level=None,
                timeout=30.0
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.row_factory = sqlite3.Row
        
        return self._local.conn
    
    @contextmanager
    def _transaction(self):
        """Explicit transaction context."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    
    def _init_schema(self, conn: sqlite3.Connection):
        """Initialize database schema."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                model TEXT NOT NULL,
                glossary_version TEXT NOT NULL,
                domain TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Create indices
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_glossary ON cache(glossary_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON cache(domain)")
        
        self.logger.info(f"Cache schema initialized: {self.db_path}")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        try:
            conn = self._get_connection()
            cur = conn.execute("SELECT * FROM cache WHERE key = ?", (key,))
            row = cur.fetchone()
            
            if row:
                return CacheEntry(**dict(row))
            return None
            
        except Exception as e:
            self.logger.error(f"Cache read error: {e}")
            return None
    
    def set(self, entry: CacheEntry):
        """Set entry."""
        try:
            with self._transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache
                    (key, source, target, source_lang, target_lang, model,
                     glossary_version, domain, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.source,
                    entry.target,
                    entry.source_lang,
                    entry.target_lang,
                    entry.model,
                    entry.glossary_version,
                    entry.domain,
                    entry.confidence,
                    entry.timestamp
                ))
        except Exception as e:
            self.logger.error(f"Cache write error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry by key."""
        try:
            with self._transaction() as conn:
                cur = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return cur.rowcount > 0
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            conn = self._get_connection()
            
            # Total count
            cur = conn.execute("SELECT COUNT(*) FROM cache")
            total = cur.fetchone()[0]
            
            # Database size
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                'total_entries': total,
                'size_mb': round(size_mb, 2),
                'db_path': str(self.db_path)
            }
        except Exception as e:
            self.logger.error(f"Stats error: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ===== Cache Manager =====

class CacheManager:
    """
    High-level cache manager with batch operations.
    
    Features:
    - Thread-safe operations
    - Batch lookups (optimization)
    - Automatic cleanup
    - Statistics tracking
    """
    
    def __init__(
        self,
        storage: SQLiteStorage,
        config: CacheConfig,
        logger: logging.Logger
    ):
        self.storage = storage
        self.config = config
        self.logger = logger
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._stats_lock = threading.Lock()
    
    def generate_key(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        glossary_version: str,
        domain: str
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            source_text: Source text
            source_lang: Source language
            target_lang: Target language
            glossary_version: Glossary version
            domain: Domain
            
        Returns:
            SHA256 hash key
        """
        key_string = f"{source_lang}:{target_lang}:{glossary_version}:{domain}:{source_text}"
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def get(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        glossary_version: str,
        domain: str
    ) -> Optional[CacheEntry]:
        """
        Get cached translation.
        
        Args:
            source_text: Source text
            source_lang: Source language
            target_lang: Target language
            glossary_version: Glossary version
            domain: Domain
            
        Returns:
            CacheEntry if found and valid, None otherwise
        """
        key = self.generate_key(source_text, source_lang, target_lang, glossary_version, domain)
        
        entry = self.storage.get(key)
        
        if entry is None:
            self._increment_stat('misses')
            return None
        
        # Check if stale
        if self._is_stale(entry.timestamp):
            self.logger.debug(f"Cache entry stale: {key}")
            self.storage.delete(key)
            self._increment_stat('misses')
            return None
        
        # Validate glossary version
        if entry.glossary_version != glossary_version:
            self.logger.debug(f"Glossary version mismatch: {key}")
            self._increment_stat('misses')
            return None
        
        self._increment_stat('hits')
        self.logger.debug(f"Cache hit: {key}")
        return entry
    
    def get_batch(self, lookup_data: List[Dict[str, str]]) -> List[Optional[CacheEntry]]:
        """
        OPTIMIZATION: Batch cache lookup with single SQL query.
        
        Args:
            lookup_data: List of dicts with keys: source_text, source_lang, 
                        target_lang, glossary_version, domain
        
        Returns:
            List of CacheEntry or None (same order as input)
        """
        if not lookup_data:
            return []
        
        try:
            # Generate keys for all requests
            keys = []
            for data in lookup_data:
                key = self.generate_key(
                    data['source_text'],
                    data['source_lang'],
                    data['target_lang'],
                    data.get('glossary_version', ''),
                    data.get('domain', '')
                )
                keys.append(key)
            
            # Batch SQL query
            with self.storage._get_connection() as conn:
                placeholders = ','.join('?' * len(keys))
                query = f"SELECT * FROM cache WHERE key IN ({placeholders})"
                cur = conn.execute(query, keys)
                
                # Build results map
                results_map = {}
                for row in cur.fetchall():
                    entry = CacheEntry(**dict(row))
                    results_map[row['key']] = entry
            
            # Return in original order
            results = []
            for key, data in zip(keys, lookup_data):
                entry = results_map.get(key)
                
                # Validate glossary version and staleness
                if entry:
                    if entry.glossary_version != data.get('glossary_version', ''):
                        entry = None
                    elif self._is_stale(entry.timestamp):
                        entry = None
                
                results.append(entry)
            
            # Update stats
            hits = sum(1 for r in results if r is not None)
            misses = len(results) - hits
            
            with self._stats_lock:
                self._hits += hits
                self._misses += misses
            
            self.logger.info(
                f"Batch cache lookup: {hits}/{len(lookup_data)} hits "
                f"({hits/len(lookup_data)*100:.1f}% hit rate)"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch cache lookup error: {e}")
            # Fallback to individual lookups
            return [
                self.get(
                    data['source_text'],
                    data['source_lang'],
                    data['target_lang'],
                    data.get('glossary_version', ''),
                    data.get('domain', '')
                )
                for data in lookup_data
            ]
    
    def set(self, entry: CacheEntry):
        """
        Set cache entry.
        
        Args:
            entry: CacheEntry to store
        """
        self.storage.set(entry)
        self.logger.debug(f"Cache set: {entry.key}")
    
    def evict_stale(self) -> int:
        """
        Remove stale cache entries.
        
        Returns:
            Number of entries removed
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.max_age_days)
            cutoff_str = cutoff_date.isoformat()
            
            with self.storage._transaction() as conn:
                cur = conn.execute(
                    "DELETE FROM cache WHERE timestamp < ?",
                    (cutoff_str,)
                )
                count = cur.rowcount
            
            if count > 0:
                self.logger.info(f"Evicted {count} stale entries")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Eviction error: {e}")
            return 0
    
    def evict_glossary(self, glossary_version: str) -> int:
        """
        Remove entries for specific glossary version.
        
        Args:
            glossary_version: Glossary version to evict
            
        Returns:
            Number of entries removed
        """
        try:
            with self.storage._transaction() as conn:
                cur = conn.execute(
                    "DELETE FROM cache WHERE glossary_version = ?",
                    (glossary_version,)
                )
                count = cur.rowcount
            
            if count > 0:
                self.logger.info(f"Evicted {count} entries for glossary {glossary_version}")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Glossary eviction error: {e}")
            return 0
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries removed
        """
        try:
            with self.storage._transaction() as conn:
                cur = conn.execute("DELETE FROM cache")
                count = cur.rowcount
            
            self.logger.info(f"Cleared {count} cache entries")
            return count
            
        except Exception as e:
            self.logger.error(f"Clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._stats_lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 2),
                'total_requests': total_requests
            }
        
        # Add storage stats
        storage_stats = self.storage.get_stats()
        stats.update(storage_stats)
        
        return stats
    
    def _is_stale(self, timestamp: str) -> bool:
        """Check if entry is stale."""
        try:
            entry_time = datetime.fromisoformat(timestamp)
            age = datetime.now(timezone.utc) - entry_time
            return age.days > self.config.max_age_days
        except Exception as e:
            self.logger.error(f"Timestamp parse error: {e}")
            return True
    
    def _increment_stat(self, stat: str):
        """Thread-safe stat increment."""
        with self._stats_lock:
            if stat == 'hits':
                self._hits += 1
            elif stat == 'misses':
                self._misses += 1
    
    def close(self):
        """Close cache manager."""
        self.storage.close()


# ===== Example Usage =====

if __name__ == "__main__":
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("cache_test")
    
    # Create cache manager
    config = CacheConfig(max_age_days=180)
    storage = SQLiteStorage(Path("test_cache.db"), logger)
    cache_mgr = CacheManager(storage, config, logger)
    
    # Create entry
    entry = CacheEntry(
        key=cache_mgr.generate_key("Hello", "en", "ru", "v1", "general"),
        source="Hello",
        target="Привет",
        source_lang="en",
        target_lang="ru",
        model="gpt-4o-mini",
        glossary_version="v1",
        domain="general",
        confidence=1.0,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    # Set
    cache_mgr.set(entry)
    print("✓ Entry stored")
    
    # Get
    retrieved = cache_mgr.get("Hello", "en", "ru", "v1", "general")
    if retrieved:
        print(f"✓ Retrieved: {retrieved.source} -> {retrieved.target}")
    
    # Batch get
    lookup_data = [
        {'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru', 
         'glossary_version': 'v1', 'domain': 'general'},
        {'source_text': 'World', 'source_lang': 'en', 'target_lang': 'ru',
         'glossary_version': 'v1', 'domain': 'general'}
    ]
    
    results = cache_mgr.get_batch(lookup_data)
    print(f"✓ Batch results: {sum(1 for r in results if r)}/{len(results)} found")
    
    # Stats
    stats = cache_mgr.get_stats()
    print(f"\n✓ Cache stats: {stats}")
    
    # Cleanup
    cache_mgr.close()
