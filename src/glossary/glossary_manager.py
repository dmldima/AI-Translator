"""
Production-ready glossary manager with thread-safety and performance optimizations.

Features:
- Thread-safe SQLite operations
- Precompiled regex patterns for performance
- Proper UNIQUE constraints
- Batch operations support
- Comprehensive validation
- Type hints throughout
"""
import os
import re
import json
import unicodedata
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler


# ===== Configuration =====
class TermStatus(Enum):
    """Term approval status."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


@dataclass
class GlossaryConfig:
    """Glossary configuration."""
    log_level: str = "INFO"
    log_path: Path = Path("glossary.log")
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5


# ===== Logging =====
def setup_logger(name: str, config: GlossaryConfig) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    handler = RotatingFileHandler(
        config.log_path,
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s [%(funcName)s:%(lineno)d] - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    
    return logger


# ===== Text Utilities =====
class TextUtil:
    """Text normalization and pattern utilities."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for consistent matching."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)
        
        # Character replacements
        replacements = {
            '\u2013': '-', '\u2014': '-',
            '\u00A0': ' ', '\u200B': ''
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Whitespace normalization
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    @staticmethod
    def norm_key(text: str, case_sensitive: bool) -> str:
        """Generate normalized lookup key."""
        normalized = TextUtil.normalize(text)
        return normalized if case_sensitive else normalized.casefold()
    
    @staticmethod
    def word_boundary_pattern(term: str, case_sensitive: bool) -> re.Pattern:
        """Create compiled regex with word boundaries."""
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = rf"(?<!\w)({re.escape(term)})(?!\w)"
        return re.compile(pattern, flags)


# ===== Data Models =====
@dataclass
class GlossaryTerm:
    """Validated glossary term."""
    source: str
    target: str
    domain: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    definition: str = ""
    usage_examples: List[str] = field(default_factory=list)
    status: TermStatus = TermStatus.DRAFT
    aliases: List[str] = field(default_factory=list)
    priority: int = 0
    case_sensitive: bool = False
    version: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize fields."""
        if not self.source or not self.target:
            raise ValueError("source and target are required")
        
        self.source = TextUtil.normalize(self.source)
        self.target = TextUtil.normalize(self.target)
        
        if isinstance(self.status, str):
            self.status = TermStatus(self.status)
        
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    @property
    def source_norm(self) -> str:
        """Normalized source for lookups."""
        return TextUtil.norm_key(self.source, self.case_sensitive)


@dataclass
class ReplacementResult:
    """Result of text replacement operation."""
    text: str
    replacements: int
    terms_applied: List[str]
    details: List[Dict[str, Any]] = field(default_factory=list)


# ===== Storage Interface =====
class GlossaryStorage(ABC):
    """Abstract glossary storage."""
    
    @abstractmethod
    def upsert(self, term: GlossaryTerm) -> int:
        """Insert or update term. Returns term ID."""
        pass
    
    @abstractmethod
    def get(self, term_id: int) -> Optional[GlossaryTerm]:
        """Get term by ID."""
        pass
    
    @abstractmethod
    def find_best(
        self,
        source: str,
        domain: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Optional[GlossaryTerm]:
        """Find best matching term."""
        pass
    
    @abstractmethod
    def list_terms(
        self,
        domain: Optional[str] = None,
        status: Optional[TermStatus] = None
    ) -> Iterator[GlossaryTerm]:
        """List terms with filters."""
        pass
    
    @abstractmethod
    def delete(self, term_id: int) -> bool:
        """Delete term by ID."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        pass


# ===== SQLite Storage =====
class SQLiteGlossary(GlossaryStorage):
    """Thread-safe SQLite glossary storage."""
    
    def __init__(self, db_path: Path, logger: logging.Logger):
        self.db_path = Path(db_path)
        self.logger = logger
        self._local = threading.local()
        self._stats = {'reads': 0, 'writes': 0, 'errors': 0}
        self._stats_lock = threading.Lock()
        
        # Initialize schema
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
    
    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema with proper constraints."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_norm TEXT NOT NULL,
                target TEXT NOT NULL,
                domain TEXT,
                source_lang TEXT,
                target_lang TEXT,
                definition TEXT DEFAULT '',
                usage_examples TEXT DEFAULT '[]',
                status TEXT DEFAULT 'draft',
                aliases TEXT DEFAULT '[]',
                priority INTEGER DEFAULT 0,
                case_sensitive INTEGER DEFAULT 0,
                version TEXT,
                author TEXT,
                timestamp TEXT NOT NULL,
                UNIQUE(source_norm, domain, source_lang, target_lang)
            )
        """)
        
        # Create indices for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_norm ON glossary(source_norm)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON glossary(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON glossary(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON glossary(priority DESC)")
        
        self.logger.info(f"Schema initialized: {self.db_path}")
    
    def _increment_stat(self, stat: str) -> None:
        """Thread-safe stat increment."""
        with self._stats_lock:
            self._stats[stat] = self._stats.get(stat, 0) + 1
    
    def _row_to_term(self, row: sqlite3.Row) -> GlossaryTerm:
        """Convert database row to GlossaryTerm."""
        return GlossaryTerm(
            source=row['source'],
            target=row['target'],
            domain=row['domain'],
            source_lang=row['source_lang'],
            target_lang=row['target_lang'],
            definition=row['definition'] or "",
            usage_examples=json.loads(row['usage_examples'] or '[]'),
            status=TermStatus(row['status']),
            aliases=json.loads(row['aliases'] or '[]'),
            priority=row['priority'],
            case_sensitive=bool(row['case_sensitive']),
            version=row['version'],
            author=row['author'],
            timestamp=row['timestamp']
        )
    
    def upsert(self, term: GlossaryTerm) -> int:
        """Insert or update term."""
        try:
            with self._transaction() as conn:
                cur = conn.execute("""
                    INSERT INTO glossary (
                        source, source_norm, target, domain, source_lang, target_lang,
                        definition, usage_examples, status, aliases, priority,
                        case_sensitive, version, author, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_norm, domain, source_lang, target_lang) 
                    DO UPDATE SET
                        source=excluded.source,
                        target=excluded.target,
                        definition=excluded.definition,
                        usage_examples=excluded.usage_examples,
                        status=excluded.status,
                        aliases=excluded.aliases,
                        priority=excluded.priority,
                        case_sensitive=excluded.case_sensitive,
                        version=excluded.version,
                        author=excluded.author,
                        timestamp=excluded.timestamp
                    RETURNING id
                """, (
                    term.source,
                    term.source_norm,
                    term.target,
                    term.domain,
                    term.source_lang,
                    term.target_lang,
                    term.definition,
                    json.dumps(term.usage_examples, ensure_ascii=False),
                    term.status.value,
                    json.dumps(term.aliases, ensure_ascii=False),
                    term.priority,
                    int(term.case_sensitive),
                    term.version,
                    term.author,
                    term.timestamp
                ))
                
                term_id = cur.fetchone()[0]
                self._increment_stat('writes')
                
                self.logger.info(
                    f"Upserted term ID={term_id}: {term.source} -> {term.target}"
                )
                
                return term_id
                
        except Exception as e:
            self._increment_stat('errors')
            self.logger.error(f"Upsert error: {e}")
            raise
    
    def get(self, term_id: int) -> Optional[GlossaryTerm]:
        """Get term by ID."""
        try:
            conn = self._get_connection()
            cur = conn.execute("SELECT * FROM glossary WHERE id = ?", (term_id,))
            row = cur.fetchone()
            
            if row:
                self._increment_stat('reads')
                return self._row_to_term(row)
            
            return None
            
        except Exception as e:
            self._increment_stat('errors')
            self.logger.error(f"Get error for ID {term_id}: {e}")
            return None
    
    def find_best(
        self,
        source: str,
        domain: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Optional[GlossaryTerm]:
        """Find best matching term with priority."""
        try:
            source_norm = TextUtil.norm_key(source, False)
            conn = self._get_connection()
            
            # Build query with proper NULL handling
            query = """
                SELECT * FROM glossary 
                WHERE source_norm = ?
                  AND (domain IS NULL OR domain = ?)
                  AND (source_lang IS NULL OR source_lang = ?)
                  AND (target_lang IS NULL OR target_lang = ?)
                  AND status != 'deprecated'
                ORDER BY 
                    priority DESC,
                    (domain = ?) DESC,
                    (source_lang = ?) DESC,
                    (target_lang = ?) DESC
                LIMIT 1
            """
            
            cur = conn.execute(
                query,
                (source_norm, domain, source_lang, target_lang, domain, source_lang, target_lang)
            )
            row = cur.fetchone()
            
            if row:
                self._increment_stat('reads')
                return self._row_to_term(row)
            
            return None
            
        except Exception as e:
            self._increment_stat('errors')
            self.logger.error(f"Find error for source '{source}': {e}")
            return None
    
    def list_terms(
        self,
        domain: Optional[str] = None,
        status: Optional[TermStatus] = None
    ) -> Iterator[GlossaryTerm]:
        """List terms with filters."""
        try:
            conn = self._get_connection()
            
            query = "SELECT * FROM glossary WHERE 1=1"
            params = []
            
            if domain is not None:
                query += " AND domain = ?"
                params.append(domain)
            
            if status is not None:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY priority DESC, source"
            
            cur = conn.execute(query, params)
            
            while True:
                rows = cur.fetchmany(100)
                if not rows:
                    break
                for row in rows:
                    yield self._row_to_term(row)
                    
        except Exception as e:
            self._increment_stat('errors')
            self.logger.error(f"List error: {e}")
    
    def delete(self, term_id: int) -> bool:
        """Delete term by ID."""
        try:
            with self._transaction() as conn:
                cur = conn.execute("DELETE FROM glossary WHERE id = ?", (term_id,))
                deleted = cur.rowcount > 0
            
            if deleted:
                self.logger.info(f"Deleted term ID={term_id}")
            
            return deleted
            
        except Exception as e:
            self._increment_stat('errors')
            self.logger.error(f"Delete error for ID {term_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            conn = self._get_connection()
            
            # Total counts
            cur = conn.execute("SELECT COUNT(*) FROM glossary")
            total = cur.fetchone()[0]
            
            # Status distribution
            cur = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM glossary 
                GROUP BY status
            """)
            by_status = {row['status']: row['count'] for row in cur.fetchall()}
            
            # Domain distribution
            cur = conn.execute("""
                SELECT domain, COUNT(*) as count 
                FROM glossary 
                GROUP BY domain
            """)
            by_domain = {row['domain']: row['count'] for row in cur.fetchall()}
            
            with self._stats_lock:
                stats = self._stats.copy()
            
            return {
                **stats,
                'total_terms': total,
                'by_status': by_status,
                'by_domain': by_domain
            }
            
        except Exception as e:
            self.logger.error(f"Stats error: {e}")
            return {'error': str(e)}
    
    def close(self) -> None:
        """Close connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ===== Glossary Manager =====
class GlossaryManager:
    """High-level glossary manager with compiled patterns."""
    
    def __init__(
        self,
        storage: GlossaryStorage,
        config: Optional[GlossaryConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.storage = storage
        self.config = config or GlossaryConfig()
        self.logger = logger or setup_logger("glossary", self.config)
        
        # Pattern cache for performance
        self._pattern_cache: Dict[Tuple[str, bool], re.Pattern] = {}
        self._cache_lock = threading.Lock()
    
    def _get_pattern(self, term: str, case_sensitive: bool) -> re.Pattern:
        """Get or create compiled pattern."""
        cache_key = (term, case_sensitive)
        
        with self._cache_lock:
            if cache_key not in self._pattern_cache:
                pattern = TextUtil.word_boundary_pattern(term, case_sensitive)
                self._pattern_cache[cache_key] = pattern
                
                # Limit cache size
                if len(self._pattern_cache) > 1000:
                    # Remove oldest 20%
                    to_remove = list(self._pattern_cache.keys())[:200]
                    for key in to_remove:
                        del self._pattern_cache[key]
            
            return self._pattern_cache[cache_key]
    
    def upsert_term(self, **kwargs) -> int:
        """Create or update term."""
        term = GlossaryTerm(**kwargs)
        return self.storage.upsert(term)
    
    def get_term(self, term_id: int) -> Optional[GlossaryTerm]:
        """Get term by ID."""
        return self.storage.get(term_id)
    
    def delete_term(self, term_id: int) -> bool:
        """Delete term."""
        return self.storage.delete(term_id)
    
    def lookup(
        self,
        source: str,
        domain: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Optional[GlossaryTerm]:
        """Lookup term with best match."""
        return self.storage.find_best(source, domain, source_lang, target_lang)
    
    def list_terms(
        self,
        domain: Optional[str] = None,
        status: Optional[TermStatus] = None
    ) -> Iterator[GlossaryTerm]:
        """List terms with filters."""
        return self.storage.list_terms(domain, status)
    
    def apply_to_text(
        self,
        text: str,
        domain: Optional[str] = None,
        strategy: str = "replace",
        status_filter: Optional[TermStatus] = TermStatus.APPROVED
    ) -> ReplacementResult:
        """
        Apply glossary terms to text.
        
        Args:
            text: Input text
            domain: Domain filter
            strategy: 'replace' or 'mark' (for markup)
            status_filter: Only use terms with this status
        
        Returns:
            ReplacementResult with processed text and statistics
        """
        if strategy not in ('replace', 'mark'):
            raise ValueError("strategy must be 'replace' or 'mark'")
        
        # Get terms sorted by length (longest first for greedy matching)
        terms = list(self.list_terms(domain=domain, status=status_filter))
        terms.sort(key=lambda t: len(t.source), reverse=True)
        
        total_replacements = 0
        applied_terms = []
        details = []
        
        for term in terms:
            pattern = self._get_pattern(term.source, term.case_sensitive)
            matches = pattern.findall(text)
            count = len(matches)
            
            if count > 0:
                total_replacements += count
                applied_terms.append(term.source)
                
                details.append({
                    'source': term.source,
                    'target': term.target,
                    'count': count,
                    'priority': term.priority
                })
                
                if strategy == 'mark':
                    text = pattern.sub(r'@@\1@@', text)
                else:
                    text = pattern.sub(term.target, text)
        
        self.logger.info(
            f"Applied {total_replacements} replacements "
            f"({len(applied_terms)} unique terms) for domain={domain}"
        )
        
        return ReplacementResult(
            text=text,
            replacements=total_replacements,
            terms_applied=applied_terms,
            details=details
        )
    
    def batch_upsert(self, terms: List[Dict[str, Any]]) -> List[int]:
        """Batch insert/update terms."""
        term_ids = []
        
        for term_data in terms:
            try:
                term_id = self.upsert_term(**term_data)
                term_ids.append(term_id)
            except Exception as e:
                self.logger.error(f"Batch upsert error: {e}")
                term_ids.append(-1)
        
        self.logger.info(f"Batch upserted {len(term_ids)} terms")
        return term_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get glossary statistics."""
        return self.storage.get_stats()
    
    def close(self) -> None:
        """Release resources."""
        self.storage.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ===== Example Usage =====
if __name__ == "__main__":
    config = GlossaryConfig(log_level="DEBUG")
    logger = setup_logger("glossary", config)
    
    storage = SQLiteGlossary(Path("glossary.db"), logger)
    
    with GlossaryManager(storage, config, logger) as gm:
        # Add terms
        gm.upsert_term(
            source="Agreement",
            target="Соглашение",
            domain="legal",
            definition="Договор между сторонами",
            status=TermStatus.APPROVED,
            priority=10
        )
        
        gm.upsert_term(
            source="contract",
            target="контракт",
            domain="legal",
            status=TermStatus.APPROVED,
            priority=5
        )
        
        # Apply to text
        text = "This Agreement is a contract between parties."
        result = gm.apply_to_text(text, domain="legal")
        
        print(f"Original: {text}")
        print(f"Processed: {result.text}")
        print(f"Replacements: {result.replacements}")
        print(f"Terms applied: {result.terms_applied}")
        print(f"Details: {result.details}")
        
        # Statistics
        stats = gm.get_stats()
        print(f"Stats: {stats}")
