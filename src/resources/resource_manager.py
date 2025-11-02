"""
Resource manager for database optimization and monitoring.
Complete implementation with connection pooling and optimization tools.
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import time


logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages database resources and optimizations.
    
    Features:
    - Index optimization
    - Query analysis
    - Performance monitoring
    - Connection pooling
    - Database maintenance
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize resource manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._query_stats = {}
        
        logger.info(f"ResourceManager initialized for {db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.
        
        Returns:
            SQLite connection for current thread
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                isolation_level=None,
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            
            logger.debug(f"Created new connection for thread {threading.get_ident()}")
        
        return self._local.conn
    
    def close_connection(self):
        """Close thread-local connection."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            logger.debug(f"Closed connection for thread {threading.get_ident()}")
    
    def optimize_indices(self, table_name: str, index_columns: List[str]):
        """
        OPTIMIZATION: Create optimal indices for cache/glossary tables.
        
        Args:
            table_name: Table name
            index_columns: List of columns to index
        """
        try:
            conn = self.get_connection()
            
            for col in index_columns:
                index_name = f"idx_{table_name}_{col}"
                
                # Check if index exists
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (index_name,)
                )
                
                if not cur.fetchone():
                    # Create index
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {table_name}({col})"
                    )
                    logger.info(f"Created index: {index_name}")
                else:
                    logger.debug(f"Index already exists: {index_name}")
            
            # Analyze for query optimization
            conn.execute(f"ANALYZE {table_name}")
            
            logger.info(f"Optimized indices for {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to optimize indices: {e}")
    
    def create_composite_index(self, table_name: str, columns: List[str], index_name: Optional[str] = None):
        """
        Create composite index on multiple columns.
        
        Args:
            table_name: Table name
            columns: List of columns for composite index
            index_name: Optional custom index name
        """
        try:
            if not index_name:
                index_name = f"idx_{table_name}_{'_'.join(columns)}"
            
            conn = self.get_connection()
            
            # Check if exists
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                (index_name,)
            )
            
            if not cur.fetchone():
                columns_str = ', '.join(columns)
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    f"ON {table_name}({columns_str})"
                )
                logger.info(f"Created composite index: {index_name} on ({columns_str})")
            
        except Exception as e:
            logger.error(f"Failed to create composite index: {e}")
    
    def vacuum_database(self) -> float:
        """
        Optimize database by reclaiming unused space.
        
        Returns:
            Size reduction in MB
        """
        try:
            # Get size before
            size_before = self.get_database_size()
            
            logger.info(f"Starting VACUUM (current size: {size_before:.2f} MB)...")
            
            conn = self.get_connection()
            conn.execute("VACUUM")
            
            # Get size after
            size_after = self.get_database_size()
            
            reduction = size_before - size_after
            logger.info(f"Database vacuumed: {reduction:.2f} MB reclaimed (new size: {size_after:.2f} MB)")
            
            return reduction
            
        except Exception as e:
            logger.error(f"Vacuum failed: {e}")
            return 0.0
    
    def analyze_database(self):
        """
        Analyze database for query optimization.
        Updates SQLite's query planner statistics.
        """
        try:
            conn = self.get_connection()
            
            logger.info("Analyzing database...")
            conn.execute("ANALYZE")
            
            logger.info("Database analysis complete")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
    
    def get_database_size(self) -> float:
        """
        Get database file size in MB.
        
        Returns:
            Size in megabytes
        """
        if not self.db_path.exists():
            return 0.0
        
        size_bytes = self.db_path.stat().st_size
        return size_bytes / (1024 * 1024)
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict with table statistics
        """
        try:
            conn = self.get_connection()
            
            # Row count
            cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            
            # Table info
            cur = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [row['name'] for row in cur.fetchall()]
            
            # Index info
            cur = conn.execute(f"PRAGMA index_list({table_name})")
            indices = [row['name'] for row in cur.fetchall()]
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'columns': columns,
                'column_count': len(columns),
                'indices': indices,
                'index_count': len(indices)
            }
            
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {
                'table_name': table_name,
                'error': str(e)
            }
    
    def get_all_tables(self) -> List[str]:
        """
        Get list of all tables in database.
        
        Returns:
            List of table names
        """
        try:
            conn = self.get_connection()
            
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            
            tables = [row['name'] for row in cur.fetchall()]
            return tables
            
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information.
        
        Returns:
            Dict with database info
        """
        try:
            conn = self.get_connection()
            
            # Page count and size
            cur = conn.execute("PRAGMA page_count")
            page_count = cur.fetchone()[0]
            
            cur = conn.execute("PRAGMA page_size")
            page_size = cur.fetchone()[0]
            
            # Journal mode
            cur = conn.execute("PRAGMA journal_mode")
            journal_mode = cur.fetchone()[0]
            
            # Get all tables
            tables = self.get_all_tables()
            
            # Table stats
            table_stats = {}
            total_rows = 0
            for table in tables:
                stats = self.get_table_stats(table)
                table_stats[table] = stats
                total_rows += stats.get('row_count', 0)
            
            return {
                'db_path': str(self.db_path),
                'size_mb': self.get_database_size(),
                'page_count': page_count,
                'page_size': page_size,
                'journal_mode': journal_mode,
                'tables': tables,
                'table_count': len(tables),
                'total_rows': total_rows,
                'table_stats': table_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {'error': str(e)}
    
    def check_integrity(self) -> bool:
        """
        Check database integrity.
        
        Returns:
            True if integrity check passed
        """
        try:
            conn = self.get_connection()
            
            logger.info("Checking database integrity...")
            cur = conn.execute("PRAGMA integrity_check")
            result = cur.fetchone()[0]
            
            if result == 'ok':
                logger.info("✓ Database integrity check passed")
                return True
            else:
                logger.error(f"✗ Database integrity check failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Integrity check error: {e}")
            return False
    
    def optimize_all_tables(self):
        """
        Run optimization on all tables in database.
        """
        try:
            tables = self.get_all_tables()
            
            logger.info(f"Optimizing {len(tables)} tables...")
            
            for table in tables:
                # Create common indices
                common_columns = ['timestamp', 'created_at', 'updated_at']
                existing_columns = self.get_table_stats(table).get('columns', [])
                
                for col in common_columns:
                    if col in existing_columns:
                        self.optimize_indices(table, [col])
            
            # Analyze database
            self.analyze_database()
            
            logger.info("All tables optimized")
            
        except Exception as e:
            logger.error(f"Failed to optimize all tables: {e}")
    
    def get_query_plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Get query execution plan (for optimization).
        
        Args:
            query: SQL query to analyze
            
        Returns:
            List of query plan steps
        """
        try:
            conn = self.get_connection()
            
            cur = conn.execute(f"EXPLAIN QUERY PLAN {query}")
            
            plan = []
            for row in cur.fetchall():
                plan.append(dict(row))
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to get query plan: {e}")
            return []
    
    def track_query(self, query_name: str, duration: float):
        """
        Track query performance.
        
        Args:
            query_name: Name/identifier for the query
            duration: Execution time in seconds
        """
        with self._lock:
            if query_name not in self._query_stats:
                self._query_stats[query_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0
                }
            
            stats = self._query_stats[query_name]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get query performance statistics.
        
        Returns:
            Dict with query stats
        """
        with self._lock:
            result = {}
            
            for query_name, stats in self._query_stats.items():
                avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                
                result[query_name] = {
                    'count': stats['count'],
                    'total_time': round(stats['total_time'], 4),
                    'avg_time': round(avg_time, 4),
                    'min_time': round(stats['min_time'], 4),
                    'max_time': round(stats['max_time'], 4)
                }
            
            return result
    
    def reset_query_stats(self):
        """Reset query statistics."""
        with self._lock:
            self._query_stats.clear()
            logger.info("Query statistics reset")
    
    def cleanup(self):
        """
        Perform database cleanup and maintenance.
        
        Returns:
            Dict with cleanup results
        """
        results = {}
        
        try:
            # Check integrity
            results['integrity_ok'] = self.check_integrity()
            
            # Vacuum
            results['space_reclaimed_mb'] = self.vacuum_database()
            
            # Analyze
            self.analyze_database()
            results['analyzed'] = True
            
            # Get final size
            results['final_size_mb'] = self.get_database_size()
            
            logger.info(f"Cleanup complete: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            results['error'] = str(e)
            return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()


# ===== Example Usage =====

if __name__ == "__main__":
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Create resource manager
    db_path = Path("test_resource.db")
    
    with ResourceManager(db_path) as manager:
        # Create test table
        conn = manager.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER,
                timestamp TEXT
            )
        """)
        
        # Insert test data
        for i in range(1000):
            conn.execute(
                "INSERT INTO test_table (name, value, timestamp) VALUES (?, ?, datetime('now'))",
                (f"item_{i}", i, )
            )
        
        print("✓ Test data inserted")
        
        # Optimize indices
        manager.optimize_indices('test_table', ['name', 'value', 'timestamp'])
        print("✓ Indices optimized")
        
        # Get table stats
        stats = manager.get_table_stats('test_table')
        print(f"\n✓ Table stats:")
        print(f"  Rows: {stats['row_count']}")
        print(f"  Columns: {stats['column_count']}")
        print(f"  Indices: {stats['index_count']}")
        
        # Get database info
        info = manager.get_database_info()
        print(f"\n✓ Database info:")
        print(f"  Size: {info['size_mb']:.2f} MB")
        print(f"  Tables: {info['table_count']}")
        print(f"  Total rows: {info['total_rows']}")
        
        # Check integrity
        integrity_ok = manager.check_integrity()
        print(f"\n✓ Integrity check: {'passed' if integrity_ok else 'failed'}")
        
        # Cleanup
        cleanup_results = manager.cleanup()
        print(f"\n✓ Cleanup results: {cleanup_results}")
    
    print("\n✓ All tests completed")
