"""
Resource management utilities to prevent leaks.
Context managers and cleanup handlers.
"""
import sqlite3
import threading
import weakref
import atexit
import logging
from contextlib import contextmanager
from typing import Optional, Callable, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class ResourceTracker:
    """
    Track and cleanup resources automatically.
    Prevents file handle and connection leaks.
    """
    
    def __init__(self):
        self._resources = weakref.WeakValueDictionary()
        self._cleanup_handlers = {}
        self._lock = threading.Lock()
        
        # Register cleanup at exit
        atexit.register(self.cleanup_all)
    
    def register(
        self, 
        resource: Any, 
        cleanup_func: Callable, 
        resource_id: Optional[str] = None
    ):
        """
        Register resource for automatic cleanup.
        
        Args:
            resource: Resource object
            cleanup_func: Function to call for cleanup
            resource_id: Optional identifier
        """
        with self._lock:
            if resource_id is None:
                resource_id = f"{type(resource).__name__}_{id(resource)}"
            
            self._resources[resource_id] = resource
            self._cleanup_handlers[resource_id] = cleanup_func
            
            logger.debug(f"Registered resource: {resource_id}")
    
    def unregister(self, resource_id: str):
        """Unregister resource."""
        with self._lock:
            if resource_id in self._resources:
                del self._resources[resource_id]
            if resource_id in self._cleanup_handlers:
                del self._cleanup_handlers[resource_id]
            
            logger.debug(f"Unregistered resource: {resource_id}")
    
    def cleanup(self, resource_id: str):
        """Cleanup specific resource."""
        with self._lock:
            if resource_id in self._cleanup_handlers:
                cleanup_func = self._cleanup_handlers[resource_id]
                try:
                    cleanup_func()
                    logger.debug(f"Cleaned up resource: {resource_id}")
                except Exception as e:
                    logger.error(f"Cleanup failed for {resource_id}: {e}")
                finally:
                    self.unregister(resource_id)
    
    def cleanup_all(self):
        """Cleanup all registered resources."""
        logger.info("Cleaning up all registered resources")
        
        with self._lock:
            resource_ids = list(self._cleanup_handlers.keys())
        
        for resource_id in resource_ids:
            self.cleanup(resource_id)
        
        logger.info(f"Cleaned up {len(resource_ids)} resources")
    
    def get_stats(self) -> dict:
        """Get resource tracking statistics."""
        with self._lock:
            return {
                'active_resources': len(self._resources),
                'resource_types': [
                    type(r).__name__ 
                    for r in self._resources.values()
                ]
            }


# Global resource tracker
_resource_tracker: Optional[ResourceTracker] = None


def get_resource_tracker() -> ResourceTracker:
    """Get global resource tracker."""
    global _resource_tracker
    if _resource_tracker is None:
        _resource_tracker = ResourceTracker()
    return _resource_tracker


class ManagedSQLiteConnection:
    """
    SQLite connection with automatic cleanup and thread safety.
    Replaces raw thread-local connections.
    """
    
    def __init__(self, db_path: Path, timeout: float = 30.0):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._local = threading.local()
        self._connections = weakref.WeakSet()
        self._lock = threading.Lock()
        
        # Register for cleanup
        tracker = get_resource_tracker()
        tracker.register(
            self, 
            self.close_all,
            f"sqlite_{db_path.name}_{id(self)}"
        )
    
    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = self._create_connection()
        
        return self._local.conn
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create new connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            isolation_level=None,
            check_same_thread=False
        )
        
        # Configure connection
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA cache_size=-64000")
        conn.row_factory = sqlite3.Row
        
        # Track connection
        with self._lock:
            self._connections.add(conn)
        
        logger.debug(f"Created SQLite connection: {self.db_path}")
        return conn
    
    def close_thread_connection(self):
        """Close current thread's connection."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                self._local.conn.close()
                logger.debug(f"Closed thread connection: {self.db_path}")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self._local.conn = None
    
    def close_all(self):
        """Close all connections."""
        logger.info(f"Closing all connections to {self.db_path}")
        
        # Close thread-local connection
        self.close_thread_connection()
        
        # Close all tracked connections
        with self._lock:
            connections = list(self._connections)
        
        for conn in connections:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        logger.info(f"Closed {len(connections)} connections")
    
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close_all()
        except Exception as e:
            logger.error(f"Error in __del__: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_thread_connection()


class ManagedFileHandle:
    """
    File handle with automatic cleanup.
    Ensures files are
