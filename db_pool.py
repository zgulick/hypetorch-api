# db_pool.py
import os
import time
import threading
import logging
from typing import Optional, Dict, List, Any
import queue

# Import db_config
from db_config import get_db_settings

# Try to import PostgreSQL driver, fall back to SQLite if unavailable
try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    import sqlite3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_pool')

class DatabaseConnectionPool:
    """
    A singleton connection pool for database connections.
    Supports both PostgreSQL and SQLite with automatic fallback.
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the connection pool."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the connection pool configuration."""
        # Only execute if this is the first instance
        if DatabaseConnectionPool._instance is None:
            # Get settings from config
            settings = get_db_settings()
            
            # Database configuration
            self.pg_pool = None
            self.sqlite_path = settings["sqlite_path"]
            self.sqlite_connections = {}
            self.sqlite_locks = {}
            
            # Pool configuration
            self.max_connections = settings["max_connections"]
            self.min_connections = settings["min_connections"]
            self.max_idle_time = settings["max_idle_time"]
            
            # Database URL
            self.db_url = settings["database_url"]
            
            # Initialize the appropriate pool
            self._initialize_pool()
            
            logger.info(f"Connection pool initialized. PostgreSQL available: {POSTGRESQL_AVAILABLE}")
            
            # Set up a thread to monitor connections
            self._start_monitor()
    
    def _initialize_pool(self):
        """Initialize the appropriate connection pool based on available drivers."""
        if POSTGRESQL_AVAILABLE and self.db_url:
            try:
                # Create a threaded connection pool
                self.pg_pool = pool.ThreadedConnectionPool(
                    minconn=self.min_connections,
                    maxconn=self.max_connections,
                    dsn=self.db_url
                )
                logger.info(f"PostgreSQL connection pool created with {self.min_connections}-{self.max_connections} connections")
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL connection pool: {e}")
                self.pg_pool = None
        else:
            logger.info("Using SQLite fallback (no PostgreSQL driver or DATABASE_URL)")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)
            
            # Initialize the connection queue for SQLite
            self.sqlite_connections = {}
            self.sqlite_locks = {}
            self._setup_sqlite()
    
    def _setup_sqlite(self):
        """Set up SQLite connection pool (simulated with a dictionary of connections)."""
        try:
            # Create a test connection to ensure SQLite works
            test_conn = sqlite3.connect(self.sqlite_path)
            test_conn.close()
            logger.info(f"SQLite connection verified at {self.sqlite_path}")
        except Exception as e:
            logger.error(f"Failed to create SQLite test connection: {e}")
    
    def _start_monitor(self):
        """Start a thread to monitor connection health."""
        monitor_thread = threading.Thread(target=self._monitor_connections, daemon=True)
        monitor_thread.start()
        logger.info("Connection monitor thread started")
    
    def _monitor_connections(self):
        """Monitor connections for health and cleanup idle connections."""
        while True:
            try:
                # Sleep for a while between checks
                time.sleep(30)
                
                # Check PostgreSQL connections if available
                if self.pg_pool:
                    # PostgreSQL pools handle this internally
                    pass
                
                # Check SQLite connections
                current_time = time.time()
                for thread_id, conn_info in list(self.sqlite_connections.items()):
                    if conn_info['last_used'] < current_time - self.max_idle_time:
                        with self._lock:
                            # Only close if it hasn't been used during our sleep
                            if thread_id in self.sqlite_connections and \
                               self.sqlite_connections[thread_id]['last_used'] < current_time - self.max_idle_time:
                                try:
                                    self.sqlite_connections[thread_id]['connection'].close()
                                    del self.sqlite_connections[thread_id]
                                    logger.debug(f"Closed idle SQLite connection for thread {thread_id}")
                                except Exception as e:
                                    logger.warning(f"Error closing idle SQLite connection: {e}")
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
    
    def get_connection(self, cursor_factory=None):
        """
        Get a database connection from the pool.
    
        Args:
            cursor_factory: Optional factory to use when creating cursors
                       (only applicable for PostgreSQL)
    
        Returns:
            A database connection
        """
        if POSTGRESQL_AVAILABLE and self.pg_pool:
            try:
                # Get connection from PostgreSQL pool
                conn = self.pg_pool.getconn(key=threading.get_ident())
            
                # If this is the first time getting this connection, set cursor factory
                if cursor_factory:
                    # Store the cursor factory directly as an attribute
                    conn._cursor_factory = cursor_factory
            
                # Check if connection is still alive
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            
                logger.debug("Retrieved PostgreSQL connection from pool")
                return conn
            except Exception as e:
                logger.error(f"Error getting PostgreSQL connection: {e}")
                if 'conn' in locals():
                    try:
                        self.pg_pool.putconn(conn)
                    except:
                        pass
            
                # Fall back to SQLite if PostgreSQL fails
                logger.warning("Falling back to SQLite after PostgreSQL connection failure")
                return self._get_sqlite_connection()
        else:
            # No PostgreSQL available, use SQLite
            return self._get_sqlite_connection()
    
    def _get_sqlite_connection(self):
        """Get a SQLite connection for the current thread."""
        thread_id = threading.get_ident()
        
        with self._lock:
            # Check if we already have a connection for this thread
            if thread_id in self.sqlite_connections:
                conn_info = self.sqlite_connections[thread_id]
                conn_info['last_used'] = time.time()
                return conn_info['connection']
            
            # Create a new connection
            try:
                conn = sqlite3.connect(self.sqlite_path)
                # Enable accessing columns by name
                conn.row_factory = sqlite3.Row
                
                # Store the connection
                self.sqlite_connections[thread_id] = {
                    'connection': conn,
                    'last_used': time.time()
                }
                
                # Create a thread-specific lock for this connection
                self.sqlite_locks[thread_id] = threading.Lock()
                
                logger.debug(f"Created new SQLite connection for thread {thread_id}")
                return conn
            except Exception as e:
                logger.error(f"Error creating SQLite connection: {e}")
                raise
    
    def return_connection(self, conn):
        """
        Return a connection to the pool.
        
        Args:
            conn: The connection to return
        """
        thread_id = threading.get_ident()
        
        # Handle PostgreSQL connections
        if POSTGRESQL_AVAILABLE and self.pg_pool:
            try:
                # Check if this is actually a PostgreSQL connection
                if hasattr(conn, 'dsn'):
                    self.pg_pool.putconn(conn, key=thread_id)
                    logger.debug("Returned PostgreSQL connection to pool")
                    return
            except Exception as e:
                logger.error(f"Error returning PostgreSQL connection to pool: {e}")
        
        # Handle SQLite connections - we don't actually close them,
        # just mark them as available for reuse by the same thread
        with self._lock:
            if thread_id in self.sqlite_connections:
                self.sqlite_connections[thread_id]['last_used'] = time.time()
                logger.debug(f"Marked SQLite connection as reusable for thread {thread_id}")
    
    def close_all(self):
        """Close all connections in the pool."""
        # Close PostgreSQL pool if available
        if POSTGRESQL_AVAILABLE and self.pg_pool:
            try:
                self.pg_pool.closeall()
                logger.info("Closed all PostgreSQL connections")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL connections: {e}")
        
        # Close all SQLite connections
        with self._lock:
            for thread_id, conn_info in list(self.sqlite_connections.items()):
                try:
                    conn_info['connection'].close()
                    logger.debug(f"Closed SQLite connection for thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Error closing SQLite connection: {e}")
            
            # Clear the dictionaries
            self.sqlite_connections.clear()
            self.sqlite_locks.clear()
            
        logger.info("Closed all database connections")

# Create a function to get a connection from the pool
def get_db_connection(cursor_factory=None):
    """
    Get a database connection from the pool.
    
    Args:
        cursor_factory: Optional cursor factory (PostgreSQL only)
    
    Returns:
        A database connection
    """
    pool = DatabaseConnectionPool.get_instance()
    return pool.get_connection(cursor_factory)

# Create a function to return a connection to the pool
def return_db_connection(conn):
    """
    Return a database connection to the pool.
    
    Args:
        conn: The connection to return
    """
    pool = DatabaseConnectionPool.get_instance()
    pool.return_connection(conn)

# Create a context manager for database connections
class DatabaseConnection:
    """Context manager for database connections."""
    
    def __init__(self, cursor_factory=None):
        """Initialize with optional cursor factory."""
        self.cursor_factory = cursor_factory
        self.conn = None
    
    def __enter__(self):
        """Get a connection from the pool."""
        self.conn = get_db_connection(self.cursor_factory)
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return the connection to the pool."""
        if self.conn:
            if exc_type is not None:
                # Exception occurred, rollback
                try:
                    self.conn.rollback()
                except:
                    pass
            return_db_connection(self.conn)

# Helper functions for common database operations
def execute_query(query, params=None, fetch=True, cursor_factory=None):
    """
    Execute a database query with proper connection handling.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query
        fetch: Whether to fetch and return results
        cursor_factory: Optional cursor factory (PostgreSQL only)
    
    Returns:
        Query results if fetch=True, otherwise None
    """
    with DatabaseConnection(cursor_factory) as conn:
        if POSTGRESQL_AVAILABLE and hasattr(conn, 'cursor') and callable(conn.cursor):
            if cursor_factory:
                cursor = conn.cursor(cursor_factory=cursor_factory)
            else:
                # Use the cursor factory stored on the connection if available
                factory = getattr(conn, '_cursor_factory', None)
                cursor = conn.cursor(cursor_factory=factory)
        else:
            cursor = conn.cursor()
        
        cursor.execute(query, params or ())
        
        if fetch:
            result = cursor.fetchall()
        else:
            result = None
            conn.commit()
        
        cursor.close()
        return result

def execute_transaction(queries):
    """
    Execute multiple queries in a single transaction.
    
    Args:
        queries: List of (query, params) tuples
    
    Returns:
        True if successful, False otherwise
    """
    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        try:
            for query, params in queries:
                cursor.execute(query, params or ())
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            return False
        finally:
            cursor.close()