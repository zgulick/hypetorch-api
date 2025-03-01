# db_wrapper.py
"""Database wrapper that gracefully handles missing dependencies"""

# Flag to track if database functionality is available
DB_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
    print("✅ Database module loaded successfully")
except ImportError:
    print("⚠️ WARNING: psycopg2 not installed - running in file-only mode")

# Dummy functions that will be used if psycopg2 is not available
def dummy_initialize():
    print("⚠️ Database initialization skipped - running in file-only mode")
    return False

def dummy_execute(*args, **kwargs):
    print("⚠️ Database query skipped - running in file-only mode")
    return []

def dummy_save(*args, **kwargs):
    print("⚠️ Database save skipped - running in file-only mode")
    return {"status": "skipped", "reason": "psycopg2 not available"}

# Only define real database functions if psycopg2 is available
if DB_AVAILABLE:
    import os
    
    # Get database URL from environment variable
    DATABASE_URL = os.environ.get("DATABASE_URL")
    
    # Environment setting (development or production)
    DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")
    
    def get_connection():
        """Create a database connection with the appropriate schema"""
        try:
            conn = psycopg2.connect(DATABASE_URL)
            
            # Set the search path to the appropriate schema
            with conn.cursor() as cursor:
                # Create schema if it doesn't exist
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_ENVIRONMENT}")
                # Set search path to our environment
                cursor.execute(f"SET search_path TO {DB_ENVIRONMENT}")
            
            conn.commit()
            return conn
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None
    
    def initialize_database():
        """Create necessary tables if they don't exist"""
        print("Starting database initialization...")
        
        if not DATABASE_URL:
            print("❌ DATABASE_URL not set - cannot initialize database")
            return False
            
        try:
            conn = get_connection()
            if not conn:
                return False
                
            with conn.cursor() as cursor:
                # Create entities table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT,
                    category TEXT,
                    subcategory TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create hype_scores table for historical tracking
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS hype_scores (
                    id SERIAL PRIMARY KEY,
                    entity_id INTEGER REFERENCES entities(id),
                    score FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    algorithm_version TEXT
                )
                """)
                
                # Create metrics table for individual data points
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    entity_id INTEGER REFERENCES entities(id),
                    metric_type TEXT NOT NULL,
                    value FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            
            conn.commit()
            conn.close()
            print(f"✅ Database initialized with {DB_ENVIRONMENT} schema")
            return True
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            return False
    
    def execute_query(query, params=None, fetch=True):
        """Execute a database query and optionally return results"""
        try:
            conn = get_connection()
            if not conn:
                return []
                
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    result = cursor.fetchall()
                else:
                    result = None
                conn.commit()
                return result
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
else:
    # If psycopg2 is not available, use dummy functions
    initialize_database = dummy_initialize
    execute_query = dummy_execute

# Export the appropriate functions
__all__ = ['DB_AVAILABLE', 'initialize_database', 'execute_query']