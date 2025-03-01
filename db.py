import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Get database URL from environment variable (we'll set this in Render)
DATABASE_URL = os.environ.get("DATABASE_URL")

# Environment setting (development or production)
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")

def get_connection():
    """Create a database connection with the appropriate schema"""
    conn = psycopg2.connect(DATABASE_URL)
    
    # Set the search path to the appropriate schema
    with conn.cursor() as cursor:
        # Create schema if it doesn't exist
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_ENVIRONMENT}")
        # Set search path to our environment
        cursor.execute(f"SET search_path TO {DB_ENVIRONMENT}")
    
    conn.commit()
    return conn

def initialize_database():
    """Create necessary tables if they don't exist"""
    conn = get_connection()
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

def execute_query(query, params=None, fetch=True):
    """Execute a database query and optionally return results"""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params or ())
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
            conn.commit()
            return result
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
        raise e
    finally:
        conn.close()