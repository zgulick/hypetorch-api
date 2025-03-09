import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(parent_dir, '.env'))

# Get database URL from environment variable (we'll set this in Render)
DATABASE_URL = os.environ.get("DATABASE_URL")

# Environment setting (development or production)
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")

def get_connection():
    """Create a database connection with the appropriate schema"""
    database_url = os.environ.get("DATABASE_URL")
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    conn = psycopg2.connect(database_url)
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
        
        # Create hype_scores table with improved time-series capabilities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS hype_scores (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id),
            score FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            time_period TEXT,
            algorithm_version TEXT
        )
        """)
        
        # Create component_metrics table for tracking individual metrics over time
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS component_metrics (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id),
            metric_type TEXT NOT NULL,
            value FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            time_period TEXT
        )
        """)
        
        # Create indexes for faster time-based queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hype_scores_timestamp 
        ON hype_scores(timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hype_scores_entity_timestamp 
        ON hype_scores(entity_id, timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_component_metrics_timestamp 
        ON component_metrics(timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_component_metrics_entity_timestamp 
        ON component_metrics(entity_id, timestamp)
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