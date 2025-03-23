# db_wrapper.py
"""Database wrapper that gracefully handles missing dependencies"""

import os
import json
import time
from db_pool import DatabaseConnection, execute_query, execute_transaction
from db_pool import get_db_connection, return_db_connection, DatabaseConnectionPool
from datetime import datetime
from pathlib import Path
from db_utils import with_retry, with_connection, transactional
import threading
from contextlib import contextmanager
import requests
import base64
import traceback
from db_config import get_db_settings, POSTGRESQL_AVAILABLE, DB_ENVIRONMENT
SCHEMA_PREFIX = f"{DB_ENVIRONMENT}." if POSTGRESQL_AVAILABLE is True else ""
from db_pool import SQLITE_AVAILABLE


# Flag to track if database functionality is available
DB_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
    print("✅ PostgreSQL database module loaded successfully")
except ImportError:
    print("⚠️ WARNING: psycopg2 not installed, falling back to SQLite")
    try:
        import sqlite3
        print("✅ SQLite database module loaded as fallback")
        DB_AVAILABLE = "SQLITE"
    except ImportError:
        print("⚠️ WARNING: Neither PostgreSQL nor SQLite available - running in file-only mode")

class ConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConnectionPool, cls).__new__(cls)
                cls._instance.pool = []
                cls._instance.max_pool_size = 10
        return cls._instance

    def get_connection(self):
        """Get a connection from the pool or create a new one"""
        with self._lock:
            if self.pool:
                return self.pool.pop()
            else:
                # Create a new connection
                if DB_AVAILABLE == True:  # PostgreSQL
                    return get_pg_connection()
                elif DB_AVAILABLE == "SQLITE":  # SQLite
                    return get_sqlite_connection()
                else:
                    raise Exception("No database available")

    def return_connection(self, conn):
        """Return a connection to the pool"""
        with self._lock:
            if len(self.pool) < self.max_pool_size:
                self.pool.append(conn)
            else:
                conn.close()

# Create a singleton instance
connection_pool = ConnectionPool()

@contextmanager
def get_connection_from_pool():
    """Context manager for getting and returning a connection"""
    conn = None
    try:
        conn = connection_pool.get_connection()
        yield conn
    finally:
        if conn:
            connection_pool.return_connection(conn)

def execute_pooled_query(query, params=None, fetch=True):
    """Execute a database query using the connection pool"""
    try:
        # This uses our new execute_query function from db_pool.py
        # which handles all the connection management internally
        result = execute_query(query, params, fetch)
        return result
    except Exception as e:
        print(f"❌ Database query error: {e}")
        raise

# Base directory for SQLite
BASE_DIR = Path("/opt/render/project/src") if os.path.exists("/opt/render/project") else Path(".")
DB_PATH = BASE_DIR / "hypetorch.db"

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Environment setting (development or production)
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")

# Setup SQLite connection if needed
def get_sqlite_connection():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    return conn

def init_sqlite_db():
    """Initialize the SQLite database with required tables"""
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        type TEXT DEFAULT 'person',
        category TEXT DEFAULT 'Sports',
        subcategory TEXT DEFAULT 'Unrivaled',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS hype_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        data_json TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entity_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        hype_score REAL,
        mentions INTEGER,
        talk_time REAL,
        wikipedia_views INTEGER,
        reddit_mentions INTEGER,
        google_trends INTEGER,
        google_news_mentions INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ SQLite database initialized successfully")

# Setup PostgreSQL connection
def get_pg_connection():
    """Create a database connection with the appropriate schema"""
    try:
        # Get a connection from the pool
        conn = get_db_connection(psycopg2.extras.RealDictCursor)
        
        # Set the search path to the appropriate schema
        cursor = conn.cursor()
        try:
            # Create schema if it doesn't exist
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_ENVIRONMENT}")
            # Set search path to our environment
            cursor.execute(f"SET search_path TO {DB_ENVIRONMENT}")
        finally:
            cursor.close()
        conn.commit()
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        # Return the connection to the pool if there was an error
        if 'conn' in locals():
            return_db_connection(conn)
        raise  # Re-raise so our retry decorator can catch it

def init_pg_db():
    """Create necessary tables if they don't exist"""
    print("Starting PostgreSQL database initialization...")
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set - cannot initialize PostgreSQL database")
        return False
        
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
        if not conn:
            return False
            
        with conn.cursor() as cursor:
            # Create hype_data table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hype_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                data_json TEXT NOT NULL
            )
            """)
            
            # Create entity_history table for historical tracking
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_history (
                id SERIAL PRIMARY KEY,
                entity_name TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                hype_score REAL,
                mentions INTEGER,
                talk_time REAL,
                wikipedia_views INTEGER,
                reddit_mentions INTEGER,
                google_trends INTEGER,
                google_news_mentions INTEGER,
                rodmn_score REAL
            )
            """)
        
        conn.commit()
        conn.close()
        print(f"✅ PostgreSQL database initialized with {DB_ENVIRONMENT} schema")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL database initialization error: {e}")
        return False

# Initialize the appropriate database
def initialize_database():
    """Initialize the database based on available modules."""
    # Get the connection pool instance
    pool = DatabaseConnectionPool.get_instance()
    
    success = False
    
    # Try to initialize PostgreSQL schema and tables
    if DB_AVAILABLE is True:
        try:
            # Get a connection with the right schema
            with DatabaseConnection() as conn:
                cursor = conn.cursor()
                
                # Get DB_ENVIRONMENT from config
                db_env = get_db_settings().get("environment", "development")
                
                # Create schema if it doesn't exist
                if DB_AVAILABLE is True:
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {db_env}")
                    cursor.execute(f"SET search_path TO {db_env}")
                # Set search path to our environment
                cursor.execute(f"SET search_path TO {db_env}")
                
                # Create the entities table if it doesn't exist
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_env}.entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT DEFAULT 'person',
                    category TEXT DEFAULT 'Sports',
                    subcategory TEXT DEFAULT 'Unrivaled',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create other required tables
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_env}.hype_scores (
                    id SERIAL PRIMARY KEY,
                    entity_id INTEGER REFERENCES entities(id),
                    score FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    time_period TEXT,
                    algorithm_version TEXT
                )
                """)
                
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_env}.component_metrics (
                    id SERIAL PRIMARY KEY,
                    entity_id INTEGER REFERENCES entities(id),
                    metric_type TEXT NOT NULL,
                    value FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    time_period TEXT
                )
                """)
                
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_env}.entity_history (
                    id SERIAL PRIMARY KEY,
                    entity_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hype_score FLOAT,
                    mentions INTEGER,
                    talk_time FLOAT,
                    wikipedia_views INTEGER,
                    reddit_mentions INTEGER,
                    google_trends INTEGER,
                    google_news_mentions INTEGER,
                    rodmn_score FLOAT
                )
                """)
                
                # Create the API keys table if it doesn't exist
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_env}.api_keys (
                    id SERIAL PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    client_name TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    permissions TEXT
                )
                """)
                
                conn.commit()
                success = True
                print(f"✅ PostgreSQL database initialized with {db_env} schema")
            
        except Exception as e:
            print(f"❌ Error initializing PostgreSQL database: {e}")
            traceback.print_exc()
    
    # Initialize SQLite if PostgreSQL failed and SQLite is available
    if not success and DB_AVAILABLE == "SQLITE":
        try:
            init_sqlite_db()
            success = True
            print("✅ SQLite database initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing SQLite database: {e}")
    
    if not success:
        print("⚠️ Database initialization failed - running in file-only mode")
    
    return success

# Save JSON data to the database
@with_retry(max_retries=3)
def save_json_data(data):
    """Save JSON data to the appropriate database"""
    try:
        # Debug info
        print(f"Debug: Data Keys: {list(data.keys())}")
        print(f"Debug: Hype scores present: {'hype_scores' in data}")
        
        # Convert data to JSON string
        data_json = json.dumps(data)
        timestamp = datetime.now().isoformat()
        
        if DB_AVAILABLE == True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")
                
            cursor = conn.cursor()
            
            # Save to hype_data table
            cursor.execute(
                "INSERT INTO hype_data (timestamp, data_json) VALUES (%s, %s)",
                (timestamp, data_json)
            )
            
            # Extract and save individual entity history
            if "hype_scores" in data:
                print(f"Debug: Found {len(data['hype_scores'])} entities in hype_scores")
                save_entity_history_pg(cursor, data, timestamp)
            
            conn.commit()
            conn.close()
            
        elif DB_AVAILABLE == "SQLITE":  # SQLite fallback
            conn = get_sqlite_connection()
            cursor = conn.cursor()
            
            # Save to hype_data table
            cursor.execute(
                "INSERT INTO hype_data (timestamp, data_json) VALUES (?, ?)",
                (timestamp, data_json)
            )
            
            # Extract and save individual entity history
            if "hype_scores" in data:
                print(f"Debug: Found {len(data['hype_scores'])} entities in hype_scores")
                save_entity_history_sqlite(cursor, data, timestamp)
            
            conn.commit()
            conn.close()
            
        else:  # File-only mode
            # Save to JSON file as fallback
            with open(BASE_DIR / "hypetorch_latest_output.json", "w") as f:
                json.dump(data, f, indent=4)
                
        return True, "Data saved successfully"
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        # Always save to file as ultimate fallback
        try:
            with open(BASE_DIR / "hypetorch_latest_output.json", "w") as f:
                json.dump(data, f, indent=4)
            return False, f"Database error: {e} (Saved to file instead)"
        except:
            return False, f"Complete failure: {e}"

def get_entity_metrics_batch(entity_ids, metrics=None):
    """
    Get metrics for multiple entities in a single query.
    
    Args:
        entity_ids: List of entity IDs
        metrics: List of metrics to retrieve (None for all)
        
    Returns:
        dict: Mapped metrics by entity and metric type
    """
    metrics = metrics or ["hype_score", "mention_counts", "talk_time_counts", "rodmn_score"]
    result = {metric: {} for metric in metrics}
    
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor()
            
            # Convert list to tuple for SQL IN clause
            if len(entity_ids) == 1:
                # Special case for single ID (needs trailing comma)
                id_tuple = f"({entity_ids[0]},)"
            else:
                id_tuple = str(tuple(entity_ids))
            
            # Customize query based on requested metrics
            metric_to_table = {
                "hype_score": "hype_scores",
                "mention_counts": "component_metrics",
                "talk_time_counts": "component_metrics",
                "rodmn_score": "component_metrics"
            }
            
            for metric in metrics:
                table = metric_to_table.get(metric)
                if not table:
                    continue
                    
                if table == "hype_scores":
                    query = f"""
                    SELECT e.name, h.score
                    FROM entities e
                    JOIN hype_scores h ON e.id = h.entity_id
                    WHERE e.id IN {id_tuple}
                    AND h.timestamp = (
                        SELECT MAX(timestamp) 
                        FROM hype_scores 
                        WHERE entity_id = h.entity_id
                    )
                    """
                    cursor.execute(query)
                    
                    for row in cursor.fetchall():
                        result["hype_score"][row["name"]] = row["score"]
                        
                elif table == "component_metrics":
                    metric_field = "metric_type"
                    metric_value = metric
                    
                    if metric == "rodmn_score":
                        metric_value = "rodmn_score"
                    
                    query = f"""
                    SELECT e.name, cm.value
                    FROM entities e
                    JOIN component_metrics cm ON e.id = cm.entity_id
                    WHERE e.id IN {id_tuple}
                    AND cm.{metric_field} = %s
                    AND cm.timestamp = (
                        SELECT MAX(timestamp) 
                        FROM component_metrics 
                        WHERE entity_id = cm.entity_id
                        AND {metric_field} = %s
                    )
                    """
                    cursor.execute(query, (metric_value, metric_value))
                    
                    for row in cursor.fetchall():
                        result[metric][row["name"]] = row["value"]
            
        return result
    except Exception as e:
        print(f"❌ Error getting entity metrics: {e}")
        return {metric: {} for metric in metrics}

# Helper functions for entity history
def save_entity_history_pg(cursor, data, timestamp):
    """Extract entity data and save to PostgreSQL entity_history table"""
    hype_scores = data.get("hype_scores", {})
    mention_counts = data.get("mention_counts", {})
    talk_time_counts = data.get("talk_time_counts", {})
    wikipedia_views = data.get("wikipedia_views", {})
    reddit_mentions = data.get("reddit_mentions", {})
    google_trends = data.get("google_trends", {})
    google_news_mentions = data.get("google_news_mentions", {})
    rodmn_scores = data.get("rodmn_scores", {})
    
    # Process all entities found in hype_scores
    for entity_name, hype_score in hype_scores.items():
        cursor.execute(
            """
            INSERT INTO entity_history 
            (entity_name, timestamp, hype_score, mentions, talk_time, 
             wikipedia_views, reddit_mentions, google_trends, google_news_mentions, rodmn_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entity_name,
                timestamp,
                hype_score,
                mention_counts.get(entity_name, 0),
                talk_time_counts.get(entity_name, 0),
                wikipedia_views.get(entity_name, 0),
                reddit_mentions.get(entity_name, 0),
                google_trends.get(entity_name, 0),
                google_news_mentions.get(entity_name, 0),
                rodmn_scores.get(entity_name, 0)
            )
        )
    print(f"✅ Saved history for {len(hype_scores)} entities to PostgreSQL")


def save_entity_history_sqlite(cursor, data, timestamp):
    """Extract entity data and save to SQLite entity_history table"""
    hype_scores = data.get("hype_scores", {})
    mention_counts = data.get("mention_counts", {})
    talk_time_counts = data.get("talk_time_counts", {})
    wikipedia_views = data.get("wikipedia_views", {})
    reddit_mentions = data.get("reddit_mentions", {})
    google_trends = data.get("google_trends", {})
    google_news_mentions = data.get("google_news_mentions", {})
    
    # Process all entities found in hype_scores
    for entity_name, hype_score in hype_scores.items():
        cursor.execute(
            """
            INSERT INTO entity_history 
            (entity_name, timestamp, hype_score, mentions, talk_time, 
             wikipedia_views, reddit_mentions, google_trends, google_news_mentions, rodmn_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_name,
                timestamp,
                hype_score,
                mention_counts.get(entity_name, 0),
                talk_time_counts.get(entity_name, 0),
                wikipedia_views.get(entity_name, 0),
                reddit_mentions.get(entity_name, 0),
                google_trends.get(entity_name, 0),
                google_news_mentions.get(entity_name, 0),
            )
        )
    print(f"✅ Saved history for {len(hype_scores)} entities to SQLite")

# Get latest data from the database
@with_retry(max_retries=3)
def get_latest_data():
    """Retrieve the most recent data from the database"""
    try:
        if DB_AVAILABLE == True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")
                
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get the most recent entry
            cursor.execute(
                "SELECT data_json FROM hype_data ORDER BY timestamp DESC LIMIT 1"
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result['data_json'])
                
        elif DB_AVAILABLE == "SQLITE":  # SQLite fallback
            conn = get_sqlite_connection()
            cursor = conn.cursor()
            
            # Get the most recent entry
            cursor.execute(
                "SELECT data_json FROM hype_data ORDER BY timestamp DESC LIMIT 1"
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result['data_json'])
        
        # Try file as fallback
        try:
            with open(BASE_DIR / "hypetorch_latest_output.json", "r") as f:
                return json.load(f)
        except:
            return {}
            
    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
        # Try file as fallback
        try:
            with open(BASE_DIR / "hypetorch_latest_output.json", "r") as f:
                return json.load(f)
        except:
            return {}

# Get entity history from the database
@with_retry(max_retries=3)
def get_entity_history(entity_name, limit=10):
    """Get historical data for a specific entity"""
    try:
        if DB_AVAILABLE == True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")
                
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                """
                SELECT * FROM entity_history 
                WHERE entity_name = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
                """,
                (entity_name, limit)
            )
            
            results = cursor.fetchall()
            conn.close()
            
        elif DB_AVAILABLE == "SQLITE":  # SQLite fallback
            conn = get_sqlite_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM entity_history 
                WHERE entity_name = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (entity_name, limit)
            )
            
            results = cursor.fetchall()
            conn.close()
            
        else:
            return []
        
        # Convert to list of dictionaries
        history = []
        for row in results:
            if isinstance(row, sqlite3.Row) or isinstance(row, dict):
                # Handle both SQLite Row objects and PostgreSQL dict-like objects
                if isinstance(row, sqlite3.Row):
                    row_dict = dict(row)
                else:
                    row_dict = row
                    
                history.append({
                    'timestamp': row_dict['timestamp'],
                    'hype_score': row_dict['hype_score'],
                    'mentions': row_dict['mentions'],
                    'talk_time': row_dict['talk_time'],
                    'wikipedia_views': row_dict['wikipedia_views'],
                    'reddit_mentions': row_dict['reddit_mentions'],
                    'google_trends': row_dict['google_trends'],
                    'google_news_mentions': row_dict['google_news_mentions']
                })
            
        return history
    except Exception as e:
        print(f"❌ Error retrieving entity history: {e}")
        return []

def add_rodmn_column():
    """Add the rodmn_score column to entity_history table if it doesn't exist."""
    print("Checking for rodmn_score column in entity_history table...")
    conn = get_pg_connection()
    if not conn:
        print("❌ Failed to connect to PostgreSQL database")
        return False
        
    try:
        with conn.cursor() as cursor:
            # Check if column exists
            cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='entity_history' AND column_name='rodmn_score'
            """)
            
            if not cursor.fetchone():
                print("Adding rodmn_score column to entity_history table...")
                cursor.execute("""
                ALTER TABLE entity_history 
                ADD COLUMN rodmn_score REAL
                """)
                
                conn.commit()
                print("✅ Successfully added rodmn_score column")
            else:
                print("✅ rodmn_score column already exists")
                
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error adding column: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

@with_connection(get_pg_connection)
@transactional
def save_to_pg(conn, data, data_json, timestamp):
    """Save data to PostgreSQL with transaction handling"""
    cursor = conn.cursor()
    
    # Save to hype_data table
    cursor.execute(
        "INSERT INTO hype_data (timestamp, data_json) VALUES (%s, %s)",
        (timestamp, data_json)
    )
    
    # Extract and save individual entity history
    if "hype_scores" in data:
        print(f"Debug: Found {len(data['hype_scores'])} entities in hype_scores")
        save_entity_history_pg(cursor, data, timestamp)
    
    print("✅ Data saved to PostgreSQL successfully")

@with_connection(get_sqlite_connection)
@transactional
def save_to_sqlite(conn, data, data_json, timestamp):
    """Save data to SQLite with transaction handling"""
    cursor = conn.cursor()
    
    # Save to hype_data table
    cursor.execute(
        "INSERT INTO hype_data (timestamp, data_json) VALUES (?, ?)",
        (timestamp, data_json)
    )
    
    # Extract and save individual entity history
    if "hype_scores" in data:
        print(f"Debug: Found {len(data['hype_scores'])} entities in hype_scores")
        save_entity_history_sqlite(cursor, data, timestamp)
    
    print("✅ Data saved to SQLite successfully")

def save_entity_history_pg(cursor, data, timestamp):
    """Extract entity data and save to PostgreSQL entity_history table"""
    hype_scores = data.get("hype_scores", {})
    mention_counts = data.get("mention_counts", {})
    talk_time_counts = data.get("talk_time_counts", {})
    wikipedia_views = data.get("wikipedia_views", {})
    reddit_mentions = data.get("reddit_mentions", {})
    google_trends = data.get("google_trends", {})
    google_news_mentions = data.get("google_news_mentions", {})
    rodmn_scores = data.get("rodmn_scores", {})
    
    # Process all entities found in hype_scores
    successful_inserts = 0
    for entity_name, hype_score in hype_scores.items():
        try:
            cursor.execute(
                """
                INSERT INTO entity_history 
                (entity_name, timestamp, hype_score, mentions, talk_time, 
                 wikipedia_views, reddit_mentions, google_trends, google_news_mentions, rodmn_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    entity_name,
                    timestamp,
                    hype_score,
                    mention_counts.get(entity_name, 0),
                    talk_time_counts.get(entity_name, 0),
                    wikipedia_views.get(entity_name, 0),
                    reddit_mentions.get(entity_name, 0),
                    google_trends.get(entity_name, 0),
                    google_news_mentions.get(entity_name, 0),
                    rodmn_scores.get(entity_name, 0)
                )
            )
            successful_inserts += 1
        except Exception as e:
            print(f"⚠️ Error saving history for {entity_name} to PostgreSQL: {e}")
            # Continue with other entities even if one fails
            continue
    
    print(f"✅ Saved history for {successful_inserts}/{len(hype_scores)} entities to PostgreSQL")

def save_entity_history_sqlite(cursor, data, timestamp):
    """Extract entity data and save to SQLite entity_history table"""
    hype_scores = data.get("hype_scores", {})
    mention_counts = data.get("mention_counts", {})
    talk_time_counts = data.get("talk_time_counts", {})
    wikipedia_views = data.get("wikipedia_views", {})
    reddit_mentions = data.get("reddit_mentions", {})
    google_trends = data.get("google_trends", {})
    google_news_mentions = data.get("google_news_mentions", {})
    rodmn_scores = data.get("rodmn_scores", {})
    
    # Process all entities found in hype_scores
    successful_inserts = 0
    for entity_name, hype_score in hype_scores.items():
        try:
            # Check if rodmn_score column exists in SQLite
            has_rodmn = True
            try:
                cursor.execute("SELECT rodmn_score FROM entity_history LIMIT 1")
            except:
                has_rodmn = False
            
            if has_rodmn:
                cursor.execute(
                    """
                    INSERT INTO entity_history 
                    (entity_name, timestamp, hype_score, mentions, talk_time, 
                     wikipedia_views, reddit_mentions, google_trends, google_news_mentions, rodmn_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_name,
                        timestamp,
                        hype_score,
                        mention_counts.get(entity_name, 0),
                        talk_time_counts.get(entity_name, 0),
                        wikipedia_views.get(entity_name, 0),
                        reddit_mentions.get(entity_name, 0),
                        google_trends.get(entity_name, 0),
                        google_news_mentions.get(entity_name, 0),
                        rodmn_scores.get(entity_name, 0)
                    )
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO entity_history 
                    (entity_name, timestamp, hype_score, mentions, talk_time, 
                     wikipedia_views, reddit_mentions, google_trends, google_news_mentions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_name,
                        timestamp,
                        hype_score,
                        mention_counts.get(entity_name, 0),
                        talk_time_counts.get(entity_name, 0),
                        wikipedia_views.get(entity_name, 0),
                        reddit_mentions.get(entity_name, 0),
                        google_trends.get(entity_name, 0),
                        google_news_mentions.get(entity_name, 0)
                    )
                )
            successful_inserts += 1
        except Exception as e:
            print(f"⚠️ Error saving history for {entity_name} to SQLite: {e}")
            # Continue with other entities even if one fails
            continue
    
    print(f"✅ Saved history for {successful_inserts}/{len(hype_scores)} entities to SQLite")    

@with_retry(max_retries=3)
def update_entity(entity_name, entity_data):
    try:
        # Connect to database
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Start a transaction
        cursor.execute("BEGIN")
        
        # Debug: Print current and new names
        print(f"🔍 Attempting to update:")
        print(f"  Current Name: {entity_name}")
        print(f"  New Name: {entity_data['name']}")
        
        # First, check if the new name already exists (excluding the current entity)
        cursor.execute("""
            SELECT id, name FROM entities 
            WHERE name = %s AND name != %s
        """, (entity_data['name'], entity_name))
        existing_entity = cursor.fetchone()
        
        if existing_entity:
            print(f"❌ Duplicate entity found: {existing_entity}")
            conn.rollback()
            return False, f"An entity with the name {entity_data['name']} already exists"
        
        # Get the ID of the current entity
        cursor.execute("SELECT id FROM entities WHERE name = %s", (entity_name,))
        entity_id = cursor.fetchone()
        
        if not entity_id:
            conn.rollback()
            print(f"❌ Original entity not found: {entity_name}")
            return False, f"Entity {entity_name} not found"
        
        entity_id = entity_id[0]
        
        # Update tables differently based on their schema
        # entity_history uses entity_name
        cursor.execute("""
            UPDATE entity_history 
            SET entity_name = %s 
            WHERE entity_name = %s
        """, (entity_data['name'], entity_name))
        
        # hype_scores and component_metrics use entity_id
        cursor.execute("""
            UPDATE hype_scores 
            SET entity_id = %s 
            WHERE entity_id = %s
        """, (entity_id, entity_id))
        
        cursor.execute("""
            UPDATE component_metrics 
            SET entity_id = %s 
            WHERE entity_id = %s
        """, (entity_id, entity_id))
        
        # Update the main entities table
        cursor.execute("""
            UPDATE entities 
            SET 
                name = %s, 
                type = %s, 
                category = %s, 
                subcategory = %s
            WHERE id = %s
        """, (
            entity_data['name'], 
            entity_data['type'], 
            entity_data['category'], 
            entity_data['subcategory'], 
            entity_id
        ))
        
        # Commit all changes
        conn.commit()
        
        # Regenerate entities.json
        export_entities_to_json()
        
        print(f"✅ Successfully updated entity from {entity_name} to {entity_data['name']}")
        return True, "Entity updated successfully across all tables"
    
    except Exception as e:
        # Rollback if anything goes wrong
        conn.rollback()
        print(f"Error updating entity: {e}")
        return False, str(e)
    finally:
        conn.close()

def create_entity(entity_data):
    """
    Create a new entity with comprehensive validation and cross-system sync.
    
    Args:
        entity_data (dict): Data for the new entity
    
    Returns:
        tuple: (success_boolean, message_string)
    """
    try:
        # Validate required fields
        name = entity_data.get("name", "").strip()
        entity_type = entity_data.get("type", "person").lower()
        category = entity_data.get("category", "Sports").strip()
        subcategory = entity_data.get("subcategory", "Unrivaled").strip()
        
        # Comprehensive input validation
        if not name:
            return False, "Entity name is required"
        
        if entity_type not in ["person", "non-person"]:
            return False, "Invalid entity type. Must be 'person' or 'non-person'"
        
        if DB_AVAILABLE == True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")
            
            try:
                cursor = conn.cursor()
                
                # Check if entity already exists
                cursor.execute("SELECT id FROM entities WHERE name = %s", (name,))
                if cursor.fetchone():
                    return False, f"Entity '{name}' already exists"
                
                # Insert new entity
                cursor.execute("""
                    INSERT INTO entities (name, type, category, subcategory) 
                    VALUES (%s, %s, %s, %s)
                """, (name, entity_type, category, subcategory))
                
                # Commit transaction
                conn.commit()
                
                # Update entities.json to reflect changes
                export_entities_to_json()
                
                return True, f"Entity '{name}' created successfully"
            
            except Exception as e:
                # Rollback transaction on any error
                conn.rollback()
                print(f"❌ Entity creation error: {e}")
                return False, f"Entity creation failed: {str(e)}"
            
            finally:
                conn.close()
        
        else:
            return False, "Database not available for entity creation"
    
    except Exception as e:
        print(f"❌ Comprehensive entity creation error: {e}")
        return False, f"Unexpected error: {str(e)}"

def delete_entity(entity_name):
    """
    Delete an entity with comprehensive cross-system sync.
    
    Args:
        entity_name (str): Name of entity to delete
    
    Returns:
        tuple: (success_boolean, message_string)
    """
    try:
        if DB_AVAILABLE == True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")
            
            try:
                cursor = conn.cursor()
                
                # Delete from related tables first
                tables_to_clean = [
                    "entity_history",
                    # Add other tables with entity references if needed
                ]
                
                for table in tables_to_clean:
                    cursor.execute(f"DELETE FROM {table} WHERE entity_name = %s", (entity_name,))
                
                # Delete from entities table
                cursor.execute("DELETE FROM entities WHERE name = %s", (entity_name,))
                
                # Commit transaction
                conn.commit()
                
                # Update entities.json to reflect changes
                export_entities_to_json()
                
                return True, f"Entity '{entity_name}' deleted successfully"
            
            except Exception as e:
                # Rollback transaction on any error
                conn.rollback()
                print(f"❌ Entity deletion error: {e}")
                return False, f"Entity deletion failed: {str(e)}"
            
            finally:
                conn.close()
        
        else:
            return False, "Database not available for entity deletion"
    
    except Exception as e:
        print(f"❌ Comprehensive entity deletion error: {e}")
        return False, f"Unexpected error: {str(e)}"

def import_entities_to_database():
    """Import all entities from entities.json file into the database"""
    try:
        # Path to entities.json
        entities_json_path = BASE_DIR / "entities.json"
        
        if not os.path.exists(entities_json_path):
            print("❌ entities.json file not found")
            return False

        # Load entities from JSON
        with open(entities_json_path, 'r') as f:
            entities_data = json.load(f)

        # Connect to database
        if DB_AVAILABLE is True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")

            cursor = conn.cursor()
            try:
                entities_imported = 0

                for category, subcategories in entities_data.items():
                    for subcategory, entity_list in subcategories.items():
                        for entity_obj in entity_list:
                            entity_name = entity_obj.get("name")
                            entity_type = entity_obj.get("type")

                            if not entity_name:
                                continue

                            # Check if entity already exists
                            cursor.execute("SELECT id FROM entities WHERE name = %s", (entity_name,))
                            if cursor.fetchone():
                                continue  # Skip existing

                            # Insert new entity
                            cursor.execute("""
                                INSERT INTO entities (name, type, category, subcategory)
                                VALUES (%s, %s, %s, %s)
                            """, (entity_name, entity_type, category, subcategory))

                            entities_imported += 1

                conn.commit()
                print(f"✅ Imported {entities_imported} entities from JSON to database")
                return True

            finally:
                cursor.close()
                conn.close()

        else:
            print("❌ Database not available for import")
            return False

    except Exception as e:
        print(f"❌ Error importing entities: {e}")
        return False

    
def export_entities_to_json():
    """Export all entities from database to entities.json files"""
    try:
        if DB_AVAILABLE == True:  # PostgreSQL
            conn = get_pg_connection()
            if not conn:
                raise Exception("Failed to connect to PostgreSQL database")
                
            cursor = conn.cursor()
            try:
                # Get all entities from database using the correct schema
                cursor.execute(f"""
                    SELECT name, type, category, subcategory
                    FROM {DB_ENVIRONMENT}.entities
                """)

                db_entities = cursor.fetchall()
            
                # Create structure for entities.json
                entities_data = {}
            
                # Group entities by category and subcategory
                for entity in db_entities:
                    name, entity_type, category, subcategory = entity  # Only unpack 4 values
                
                    if category not in entities_data:
                        entities_data[category] = {}
                
                    if subcategory not in entities_data[category]:
                        entities_data[category][subcategory] = []
                
                    # Create empty lists for missing columns
                    aliases_list = []  # Define this variable explicitly
                    related_entities_list = [subcategory]  # Define this variable explicitly
                
                    # Create entity object with default values for missing columns
                    entity_obj = {
                        "name": name,
                        "type": entity_type,
                        "gender": "female" if entity_type == "person" else "neutral",
                        "aliases": aliases_list,  # Use the defined variable
                        "related_entities": related_entities_list  # Use the defined variable
                    }
                
                    # Add to appropriate list
                    entities_data[category][subcategory].append(entity_obj)
            
                # Path to export
                export_path = BASE_DIR / "entities.json"
            
                # Save to file
                with open(export_path, 'w') as f:
                    json_content = json.dumps(entities_data, indent=4)
                    f.write(json_content)
                print(f"✅ Exported entities to {export_path}")
            
                # Push to GitHub
                push_success = push_to_github(json_content)
                print(f"GitHub push {'succeeded' if push_success else 'failed'}")

            finally:    
                cursor.close()
                conn.close()
                print(f"✅ Exported {len(db_entities)} entities from database to JSON")
            return True
                
        else:
            print("❌ Database not available for export")
            return False
            
    except Exception as e:
        print(f"❌ Error exporting entities: {e}")
        return False

def push_to_github(file_content):
    """Push the entities.json file to GitHub repositories."""
    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            print("❌ GITHUB_TOKEN not found in environment variables")
            return False
            
        # GitHub API headers
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Repositories to update
        repos = [
            {"owner": "zgulick", "repo": "hypetorch-scripts", "path": "entities.json"},
            {"owner": "zgulick", "repo": "hypetorch-api", "path": "entities.json"}
        ]
        
        success = True
        for repo in repos:
            # First, get the current file (to get the SHA, which is required for updates)
            url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/contents/{repo['path']}"
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    # File exists, get its SHA
                    sha = response.json()["sha"]
                    
                    # Update the file
                    payload = {
                        "message": "Update entities.json via API",
                        "content": base64.b64encode(file_content.encode()).decode(),
                        "sha": sha
                    }
                    
                    update_response = requests.put(url, headers=headers, json=payload)
                    if update_response.status_code not in [200, 201]:
                        print(f"❌ Failed to update {repo['path']} in {repo['repo']}: {update_response.json()}")
                        success = False
                    else:
                        print(f"✅ Updated {repo['path']} in {repo['repo']}")
                        
                elif response.status_code == 404:
                    # File doesn't exist yet, create it
                    payload = {
                        "message": "Create entities.json via API",
                        "content": base64.b64encode(file_content.encode()).decode()
                    }
                    
                    create_response = requests.put(url, headers=headers, json=payload)
                    if create_response.status_code not in [200, 201]:
                        print(f"❌ Failed to create {repo['path']} in {repo['repo']}: {create_response.json()}")
                        success = False
                    else:
                        print(f"✅ Created {repo['path']} in {repo['repo']}")
                        
                else:
                    print(f"❌ Error getting file from {repo['repo']}: {response.json()}")
                    success = False
                    
            except Exception as e:
                print(f"❌ Error processing {repo['repo']}: {e}")
                success = False
                
        return success
        
    except Exception as e:
        print(f"❌ Error pushing to GitHub: {e}")
        return False

# Initialize database on module import
initialize_database()