# db_wrapper.py
"""Database wrapper that gracefully handles missing dependencies"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from db_utils import with_retry, with_connection, transactional

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

# Base directory for SQLite
BASE_DIR = Path("/opt/render/project/src") if os.path.exists("/opt/render/project") else Path(".")
DB_PATH = BASE_DIR / "hypetorch.db"

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Environment setting (development or production)
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")

# Setup SQLite connection if needed
def get_sqlite_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_sqlite_db():
    """Initialize the SQLite database with required tables"""
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    
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
        print(f"❌ PostgreSQL connection error: {e}")
        raise  # Re-raise so our retry decorator can catch it

def init_pg_db():
    """Create necessary tables if they don't exist"""
    print("Starting PostgreSQL database initialization...")
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set - cannot initialize PostgreSQL database")
        return False
        
    try:
        conn = get_pg_connection()
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
    """Initialize the database based on available modules"""
    if DB_AVAILABLE == True:  # PostgreSQL
        return init_pg_db()
    elif DB_AVAILABLE == "SQLITE":  # SQLite fallback
        return init_sqlite_db()
    else:
        print("⚠️ Database initialization skipped - running in file-only mode")
        return False

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

# Initialize database on module import
initialize_database()