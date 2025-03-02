import sqlite3
import os
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Prints to console
        logging.FileHandler('/opt/render/project/src/hypetorch_database.log', mode='a')  # Logs to file, append mode
    ]
)
logger = logging.getLogger(__name__)

# Database location - uses /opt/render/project/src for Render, or current directory for local dev
BASE_DIR = os.environ.get('RENDER_DATA_DIR', '/opt/render/project/src')
DB_PATH = Path(BASE_DIR) / "hypetorch.db"

def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_db():
    """Initialize the database with required tables"""
    logger.info("🔍 Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
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
        
        # Log existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Existing tables: {[table[0] for table in tables]}")
        
        conn.commit()
        logger.info("✅ Database tables created/verified successfully")
    except sqlite3.Error as e:
        logger.error(f"❌ Database initialization error: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed.")

# Initialize the database when this module is imported
init_db()