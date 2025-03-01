import sqlite3
import os
import json
from pathlib import Path

# Database location - uses /opt/render/project/src for Render, or current directory for local dev
BASE_DIR = Path("/opt/render/project/src") if os.path.exists("/opt/render/project") else Path(".")
DB_PATH = BASE_DIR / "hypetorch.db"

def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
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
    print("✅ Database initialized successfully")

# Initialize the database when this module is imported
init_db()