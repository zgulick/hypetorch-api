import json
import time
import sqlite3
from datetime import datetime
from db_sqlite import get_db_connection
import logging
import os
from datetime import datetime, timedelta
from shutil import copy2

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Prints to console
        logging.FileHandler('/opt/render/project/src/hypetorch_operations.log', mode='a')  # Logs to file, append mode
    ]
)

def clean_old_database_entries(conn):
    """Remove database entries older than 30 days"""
    try:
        cursor = conn.cursor()
        
        # Calculate date 30 days ago
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        # Delete old entries from hype_data
        cursor.execute("DELETE FROM hype_data WHERE timestamp < ?", (thirty_days_ago,))
        
        # Delete old entries from entity_history
        cursor.execute("DELETE FROM entity_history WHERE timestamp < ?", (thirty_days_ago,))
        
        conn.commit()
        logging.info(f"Cleaned database entries older than {thirty_days_ago}")
        return cursor.rowcount  # Return number of rows deleted
    except Exception as e:
        logging.error(f"Error cleaning database: {e}")
        return 0

def backup_database():
    """Create a backup of the database file"""
    try:
        # Ensure backup directory exists
        backup_dir = "/opt/render/project/src/database_backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"hypetorch_backup_{timestamp}.db")
        
        # Use the DB_PATH from db_sqlite
        from db_sqlite import DB_PATH
        
        copy2(DB_PATH, backup_path)
        logging.info(f"Database backed up to {backup_path}")
        return backup_path
    except Exception as e:
        logging.error(f"Error creating database backup: {e}")
        return None

def save_json_data(data):
    """Save JSON data to the database with enhanced logging and persistence"""
    try:
        conn = get_db_connection()
        
        # Debug info
        logging.info(f"Debug: Data Keys: {list(data.keys())}")
        logging.info(f"Debug: Hype scores present: {'hype_scores' in data}")
        
        # Ensure data is not empty
        if not data or not data.get('hype_scores'):
            logging.warning("Attempted to save empty or invalid data")
            return False, "No valid data to save"
        
        # Clean old database entries before inserting new data
        cleaned_entries = clean_old_database_entries(conn)
        logging.info(f"Cleaned {cleaned_entries} old database entries")
        
        # Backup database before inserting new data
        backup_path = backup_database()
        if backup_path:
            logging.info(f"Database backed up to {backup_path}")
        
        # Convert data to JSON string
        data_json = json.dumps(data)
        
        # Save to hype_data table
        timestamp = datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO hype_data (timestamp, data_json) VALUES (?, ?)",
            (timestamp, data_json)
        )
        
        # Extract and save individual entity history
        if "hype_scores" in data:
            logging.info(f"Found {len(data['hype_scores'])} entities in hype_scores")
            save_entity_history(cursor, data, timestamp)
        else:
            logging.warning("No hype_scores found in data!")
        
        conn.commit()
        conn.close()
        
        return True, "Data saved successfully"
    except Exception as e:
        logging.error(f"❌ Error saving data: {e}")
        return False, f"Error saving data: {e}"
    
def save_entity_history(cursor, data, timestamp):
    """Extract entity data and save to entity_history table"""
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
    print(f"✅ Saved history for {len(hype_scores)} entities")

def get_latest_data():
    """Retrieve the most recent data from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent entry
        cursor.execute(
            "SELECT data_json, timestamp FROM hype_data ORDER BY timestamp DESC LIMIT 1"
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            print(f"✅ Found database entry from: {result['timestamp']}")
            data = json.loads(result['data_json'])
            print(f"📊 Loaded data keys: {list(data.keys())}")
            print(f"🏀 Number of hype scores: {len(data.get('hype_scores', {}))}")
            return data
        
        print("❌ No data found in database")
        return {}
    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
        return {}

def get_entity_history(entity_name, limit=10):
    """Get historical data for a specific entity"""
    try:
        conn = get_db_connection()
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
        
        # Convert to list of dictionaries
        history = []
        for row in results:
            history.append({
                'timestamp': row['timestamp'],
                'hype_score': row['hype_score'],
                'mentions': row['mentions'],
                'talk_time': row['talk_time'],
                'wikipedia_views': row['wikipedia_views'],
                'reddit_mentions': row['reddit_mentions'],
                'google_trends': row['google_trends'],
                'google_news_mentions': row['google_news_mentions']
            })
            
        return history
    except Exception as e:
        print(f"❌ Error retrieving entity history: {e}")
        return []