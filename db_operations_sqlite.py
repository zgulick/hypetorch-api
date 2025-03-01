import json
import time
import sqlite3
from datetime import datetime
from db_sqlite import get_db_connection

def save_json_data(data):
    """Save JSON data to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Debug info
        print(f"Debug: Data Keys: {list(data.keys())}")
        print(f"Debug: Hype scores present: {'hype_scores' in data}")
        
        # Convert data to JSON string
        data_json = json.dumps(data)
        
        # Save to hype_data table
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO hype_data (timestamp, data_json) VALUES (?, ?)",
            (timestamp, data_json)
        )
        
        # Extract and save individual entity history
        if "hype_scores" in data:
            print(f"Debug: Found {len(data['hype_scores'])} entities in hype_scores")
            save_entity_history(cursor, data, timestamp)
        else:
            print("Debug: No hype_scores found in data!")
        
        conn.commit()
        conn.close()
        return True, "Data saved successfully"
    except Exception as e:
        print(f"❌ Error saving data: {e}")
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
            "SELECT data_json FROM hype_data ORDER BY timestamp DESC LIMIT 1"
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result['data_json'])
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