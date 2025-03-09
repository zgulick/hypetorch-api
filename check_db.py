# check_db.py
from db import get_connection
from psycopg2.extras import RealDictCursor
import json

def check_entity_data(entity_name):
    """Check if an entity exists in the database and print its data."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if entity exists
        cursor.execute(
            "SELECT id, name, type, category, subcategory FROM entities WHERE name = %s",
            (entity_name,)
        )
        entity = cursor.fetchone()
        
        if not entity:
            print(f"❌ Entity '{entity_name}' not found in database.")
            conn.close()
            return
        
        print(f"✅ Found entity: {json.dumps(dict(entity), indent=2)}")
        
        # Get HYPE scores
        cursor.execute(
            """
            SELECT score, timestamp, time_period
            FROM hype_scores
            WHERE entity_id = %s
            ORDER BY timestamp DESC
            """,
            (entity['id'],)
        )
        hype_scores = cursor.fetchall()
        
        print(f"Found {len(hype_scores)} HYPE scores:")
        for score in hype_scores:
            print(f"  • Score: {score['score']}, Time: {score['timestamp']}, Period: {score['time_period']}")
        
        # Get component metrics
        cursor.execute(
            """
            SELECT metric_type, value, timestamp, time_period
            FROM component_metrics
            WHERE entity_id = %s
            ORDER BY metric_type, timestamp DESC
            """,
            (entity['id'],)
        )
        metrics = cursor.fetchall()
        
        print(f"Found {len(metrics)} component metrics:")
        for metric in metrics:
            print(f"  • {metric['metric_type']}: {metric['value']}, Time: {metric['timestamp']}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

if __name__ == "__main__":
    # Check for Breanna Stewart
    print("\n🔍 Checking database for Breanna Stewart:\n")
    check_entity_data("Breanna Stewart")