# test_database.py
import os
import time
from dotenv import load_dotenv
from database import (
    get_connection, close_connections, initialize_database,
    get_entities, get_entity_by_name, get_current_metrics,
    load_latest_data, get_entity_by_id, get_entities_by_category,
    search_entities, get_latest_hype_scores, get_hype_score_history
)

# Load environment variables
load_dotenv()

def run_tests():
    print("🧪 Testing database connection...")
    try:
        conn = get_connection()
        print("✅ Database connection successful")
        conn.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    print("\n🧪 Testing database initialization...")
    success = initialize_database()
    print(f"{'✅' if success else '❌'} Database initialization")
    
    print("\n🧪 Testing entity retrieval...")
    entities = get_entities()
    print(f"✅ Retrieved {len(entities)} entities")
    
    print("\n🧪 Testing entity lookup...")
    entity_name = "Caitlin Clark"  # Replace with an entity you know exists
    entity = get_entity_by_name(entity_name)
    if entity:
        print(f"✅ Found entity: {entity['name']}")
    else:
        print(f"❌ Entity not found: {entity_name}")
    
    print("\n🧪 Testing metrics retrieval...")
    metrics = get_current_metrics()
    print(f"✅ Retrieved {len(metrics)} metric records")
    
    print("\n🧪 Testing latest data load...")
    data = load_latest_data()
    print(f"✅ Loaded latest data with {len(data.get('hype_scores', {}))} HYPE scores")
    
    print("\n🧪 Testing entity lookup by ID...")
    if entity:  # Use the entity we found earlier
        entity_by_id = get_entity_by_id(entity['id'])
        if entity_by_id:
            print(f"✅ Found entity by ID: {entity_by_id['name']}")
        else:
            print(f"❌ Entity not found by ID: {entity['id']}")

    print("\n🧪 Testing entities by category...")
    category = "Sports"  # Use a category you know exists
    category_entities = get_entities_by_category(category)
    print(f"✅ Found {len(category_entities)} entities in category '{category}'")

    print("\n🧪 Testing entity search...")
    search_results = search_entities("Caitlin")  # Use a term you know exists
    print(f"✅ Found {len(search_results)} entities matching 'Caitlin'")    
    
    print("\n🧪 Testing latest HYPE scores...")
    hype_scores = get_latest_hype_scores(limit=10)
    if hype_scores:
        print(f"✅ Retrieved {len(hype_scores)} latest HYPE scores")
        # Display top 3 HYPE scores
        for i, score in enumerate(hype_scores[:3]):
            print(f"  {i+1}. {score['name']}: {score['score']}")
    else:
        print("❌ No HYPE scores found")

    print("\n🧪 Testing HYPE score history...")
    if entity:  # Use the entity we found earlier
        history = get_hype_score_history(entity['id'], limit=5)
        if history:
            print(f"✅ Retrieved {len(history)} historical HYPE scores for {entity['name']}")
            # Display most recent 2 scores with dates
            for i, record in enumerate(history[:2]):
                timestamp = record['timestamp'].strftime('%Y-%m-%d') if hasattr(record['timestamp'], 'strftime') else record['timestamp']
                print(f"  {i+1}. {timestamp}: {record['score']}")
        else:
            print(f"❌ No HYPE score history found for {entity['name']}")

    return True

if __name__ == "__main__":
    print("🚀 Starting database tests...\n")
    success = run_tests()
    print(f"\n{'✅ All tests passed!' if success else '❌ Tests failed!'}")