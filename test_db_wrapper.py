# test_db_wrapper.py
import json
from datetime import datetime
from db_wrapper import get_latest_data, save_json_data, get_entity_history

def test_database_operations():
    """Test the database wrapper functions with retry capabilities"""
    print("\n===== Testing Database Operations =====")
    
    # 1. Create some test data
    test_data = {
        "hype_scores": {
            "Test Entity 1": 85.5,
            "Test Entity 2": 92.1
        },
        "mention_counts": {
            "Test Entity 1": 120,
            "Test Entity 2": 150
        },
        "talk_time_counts": {
            "Test Entity 1": 5.2,
            "Test Entity 2": 7.8
        }
    }
    
    # 2. Test saving data
    print("\nTesting save_json_data()...")
    success, message = save_json_data(test_data)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    print(f"Message: {message}")
    
    # 3. Test retrieving data
    print("\nTesting get_latest_data()...")
    retrieved_data = get_latest_data()
    if retrieved_data:
        print(f"✅ Successfully retrieved data with {len(retrieved_data.keys())} keys")
        print(f"Retrieved keys: {list(retrieved_data.keys())}")
    else:
        print("❌ Failed to retrieve data or data is empty")
    
    # 4. Test entity history
    print("\nTesting get_entity_history()...")
    entity_name = "Test Entity 1"
    history = get_entity_history(entity_name, limit=5)
    if history:
        print(f"✅ Successfully retrieved history for {entity_name}")
        print(f"Found {len(history)} history records")
    else:
        print(f"❌ No history found for {entity_name}")

if __name__ == "__main__":
    test_database_operations()
    print("\n===== Test completed =====")