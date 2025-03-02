# db_operations.py
"""Simplified database operations for HypeTorch API"""

from db_wrapper import save_json_data, get_latest_data, get_entity_history

# These functions are now wrappers around the db_wrapper functions 
# that handle all the database logic and fallbacks

def save_all_data(data):
    """Save all HypeTorch data to the database"""
    return save_json_data(data)

def load_latest_data():
    """Load the latest HypeTorch data from the database"""
    return get_latest_data()

def get_entity_history_data(entity_name, limit=30):
    """Get historical data for a specific entity"""
    return get_entity_history(entity_name, limit)