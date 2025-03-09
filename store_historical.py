
# store_historical.py
import os
import json
import sys
from dotenv import load_dotenv
load_dotenv()

# Import from current directory
try:
    from db_historical import store_hype_data
except ImportError:
    print("Failed to import db_historical")
    sys.exit(1)

# Path to the output JSON
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hypetorch_latest_output.json'))

print(f"Looking for output file at: {json_path}")

try:
    # Load the JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Get time period
    time_period = "last_30_days"  # Default
    if len(sys.argv) > 1:
        time_period = sys.argv[1]
    
    print(f"Using time period: {time_period}")
    
    # Store in database
    success = store_hype_data(data, time_period)
    
    if success:
        print("✅ Historical data stored successfully in the database")
    else:
        print("⚠️ Failed to store historical data")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
