from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from db_sqlite import init_db
from db_operations_sqlite import save_json_data, get_entity_history, get_latest_data

# Initialize SQLite database
try:
    init_db()
    print("✅ SQLite database initialization complete")
except Exception as e:
    print(f"❌ SQLite database initialization error: {e}")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
import json

# ✅ Set the correct directory path for Render
BASE_DIR = Path("/opt/render/project/src")
DATA_FILE = BASE_DIR / "hypetorch_latest_output.json"

# ✅ Ensure the data file exists (create a default if missing)
if not DATA_FILE.exists():
    print(f"❌ ERROR: {DATA_FILE} not found! Creating a default file.")
    with open(DATA_FILE, "w") as f:
        json.dump({"message": "Default data"}, f)


def load_data():
    """Loads data from the database only, without falling back to JSON file."""
    # Try to get data from database
    db_data = get_latest_data()
    if db_data:
        print("✅ Data loaded from database successfully")
        return db_data
    
    # If database is empty, return empty data
    print("❌ WARNING: No data found in database!")
    return {}
    
# ✅ API Endpoints

@app.get("/api/entities")
def get_entities():
    """Returns a list of all tracked entities (players, teams, brands, etc.)."""
    data = load_data()
    return list(data.get("hype_scores", {}).keys())

@app.get("/api/entities/{entity_id}")
def get_entity_details(entity_id: str):
    """Returns detailed hype data for a specific entity."""
    data = load_data()
    entity_name = entity_id.replace("_", " ")  # Convert underscores to spaces
    
    # Case-sensitive direct lookup
    if entity_name in data.get("hype_scores", {}):
        return {
            "name": entity_name,
            "hype_score": data.get("hype_scores", {}).get(entity_name, "N/A"),
            "mentions": data.get("mention_counts", {}).get(entity_name, 0),
            "talk_time": data.get("talk_time_counts", {}).get(entity_name, 0),
            "sentiment": data.get("player_sentiment_scores", {}).get(entity_name, [])
        }
    
    # Case-insensitive lookup (fallback)
    for key in data.get("hype_scores", {}):
        if key.lower() == entity_name.lower():
            return {
                "name": key,  # Return the original case
                "hype_score": data.get("hype_scores", {}).get(key, "N/A"),
                "mentions": data.get("mention_counts", {}).get(key, 0),
                "talk_time": data.get("talk_time_counts", {}).get(key, 0),
                "sentiment": data.get("player_sentiment_scores", {}).get(key, [])
            }
    
    # If we get here, the entity wasn't found with either method
    raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found.")

@app.get("/api/hype_scores")
def get_hype_scores():
    """Returns all hype scores from the JSON file."""
    data = load_data()
    if "hype_scores" not in data:
        raise HTTPException(status_code=500, detail="❌ ERROR: 'hype_scores' field missing in JSON file.")
    return data["hype_scores"]

@app.get("/api/entities/{entity_id}/metrics")
def get_entity_metrics(entity_id: str):
    """Returns engagement metrics for a specific entity."""
    data = load_data()
    entity_name = entity_id.replace("_", " ")
    
    # Case-sensitive direct lookup
    if entity_name in data.get("mention_counts", {}):
        return {
            "mentions": data.get("mention_counts", {}).get(entity_name, 0),
            "talk_time": data.get("talk_time_counts", {}).get(entity_name, 0),
            "sentiment": data.get("player_sentiment_scores", {}).get(entity_name, [])
        }
    
    # Case-insensitive lookup (fallback)
    for key in data.get("mention_counts", {}):
        if key.lower() == entity_name.lower():
            return {
                "mentions": data.get("mention_counts", {}).get(key, 0),
                "talk_time": data.get("talk_time_counts", {}).get(key, 0),
                "sentiment": data.get("player_sentiment_scores", {}).get(key, [])
            }
    
    # If we get here, return zeros rather than an error
    return {
        "mentions": 0,
        "talk_time": 0,
        "sentiment": []
    }

@app.get("/api/entities/{entity_id}/trending")
def get_entity_trending(entity_id: str):
    """Returns trending data for a specific entity."""
    data = load_data()
    entity_name = entity_id.replace("_", " ")  # Convert underscores to spaces
    
    # Case-sensitive direct lookup
    if entity_name in data.get("google_trends", {}):
        return {
            "google_trends": data.get("google_trends", {}).get(entity_name, 0),
            "wikipedia_views": data.get("wikipedia_views", {}).get(entity_name, 0),
            "reddit_mentions": data.get("reddit_mentions", {}).get(entity_name, 0),
            "google_news_mentions": data.get("google_news_mentions", {}).get(entity_name, 0)
        }
    
    # Case-insensitive lookup (fallback)
    for key in data.get("google_trends", {}):
        if key.lower() == entity_name.lower():
            return {
                "google_trends": data.get("google_trends", {}).get(key, 0),
                "wikipedia_views": data.get("wikipedia_views", {}).get(key, 0),
                "reddit_mentions": data.get("reddit_mentions", {}).get(key, 0),
                "google_news_mentions": data.get("google_news_mentions", {}).get(key, 0)
            }
    
    # If we get here, return zeros rather than an error
    return {
        "google_trends": 0,
        "wikipedia_views": 0,
        "reddit_mentions": 0,
        "google_news_mentions": 0
    }

@app.get("/api/last_updated")
def get_last_updated():
    """Returns the last modified timestamp of the JSON file."""
    if DATA_FILE.exists():
        return {"last_updated": os.path.getmtime(DATA_FILE)}
    return {"message": "No data available."}

@app.get("/api/entities/{entity_id}/history")
def get_entity_history_endpoint(entity_id: str, limit: int = 30):
    """Returns historical HYPE score data for a specific entity."""
    entity_name = entity_id.replace("_", " ")
    history = get_entity_history(entity_name, limit)
    return {"name": entity_name, "history": history}

#@app.get("/api/entities/{entity_id}/metric_history")
#def get_metric_history_endpoint(entity_id: str, metric_type: str, days: int = 30):
#    """Returns historical metric data for a specific entity."""
#    entity_name = entity_id.replace("_", " ")
#    history = get_metric_history(entity_name, metric_type, days)
#    return {"name": entity_name, "metric": metric_type, "history": history}

# @app.get("/api/top_entities")
#def get_top_entities_endpoint(limit: int = 10):
#    """Returns top entities by latest HYPE score."""
#    return get_top_entities(limit)

@app.post("/api/upload_json")
def upload_json(file: UploadFile = File(...)):
    """Uploads a new JSON file, saves to database, and replaces the current file."""
    try:
        content = file.file.read()
        json_data = json.loads(content)
        
        # Save to database
        success, message = save_json_data(json_data)
        
        # Also save to file for backward compatibility
        with open(DATA_FILE, "w") as f:
            json.dump(json_data, f, indent=4)
        
        return {
            "message": "✅ File uploaded successfully!",
            "database_saved": success,
            "details": message
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="❌ ERROR: Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ ERROR processing file: {str(e)}")
        
@app.get("/api/debug")
def debug_json():
    """Debug endpoint to inspect JSON file contents."""
    data = load_data()
    return data  # Returns full JSON for debugging

@app.get("/api/debug/uploaded_json")
def debug_uploaded_json():
    """Debug endpoint to check the structure of the last uploaded JSON."""
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            
            # Return a summary of the data structure
            return {
                "file_exists": True,
                "keys_present": list(data.keys()),
                "hype_scores_present": "hype_scores" in data,
                "num_entities": len(data.get("hype_scores", {})),
                "first_few_entities": list(data.get("hype_scores", {}).keys())[:5]
            }
        else:
            return {"file_exists": False}
    except Exception as e:
        return {"error": str(e)}
@app.get("/api/debug/entity/{entity_id}")
def debug_entity_history(entity_id: str, limit: int = 30):
    """Debug endpoint to check entity history function."""
    from db_operations_sqlite import get_entity_history
    
    entity_name = entity_id.replace("_", " ")
    
    # Try directly querying SQLite
    try:
        from db_sqlite import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM entity_history WHERE entity_name = ? LIMIT ?",
            (entity_name, limit)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        direct_query = []
        for row in results:
            row_dict = {}
            for key in row.keys():
                row_dict[key] = row[key] 
            direct_query.append(row_dict)
            
        # Normal function call
        history = get_entity_history(entity_name, limit)
        
        return {
            "entity_name": entity_name,
            "direct_query_results": direct_query,
            "function_results": history,
            "direct_count": len(direct_query),
            "function_count": len(history)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug/database")
def debug_database():
    """Debug endpoint to check database information."""
    from pathlib import Path
    import os
    from db_sqlite import DB_PATH
    
    db_exists = os.path.exists(DB_PATH)
    
    return {
        "database_path": str(DB_PATH),
        "database_exists": db_exists,
        "database_size_bytes": os.path.getsize(DB_PATH) if db_exists else 0,
        "current_directory": os.getcwd(),
        "render_directory_exists": os.path.exists("/opt/render/project/src")
    }

# Initialize SQLite database on startup
print("\n🚀 Initializing SQLite database...")
init_db()

# ✅ Load Data on API Startup to Confirm Database Load
print("\n🚀 Attempting to load data at startup...")
startup_data = load_data()
print(f"\n✅ DEBUG: Startup Data Summary:")
print(f"Total keys: {len(startup_data)}")
print(f"Hype Scores: {len(startup_data.get('hype_scores', {}))}")

# ✅ Load Data on API Startup to Confirm JSON File is Accessible
print("\n🚀 DEBUG: Testing JSON Load at Startup...")
startup_data = load_data()
print(f"\n✅ DEBUG: JSON Data Loaded at Startup (First 500 characters):\n{json.dumps(startup_data, indent=4)[:500]}")
