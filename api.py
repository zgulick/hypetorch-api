from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import time

# Import our new database functions
from db_wrapper import initialize_database, DB_AVAILABLE
from db_operations import save_all_data, load_latest_data, get_entity_history_data

# Initialize the database
try:
    db_initialized = initialize_database()
    if DB_AVAILABLE == True:
        print("✅ PostgreSQL database initialized successfully")
    elif DB_AVAILABLE == "SQLITE":
        print("✅ SQLite database initialized successfully")
    else:
        print("⚠️ Running in file-only mode - no database available")
except Exception as e:
    print(f"❌ Database initialization error: {e}")
    db_initialized = False

# Set the correct directory path for Render
BASE_DIR = Path("/opt/render/project/src") if os.path.exists("/opt/render/project") else Path(".")
DATA_FILE = BASE_DIR / "hypetorch_latest_output.json"

# Ensure the data file exists (create a default if missing)
if not DATA_FILE.exists():
    print(f"❌ ERROR: {DATA_FILE} not found! Creating a default file.")
    with open(DATA_FILE, "w") as f:
        json.dump({"message": "Default data"}, f)

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

def load_data():
    """Load data through our database operations module"""
    data = load_latest_data()
    if data:
        print("✅ Data loaded successfully")
        return data
    
    print("❌ WARNING: No data found!")
    return {}
    
# API Endpoints

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
        raise HTTPException(status_code=500, detail="❌ ERROR: 'hype_scores' field missing in data.")
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
    history = get_entity_history_data(entity_name, limit)
    return {"name": entity_name, "history": history}

@app.post("/api/upload_json")
def upload_json(file: UploadFile = File(...)):
    """Uploads a new JSON file, saves to database, and replaces the current file."""
    try:
        content = file.file.read()
        json_data = json.loads(content)
        
        # Save to database
        success, message = save_all_data(json_data)
        
        # Also save to file for backward compatibility
        with open(DATA_FILE, "w") as f:
            json.dump(json_data, f, indent=4)
        
        return {
            "message": "✅ File uploaded successfully!",
            "database_saved": success,
            "details": message,
            "db_available": DB_AVAILABLE
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

@app.get("/api/debug/database")
def debug_database():
    """Debug endpoint to check database information."""
    return {
        "database_available": DB_AVAILABLE,
        "database_type": "PostgreSQL" if DB_AVAILABLE == True else 
                        "SQLite" if DB_AVAILABLE == "SQLITE" else "None",
        "database_path": str(BASE_DIR),
        "database_file_exists": os.path.exists(DATA_FILE),
        "database_file_size_bytes": os.path.getsize(DATA_FILE) if os.path.exists(DATA_FILE) else 0,
        "current_directory": os.getcwd(),
        "render_directory_exists": os.path.exists("/opt/render/project/src")
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database": DB_AVAILABLE
    }

# Load Data on API Startup to Confirm Access
print("\n🚀 DEBUG: Testing Data Load at Startup...")
startup_data = load_data()
print(f"\n✅ DEBUG: Data Loaded at Startup (Keys): {list(startup_data.keys())}")