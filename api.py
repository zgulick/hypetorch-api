from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import json
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import time
import traceback
import psycopg2
from psycopg2 import extras
# Import our new database functions
from db_wrapper import initialize_database, DB_AVAILABLE, add_rodmn_column, execute_pooled_query, update_entity, create_entity, delete_entity, import_entities_to_database
from db_operations import save_all_data, load_latest_data, get_entity_history_data
from typing import Optional
from db_wrapper import get_pg_connection
from fastapi import Query
from db_historical import (
    store_hype_data, 
    get_entity_history, 
    get_entity_metrics_history,
    get_trending_entities
)
# Add these new imports
from auth_middleware import get_api_key, api_key_required
from api_key_routes import router as api_key_router

# Initialize the database
try:
    db_initialized = initialize_database()
    # Also check and add rodmn_score column if needed
    add_rodmn_column()
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

# Register the API key management routes
app.include_router(api_key_router)

# Set the admin secret from environment variable or use a default for development
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "temporary-dev-secret")

def load_data():
    """Load data directly from the database instead of JSON file."""
    try:
        # Connect to database
        conn = get_pg_connection()
        if not conn:
            raise Exception("Failed to connect to PostgreSQL database")
            
        # Get all entities from database
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT id, name, type, category, subcategory FROM entities")
        entities = cursor.fetchall()
        
        # Create a data structure matching the expected format
        data = {
            "hype_scores": {},
            "mention_counts": {},
            "talk_time_counts": {},
            "player_sentiment_scores": {},
            "rodmn_scores": {}
        }
        
        # Fill with data from entities table
        for entity in entities:
            entity_name = entity["name"]
            data["hype_scores"][entity_name] = 50  # Default score
            data["mention_counts"][entity_name] = 0
            data["talk_time_counts"][entity_name] = 0
            data["player_sentiment_scores"][entity_name] = []
            data["rodmn_scores"][entity_name] = 5
        
        # Get actual metric data if available
        for entity in entities:
            entity_id = entity["id"]
            entity_name = entity["name"]
            
            # Get latest hype score
            cursor.execute(
                "SELECT score FROM hype_scores WHERE entity_id = %s ORDER BY timestamp DESC LIMIT 1",
                (entity_id,)
            )
            hype_score = cursor.fetchone()
            if hype_score:
                data["hype_scores"][entity_name] = hype_score["score"]
                
            # Get latest metrics
            cursor.execute(
                """
                SELECT metric_type, value 
                FROM component_metrics 
                WHERE entity_id = %s 
                ORDER BY timestamp DESC
                """,
                (entity_id,)
            )
            metrics = cursor.fetchall()
            
            for metric in metrics:
                metric_type = metric["metric_type"]
                value = metric["value"]
                
                if metric_type == "mentions":
                    data["mention_counts"][entity_name] = value
                elif metric_type == "talk_time_counts":
                    data["talk_time_counts"][entity_name] = value
                elif metric_type == "rodmn_score":
                    data["rodmn_scores"][entity_name] = value
        
        conn.close()
        print("✅ Data loaded directly from database")
        return data
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        # As fallback, try to use load_latest_data
        return load_latest_data()
    
# API Endpoints

@app.get("/api/entities")
def get_entities():
    """Returns a list of all tracked entities (players, teams, brands, etc.)."""
    data = load_data()
    return list(data.get("hype_scores", {}).keys())

@app.get("/api/entities/{entity_id}")
def get_entity_details(entity_id: str, key_info: dict = Depends(get_api_key)):
    try:
        print(f"🔍 Fetching details for: {entity_id}")
        
        # Connect directly to database
        conn = get_pg_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        entity_name = entity_id.replace("_", " ")  # Convert underscores to spaces
        
        # Fetch entity with exact and case-insensitive match
        cursor.execute("""
            SELECT id, name, type, category, subcategory 
            FROM entities 
            WHERE LOWER(name) = LOWER(%s)
        """, (entity_name,))
        
        entity_details = cursor.fetchone()
        
        # If not found, try fuzzy search
        if not entity_details:
            cursor.execute("""
                SELECT id, name, type, category, subcategory 
                FROM entities 
                WHERE name ILIKE %s
                LIMIT 1
            """, (f"%{entity_name}%",))
            entity_details = cursor.fetchone()
        
        if not entity_details:
            print(f"❌ No entity found for {entity_name}")
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found.")
        
        # Load all entity data from database
        data = load_data()
        
        # Use entity name from database for lookups
        correct_name = entity_details['name']
        
        # Construct response
        response = {
            "name": correct_name,
            "type": entity_details['type'],
            "category": entity_details['category'],
            "subcategory": entity_details['subcategory'],
            "hype_score": data.get("hype_scores", {}).get(correct_name, 0),
            "rodmn_score": data.get("rodmn_scores", {}).get(correct_name, 0),
            "mentions": data.get("mention_counts", {}).get(correct_name, 0),
            "talk_time": data.get("talk_time_counts", {}).get(correct_name, 0),
            "sentiment": data.get("player_sentiment_scores", {}).get(correct_name, [])
        }
        
        conn.close()
        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error processing entity request for {entity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error processing entity data: {str(e)}")

@app.put("/api/entities/{entity_id}")
async def update_entity_endpoint(entity_id: str, entity_data: dict):
    """Updates details for a specific entity."""
    try:
        # Convert underscores to spaces in entity name
        entity_name = entity_id.replace("_", " ")
        
        # Validate input data
        if not entity_data:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        # Ensure data contains all necessary fields with defaults
        update_data = {
            "name": entity_data.get("name", entity_name),
            "type": entity_data.get("type", "person"),
            "category": entity_data.get("category", "Sports"),
            "subcategory": entity_data.get("subcategory", "Unrivaled")
        }
        
        # Call the database function to update the entity
        success, message = update_entity(entity_name, update_data)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
            
        return {"success": True, "message": message}
    
    except Exception as e:
        # Log the full error for debugging
        print(f"❌ Full error updating entity: {traceback.format_exc()}")
        
        # Specific error handling
        if "violates foreign key constraint" in str(e):
            raise HTTPException(
                status_code=409, 
                detail="Cannot update entity due to existing data references."
            )
        
        # Generic error handling
        raise HTTPException(status_code=500, detail=f"Error updating entity: {str(e)}")

@app.post("/api/entities")
async def create_entity_endpoint(entity_data: dict):
    """Creates a new entity."""
    try:
        # Validate required fields
        if not entity_data.get("name"):
            raise HTTPException(status_code=400, detail="Entity name is required")
        
        # Ensure data contains all necessary fields with defaults
        new_entity_data = {
            "name": entity_data.get("name"),
            "type": entity_data.get("type", "person"),
            "category": entity_data.get("category", "Sports"),
            "subcategory": entity_data.get("subcategory", "Unrivaled")
        }
        
        # Call the database function to create the entity
        success, message = create_entity(new_entity_data)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
            
        return {"success": True, "message": message}
    
    except Exception as e:
        print(f"❌ Error creating entity: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error creating entity: {str(e)}")

@app.get("/api/hype_scores")
def get_hype_scores(key_info: dict = Depends(get_api_key)):
    """Returns all hype scores from the JSON file."""
    try:
        data = load_data()
        
        if not data:
            raise HTTPException(status_code=500, detail="Failed to load data")
            
        if "hype_scores" not in data:
            raise HTTPException(status_code=500, detail="❌ ERROR: 'hype_scores' field missing in data.")
            
        return data["hype_scores"]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error retrieving hype scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error retrieving hype scores: {str(e)}")

@app.get("/api/rodmn_scores")
def get_rodmn_scores():
    """Returns all RODMN scores from the JSON file."""
    data = load_data()
    if "rodmn_scores" not in data:
        # Return empty dictionary instead of throwing error
        print("⚠️ WARNING: 'rodmn_scores' field missing in data.")
        return {}
    return data["rodmn_scores"]

@app.get("/api/controversial")
def get_controversial_entities(limit: int = 10):
    """Returns entities sorted by RODMN score (most controversial first)."""
    data = load_data()
    if "rodmn_scores" not in data:
        # Return empty list instead of throwing error
        print("⚠️ WARNING: 'rodmn_scores' field missing in data.")
        return []
    
    # Sort entities by RODMN score and take top 'limit'
    rodmn_scores = data["rodmn_scores"]
    sorted_entities = sorted(rodmn_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    # Format the response
    return [{"name": name, "rodmn_score": score} for name, score in sorted_entities]

@app.get("/api/entities/{entity_id}/metrics")
def get_entity_metrics(entity_id: str):
    """Returns engagement metrics for a specific entity."""
    try:
        data = load_data()
        if not data:
            raise HTTPException(status_code=500, detail="Failed to load data")
            
        entity_name = entity_id.replace("_", " ")
        
        # Create case-insensitive maps of all the data dictionaries
        mention_counts_lower = {k.lower(): v for k, v in data.get("mention_counts", {}).items()}
        talk_time_lower = {k.lower(): v for k, v in data.get("talk_time_counts", {}).items()}
        sentiment_lower = {k.lower(): v for k, v in data.get("player_sentiment_scores", {}).items()}
        rodmn_lower = {k.lower(): v for k, v in data.get("rodmn_scores", {}).items()}
        
        # Use lowercase key for all lookups
        entity_lower = entity_name.lower()
        
        # Look up all metrics using case-insensitive keys
        return {
            "mentions": mention_counts_lower.get(entity_lower, 0),
            "talk_time": talk_time_lower.get(entity_lower, 0),
            "sentiment": sentiment_lower.get(entity_lower, []),
            "rodmn_score": rodmn_lower.get(entity_lower, 0)
        }
    except Exception as e:
        print(f"Error retrieving metrics for entity {entity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error retrieving entity metrics: {str(e)}")
    
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

@app.post("/api/store_historical")
def store_historical_data(time_period: str = Query("last_30_days"), file: UploadFile = File(...)):
    """Store current HYPE data as a historical snapshot."""
    try:
        content = file.file.read()
        json_data = json.loads(content)
        
        # Store in database
        success = store_hype_data(json_data, time_period)
        
        return {
            "message": "✅ Historical data saved successfully" if success else "❌ Failed to save historical data",
            "success": success,
            "time_period": time_period
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="❌ ERROR: Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ ERROR processing file: {str(e)}")

@app.get("/api/entities/{entity_id}/history")
def get_entity_hype_history(
    entity_id: str, 
    limit: int = 30, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
):
    """Returns historical HYPE score data for a specific entity."""
    entity_name = entity_id.replace("_", " ")
    history = get_entity_history(entity_name, limit, start_date, end_date)
    return {"name": entity_name, "history": history}

@app.get("/api/entities/{entity_id}/metrics/{metric_type}/history")
def get_entity_metric_history(
    entity_id: str, 
    metric_type: str,
    limit: int = 30, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
):
    """Returns historical metrics data for a specific entity and metric type."""
    entity_name = entity_id.replace("_", " ")
    history = get_entity_metrics_history(entity_name, metric_type, limit, start_date, end_date)
    return {
        "name": entity_name, 
        "metric": metric_type, 
        "history": history
    }

@app.get("/api/trending")
def get_trending_entities_endpoint(
    metric: str = Query("hype_scores"),
    limit: int = 10,
    time_period: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None
):
    """Returns trending entities based on recent changes in metrics."""
    trending = get_trending_entities(metric, limit, time_period, category, subcategory)
    return {"trending": trending}

@app.get("/api/update_entities_json")
def update_entities_json():
    """Trigger a manual update of the entities.json file from the database."""
    try:
        from db_wrapper import export_entities_to_json
        success = export_entities_to_json()
        return {
            "success": success,
            "message": "Entities JSON updated successfully" if success else "Failed to update entities JSON"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating entities JSON: {str(e)}")

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

@app.get("/api/health-check")
def health_check():
    """Comprehensive health check endpoint for the API and database"""
    health_info = {
        "status": "healthy",
        "timestamp": time.time(),
        "database": {
            "type": "PostgreSQL" if DB_AVAILABLE == True else 
                    "SQLite" if DB_AVAILABLE == "SQLITE" else "None",
            "connection": "unknown"
        },
        "api_version": "1.0.0"
    }
    
    # Test database connection
    try:
        if DB_AVAILABLE == True or DB_AVAILABLE == "SQLITE":
            # Use our new pooled query function
            start_time = time.time()
            execute_pooled_query("SELECT 1")
            query_time = time.time() - start_time
            
            health_info["database"]["connection"] = "connected"
            health_info["database"]["query_time_ms"] = round(query_time * 1000, 2)
        else:
            health_info["database"]["connection"] = "disabled"
    except Exception as e:
        health_info["status"] = "unhealthy"
        health_info["database"]["connection"] = "error"
        health_info["database"]["error"] = str(e)
    
    # Add memory usage if available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        health_info["memory"] = {
            "rss_mb": round(memory_info.rss / (1024 * 1024), 2)
        }
    except ImportError:
        # psutil not available, skip memory info
        pass
    
    return health_info

@app.delete("/api/entities/{entity_id}")
async def delete_entity_endpoint(entity_id: str):
    """Deletes a specific entity."""
    try:
        # Convert underscores to spaces in entity name
        entity_name = entity_id.replace("_", " ")
        
        # Call the database function to delete the entity
        success, message = delete_entity(entity_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
            
        return {"success": True, "message": message}
    
    except Exception as e:
        print(f"❌ Error deleting entity: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error deleting entity: {str(e)}")
    
# Import entities from JSON to database on startup
print("\n🔄 Importing entities from JSON to database...")
from db_wrapper import import_entities_to_database
import_entities_to_database()

# Load Data on API Startup to Confirm Access
print("\n🚀 DEBUG: Testing Data Load at Startup...")
startup_data = load_data()
print(f"\n✅ DEBUG: Data Loaded at Startup (Keys): {list(startup_data.keys())}")