from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import json
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# ✅ Allow only hypetorch.com to access the API
origins = [
    "https://hypetorch.com",
    "https://www.hypetorch.com"
]

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def check_api_origin(request: Request, call_next):
    allowed_origins = ["hypetorch.com", "www.hypetorch.com"]
    
    # ✅ Check if the request came from the correct domain
    referer = request.headers.get("referer", "")
    if not any(origin in referer for origin in allowed_origins):
        raise HTTPException(status_code=403, detail="Forbidden: Unauthorized origin")

    response = await call_next(request)
    return response

# ✅ Correct JSON file location for Render
BASE_DIR = Path("/opt/render/project/src")
DATA_FILE = BASE_DIR / "hypetorch_latest_output.json"

# Ensure the data folder exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Function to load JSON data safely
def load_data():
    """Loads the JSON file if it exists, otherwise returns an empty dictionary."""
    if not DATA_FILE.exists():
        print(f"❌ ERROR: {DATA_FILE} not found!")
        return {}
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        print("✅ JSON Data Loaded Successfully")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: JSON file is corrupted: {e}")
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

    if entity_name not in data.get("hype_scores", {}):
        raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found.")

    return {
        "name": entity_name,
        "hype_score": data.get("hype_scores", {}).get(entity_name, "N/A"),
        "mentions": data.get("mention_counts", {}).get(entity_name, 0),
        "talk_time": data.get("talk_time_counts", {}).get(entity_name, 0),
        "sentiment": data.get("player_sentiment_scores", {}).get(entity_name, [])
    }

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

    return {
        "mentions": data.get("mention_counts", {}).get(entity_name, 0),
        "talk_time": data.get("talk_time_counts", {}).get(entity_name, 0),
        "sentiment": data.get("player_sentiment_scores", {}).get(entity_name, [])
    }

@app.get("/api/entities/{entity_id}/trending")
def get_entity_trending(entity_id: str):
    """Returns trending data for a specific entity."""
    data = load_data()
    entity_name = entity_id.replace("_", " ")

    return {
        "google_trends": data.get("google_trends", {}).get(entity_name, 0),
        "wikipedia_views": data.get("wikipedia_views", {}).get(entity_name, 0),
        "reddit_mentions": data.get("reddit_mentions", {}).get(entity_name, 0),
        "google_news_mentions": data.get("google_news_mentions", {}).get(entity_name, 0)
    }

@app.get("/api/last_updated")
def get_last_updated():
    """Returns the last modified timestamp of the JSON file."""
    if DATA_FILE.exists():
        return {"last_updated": os.path.getmtime(DATA_FILE)}
    return {"message": "No data available."}

@app.post("/api/upload_json")
def upload_json(file: UploadFile = File(...)):
    """Uploads a new JSON file and replaces the current one."""
    try:
        content = file.file.read()
        json_data = json.loads(content)
        with open(DATA_FILE, "w") as f:
            json.dump(json_data, f, indent=4)
        return {"message": "✅ File uploaded successfully!"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="❌ ERROR: Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ ERROR processing file: {str(e)}")

@app.get("/api/debug")
def debug_json():
    """Debug endpoint to inspect JSON file contents."""
    data = load_data()
    return data  # Returns full JSON for debugging

# ✅ Load Data on API Startup to Confirm JSON File is Accessible
print("\n🚀 DEBUG: Testing JSON Load at Startup...")
startup_data = load_data()
print(f"\n✅ DEBUG: JSON Data Loaded at Startup (First 500 characters):\n{json.dumps(startup_data, indent=4)[:500]}")
