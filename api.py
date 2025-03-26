from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Request
import json
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import time
import traceback
import psycopg2
from psycopg2 import extras
from psycopg2.extras import RealDictCursor
# Import our new database functions
from db_wrapper import initialize_database, DB_AVAILABLE, add_rodmn_column, execute_pooled_query, update_entity, create_entity, delete_entity, import_entities_to_database
from db_operations import save_all_data, load_latest_data, get_entity_history_data
from typing import Optional
from datetime import datetime
from fastapi import Query
from db_historical import (
    store_hype_data, 
    get_entity_history, 
    get_entity_metrics_history,
    get_trending_entities
)
from auth_middleware import get_api_key, api_key_required
from api_key_routes import router as api_key_router
from api_key_manager import create_api_key, get_api_keys, revoke_api_key
from db_pool import execute_query, DatabaseConnection
from api_utils import StandardResponse
from api_v1 import v1_router
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
import uuid
import logging
from api_models import EntityCreate, EntityUpdate
import hashlib
from fastapi.responses import Response

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

# Run migrations on startup
try:
    from run_migrations import run_migrations
    migrations_success = run_migrations()
    print(f"Database migrations: {'✅ Success' if migrations_success else '❌ Failed'}")
except Exception as e:
    print(f"Error running migrations: {e}")

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

# Add this middleware to api.py after your existing middlewares
@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    """Add cache-control headers to responses."""
    response = await call_next(request)
    
    # Skip cache headers for dynamically changing content
    if request.url.path.startswith("/api/admin"):
        # Admin endpoints shouldn't be cached
        response.headers["Cache-Control"] = "no-store, max-age=0"
    elif request.method == "GET" and request.url.path.startswith("/api/v1/entities"):
        # Entity data can be cached briefly
        response.headers["Cache-Control"] = "public, max-age=60"  # 1 minute
        
        # Add ETag headers for conditional requests (simple implementation)
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
            
        # Generate ETag from response content
        etag = hashlib.md5(response_body).hexdigest()
        response.headers["ETag"] = f'"{etag}"'
        
        # Check If-None-Match header for conditional requests
        if_none_match = request.headers.get("If-None-Match")
        if if_none_match and if_none_match == f'"{etag}"':
            return Response(status_code=304)  # Not Modified
        
        # Reconstruct response with the original body
        new_response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        return new_response
    
    return response

@app.middleware("http")
async def token_tracking_middleware(request: Request, call_next):
    """Middleware to track and deduct tokens for API usage."""
    # Skip token tracking for non-API routes and health checks
    if not request.url.path.startswith("/api/") or request.url.path == "/api/health-check":
        return await call_next(request)
    
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    
    # DEVELOPMENT MODE: Always allow requests regardless of token balance
    # This ensures all API endpoints work for development
    dev_mode = os.environ.get("DEVELOPMENT_MODE", "false").lower() == "true"
    dev_key = os.environ.get("DEV_API_KEY", "")
    
    # Allow all requests from development mode or with dev key
    if dev_mode or (api_key and api_key == dev_key):
        # Process the request without any token deduction
        response = await call_next(request)
        return response
    
    # For production: Normal token processing continues below
    from token_manager import calculate_token_cost, deduct_tokens
    import uuid
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Get client information
    client_ip = request.client.host if request.client else "unknown"
    
    api_key_id = None
    # Process only if API key is provided
    if api_key:
        # Get API key ID from the database
        try:
            from api_key_manager import validate_api_key
            key_info = validate_api_key(api_key)
            if key_info:
                api_key_id = key_info.get("id")
        except Exception as e:
            print(f"Error validating API key for token tracking: {e}")
    
    # Calculate token cost
    endpoint = request.url.path
    token_cost = calculate_token_cost(endpoint)
    
    # Create metadata for tracking
    metadata = {
        "method": request.method,
        "query_params": str(request.query_params)
    }
    
    # Check and deduct tokens if we have a valid API key ID
    token_deduction_success = False
    deduction_message = "No API key provided"
    
    if api_key_id:
        token_deduction_success, deduction_message = deduct_tokens(
            api_key_id, token_cost, endpoint, request_id, client_ip, metadata
        )
        
        # If insufficient tokens, return error response
        if not token_deduction_success and "Insufficient token balance" in deduction_message:
            return JSONResponse(
                status_code=402,  # Payment Required
                content={
                    "status": "error",
                    "error": {
                        "code": 402,
                        "message": "Insufficient token balance. Please purchase more tokens.",
                        "details": deduction_message
                    },
                    "metadata": {
                        "timestamp": time.time(),
                        "request_id": request_id
                    }
                }
            )
    
    # Process the request
    response = await call_next(request)
    
    # Add token tracking headers to the response
    if api_key_id:
        response.headers["X-Tokens-Used"] = str(token_cost)
        response.headers["X-Tokens-Status"] = "deducted" if token_deduction_success else "not-deducted"
        
    return response

# Register the API key management routes
app.include_router(api_key_router)  # Remove the prefix here

from token_routes import router as token_router
app.include_router(token_router)

from client_token_routes import router as client_token_router
app.include_router(client_token_router)

# Set the admin secret from environment variable or use a default for development
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "temporary-dev-secret")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api')

# Add to imports
import logging

# Set up logging if not already done
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api')

# Add this middleware before any routes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Extract client info
    client_host = request.client.host if request.client else "unknown"
    
    # Log request start
    logger.info(f"Request {request_id} started: {request.method} {request.url.path} from {client_host}")
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Add timing header
        response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
        response.headers["X-Request-ID"] = request_id
        
        # Log request completion
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.2f}ms")
        
        return response
    except Exception as e:
        # Log request error
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to enforce rate limits on API endpoints.
    Rate limits are based on API key and endpoint.
    """
    start_time = time.time()
    
    # Skip rate limiting for non-API routes
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        # No API key, use IP address as identifier
        client_id = request.client.host if request.client else "unknown"
    else:
        client_id = api_key
    
    # Check rate limit
    from rate_limiter import rate_limiter
    allowed, rate_limit_info = rate_limiter.check_rate_limit(client_id, request.url.path)
    
    if not allowed:
        # Rate limit exceeded
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded. Please slow down your requests.",
                    "details": {
                        "retry_after": rate_limit_info["X-RateLimit-Reset"]
                    }
                },
                "metadata": {
                    "timestamp": time.time()
                }
            },
            headers={
                "Retry-After": rate_limit_info["X-RateLimit-Reset"],
                **rate_limit_info
            }
        )
    
    # Process the request
    response = await call_next(request)
    
    # Add rate limit headers to the response
    for header, value in rate_limit_info.items():
        response.headers[header] = value
    
    return response


def load_data(entity_id=None):
    """Load data with optional entity filtering."""
    try:
        # Connect to database
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get the correct schema from config
            db_env = os.environ.get("DB_ENVIRONMENT", "development")
            
            # Get entities with optional filter
            if entity_id:
                cursor.execute(f"SELECT id, name, type, category, subcategory FROM {db_env}.entities WHERE name = %s", (entity_id,))
            else:
                cursor.execute(f"SELECT id, name, type, category, subcategory FROM {db_env}.entities")
            
            entities = cursor.fetchall()
            
            # Create base data structure
            data = {
                "hype_scores": {},
                "mention_counts": {},
                "talk_time_counts": {},
                "player_sentiment_scores": {},
                "rodmn_scores": {}
            }
            
            # Construct ID list for efficient querying
            entity_ids = [entity["id"] for entity in entities]
            entity_names = [entity["name"] for entity in entities]
            
            if not entity_ids:
                return data
                
            # Use IN clause for efficiency
            placeholders = ','.join(['%s'] * len(entity_ids))
            
            # Get HYPE scores with a single query
            cursor.execute(
                f"""
                SELECT e.name, h.score
                FROM entities e
                JOIN hype_scores h ON e.id = h.entity_id
                WHERE e.id IN ({placeholders})
                AND h.timestamp = (
                    SELECT MAX(timestamp) 
                    FROM hype_scores 
                    WHERE entity_id = h.entity_id
                )
                """,
                entity_ids
            )
            
            # Process results
            for row in cursor.fetchall():
                data["hype_scores"][row["name"]] = row["score"]
                
            # Similar approach for other metrics...
            
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return load_latest_data()  # Fallback


# API Endpoints
@app.get("/api/entities")
def get_entities(key_info: dict = Depends(get_api_key)):
    """Returns a list of all tracked entities (players, teams, brands, etc.)."""
    start_time = time.time()
    
    try:
        data = load_data()
        entities_list = list(data.get("hype_scores", {}).keys())
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return StandardResponse.success(
            data=entities_list,
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "count": len(entities_list)
            }
        )
    except Exception as e:
        print(f"Error retrieving entities: {str(e)}")
        return StandardResponse.error(
            message="Failed to retrieve entities",
            status_code=500,
            details=str(e)
        )

@app.get("/api/entities/{entity_id}")
def get_entity_details(entity_id: str, key_info: dict = Depends(get_api_key)):
    start_time = time.time()
    
    try:
        print(f"🔍 Fetching details for: {entity_id}")
        
        entity_name = entity_id.replace("_", " ")  # Convert underscores to spaces
        
        # Connect directly to database
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
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
            return StandardResponse.error(
                message=f"Entity '{entity_name}' not found",
                status_code=404
            )
        
        # Load all entity data from database
        data = load_data()
        
        # Use entity name from database for lookups
        correct_name = entity_details['name']
        
        # Construct response
        entity_data = {
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
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return StandardResponse.success(
            data=entity_data,
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "entity_id": entity_details['id']
            }
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error processing entity request for {entity_id}: {str(e)}")
        return StandardResponse.error(
            message="Server error processing entity data",
            status_code=500,
            details=str(e)
        )
        
@app.put("/api/entities/{entity_id}")
async def update_entity_endpoint(
    entity_id: str,
    entity_data: EntityUpdate,  # Use Pydantic model
    key_info: dict = Depends(get_api_key)
):
    """Updates details for a specific entity."""
    start_time = time.time()
    
    try:
        # Convert underscores to spaces
        entity_name = entity_id.replace("_", " ")
        
        # Filter out None values
        update_data = {k: v for k, v in entity_data.dict().items() if v is not None}
        
        if not update_data:
            return StandardResponse.error(
                message="No update data provided",
                status_code=400
            )
        
        # Call the database function
        success, message = update_entity(entity_name, update_data)
        
        if not success:
            return StandardResponse.error(
                message=message,
                status_code=400
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data={"message": message},
            metadata={
                "processing_time_ms": round(processing_time, 2)
            }
        )
    except Exception as e:
        print(f"❌ Error updating entity: {str(e)}")
        return StandardResponse.error(
            message="Error updating entity",
            status_code=500,
            details=str(e)
        )

@app.post("/api/entities")
async def create_entity_endpoint(
    entity_data: EntityCreate,  # Use Pydantic model
    key_info: dict = Depends(get_api_key)
):
    """Creates a new entity."""
    start_time = time.time()
    
    try:
        # Convert Pydantic model to dict
        entity_dict = entity_data.dict()
        
        # Call the database function
        success, message = create_entity(entity_dict)
        
        if not success:
            return StandardResponse.error(
                message=message,
                status_code=400
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data={"name": entity_data.name, "message": message},
            metadata={
                "processing_time_ms": round(processing_time, 2)
            }
        )
    except Exception as e:
        print(f"❌ Error creating entity: {str(e)}")
        return StandardResponse.error(
            message="Error creating entity",
            status_code=500,
            details=str(e)
        )

@app.get("/api/hype_scores")
def get_hype_scores(key_info: dict = Depends(get_api_key)):
    """Returns all hype scores from the JSON file."""
    start_time = time.time()
    
    try:
        data = load_data()
        
        if not data:
            return StandardResponse.error(
                message="Failed to load data",
                status_code=500
            )
            
        if "hype_scores" not in data:
            return StandardResponse.error(
                message="'hype_scores' field missing in data",
                status_code=500
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return StandardResponse.success(
            data=data["hype_scores"],
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "count": len(data["hype_scores"])
            }
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error retrieving hype scores: {str(e)}")
        return StandardResponse.error(
            message="Server error retrieving hype scores",
            status_code=500,
            details=str(e)
        )
    
@app.get("/api/rodmn_scores")
def get_rodmn_scores(key_info: dict = Depends(get_api_key)):
    """Returns all RODMN scores from the JSON file."""
    start_time = time.time()
    
    try:
        data = load_data()
        
        if "rodmn_scores" not in data:
            # Return empty dictionary instead of throwing error
            print("⚠️ WARNING: 'rodmn_scores' field missing in data.")
            rodmn_scores = {}
        else:
            rodmn_scores = data["rodmn_scores"]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return StandardResponse.success(
            data=rodmn_scores,
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "count": len(rodmn_scores)
            }
        )
    except Exception as e:
        print(f"Error retrieving RODMN scores: {str(e)}")
        return StandardResponse.error(
            message="Server error retrieving RODMN scores",
            status_code=500,
            details=str(e)
        )

@app.get("/api/controversial")
def get_controversial_entities(limit: int = 10, key_info: dict = Depends(get_api_key)):
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
def get_entity_metrics(entity_id: str, key_info: dict = Depends(get_api_key)):
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
def get_entity_trending(entity_id: str, key_info: dict = Depends(get_api_key)):
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
def get_last_updated(key_info: dict = Depends(get_api_key)):
    """Returns the last modified timestamp of the JSON file."""
        # Try to get last update time from the database
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT MAX(timestamp) AS last_ts FROM hype_data;")
            result = cursor.fetchone()
            if result and result["last_ts"]:
                # If it's a datetime object, convert to epoch seconds
                last_ts = result["last_ts"]
                if hasattr(last_ts, "timestamp"):
                    last_ts = last_ts.timestamp()
                return {"last_updated": last_ts}
    except Exception:
        # If there's any error (no DB or query failed), just use file fallback
        pass
    if DATA_FILE.exists():
        return {"last_updated": os.path.getmtime(DATA_FILE)}
    return {"message": "No data available."}

@app.get("/api/entities/{entity_id}/history")
def get_entity_history_endpoint(entity_id: str, limit: int = 30, key_info: dict = Depends(get_api_key)):
    """Returns historical HYPE score data for a specific entity."""
    entity_name = entity_id.replace("_", " ")
    history = get_entity_history_data(entity_name, limit)
    return {"name": entity_name, "history": history}

@app.post("/api/store_historical")
def store_historical_data(time_period: str = Query("last_30_days"), file: UploadFile = File(...), key_info: dict = Depends(get_api_key)):
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
    end_date: Optional[str] = None,
    key_info: dict = Depends(get_api_key)
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
    end_date: Optional[str] = None,
    key_info: dict = Depends(get_api_key)
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
    subcategory: Optional[str] = None,
    key_info: dict = Depends(get_api_key)
):
    """Returns trending entities based on recent changes in metrics."""
    trending = get_trending_entities(metric, limit, time_period, category, subcategory)
    return {"trending": trending}

@app.get("/api/update_entities_json")
def update_entities_json(key_info: dict = Depends(get_api_key)):
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
def upload_json(file: UploadFile = File(...), key_info: dict = Depends(get_api_key)):
    """Uploads a new JSON file, saves to database, and replaces the current file."""
    start_time = time.time()
    
    try:
        content = file.file.read()
        json_data = json.loads(content)
        
        # Save to database
        success, message = save_all_data(json_data)
        
        # Also save to file for backward compatibility
        with open(DATA_FILE, "w") as f:
            json.dump(json_data, f, indent=4)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return StandardResponse.success(
            data={
                "message": "File uploaded successfully!",
                "database_saved": success,
                "details": message,
                "db_available": DB_AVAILABLE
            },
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "file_size_bytes": len(content),
                "entity_count": len(json_data.get("hype_scores", {}))
            }
        )
    except json.JSONDecodeError:
        return StandardResponse.error(
            message="Invalid JSON format",
            status_code=400
        )
    except Exception as e:
        return StandardResponse.error(
            message="Error processing file",
            status_code=500,
            details=str(e)
        )
            
@app.get("/api/debug")
def debug_json(key_info: dict = Depends(get_api_key)):
    """Debug endpoint to inspect JSON file contents."""
    data = load_data()
    return data  # Returns full JSON for debugging

@app.get("/api/debug/database")
def debug_database(key_info: dict = Depends(get_api_key)):
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
            # Use our imported execute_query function instead of execute_pooled_query
            start_time = time.time()
            execute_query("SELECT 1")
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
async def delete_entity_endpoint(entity_id: str, keyinfo: dict = Depends(get_api_key)):
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

@app.get("/api/admin/rate-limits")
def get_rate_limits(key_info: dict = Depends(get_api_key)):
    """
    Get rate limit usage for all clients.
    Admin only endpoint.
    """
    from rate_limiter import rate_limiter
    
    # Example of how you might get all API keys
    all_clients = []
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT key_hash, client_name FROM api_keys WHERE is_active = TRUE")
            all_clients = cursor.fetchall()
    except Exception as e:
        print(f"Error fetching API keys: {e}")
    
    # Get usage for each client
    client_usage = {}
    for client in all_clients:
        client_id = client["key_hash"]
        client_name = client["client_name"]
        usage = rate_limiter.get_client_usage(client_id)
        if usage:  # Only include clients with usage
            client_usage[client_name] = usage
    
    return {
        "status": "success",
        "data": client_usage,
        "metadata": {
            "timestamp": time.time(),
            "client_count": len(client_usage)
        }
    }

# Add these to api.py
@app.get("/api/test")
def test_api():
    """Simple test endpoint to verify API is working"""
    return {"status": "API is working"}

@app.get("/api/admin/keys/test")
def test_admin_keys():
    """Test endpoint to verify admin keys path is working"""
    return {"status": "Admin keys path is accessible"}

@app.get("/api/admin/settings")
def get_settings(key_info: dict = Depends(get_api_key)):
    """Returns dashboard settings from the database."""
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
            # Get settings from database
            cursor.execute("""
                SELECT * FROM system_settings 
                WHERE id = 1
            """)
            
            settings = cursor.fetchone()
        
        # If no settings found, create default settings
        if not settings:
            # Return defaults since we haven't saved any settings yet
            return {
                "dashboardTitle": "HYPE Analytics Dashboard",
                "featuredEntities": "Caitlin Clark, Angel Reese, Breanna Stewart, Sabrina Ionescu, WNBA",
                "defaultTimeframe": "last_30_days",
                "enableRodmnScore": True,
                "enableSentimentAnalysis": True,
                "enableTalkTimeMetric": True,
                "enableWikipediaViews": True,
                "enableRedditMentions": True,
                "enableGoogleTrends": True,
                "minEntityDisplayCount": 5,
                "maxEntityDisplayCount": 10,
                "refreshInterval": 0,
                "publicDashboard": True,
            }
        
        return settings
    except Exception as e:
        print(f"Error retrieving settings: {str(e)}")
        # Return default settings if there's an error
        return {
            "dashboardTitle": "HYPE Analytics Dashboard",
            "featuredEntities": "Caitlin Clark, Angel Reese, Breanna Stewart, Sabrina Ionescu, WNBA",
            "defaultTimeframe": "last_30_days",
            "enableRodmnScore": True,
            "enableSentimentAnalysis": True,
            "enableTalkTimeMetric": True,
            "enableWikipediaViews": True,
            "enableRedditMentions": True,
            "enableGoogleTrends": True,
            "minEntityDisplayCount": 5,
            "maxEntityDisplayCount": 10,
            "refreshInterval": 0,
            "publicDashboard": True,
        }
    
@app.get("/api/compare")
def compare_entities(
    entities: str,  # Comma-separated list of entity names
    metrics: Optional[str] = None,  # Comma-separated list of metrics
    start_date: Optional[str] = None,  # Format: YYYY-MM-DD
    end_date: Optional[str] = None,  # Format: YYYY-MM-DD
    include_history: bool = False,  # Whether to include historical data
    time_period: Optional[str] = None,  # e.g., "last_30_days"
    key_info: dict = Depends(get_api_key)
):
    """
    Compare multiple entities across various metrics with optional time filtering.
    Returns both current values and historical trends.
    """
    try:
        print(f"📊 Entity comparison request: entities={entities}, metrics={metrics}")
        
        # Parse parameters
        entity_list = [e.strip() for e in entities.split(",")]
        metrics_list = [m.strip() for m in metrics.split(",")] if metrics else [
            "hype_scores", "rodmn_scores", "mentions", "talk_time", 
            "sentiment", "wikipedia_views", "reddit_mentions", "google_trends"
        ]
        
        print(f"🔍 Parsed entities: {entity_list}")
        print(f"🔍 Parsed metrics: {metrics_list}")
        
        # Initialize results
        results = {"entities": {}, "metadata": {}}
        for entity in entity_list:
            results["entities"][entity] = {}
        
        # Base data loading
        data = load_data()
        
        # Process each requested entity
        for entity in entity_list:
            entity_data = {}
            
            # Get normalized entity name (case-insensitive lookup)
            normalized_entity = None
            for key in data.get("hype_scores", {}):
                if key.lower() == entity.lower():
                    normalized_entity = key
                    break
            
            if not normalized_entity:
                print(f"⚠️ Entity not found: {entity}")
                results["entities"][entity] = {"error": "Entity not found"}
                continue
            
            # Process each requested metric
            for metric in metrics_list:
                base_metric = metric.rstrip("s")  # Remove plural 's' if present
                
                # Map metric names to data dictionary keys
                metric_map = {
                    "hype_score": "hype_scores",
                    "rodmn_score": "rodmn_scores",
                    "mention": "mention_counts",
                    "talk_time": "talk_time_counts",
                    "sentiment": "player_sentiment_scores",
                    "wikipedia_view": "wikipedia_views",
                    "reddit_mention": "reddit_mentions",
                    "google_trend": "google_trends",
                    "google_news": "google_news_mentions"
                }
                
                data_key = metric_map.get(base_metric, metric)
                
                # Extract value from the appropriate data dictionary
                if data_key in data:
                    metric_data = data[data_key]
                    # Case-insensitive lookup
                    for key, value in metric_data.items():
                        if key.lower() == normalized_entity.lower():
                            if isinstance(value, list):
                                # For sentiment scores, calculate average
                                entity_data[base_metric] = sum(value) / len(value) if value else 0
                            else:
                                entity_data[base_metric] = value
                            break
                    
                    # If not found, set to 0 or empty
                    if base_metric not in entity_data:
                        entity_data[base_metric] = 0 if base_metric != "sentiment" else []
            
            # Add historical data if requested
            if include_history:
                entity_data["history"] = {}
                
                # Add HYPE score history
                if "hype_score" in entity_data or "hype_scores" in metrics_list:
                    history = get_entity_history(normalized_entity, limit=30, start_date=start_date, end_date=end_date)
                    entity_data["history"]["hype_score"] = history
                
                # Add metric history for each requested metric
                for metric in metrics_list:
                    base_metric = metric.rstrip("s")
                    if base_metric != "hype_score" and base_metric in metric_map:
                        data_key = metric_map[base_metric]
                        if data_key in data:
                            metric_history = get_entity_metrics_history(
                                normalized_entity, data_key, limit=30, 
                                start_date=start_date, end_date=end_date
                            )
                            entity_data["history"][base_metric] = metric_history
            
            results["entities"][entity] = entity_data
        
        # Add metadata
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "metrics_included": metrics_list,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "time_period": time_period
            }
        }
        
        return results
    
    except Exception as e:
        print(f"❌ Error comparing entities: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing entities: {str(e)}")

@app.post("/api/admin/settings")
def save_settings(settings: dict, key_info: dict = Depends(get_api_key)):
    """Saves dashboard settings to the database."""
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
            # Check if settings table exists, create it if not
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    id INTEGER PRIMARY KEY,
                    dashboardTitle TEXT,
                    featuredEntities TEXT,
                    defaultTimeframe TEXT,
                    enableRodmnScore BOOLEAN,
                    enableSentimentAnalysis BOOLEAN,
                    enableTalkTimeMetric BOOLEAN,
                    enableWikipediaViews BOOLEAN,
                    enableRedditMentions BOOLEAN,
                    enableGoogleTrends BOOLEAN,
                    minEntityDisplayCount INTEGER,
                    maxEntityDisplayCount INTEGER,
                    refreshInterval INTEGER,
                    publicDashboard BOOLEAN,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if settings already exist
            cursor.execute("SELECT id FROM system_settings WHERE id = 1")
            exists = cursor.fetchone()
            
            if exists:
                # Update existing settings
                cursor.execute("""
                    UPDATE system_settings SET
                        dashboardTitle = %s,
                        featuredEntities = %s,
                        defaultTimeframe = %s,
                        enableRodmnScore = %s,
                        enableSentimentAnalysis = %s,
                        enableTalkTimeMetric = %s,
                        enableWikipediaViews = %s,
                        enableRedditMentions = %s,
                        enableGoogleTrends = %s,
                        minEntityDisplayCount = %s,
                        maxEntityDisplayCount = %s,
                        refreshInterval = %s,
                        publicDashboard = %s,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (
                    settings.get("dashboardTitle", "HYPE Analytics Dashboard"),
                    settings.get("featuredEntities", ""),
                    settings.get("defaultTimeframe", "last_30_days"),
                    settings.get("enableRodmnScore", True),
                    settings.get("enableSentimentAnalysis", True),
                    settings.get("enableTalkTimeMetric", True),
                    settings.get("enableWikipediaViews", True),
                    settings.get("enableRedditMentions", True),
                    settings.get("enableGoogleTrends", True),
                    settings.get("minEntityDisplayCount", 5),
                    settings.get("maxEntityDisplayCount", 10),
                    settings.get("refreshInterval", 0),
                    settings.get("publicDashboard", True)
                ))
            else:
                # Insert new settings
                cursor.execute("""
                    INSERT INTO system_settings (
                        id,
                        dashboardTitle,
                        featuredEntities,
                        defaultTimeframe,
                        enableRodmnScore,
                        enableSentimentAnalysis,
                        enableTalkTimeMetric,
                        enableWikipediaViews,
                        enableRedditMentions,
                        enableGoogleTrends,
                        minEntityDisplayCount,
                        maxEntityDisplayCount,
                        refreshInterval,
                        publicDashboard
                    ) VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    settings.get("dashboardTitle", "HYPE Analytics Dashboard"),
                    settings.get("featuredEntities", ""),
                    settings.get("defaultTimeframe", "last_30_days"),
                    settings.get("enableRodmnScore", True),
                    settings.get("enableSentimentAnalysis", True),
                    settings.get("enableTalkTimeMetric", True),
                    settings.get("enableWikipediaViews", True),
                    settings.get("enableRedditMentions", True),
                    settings.get("enableGoogleTrends", True),
                    settings.get("minEntityDisplayCount", 5),
                    settings.get("maxEntityDisplayCount", 10),
                    settings.get("refreshInterval", 0),
                    settings.get("publicDashboard", True)
                ))
            
            conn.commit()
        
        return {"success": True, "message": "Settings saved successfully"}
    except Exception as e:
        print(f"Error saving settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error saving settings: {str(e)}")

# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    error_id = str(uuid.uuid4())
    error_details = []
    
    for error in exc.errors():
        error_details.append({
            "location": error["loc"],
            "message": error["msg"],
            "type": error["type"]
        })
    
    # Log the error
    print(f"❌ Validation Error {error_id}: {exc}")
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": {
                "code": 422,
                "message": "Validation error",
                "details": error_details,
                "error_id": error_id
            },
            "metadata": {
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    error_id = str(uuid.uuid4())
    
    # Log the error
    print(f"❌ HTTP Exception {error_id} ({exc.status_code}): {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.status_code,
                "message": str(exc.detail),
                "error_id": error_id
            },
            "metadata": {
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions."""
    error_id = str(uuid.uuid4())
    
    # Log the exception
    print(f"❌ Unhandled Exception {error_id}: {str(exc)}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": 500,
                "message": "Internal server error",
                "error_id": error_id,
                "details": str(exc) if os.environ.get("DEBUG") == "true" else None
            },
            "metadata": {
                "timestamp": time.time()
            }
        }
    )


# Import entities from JSON to database on startup
print("\n🔄 Importing entities from JSON to database...")
from db_wrapper import import_entities_to_database
import_entities_to_database()

# Include versioned routes
app.include_router(v1_router, prefix="/api")

# Initialize scheduler if enabled
if os.environ.get("ENABLE_SCHEDULED_MAINTENANCE", "true").lower() == "true":
    try:
        from scheduled_maintenance import start_scheduler
        start_scheduler()
        print("✅ Scheduled maintenance initialized")
    except Exception as e:
        print(f"❌ Error initializing scheduled maintenance: {e}")

# Announce available routes
print("\n🚀 API versions available:")
print("✅ Legacy routes: /api/...")
print("✅ V1 routes: /api/v1/...")

# Load Data on API Startup to Confirm Access
print("\n🚀 DEBUG: Testing Data Load at Startup...")
startup_data = load_data()
print(f"\n✅ DEBUG: Data Loaded at Startup (Keys): {list(startup_data.keys())}")