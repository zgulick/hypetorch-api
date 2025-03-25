# api_v1.py
from fastapi import APIRouter, Depends, Query, Request, HTTPException, File, UploadFile, Body
from typing import Optional, List
import time
import json
from api_utils import StandardResponse
from auth_middleware import get_api_key
import psycopg2
from db_operations import save_all_data, load_latest_data
from api_models import BulkEntityQuery
from db_historical import get_entity_history
from fastapi.responses import JSONResponse
from db_wrapper import (
    get_db_connection,
    DatabaseConnection,
    get_entity_metrics_batch,
    get_entities_with_status_metrics,
    get_entities_with_data_metrics,
    get_entities_with_metadata_metrics,
    create_entity_relationship,
    get_entity_relationships,
    delete_entity_relationship,
    get_entity_with_metadata,
    find_related_entities,
)
from cache_manager import cache_manager

# Create v1 router
v1_router = APIRouter(prefix="/v1")

# Versioned endpoints
@v1_router.get("/entities")
def get_entities_v1(
    key_info: dict = Depends(get_api_key),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = None,
    subcategory: Optional[str] = None
):
    """
    Get a list of all tracked entities with pagination and filtering.
    
    Parameters:
    - page: Page number (starts at 1)
    - page_size: Number of items per page
    - category: Filter by category (e.g., "Sports")
    - subcategory: Filter by subcategory (e.g., "Unrivaled")
    """
    start_time = time.time()
    
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            
            # Build query with filters
            query = "SELECT id, name, type, category, subcategory FROM entities"
            params = []
            
            if category or subcategory:
                query += " WHERE"
                
                if category:
                    query += " category = %s"
                    params.append(category)
                    
                if subcategory:
                    if category:
                        query += " AND"
                    query += " subcategory = %s"
                    params.append(subcategory)
            
            # Get total count first
            count_query = f"SELECT COUNT(*) FROM ({query}) AS filtered_entities"
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()['count']
            
            # Add pagination
            query += " ORDER BY name LIMIT %s OFFSET %s"
            offset = (page - 1) * page_size
            params.extend([page_size, offset])
            
            # Execute final query
            cursor.execute(query, params)
            entities = cursor.fetchall()
            
            # Format entities for response
            entity_list = [dict(entity) for entity in entities]
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            return StandardResponse.paginate(
                items=entity_list,
                page=page,
                page_size=page_size,
                total_items=total_count,
                metadata={
                    "processing_time_ms": round(processing_time, 2),
                    "filters": {
                        "category": category,
                        "subcategory": subcategory
                    }
                }
            )
    except Exception as e:
        print(f"Error retrieving entities: {e}")
        return StandardResponse.error(
            message="Failed to retrieve entities",
            status_code=500,
            details=str(e)
        )

@v1_router.get("/entities/{entity_id}")
def get_entity_details_v1(
    entity_id: str,
    key_info: dict = Depends(get_api_key),
    include_metrics: bool = True,
    include_history: bool = False,
    include_relationships: bool = False,
    history_limit: int = Query(30, ge=1, le=100)
):
    """
    Get detailed information about a specific entity.
    
    Parameters:
    - entity_id: Entity name or ID
    - include_metrics: Include current metrics
    - include_history: Include historical data
    - include_relationships: Include entity relationships
    - history_limit: Maximum number of history records
    """
    start_time = time.time()
    
    try:
        # Replace underscores with spaces in entity name
        entity_name = entity_id.replace("_", " ")
        
        # Check if relationships are requested
        if include_relationships:
            # Use the get_entity_with_metadata function for the complete data
            entity_data = get_entity_with_metadata(entity_name, include_metrics, include_relationships)
            
            if not entity_data:
                return StandardResponse.error(
                    message=f"Entity '{entity_name}' not found",
                    status_code=404
                )
            
            # Convert to response format
            response_data = entity_data
            
            # Add history if requested
            if include_history:
                from db_historical import get_entity_history
                
                response_data['history'] = {
                    'hype_score': get_entity_history(entity_data['name'], history_limit)
                }
        else:
            # Use existing code path for non-relationship requests
            with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Fetch entity
                cursor.execute("""
                    SELECT id, name, type, category, subcategory 
                    FROM entities 
                    WHERE LOWER(name) = LOWER(%s)
                """, (entity_name,))
                
                entity_details = cursor.fetchone()
                
                if not entity_details:
                    # Try fuzzy search
                    cursor.execute("""
                        SELECT id, name, type, category, subcategory 
                        FROM entities 
                        WHERE name ILIKE %s
                        LIMIT 1
                    """, (f"%{entity_name}%",))
                    entity_details = cursor.fetchone()
                
                if not entity_details:
                    return StandardResponse.error(
                        message=f"Entity '{entity_name}' not found",
                        status_code=404
                    )
                
                # Build response with base entity data
                response_data = dict(entity_details)
                
                # Add metrics if requested
                if include_metrics:
                    # Get latest metrics for this entity
                    entity_id = entity_details['id']
                    
                    # Get hype score
                    cursor.execute("""
                        SELECT score, timestamp 
                        FROM hype_scores 
                        WHERE entity_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (entity_id,))
                    hype_score = cursor.fetchone()
                    
                    if hype_score:
                        response_data['hype_score'] = hype_score['score']
                        response_data['hype_score_updated'] = hype_score['timestamp'].isoformat()
                    
                    # Get component metrics
                    cursor.execute("""
                        SELECT metric_type, value, timestamp 
                        FROM component_metrics 
                        WHERE entity_id = %s 
                        AND timestamp = (
                            SELECT MAX(timestamp) 
                            FROM component_metrics 
                            WHERE entity_id = %s
                        )
                    """, (entity_id, entity_id))
                    
                    metrics = cursor.fetchall()
                    
                    # Convert to a metrics object
                    response_data['metrics'] = {}
                    for metric in metrics:
                        response_data['metrics'][metric['metric_type']] = {
                            'value': metric['value'],
                            'updated': metric['timestamp'].isoformat()
                        }
                
                # Add history if requested
                if include_history:
                    from db_historical import get_entity_history
                    
                    response_data['history'] = {
                        'hype_score': get_entity_history(entity_details['name'], history_limit)
                    }
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=response_data,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "include_metrics": include_metrics,
                "include_history": include_history,
                "include_relationships": include_relationships
            }
        )
    except Exception as e:
        print(f"Error retrieving entity details: {e}")
        return StandardResponse.error(
            message="Failed to retrieve entity details",
            status_code=500,
            details=str(e)
        )

# Add more versioned endpoints following this pattern
@v1_router.post("/bulk")
def bulk_query_v1(
    query: BulkEntityQuery, 
    key_info: dict = Depends(get_api_key)
):
    """
    Bulk fetch data for multiple entities in one request.
    """
    start_time = time.time()
    try:
        # Load all data (this is a simpler approach that uses your existing functions)
        data = load_latest_data()
        
        # Extract needed data collections
        hype_scores = data.get("hype_scores", {})
        mentions = data.get("mention_counts", {})
        talk_time = data.get("talk_time_counts", {})
        rodmn_scores = data.get("rodmn_scores", {})
        
        # Process requested entities
        results = []
        for entity_name in query.entities:
            # Try case-insensitive matching
            matched_entity = None
            for existing_name in hype_scores.keys():
                if entity_name.lower() == existing_name.lower():
                    matched_entity = existing_name
                    break
            
            if not matched_entity:
                continue
                
            # Create entity result
            entity_result = {
                "name": matched_entity,
                "metrics": {}
            }
            
            # Add requested metrics
            requested_metrics = query.metrics or ["hype_score", "mentions", "talk_time"]
            
            if "hype_score" in requested_metrics:
                entity_result["metrics"]["hype_score"] = hype_scores.get(matched_entity, 0)
                
            if "mentions" in requested_metrics:
                entity_result["metrics"]["mentions"] = mentions.get(matched_entity, 0)
                
            if "talk_time" in requested_metrics:
                entity_result["metrics"]["talk_time"] = talk_time.get(matched_entity, 0)
                
            if "rodmn_score" in requested_metrics:
                entity_result["metrics"]["rodmn_score"] = rodmn_scores.get(matched_entity, 0)
            
            # Add history if requested
            if query.include_history:
                from db_historical import get_entity_history
                entity_result["history"] = {
                    "hype_score": get_entity_history(matched_entity, query.history_limit or 30)
                }
            
            results.append(entity_result)
        
        # Check if we found any results
        if not results:
            return StandardResponse.error(
                message="No matching entities found",
                status_code=404
            )
        
        # Return success response
        return StandardResponse.success(
            data=results,
            metadata={
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "entity_count": len(results),
                "metrics_included": query.metrics or ["hype_score", "mentions", "talk_time"]
            }
        )
    except Exception as e:
        print(f"Error in bulk_query_v1: {e}")
        import traceback
        traceback.print_exc()
        return StandardResponse.error(
            message="Failed to retrieve bulk entity data",
            status_code=500,
            details=str(e)
        )

@v1_router.get("/bulk")
def bulk_query_get_v1(
    entities: str = Query(..., description="Comma-separated list of entity names"),
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics"),
    include_history: bool = Query(False, description="Include historical data"),
    history_limit: int = Query(30, ge=1, le=100, description="Max history points"),
    time_period: Optional[str] = Query(None, description="Time period"),
    key_info: dict = Depends(get_api_key)
):
    """
    GET version of the bulk query endpoint. 
    Takes comma-separated query parameters instead of a JSON body.
    """
    # Convert string parameters to lists
    entity_list = [e.strip() for e in entities.split(",")]
    metrics_list = [m.strip() for m in metrics.split(",")] if metrics else None
    
    # Create a BulkEntityQuery object
    query = BulkEntityQuery(
        entities=entity_list,
        metrics=metrics_list,
        include_history=include_history,
        history_limit=history_limit,
        time_period=time_period
    )
    
    # Call the POST version with the converted parameters
    return bulk_query_v1(query, key_info)

@v1_router.get("/entities/{entity_id}/metrics")
def get_entity_metrics_v1(
    entity_id: str,
    key_info: dict = Depends(get_api_key)
):
    """Returns engagement metrics for a specific entity."""
    try:
        # Convert underscores to spaces in entity name
        entity_name = entity_id.replace("_", " ")
        
        # Load data to get the metrics
        data = load_latest_data()
        
        # Create case-insensitive maps for lookups
        mention_counts_lower = {k.lower(): v for k, v in data.get("mention_counts", {}).items()}
        talk_time_lower = {k.lower(): v for k, v in data.get("talk_time_counts", {}).items()}
        sentiment_lower = {k.lower(): v for k, v in data.get("player_sentiment_scores", {}).items()}
        rodmn_lower = {k.lower(): v for k, v in data.get("rodmn_scores", {}).items()}
        
        # Use lowercase key for lookups
        entity_lower = entity_name.lower()
        
        # Get metrics using case-insensitive keys
        return {
            "mentions": mention_counts_lower.get(entity_lower, 0),
            "talk_time": talk_time_lower.get(entity_lower, 0),
            "sentiment": sentiment_lower.get(entity_lower, []),
            "rodmn_score": rodmn_lower.get(entity_lower, 0)
        }
    except Exception as e:
        print(f"Error retrieving metrics for entity {entity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error retrieving entity metrics: {str(e)}")
    

@v1_router.get("/entities/{entity_id}/trending")
def get_entity_trending_v1(
    entity_id: str,
    key_info: dict = Depends(get_api_key)
):
    """Returns trending data for a specific entity."""
    try:
        data = load_latest_data()
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
    except Exception as e:
        print(f"Error retrieving trending data for entity {entity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error retrieving entity trending data: {str(e)}")

@v1_router.get("/entities/{entity_id}/relationships")
def get_entity_relationships_endpoint(
    entity_id: str,
    relationship_type: Optional[str] = None,
    direction: str = Query("both", regex="^(outgoing|incoming|both)$"),
    key_info: dict = Depends(get_api_key)
):
    """Get relationships for an entity."""
    start_time = time.time()
    
    try:
        # Replace underscores with spaces in entity name
        entity_name = entity_id.replace("_", " ")
        
        # Get relationships
        relationships = get_entity_relationships(entity_name, relationship_type, direction)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=relationships,
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "entity": entity_name,
                "relationship_type": relationship_type,
                "direction": direction,
                "count": len(relationships)
            }
        )
    except Exception as e:
        print(f"Error getting entity relationships: {e}")
        return StandardResponse.error(
            message="Failed to retrieve entity relationships",
            status_code=500,
            details=str(e)
        )

@v1_router.post("/entities/relationships")
def create_entity_relationship_endpoint(
    source_entity: str = Body(..., description="Source entity name"),
    target_entity: str = Body(..., description="Target entity name"),
    relationship_type: str = Body(..., description="Relationship type"),
    strength: Optional[float] = Body(None, description="Relationship strength (0-1)"),
    metadata: Optional[dict] = Body(None, description="Additional metadata"),
    key_info: dict = Depends(get_api_key)
):
    """Create a relationship between two entities."""
    start_time = time.time()
    
    try:
        # Create relationship
        success, result = create_entity_relationship(
            source_entity, target_entity, relationship_type, strength, metadata
        )
        
        if not success:
            return StandardResponse.error(
                message=result,
                status_code=400
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data={
                "relationship_id": result,
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relationship_type": relationship_type
            },
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2)
            }
        )
    except Exception as e:
        print(f"Error creating entity relationship: {e}")
        return StandardResponse.error(
            message="Failed to create entity relationship",
            status_code=500,
            details=str(e)
        )

@v1_router.delete("/entities/relationships/{relationship_id}")
def delete_entity_relationship_endpoint(
    relationship_id: int,
    key_info: dict = Depends(get_api_key)
):
    """Delete an entity relationship."""
    start_time = time.time()
    
    try:
        # Delete relationship
        success, message = delete_entity_relationship(relationship_id)
        
        if not success:
            return StandardResponse.error(
                message=message,
                status_code=404 if "not found" in message else 400
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data={"message": message},
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "relationship_id": relationship_id
            }
        )
    except Exception as e:
        print(f"Error deleting entity relationship: {e}")
        return StandardResponse.error(
            message="Failed to delete entity relationship",
            status_code=500,
            details=str(e)
        )

@v1_router.get("/entities/{entity_id}/related")
def find_related_entities_endpoint(
    entity_id: str,
    relationship_types: Optional[str] = Query(None, description="Comma-separated list of relationship types"),
    min_strength: Optional[float] = Query(None, ge=0, le=1, description="Minimum relationship strength"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    key_info: dict = Depends(get_api_key)
):
    """Find entities related to the given entity."""
    start_time = time.time()
    
    try:
        # Replace underscores with spaces in entity name
        entity_name = entity_id.replace("_", " ")
        
        # Process relationship types
        types_list = None
        if relationship_types:
            types_list = [t.strip() for t in relationship_types.split(",")]
        
        # Find related entities
        related = find_related_entities(entity_name, types_list, min_strength, max_results)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=related,
            metadata={
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time, 2),
                "entity": entity_name,
                "relationship_types": types_list,
                "min_strength": min_strength,
                "count": len(related)
            }
        )
    except Exception as e:
        print(f"Error finding related entities: {e}")
        return StandardResponse.error(
            message="Failed to find related entities",
            status_code=500,
            details=str(e)
        )

router = APIRouter()

@router.get("/entities/status/metrics")
async def get_entity_status_metrics(request: Request):
    try:
        results = get_entities_with_status_metrics()
        return JSONResponse(content={
            "status": "success",
            "data": results["data"],
            "metadata": results["metadata"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Failed to load status metrics: {str(e)}"
        })

@router.get("/entities/data/metrics")
async def get_entity_data_metrics(request: Request):
    try:
        results = get_entities_with_data_metrics()
        return JSONResponse(content={
            "status": "success",
            "data": results["data"],
            "metadata": results["metadata"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Failed to load data metrics: {str(e)}"
        })

@router.get("/entities/metadata/metrics")
async def get_entity_metadata_metrics(request: Request):
    try:
        results = get_entities_with_metadata_metrics()
        return JSONResponse(content={
            "status": "success",
            "data": results["data"],
            "metadata": results["metadata"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Failed to load metadata metrics: {str(e)}"
        })

# Add this new endpoint
@v1_router.get("/admin/cache/stats")
def get_cache_stats(key_info: dict = Depends(get_api_key)):
    """Get cache statistics."""
    return StandardResponse.success(
        data=cache_manager.get_stats(),
        metadata={
            "timestamp": time.time()
        }
    )

@v1_router.post("/admin/cache/clear")
def clear_cache(key_info: dict = Depends(get_api_key)):
    """Clear the entire cache."""
    count = cache_manager.clear()
    return StandardResponse.success(
        data={"cleared_items": count},
        metadata={
            "timestamp": time.time()
        }
    )

v1_router.include_router(router)    
