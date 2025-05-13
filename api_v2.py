# api_v2.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
import logging
from auth_middleware import get_api_key
from database import (
    get_entities,
    get_entity_by_name,
    get_current_metrics,
    get_historical_metrics,
    get_entity_current_metrics, 
    load_latest_data,
    get_hype_score_history,
    get_entities_by_category,
    get_metric_history,
    get_latest_hype_scores
)
from api_utils import StandardResponse
from models_v2 import BulkEntitiesRequest, EntityData, EntityMetrics
from pydantic import BaseModel, Field
from database import (
    initialize_database,
    create_entity,
    get_entity_by_name,
    get_entity_by_id,
    update_entity,
    delete_entity,
    create_relationship,
    get_entity_relationships,
    find_related_entities,
    get_entity_domains,
    search_entities,
    save_metric
)

logger = logging.getLogger('api_v2')

# Create the v2 router
v2_router = APIRouter(prefix="/v2")

# Health check endpoint
@v2_router.get("/health")
def health_check_v2():
    """Check the health of the API v2."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time()
    }

# Auth endpoints
@v2_router.get("/auth/verify")
def verify_api_key(key_info: dict = Depends(get_api_key)):
    """Verify API key is valid and return basic info."""
    return StandardResponse.success(
        data={
            "valid": True,
            "client_name": key_info.get("client_name"),
            "created_at": key_info.get("created_at"),
            "token_balance": key_info.get("token_balance", 0)
        },
        metadata={
            "timestamp": time.time()
        }
    )

# Entity endpoints
@v2_router.get("/entities")
def get_all_entities(
    include_metrics: bool = Query(False, description="Include current metrics"),
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    key_info: dict = Depends(get_api_key)
):
    """Get all entities with optional filtering and metrics."""
    start_time = time.time()
    
    try:
        # Get entities based on filters
        if category:
            entities = get_entities_by_category(category, subcategory)
        else:
            entities = get_entities()
        
        # If metrics requested, get them efficiently
        if include_metrics:
            # Get all current metrics in one query
            all_metrics = get_current_metrics()
            
            # Group metrics by entity_id
            metrics_by_entity = {}
            for metric in all_metrics:
                entity_id = metric["entity_id"]
                if entity_id not in metrics_by_entity:
                    metrics_by_entity[entity_id] = {}
                metrics_by_entity[entity_id][metric["metric_type"]] = metric["value"]
            
            # Add metrics to entities
            for entity in entities:
                entity_metrics = metrics_by_entity.get(entity["id"], {})
                entity["metrics"] = {
                    "hype_score": entity_metrics.get("hype_score"),
                    "rodmn_score": entity_metrics.get("rodmn_score"),
                    "talk_time": entity_metrics.get("talk_time"),
                    "mentions": entity_metrics.get("mentions"),
                    "wikipedia_views": entity_metrics.get("wikipedia_views"),
                    "reddit_mentions": entity_metrics.get("reddit_mentions"),
                    "google_trends": entity_metrics.get("google_trends")
                }
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=entities,
            metadata={
                "count": len(entities),
                "processing_time_ms": round(processing_time, 2),
                "filters": {
                    "category": category,
                    "subcategory": subcategory,
                    "include_metrics": include_metrics
                }
            }
        )
    except Exception as e:
        return StandardResponse.error(
            message="Failed to retrieve entities",
            status_code=500,
            details=str(e)
        )

# Analytics endpoints
@v2_router.get("/analytics/dashboard")
def get_dashboard_data(key_info: dict = Depends(get_api_key)):
    """Get pre-configured dashboard data - everything needed for the main view."""
    start_time = time.time()
    
    try:
        # Get top entities by HYPE score
        top_hype = get_latest_hype_scores(limit=10)
        
        # Get all entities
        entities = get_entities()
        
        # Get ALL current metrics in one query
        all_metrics = get_current_metrics()
        
        # Debug print to see what we're getting
        print(f"Found {len(all_metrics)} metrics")
        if all_metrics:
            print(f"Sample metric: {all_metrics[0]}")
        
        # Group metrics by entity_id
        metrics_by_entity = {}
        for metric in all_metrics:
            entity_id = metric["entity_id"]
            metric_type = metric["metric_type"]
            value = metric["value"]
            
            if entity_id not in metrics_by_entity:
                metrics_by_entity[entity_id] = {}
            
            metrics_by_entity[entity_id][metric_type] = value
        
        # Debug print
        print(f"Metrics grouped for {len(metrics_by_entity)} entities")
        
        # Add metrics to entities
        entities_with_metrics = []
        for entity in entities:
            entity_data = dict(entity)
            entity_id = entity["id"]
            
            # Get metrics for this entity
            entity_metrics = metrics_by_entity.get(entity_id, {})
            
            # Debug print for first few entities
            if len(entities_with_metrics) < 3:
                print(f"Entity {entity['name']} (ID: {entity_id}) has metrics: {entity_metrics}")
            
            entity_data["metrics"] = {
                "hype_score": entity_metrics.get("hype_score"),
                "rodmn_score": entity_metrics.get("rodmn_score"),
                "talk_time": entity_metrics.get("talk_time"),
                "mentions": entity_metrics.get("mentions")
            }
            entities_with_metrics.append(entity_data)
        
        # Build dashboard response
        dashboard = {
            "all_entities": entities_with_metrics,
            "top_hype": [{"name": item["name"], "score": item["score"]} for item in top_hype],
            "summary": {
                "total_entities": len(entities),
                "total_with_scores": len([e for e in entities_with_metrics if e["metrics"].get("hype_score") is not None]),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=dashboard,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "debug": {
                    "metrics_found": len(all_metrics),
                    "entities_found": len(entities),
                    "metrics_by_entity_count": len(metrics_by_entity)
                }
            }
        )
    except Exception as e:
        import traceback
        print(f"Error in dashboard: {traceback.format_exc()}")
        return StandardResponse.error(
            message="Failed to retrieve dashboard data",
            status_code=500,
            details=str(e)
        )
# Add this class definition before the bulk_entities_v2 function
class BulkEntityQueryV2(BaseModel):
    """Request model for V2 bulk entity query."""
    entity_names: List[str] = Field(..., description="List of entity names to query")
    metrics: List[str] = Field(
        default=["hype_score", "rodmn_score", "mentions", "talk_time"],
        description="Metrics to retrieve"
    )
    include_history: bool = Field(default=False, description="Include historical data")
    history_limit: int = Field(default=30, ge=1, le=100, description="Max history points per metric")

@v2_router.post("/entities/bulk")
def bulk_entities_v2(
    query: BulkEntityQueryV2,
    key_info: dict = Depends(get_api_key)
):
    """
    Bulk fetch data for multiple entities in one request (V2).
    Uses the new database layer for better performance.
    """
    start_time = time.time()
    
    try:
        results = []
        
        for entity_name in query.entity_names:
            # Get entity from database
            entity = get_entity_by_name(entity_name)
            
            if not entity:
                results.append({
                    "name": entity_name,
                    "error": "Entity not found"
                })
                continue
            
            # Get current metrics
            entity_metrics = get_entity_current_metrics(entity['id'], query.metrics)
            
            # Build entity result
            entity_result = {
                "id": entity['id'],
                "name": entity['name'],
                "type": entity.get('type'),
                "category": entity.get('category'),
                "subcategory": entity.get('subcategory'),
                "metadata": entity.get('metadata', {}),
                "metrics": entity_metrics
            }
            
            # Add history if requested
            if query.include_history:
                history = {}
                for metric in query.metrics:
                    # Use the function we just added
                    metric_history = get_metric_history(
                        entity['id'], 
                        metric,
                        query.history_limit
                    )
                    if metric_history:
                        history[metric] = metric_history
                
                entity_result["history"] = history
            
            results.append(entity_result)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=results,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "entity_count": len(results),
                "metrics_included": query.metrics
            }
        )
    except Exception as e:
        logger.error(f"Error in bulk query v2: {e}")
        return StandardResponse.error(
            message="Failed to retrieve bulk entity data",
            status_code=500,
            details=str(e)
        )
        
@v2_router.post("/metrics/compare")
def compare_entities(
    request: BulkEntitiesRequest,
    key_info: dict = Depends(get_api_key)
):
    """Compare multiple entities across requested metrics."""
    start_time = time.time()
    
    try:
        # Get latest data
        latest_data = load_latest_data()
        
        # Build comparison data
        comparison = {
            "entities": request.entity_names,
            "metrics": {}
        }
        
        # If no metrics specified, use defaults
        metrics_to_compare = request.metrics or ["hype_score", "rodmn_score", "mentions", "talk_time"]
        
        # For each metric, get values for all entities
        for metric in metrics_to_compare:
            comparison["metrics"][metric] = {}
            
            for entity_name in request.entity_names:
                if metric == "hype_score":
                    value = latest_data.get("hype_scores", {}).get(entity_name, 0)
                elif metric == "rodmn_score":
                    value = latest_data.get("rodmn_scores", {}).get(entity_name, 0)
                elif metric == "talk_time":
                    value = latest_data.get("talk_time_counts", {}).get(entity_name, 0)
                elif metric == "mentions":
                    value = latest_data.get("mention_counts", {}).get(entity_name, 0)
                elif metric == "wikipedia_views":
                    value = latest_data.get("wikipedia_views", {}).get(entity_name, 0)
                elif metric == "reddit_mentions":
                    value = latest_data.get("reddit_mentions", {}).get(entity_name, 0)
                elif metric == "google_trends":
                    value = latest_data.get("google_trends", {}).get(entity_name, 0)
                else:
                    value = 0
                    
                comparison["metrics"][metric][entity_name] = value
        
        # Add rankings for each metric
        comparison["rankings"] = {}
        for metric, values in comparison["metrics"].items():
            # Sort entities by this metric
            sorted_entities = sorted(values.items(), key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = [
                {"rank": i+1, "entity": name, "value": value}
                for i, (name, value) in enumerate(sorted_entities)
            ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=comparison,
            metadata={
                "entity_count": len(request.entity_names),
                "metric_count": len(metrics_to_compare),
                "processing_time_ms": round(processing_time, 2)
            }
        )
    except Exception as e:
        return StandardResponse.error(
            message="Failed to compare entities",
            status_code=500,
            details=str(e)
        )    