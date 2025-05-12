# api_v2.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
from auth_middleware import get_api_key
from database import get_entities, get_connection
from api_utils import StandardResponse
from models_v2 import BulkEntitiesRequest, EntityData, EntityMetrics
from database import (
    get_entities_by_category,
    get_latest_hype_scores,
    get_entity_by_name,
    get_current_metrics
)
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
        
        # Get all entities with current metrics
        entities = get_entities()
        all_metrics = get_current_metrics()
        
        # Group metrics by entity
        metrics_by_entity = {}
        for metric in all_metrics:
            entity_id = metric["entity_id"]
            if entity_id not in metrics_by_entity:
                metrics_by_entity[entity_id] = {}
            metrics_by_entity[entity_id][metric["metric_type"]] = metric["value"]
        
        # Add metrics to entities
        entities_with_metrics = []
        for entity in entities:
            entity_data = dict(entity)
            entity_metrics = metrics_by_entity.get(entity["id"], {})
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
                "total_with_scores": len([e for e in entities_with_metrics if e["metrics"].get("hype_score")]),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=dashboard,
            metadata={
                "processing_time_ms": round(processing_time, 2)
            }
        )
    except Exception as e:
        return StandardResponse.error(
            message="Failed to retrieve dashboard data",
            status_code=500,
            details=str(e)
        )