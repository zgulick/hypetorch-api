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
    get_entity_by_id,
    get_current_metrics,
    get_historical_metrics,
    get_entity_current_metrics,
    load_latest_data,
    get_hype_score_history,
    get_entities_by_category,
    get_entities_by_subcategory,
    get_metric_history,
    get_latest_hype_scores,
    search_entities_by_category,
    get_trending_entities,
    execute_query
)

from api_utils import StandardResponse
from models_v2 import BulkEntitiesRequest, EntityData, EntityMetrics
from pydantic import BaseModel, Field
import numpy as np
# from quality_gates import QualityGateManager  # Commented out - not needed for basic API endpoints

logger = logging.getLogger('api_v2')

# Create the v2 router
v2_router = APIRouter(prefix="/v2")

# Helper function to load latest data
def load_data():
    """Load the latest analysis data."""
    try:
        return load_latest_data()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return {}

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

@v2_router.get("/entities/search")
def search_entities_v2(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    key_info: dict = Depends(get_api_key)
):
    """
    Search for entities by name.
    """
    start_time = time.time()
    
    try:
        # Debug logging
        logger.info(f"Search params: q={q}, category={category}, limit={limit}")
        
        # Try with explicit keyword arguments
        results = search_entities_by_category(query=q, category=category, limit=limit)
        
        logger.info(f"Search returned {len(results)} results")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=results,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "query": q,
                "category": category,
                "count": len(results)
            }
        )
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return StandardResponse.error(
            message="Failed to search entities",
            status_code=500,
            details=str(e)
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
        elif subcategory:
            entities = get_entities_by_subcategory(subcategory)
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

@v2_router.get("/verticals")
def get_available_verticals(
    key_info: dict = Depends(get_api_key)
):
    """
    Get list of available verticals (subcategories) with entity counts.
    Used by frontend to dynamically populate vertical selector.

    This endpoint queries the entities table to find all unique subcategories
    and counts how many entities belong to each one.

    Returns:
        StandardResponse with verticals list including:
        - key: The subcategory identifier (e.g., "NBA", "Unrivaled")
        - label: Display label (same as key for now)
        - category: Parent category (e.g., "Sports", "Crypto")
        - entity_count: Number of entities in this subcategory

    Example Response:
        {
            "success": true,
            "data": {
                "verticals": [
                    {
                        "key": "Unrivaled",
                        "label": "Unrivaled",
                        "category": "Sports",
                        "entity_count": 37
                    }
                ],
                "total_verticals": 1
            }
        }
    """
    try:
        # Query to get all subcategories with entity counts
        query = """
            SELECT
                category,
                subcategory,
                COUNT(*) as entity_count
            FROM entities
            WHERE subcategory IS NOT NULL
            GROUP BY category, subcategory
            ORDER BY category, subcategory
        """

        results = execute_query(query)

        # Handle case where no entities exist
        if not results:
            return StandardResponse.success(
                data={
                    "verticals": [],
                    "total_verticals": 0
                }
            )

        # Format results into vertical structure
        verticals = []
        for row in results:
            verticals.append({
                "key": row['subcategory'],
                "label": row['subcategory'],
                "category": row['category'],
                "entity_count": row['entity_count']
            })

        return StandardResponse.success(
            data={
                "verticals": verticals,
                "total_verticals": len(verticals)
            }
        )

    except Exception as e:
        logger.error(f"Error retrieving verticals: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return StandardResponse.error(
            message="Failed to retrieve verticals",
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

@v2_router.get("/entities/{entity_id}")
def get_entity_details_v2(
    entity_id: str,
    include_metrics: bool = True,
    include_history: bool = False,
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics"),
    history_limit: int = Query(30, ge=1, le=100),
    key_info: dict = Depends(get_api_key)
):
    """
    Get detailed information about a specific entity (V2).
    """
    start_time = time.time()
    
    try:
        # Try to get entity by name
        entity = get_entity_by_name(entity_id.replace("_", " "))
        
        # If not found and it's a digit, try by ID
        if not entity and entity_id.isdigit():
            entity = get_entity_by_id(int(entity_id))
        
        if not entity:
            return StandardResponse.error(
                message=f"Entity '{entity_id}' not found",
                status_code=404
            )
        
        # Build response
        response_data = {
            "id": entity['id'],
            "name": entity['name'],
            "type": entity.get('type'),
            "category": entity.get('category'),
            "subcategory": entity.get('subcategory'),
            "metadata": entity.get('metadata', {})
        }
        
        # Add metrics if requested
        if include_metrics:
            requested_metrics = None
            if metrics:
                requested_metrics = [m.strip() for m in metrics.split(",")]
            
            entity_metrics = get_entity_current_metrics(entity['id'], requested_metrics)
            response_data['metrics'] = entity_metrics
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=response_data,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "include_metrics": include_metrics,
                "include_history": include_history
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving entity details: {e}")
        return StandardResponse.error(
            message="Failed to retrieve entity details",
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

# New endpoints for website rebuild

@v2_router.get("/trending")
def get_trending_entities_v2(
    metric: str = Query("hype_score", description="Metric to analyze for trending"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    subcategory: Optional[str] = Query(None, description="Filter by subcategory"),
    time_period: Optional[str] = Query(None, description="Filter by time period"),
    key_info: dict = Depends(get_api_key)
):
    """
    Get trending entities based on recent metric changes.
    Returns entities with biggest percentage changes.
    """
    start_time = time.time()
    
    try:
        trending_data = get_trending_entities(
            metric_type=metric,
            limit=limit,
            time_period=time_period,
            category=category,
            subcategory=subcategory
        )
        
        # Format the response
        formatted_data = []
        for row in trending_data:
            formatted_data.append({
                "name": row["entity_name"],
                "current_value": round(float(row["current_value"]), 2),
                "previous_value": round(float(row["previous_value"]), 2),
                "percent_change": round(float(row["percent_change"]), 1),
                "trend_direction": "up" if row["percent_change"] > 0 else "down",
                "current_timestamp": row["current_timestamp"].isoformat() if row["current_timestamp"] else None,
                "previous_timestamp": row["previous_timestamp"].isoformat() if row["previous_timestamp"] else None
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=formatted_data,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "metric": metric,
                "count": len(formatted_data),
                "filters": {
                    "category": category,
                    "subcategory": subcategory,
                    "time_period": time_period
                }
            }
        )
    except Exception as e:
        logger.error(f"Error getting trending entities: {e}")
        return StandardResponse.error(
            message="Failed to retrieve trending entities",
            status_code=500,
            details=str(e)
        )

@v2_router.get("/metrics/recent")
def get_recent_metrics_v2(
    period: str = Query("current", description="Time period (current, week_2025_07_27, last_7_days)"),
    entities: Optional[str] = Query(None, description="Comma-separated entity names to filter"),
    metrics: Optional[str] = Query(None, description="Comma-separated metrics to include"),
    limit: int = Query(20, ge=1, le=100, description="Maximum entities to return"),
    key_info: dict = Depends(get_api_key)
):
    """
    Get recent metrics data for dashboard widgets and charts.
    """
    start_time = time.time()
    
    try:
        # Parse entity filter
        entity_filter = []
        if entities:
            entity_filter = [name.strip() for name in entities.split(",")]
        
        # Parse metrics filter
        metrics_filter = None
        if metrics:
            metrics_filter = [metric.strip() for metric in metrics.split(",")]
        else:
            # Default metrics for dashboard
            metrics_filter = ["hype_score", "rodmn_score", "mentions", "talk_time", "wikipedia_views", "reddit_mentions", "google_trends", "google_news_mentions"]
        
        # Build query based on period
        if period == "current":
            # Get current metrics
            query = """
                SELECT 
                    e.name as entity_name,
                    cm.metric_type,
                    cm.value,
                    cm.timestamp,
                    cm.time_period
                FROM entities e
                JOIN current_metrics cm ON e.id = cm.entity_id
                WHERE 1=1
            """
            params = []
            
            if entity_filter:
                placeholders = ",".join(["%s"] * len(entity_filter))
                query += f" AND e.name IN ({placeholders})"
                params.extend(entity_filter)
            
            if metrics_filter:
                placeholders = ",".join(["%s"] * len(metrics_filter))
                query += f" AND cm.metric_type IN ({placeholders})"
                params.extend(metrics_filter)
                
            query += " ORDER BY e.name, cm.metric_type"
            
        else:
            # Get historical metrics for specific period
            query = """
                SELECT 
                    e.name as entity_name,
                    hm.metric_type,
                    hm.value,
                    hm.timestamp,
                    hm.time_period
                FROM entities e
                JOIN historical_metrics hm ON e.id = hm.entity_id
                WHERE 1=1
            """
            params = []
            
            if period.startswith("week_"):
                query += " AND hm.time_period = %s"
                params.append(period)
            
            if entity_filter:
                placeholders = ",".join(["%s"] * len(entity_filter))
                query += f" AND e.name IN ({placeholders})"
                params.extend(entity_filter)
            
            if metrics_filter:
                placeholders = ",".join(["%s"] * len(metrics_filter))
                query += f" AND hm.metric_type IN ({placeholders})"
                params.extend(metrics_filter)
                
            query += " ORDER BY e.name, hm.metric_type"
        
        # Execute query
        raw_data = execute_query(query, params)
        
        # Format response by entity
        entities_data = {}
        for row in raw_data:
            entity_name = row["entity_name"]
            if entity_name not in entities_data:
                entities_data[entity_name] = {
                    "name": entity_name,
                    "metrics": {},
                    "time_period": row["time_period"],
                    "last_updated": None
                }
            
            entities_data[entity_name]["metrics"][row["metric_type"]] = round(float(row["value"]), 2)
            
            # Keep track of most recent timestamp
            if row["timestamp"]:
                entities_data[entity_name]["last_updated"] = row["timestamp"].isoformat()
        
        # Convert to list and apply limit
        result = list(entities_data.values())[:limit]
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=result,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "period": period,
                "count": len(result),
                "metrics_included": metrics_filter,
                "entity_filter": entity_filter
            }
        )
    except Exception as e:
        logger.error(f"Error getting recent metrics: {e}")
        return StandardResponse.error(
            message="Failed to retrieve recent metrics",
            status_code=500,
            details=str(e)
        )

@v2_router.get("/dashboard/widgets")
def get_dashboard_widgets_v2(
    key_info: dict = Depends(get_api_key)
):
    """
    Get data for all dashboard widgets in one request.
    Includes top movers, narrative alerts, and story opportunities.
    """
    start_time = time.time()
    
    try:
        # Dynamic query - get latest 2 time periods automatically
        top_movers_query = """
            WITH latest_periods AS (
                SELECT DISTINCT time_period
                FROM historical_metrics 
                WHERE time_period LIKE 'week_2025_%'
                ORDER BY time_period DESC
                LIMIT 2
            ),
            player_data AS (
                SELECT 
                    e.name,
                    hm.time_period,
                    hm.value,
                    ROW_NUMBER() OVER (PARTITION BY e.name ORDER BY hm.time_period DESC) as period_rank
                FROM historical_metrics hm
                JOIN entities e ON hm.entity_id = e.id
                JOIN latest_periods lp ON hm.time_period = lp.time_period
                WHERE hm.metric_type = 'hype_score'
                  AND e.category = 'Sports'
                  AND hm.value IS NOT NULL
            ),
            changes AS (
                SELECT 
                    name,
                    MAX(CASE WHEN period_rank = 1 THEN value END) as current_week,
                    MAX(CASE WHEN period_rank = 2 THEN value END) as previous_week
                FROM player_data
                WHERE period_rank <= 2
                GROUP BY name
            )
            SELECT 
                name as entity_name,
                current_week as current_value,
                COALESCE(previous_week, 0) as previous_value,
                CASE 
                    WHEN previous_week IS NOT NULL AND previous_week > 0
                    THEN ((current_week - previous_week) / previous_week) * 100
                    ELSE 0
                END as percent_change
            FROM changes
            WHERE current_week IS NOT NULL
            ORDER BY ABS(CASE 
                WHEN previous_week IS NOT NULL AND previous_week > 0
                THEN ((current_week - previous_week) / previous_week) * 100
                ELSE 0
            END) DESC
            LIMIT 5
        """
        try:
            top_movers_raw = execute_query(top_movers_query)
            logger.info(f"Top movers query returned {len(top_movers_raw) if top_movers_raw else 0} rows")
        except Exception as e:
            logger.error(f"Top movers execute_query failed: {e}")
            raise
        
        # Check if we got results
        if not top_movers_raw:
            logger.warning("No top movers data found")
        
        top_movers = []
        for row in top_movers_raw:
            try:
                # Defensive parsing - handle both dict and tuple formats
                if isinstance(row, dict):
                    entity_name = row.get("entity_name")
                    current_value = row.get("current_value")
                    percent_change = row.get("percent_change")
                else:
                    # Handle tuple format
                    entity_name = row[0] if len(row) > 0 else None
                    current_value = row[1] if len(row) > 1 else None
                    percent_change = row[3] if len(row) > 3 else None
                
                if not entity_name or current_value is None:
                    continue
                    
                percent_change_val = float(percent_change) if percent_change is not None else 0
                current_value_val = float(current_value) if current_value is not None else 0
                
                top_movers.append({
                    "name": entity_name,
                    "current_score": round(current_value_val, 1),
                    "change": round(percent_change_val, 1),
                    "trend": "up" if percent_change_val > 0 else "down"
                })
            except Exception as e:
                logger.error(f"Error processing top mover row {row}: {e}")
                continue
        
        # Get top 5 RODMN scores from latest period (no minimum threshold)
        narrative_query = """
            WITH latest_period AS (
                SELECT time_period
                FROM historical_metrics 
                WHERE time_period LIKE 'week_2025_%'
                ORDER BY time_period DESC
                LIMIT 1
            ),
            deduplicated_metrics AS (
                SELECT 
                    e.name as entity_name,
                    hm.value as rodmn_score,
                    hm.time_period,
                    ROW_NUMBER() OVER (PARTITION BY e.id ORDER BY hm.timestamp DESC) as row_num
                FROM historical_metrics hm
                JOIN entities e ON hm.entity_id = e.id
                JOIN latest_period lp ON hm.time_period = lp.time_period
                WHERE hm.metric_type = 'rodmn_score'
                    AND e.category = 'Sports'
                    AND hm.value IS NOT NULL
            )
            SELECT 
                entity_name,
                rodmn_score,
                time_period
            FROM deduplicated_metrics
            WHERE row_num = 1
            ORDER BY rodmn_score DESC
            LIMIT 5
        """
        try:
            narrative_alerts_raw = execute_query(narrative_query)
            logger.info(f"Narrative alerts query returned {len(narrative_alerts_raw) if narrative_alerts_raw else 0} rows")
        except Exception as e:
            logger.error(f"Narrative alerts execute_query failed: {e}")
            raise
        
        # Check if we got results
        if not narrative_alerts_raw:
            logger.warning("No narrative alerts data found")
        
        narrative_alerts = []
        for row in narrative_alerts_raw:
            try:
                # Defensive parsing - handle both dict and tuple formats
                if isinstance(row, dict):
                    entity_name = row.get("entity_name")
                    rodmn_score = row.get("rodmn_score")
                else:
                    # Handle tuple format
                    entity_name = row[0] if len(row) > 0 else None
                    rodmn_score = row[1] if len(row) > 1 else None
                
                if not entity_name or rodmn_score is None:
                    continue
                    
                rodmn_score_val = float(rodmn_score) if rodmn_score is not None else 0
                
                narrative_alerts.append({
                    "name": entity_name,
                    "rodmn_score": round(rodmn_score_val, 1),
                    "alert_level": "high" if rodmn_score_val > 30 else "medium" if rodmn_score_val > 15 else "low",
                    "context": f"RODMN score {round(rodmn_score_val, 1)} - {('High' if rodmn_score_val > 30 else 'Medium' if rodmn_score_val > 15 else 'Low')} controversy level"
                })
            except Exception as e:
                logger.error(f"Error processing narrative alert row {row}: {e}")
                continue
        
        # Get story opportunities from latest period
        story_query = """
            WITH latest_period AS (
                SELECT time_period
                FROM historical_metrics 
                WHERE time_period LIKE 'week_2025_%'
                ORDER BY time_period DESC
                LIMIT 1
            )
            SELECT 
                e.name as entity_name,
                MAX(CASE WHEN hm.metric_type = 'hype_score' THEN hm.value END) as hype,
                MAX(CASE WHEN hm.metric_type = 'mentions' THEN hm.value END) as mentions,
                MAX(CASE WHEN hm.metric_type = 'talk_time' THEN hm.value END) as talk_time,
                MAX(hm.timestamp) as last_updated
            FROM historical_metrics hm
            JOIN entities e ON hm.entity_id = e.id
            JOIN latest_period lp ON hm.time_period = lp.time_period
            WHERE e.category = 'Sports'
                AND hm.metric_type IN ('hype_score', 'mentions', 'talk_time')
            GROUP BY e.name
            HAVING MAX(CASE WHEN hm.metric_type = 'hype_score' THEN hm.value END) IS NOT NULL
            ORDER BY MAX(CASE WHEN hm.metric_type = 'hype_score' THEN hm.value END) DESC
            LIMIT 5
        """
        try:
            story_opportunities_raw = execute_query(story_query)
            logger.info(f"Story opportunities query returned {len(story_opportunities_raw) if story_opportunities_raw else 0} rows")
        except Exception as e:
            logger.error(f"Story opportunities execute_query failed: {e}")
            raise
        
        # Check if we got results
        if not story_opportunities_raw:
            logger.warning("No story opportunities data found")
        
        story_opportunities = []
        for row in story_opportunities_raw:
            try:
                # Defensive parsing - handle both dict and tuple formats
                if isinstance(row, dict):
                    entity_name = row.get("entity_name")
                    hype = row.get("hype")
                    mentions = row.get("mentions")
                    talk_time = row.get("talk_time")
                else:
                    # Handle tuple format
                    entity_name = row[0] if len(row) > 0 else None
                    hype = row[1] if len(row) > 1 else None
                    mentions = row[2] if len(row) > 2 else None
                    talk_time = row[3] if len(row) > 3 else None
                
                if not entity_name:
                    continue
                    
                hype_val = float(hype) if hype is not None else 0
                mentions_val = int(mentions) if mentions is not None else 0
                talk_time_val = float(talk_time) if talk_time is not None else 0
                
                # Generate relevant angle based on metrics
                if hype_val > 60:
                    angle = "High engagement - prime for feature coverage"
                elif mentions_val > 50:
                    angle = "Frequently mentioned - trending storyline"
                elif talk_time_val > 10:
                    angle = "Extended discussion time - in-depth story potential"
                else:
                    angle = "Emerging storyline with growth potential"
                    
                story_opportunities.append({
                    "name": entity_name,
                    "hype_score": round(hype_val, 1),
                    "mentions": mentions_val,
                    "talk_time": round(talk_time_val, 1),
                    "angle": angle
                })
            except Exception as e:
                logger.error(f"Error processing story opportunity row {row}: {e}")
                continue
        
        # Format the response
        widgets = {
            "top_movers": top_movers,
            "narrative_alerts": narrative_alerts,
            "story_opportunities": story_opportunities
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=widgets,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "widget_count": 3,
                "data_updated": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error getting dashboard widgets: {e}")
        return StandardResponse.error(
            message="Failed to retrieve dashboard widgets",
            status_code=500,
            details=str(e)
        )

@v2_router.get("/time-periods")
def get_available_time_periods_v2(
    key_info: dict = Depends(get_api_key)
):
    """
    Get available time periods for historical data.
    """
    start_time = time.time()
    
    try:
        # Use your proven working query pattern
        query = """
            SELECT 
                time_period,
                COUNT(DISTINCT entity_id) as entity_count,
                COUNT(DISTINCT metric_type) as metric_count,
                COUNT(*) as total_records,
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data
            FROM historical_metrics 
            WHERE time_period LIKE 'week_2025_%'
            GROUP BY time_period
            ORDER BY time_period DESC
        """
        logger.info(f"Executing time periods query: {query}")
        try:
            periods_data = execute_query(query)
            logger.info(f"Time periods query returned {len(periods_data) if periods_data else 0} rows")
        except Exception as e:
            logger.error(f"Time periods execute_query failed: {e}")
            raise
        
        # Check if we got results
        if not periods_data:
            logger.warning("No time periods found in historical_metrics")
            return StandardResponse.success(
                data=[],
                metadata={
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "period_count": 0,
                    "message": "No historical time periods found"
                }
            )
        
        # Format periods with labels
        formatted_periods = []
        for row in periods_data:
            try:
                # Defensive parsing - handle both dict and tuple formats
                if isinstance(row, dict):
                    time_period = row.get("time_period")
                    entity_count = row.get("entity_count", 0)
                    metric_count = row.get("metric_count", 0)
                    earliest_data = row.get("earliest_data")
                    latest_data = row.get("latest_data")
                else:
                    # Handle tuple format
                    time_period = row[0] if len(row) > 0 else None
                    entity_count = row[1] if len(row) > 1 else 0
                    metric_count = row[2] if len(row) > 2 else 0
                    earliest_data = row[4] if len(row) > 4 else None
                    latest_data = row[5] if len(row) > 5 else None
                
                if not time_period:
                    continue
                    
                # Parse week format: week_2025_07_27
                if time_period.startswith("week_"):
                    parts = time_period.split("_")
                    if len(parts) >= 4:
                        year = parts[1]
                        month = parts[2]
                        day = parts[3]
                        
                        # Format for display
                        month_names = {
                            "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
                            "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
                            "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
                        }
                        
                        display_label = f"Week of {month_names.get(month, month)} {day}, {year}"
                        
                        formatted_periods.append({
                            "time_period": time_period,
                            "display_label": display_label,
                            "entity_count": entity_count,
                            "metric_count": metric_count,
                            "date_range": {
                                "start": earliest_data.isoformat() if earliest_data else None,
                                "end": latest_data.isoformat() if latest_data else None
                            }
                        })
            except Exception as e:
                logger.error(f"Error processing time period row {row}: {e}")
                continue
        
        processing_time = (time.time() - start_time) * 1000
        
        return StandardResponse.success(
            data=formatted_periods,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "period_count": len(formatted_periods)
            }
        )
    except Exception as e:
        logger.error(f"Error getting time periods: {e}")
        return StandardResponse.error(
            error=f"Failed to get time periods: {str(e)}",
            status_code=500
        )




# ===== CONFIDENCE & AI QUALITY ENDPOINTS (V2) ===== 
# COMMENTED OUT: These endpoints are not currently used by the website and cause deployment issues
# due to AI system dependencies. Can be re-enabled later if needed.
# 
# Endpoints that were removed:
# - /confidence/entities - Get AI-enhanced confidence scores  
# - /confidence/distribution - Get confidence distribution stats
# - /entities/by-confidence - Filter entities by confidence threshold
# 
# These can be added back when there's a real need for them.
