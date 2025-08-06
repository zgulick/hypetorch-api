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
    get_metric_history,
    get_latest_hype_scores,
    search_entities_by_category,
    get_trending_entities,
    execute_query
)

from api_utils import StandardResponse
from models_v2 import BulkEntitiesRequest, EntityData, EntityMetrics
from pydantic import BaseModel, Field

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
                current_timestamp = entities_data[entity_name]["last_updated"]
                if not current_timestamp or row["timestamp"] > current_timestamp:
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
        # Get top movers with actual percentage change calculation
        top_movers_query = """
            WITH current_scores AS (
                SELECT 
                    e.id,
                    e.name as entity_name,
                    cm.value as current_value,
                    cm.time_period as current_period
                FROM entities e
                JOIN current_metrics cm ON e.id = cm.entity_id
                WHERE cm.metric_type = 'hype_score'
                    AND e.category = 'Sports'
                    AND cm.value IS NOT NULL
            ),
            previous_scores AS (
                SELECT 
                    e.id,
                    e.name as entity_name,
                    hm.value as previous_value,
                    hm.time_period,
                    ROW_NUMBER() OVER (PARTITION BY e.id ORDER BY hm.timestamp DESC) as rn
                FROM entities e
                JOIN historical_metrics hm ON e.id = hm.entity_id
                WHERE hm.metric_type = 'hype_score'
                    AND e.category = 'Sports'
                    AND hm.value IS NOT NULL
            )
            SELECT 
                c.entity_name,
                c.current_value,
                COALESCE(p.previous_value, 0) as previous_value,
                CASE 
                    WHEN COALESCE(p.previous_value, 0) > 0 
                    THEN ((c.current_value - p.previous_value) / p.previous_value) * 100
                    ELSE 0
                END as percent_change
            FROM current_scores c
            LEFT JOIN previous_scores p ON c.id = p.id AND p.rn = 1
            WHERE c.current_value > 0
            ORDER BY ABS(CASE 
                WHEN COALESCE(p.previous_value, 0) > 0 
                THEN ((c.current_value - p.previous_value) / p.previous_value) * 100
                ELSE 0
            END) DESC
            LIMIT 5
        """
        top_movers_raw = execute_query(top_movers_query)
        
        top_movers = []
        for row in top_movers_raw:
            percent_change = float(row["percent_change"]) if row["percent_change"] else 0
            top_movers.append({
                "name": row["entity_name"],
                "current_score": round(float(row["current_value"]), 1),
                "change": round(percent_change, 1),
                "trend": "up" if percent_change > 0 else "down"
            })
        
        # Get narrative alerts (high RODMN scores)
        narrative_query = """
            SELECT 
                e.name as entity_name,
                cm.value as rodmn_score,
                cm.timestamp,
                cm.time_period
            FROM entities e
            JOIN current_metrics cm ON e.id = cm.entity_id
            WHERE cm.metric_type = 'rodmn_score'
                AND cm.value > 20
                AND e.category = 'Sports'
            ORDER BY cm.value DESC
            LIMIT 5
        """
        narrative_alerts_raw = execute_query(narrative_query)
        
        narrative_alerts = []
        for row in narrative_alerts_raw:
            narrative_alerts.append({
                "name": row["entity_name"],
                "rodmn_score": round(float(row["rodmn_score"]), 1),
                "alert_level": "high" if row["rodmn_score"] > 60 else "medium" if row["rodmn_score"] > 40 else "low",
                "context": f"Controversy discussions detected - RODMN score {round(float(row['rodmn_score']), 1)}"
            })
        
        # Get story opportunities (entities with interesting patterns)
        story_query = """
            SELECT 
                e.name as entity_name,
                MAX(CASE WHEN cm.metric_type = 'hype_score' THEN cm.value END) as hype,
                MAX(CASE WHEN cm.metric_type = 'mentions' THEN cm.value END) as mentions,
                MAX(CASE WHEN cm.metric_type = 'talk_time' THEN cm.value END) as talk_time,
                MAX(cm.timestamp) as last_updated
            FROM entities e
            JOIN current_metrics cm ON e.id = cm.entity_id
            WHERE e.category = 'Sports'
                AND cm.metric_type IN ('hype_score', 'mentions', 'talk_time')
            GROUP BY e.name, e.id
            HAVING MAX(CASE WHEN cm.metric_type = 'hype_score' THEN cm.value END) > 20
                OR MAX(CASE WHEN cm.metric_type = 'mentions' THEN cm.value END) > 5
            ORDER BY MAX(CASE WHEN cm.metric_type = 'hype_score' THEN cm.value END) DESC
            LIMIT 5
        """
        story_opportunities_raw = execute_query(story_query)
        
        story_opportunities = []
        for row in story_opportunities_raw:
            hype_val = float(row["hype"]) if row["hype"] else 0
            mentions_val = int(row["mentions"]) if row["mentions"] else 0  
            talk_time_val = float(row["talk_time"]) if row["talk_time"] else 0
            
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
                "name": row["entity_name"],
                "hype_score": round(hype_val, 1),
                "mentions": mentions_val,
                "talk_time": round(talk_time_val, 1),
                "angle": angle
            })
        
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
        query = """
            SELECT DISTINCT 
                time_period,
                COUNT(DISTINCT entity_id) as entity_count,
                COUNT(*) as metric_count,
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data
            FROM historical_metrics 
            WHERE time_period IS NOT NULL
                AND time_period LIKE 'week_%'
            GROUP BY time_period
            ORDER BY time_period DESC
        """
        
        periods_data = execute_query(query)
        
        # Format periods with labels
        formatted_periods = []
        for row in periods_data:
            # Parse week format: week_2025_07_27
            time_period = row["time_period"]
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
                        "entity_count": row["entity_count"],
                        "metric_count": row["metric_count"],
                        "date_range": {
                            "start": row["earliest_data"].isoformat() if row["earliest_data"] else None,
                            "end": row["latest_data"].isoformat() if row["latest_data"] else None
                        }
                    })
        
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
            message="Failed to retrieve time periods",
            status_code=500,
            details=str(e)
        )