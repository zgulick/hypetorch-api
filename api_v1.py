# api_v1.py
from fastapi import APIRouter, Depends, Query, HTTPException, File, UploadFile
from typing import Optional, List
import time
import json

from api_utils import StandardResponse
from auth_middleware import get_api_key
from db_wrapper import get_db_connection, DatabaseConnection
import psycopg2
from db_operations import save_all_data, load_latest_data

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
    history_limit: int = Query(30, ge=1, le=100)
):
    """
    Get detailed information about a specific entity.
    
    Parameters:
    - entity_id: Entity name or ID
    - include_metrics: Include current metrics
    - include_history: Include historical data
    - history_limit: Maximum number of history records
    """
    start_time = time.time()
    
    try:
        # [Same logic as your current get_entity_details but with improved structure]
        entity_name = entity_id.replace("_", " ")
        
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
                    "include_history": include_history
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