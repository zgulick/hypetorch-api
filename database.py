# database.py - Consolidated database layer for HypeTorch
import os
import json
import time
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hypetorch_database.log', mode='a')
    ]
)
logger = logging.getLogger('database')

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")
DB_SCHEMA = os.environ.get("DB_SCHEMA", None)  # Customer-specific schema
CONNECTION_TIMEOUT = 30  # seconds

# Cache for database connections
_connection_cache = {}
_last_connection_time = 0

def get_connection(schema_name=None):
    """Get a database connection with the appropriate schema."""
    global _connection_cache, _last_connection_time
    
    # Determine which schema to use
    target_schema = schema_name or DB_SCHEMA or DB_ENVIRONMENT
    
    # Check if we have a cached connection and it's recent
    current_time = time.time()
    cache_key = f"conn_{target_schema}"
    
    if cache_key in _connection_cache and current_time - _last_connection_time < 60:  # 1 minute timeout
        try:
            # Test if connection is still alive
            conn = _connection_cache.get(cache_key)
            if conn and not conn.closed:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
        except Exception:
            # Connection is dead, create a new one
            pass
    
    # Create a new connection
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=CONNECTION_TIMEOUT)
    
    # Set the search path to the appropriate schema
    with conn.cursor() as cursor:
        # Create schema if it doesn't exist (only for non-customer schemas)
        if not target_schema.startswith('customer_'):
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")
        # Set search path to our schema
        cursor.execute(f"SET search_path TO {target_schema}")
    
    conn.commit()
    
    # Cache the connection
    _connection_cache[cache_key] = conn
    _last_connection_time = current_time
    
    return conn

def close_connections():
    """Close any open database connections."""
    global _connection_cache
    
    if 'conn' in _connection_cache:
        try:
            _connection_cache['conn'].close()
        except Exception:
            pass
        
    _connection_cache = {}

def execute_query(query, params=None, fetch=True, schema_name=None):
    """
    Execute a database query and optionally return results.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query
        fetch: Whether to fetch and return results
        schema_name: Optional schema to use for this query
    
    Returns:
        Query results if fetch=True, otherwise None
    """
    conn = None
    try:
        conn = get_connection(schema_name)
        with conn.cursor() as cursor:
            # Set search path for schema (redundant but safe)
            target_schema = schema_name or DB_SCHEMA or DB_ENVIRONMENT
            cursor.execute(f"SET search_path TO {target_schema};")
            
            # Log query details for debugging
            logger.info(f"Executing query in schema '{target_schema}' with params: {params}")
            
            # Handle params safely
            if params is None:
                cursor.execute(query)
            else:
                if not isinstance(params, (tuple, list)):
                    params = (params,)
                cursor.execute(query, params)
            if fetch:
                rows = cursor.fetchall()
                logger.info(f"Query returned {len(rows)} rows")
                
                # Convert to list of dicts manually
                result = []
                if rows and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        result.append(dict(zip(columns, row)))
                
                # Commit if this was an INSERT/UPDATE/DELETE with RETURNING
                if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')) and 'RETURNING' in query.upper():
                    conn.commit()
            else:
                result = None
                conn.commit()
            return result
    except Exception as e:
        logger.error(f"Database error: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if conn:
            conn.rollback()
        raise e

def execute_transaction(queries):
    """
    Execute multiple queries in a single transaction.
    
    Args:
        queries: List of (query, params) tuples
    
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Set search path for schema
            if DB_ENVIRONMENT:
                cursor.execute(f"SET search_path TO {DB_ENVIRONMENT};")
            
            for query, params in queries:
                cursor.execute(query, params or ())
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Transaction error: {e}")
        if conn:
            conn.rollback()
        return False

def initialize_database():
    """Create necessary tables if they don't exist."""
    try:
        logger.info("Initializing database...")
        
        # Read the schema SQL file
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute the schema SQL
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(schema_sql)
        
        conn.commit()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

# Entity operations
def get_entities(category=None, subcategory=None, schema_name=None):
    """
    Get a list of entities, optionally filtered by category and subcategory.
    
    Args:
        category: Optional category filter
        subcategory: Optional subcategory filter
        schema_name: Optional schema to query
    
    Returns:
        List of entity dictionaries
    """
    query = "SELECT id, name, type, category, subcategory FROM entities"
    params = []
    
    # Add filters if provided
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
    
    query += " ORDER BY name"
    
    return execute_query(query, params, schema_name=schema_name)

def get_entity_current_metrics(entity_id, metric_types=None):
    """Get current metrics for an entity from the current_metrics table."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if metric_types:
            # Get specific metrics
            placeholders = ', '.join(['%s'] * len(metric_types))
            cursor.execute(f"""
                SELECT metric_type, value
                FROM current_metrics
                WHERE entity_id = %s AND metric_type IN ({placeholders})
            """, [entity_id] + metric_types)
        else:
            # Get all metrics
            cursor.execute("""
                SELECT metric_type, value
                FROM current_metrics
                WHERE entity_id = %s
            """, (entity_id,))
        
        results = cursor.fetchall()
        
        # Convert to dictionary
        metrics = {row['metric_type']: row['value'] for row in results}
        
        cursor.close()
        conn.close()
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        return {}

def get_metric_history(entity_id, metric_type, limit=30):
    """Get historical values for a specific metric."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT value, timestamp
            FROM historical_metrics
            WHERE entity_id = %s AND metric_type = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (entity_id, metric_type, limit))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format for API response
        return [
            {
                "value": row['value'],
                "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
            }
            for row in results
        ]
    except Exception as e:
        logger.error(f"Error getting metric history: {e}")
        return []

def delete_entity(entity_id):
    """Delete an entity and all its related data."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Delete related data first (foreign key constraints)
        cursor.execute("DELETE FROM historical_metrics WHERE entity_id = %s", (entity_id,))
        cursor.execute("DELETE FROM current_metrics WHERE entity_id = %s", (entity_id,))
        cursor.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, "Entity deleted successfully"
    except Exception as e:
        logger.error(f"Error deleting entity: {e}")
        if conn:
            conn.rollback()
        return False, str(e)

def search_entities_by_category(query, category=None, limit=20):
    """Search for entities by name."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if category:
            cursor.execute("""
                SELECT * FROM entities 
                WHERE name ILIKE %s AND category = %s
                ORDER BY name 
                LIMIT %s
            """, (f"%{query}%", category, limit))
        else:
            cursor.execute("""
                SELECT * FROM entities 
                WHERE name ILIKE %s
                ORDER BY name 
                LIMIT %s
            """, (f"%{query}%", limit))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return []

def get_entity_domains():
    """Get all unique domains/categories."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT category as domain 
            FROM entities 
            WHERE category IS NOT NULL
            ORDER BY category
        """)
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [row['domain'] for row in results]
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        return []

def get_entity_by_id(entity_id):
    """Get entity by database ID."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
        result = cursor.fetchone()
        
        cursor.close() 
        conn.close()
        
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Error getting entity by ID: {e}")
        return None

def create_relationship(source_id, target_id, relationship_type, metadata=None):
    """Stub function - relationships not implemented yet."""
    return False, "Relationships feature not implemented"

def get_entity_relationships(entity_id):
    """Stub function - relationships not implemented yet."""
    return []

def find_related_entities(entity_id, relationship_types=None, limit=10):
    """Stub function - relationships not implemented yet."""
    return []

def get_entity_by_name(name, schema_name=None):
    """
    Get entity details by name.
    
    Args:
        name: Entity name
        schema_name: Optional schema to query
    
    Returns:
        Entity dictionary or None if not found
    """
    # Try exact match first
    query = "SELECT id, name, type, category, subcategory FROM entities WHERE name = %s"
    result = execute_query(query, (name,), schema_name=schema_name)
    
    if result:
        return result[0]
    
    # Try case-insensitive match
    query = "SELECT id, name, type, category, subcategory FROM entities WHERE LOWER(name) = LOWER(%s)"
    result = execute_query(query, (name,), schema_name=schema_name)
    
    if result:
        return result[0]
    
    # Try partial match as last resort
    query = "SELECT id, name, type, category, subcategory FROM entities WHERE name ILIKE %s LIMIT 1"
    result = execute_query(query, (f"%{name}%",), schema_name=schema_name)
    
    if result:
        return result[0]
    
    return None

def get_entity_by_id(entity_id):
    """
    Get entity details by ID.
    
    Args:
        entity_id: Entity ID
    
    Returns:
        Entity dictionary or None if not found
    """
    try:
        query = "SELECT id, name, type, category, subcategory FROM entities WHERE id = %s"
        result = execute_query(query, (entity_id,))
        
        if result:
            return result[0]
        
        return None
    except Exception as e:
        logger.error(f"Error getting entity by ID: {e}")
        return None
    
def get_entities_by_category(category, subcategory=None):
    """
    Get entities filtered by category and optional subcategory.
    
    Args:
        category: Category to filter by
        subcategory: Optional subcategory filter
    
    Returns:
        List of entity dictionaries
    """
    try:
        if subcategory:
            query = """
                SELECT id, name, type, category, subcategory 
                FROM entities 
                WHERE category = %s AND subcategory = %s
                ORDER BY name
            """
            result = execute_query(query, (category, subcategory))
        else:
            query = """
                SELECT id, name, type, category, subcategory 
                FROM entities 
                WHERE category = %s
                ORDER BY name
            """
            result = execute_query(query, (category,))
        
        return result
    except Exception as e:
        logger.error(f"Error getting entities by category: {e}")
        return []

def search_entities(search_term, limit=10):
    """
    Search for entities by name, category, or subcategory.
    
    Args:
        search_term: Term to search for
        limit: Maximum number of results
    
    Returns:
        List of entity dictionaries
    """
    try:
        query = """
            SELECT id, name, type, category, subcategory 
            FROM entities 
            WHERE name ILIKE %s OR category ILIKE %s OR subcategory ILIKE %s
            ORDER BY 
                CASE WHEN name ILIKE %s THEN 1
                     WHEN name ILIKE %s THEN 2
                     ELSE 3
                END,
                name
            LIMIT %s
        """
        search_pattern = f"%{search_term}%"
        exact_pattern = f"{search_term}"
        start_pattern = f"{search_term}%"
        
        params = (search_pattern, search_pattern, search_pattern, 
                  exact_pattern, start_pattern, limit)
        
        result = execute_query(query, params)
        return result
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return []        

def get_latest_hype_scores(limit=None, category=None, subcategory=None):
    """
    Get the latest HYPE scores, optionally filtered and limited.
    
    Args:
        limit: Maximum number of scores to return
        category: Optional category filter
        subcategory: Optional subcategory filter
    
    Returns:
        List of dictionaries with entity name and HYPE score
    """
    try:
        # Base query
        query = """
            SELECT DISTINCT e.name, h.score, h.timestamp
            FROM entities e
            JOIN (
                SELECT entity_id, MAX(timestamp) as max_timestamp
                FROM hype_scores
                GROUP BY entity_id
            ) latest ON e.id = latest.entity_id
            JOIN hype_scores h ON e.id = h.entity_id AND latest.max_timestamp = h.timestamp
            WHERE 1=1
        """
        params = []
        
        # Add filters
        if category:
            query += " AND e.category = %s"
            params.append(category)
            
        if subcategory:
            query += " AND e.subcategory = %s"
            params.append(subcategory)
        
        # Add ordering and limit
        query += " ORDER BY h.score DESC"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        result = execute_query(query, params)
        return result
    except Exception as e:
        logger.error(f"Error getting latest HYPE scores: {e}")
        return []

def get_hype_score_history(entity_id, limit=30):
    """Get historical HYPE scores with run numbers for duplicate dates."""
    try:
        query = """
            SELECT
                entity_id,
                score,
                timestamp,
                time_period,
                ROW_NUMBER() OVER (PARTITION BY DATE(timestamp) ORDER BY timestamp DESC) as run_number
            FROM hype_scores
            WHERE entity_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        
        result = execute_query(query, (entity_id, limit))
        return result
    except Exception as e:
        logger.error(f"Error getting HYPE score history: {e}")
        return []

def create_entity(entity_data, schema_name=None):
    """
    Create a new entity.
    
    Args:
        entity_data: Dictionary with entity details
        schema_name: Optional schema to create entity in
    
    Returns:
        (success, message) tuple
    """
    try:
        name = entity_data.get("name", "").strip()
        if not name:
            return False, "Entity name is required"
        
        entity_type = entity_data.get("type", "person")
        category = entity_data.get("category", "Sports")
        subcategory = entity_data.get("subcategory", "Unrivaled")
        metadata = json.dumps(entity_data.get("metadata", {}))
        
        # Check if entity already exists
        existing = get_entity_by_name(name, schema_name=schema_name)
        if existing:
            return False, f"Entity '{name}' already exists"
        
        # Insert new entity
        query = """
            INSERT INTO entities (name, type, category, subcategory, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        result = execute_query(query, (name, entity_type, category, subcategory, metadata), fetch=True, schema_name=schema_name)
        
        if result:
            return True, f"Entity '{name}' created successfully"
        else:
            return False, "Failed to create entity"
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        return False, str(e)

def update_entity(entity_name, entity_data):
    """
    Update an existing entity.
    
    Args:
        entity_name: Current entity name
        entity_data: Dictionary with updated entity details
    
    Returns:
        (success, message) tuple
    """
    try:
        # Get the entity ID
        entity = get_entity_by_name(entity_name)
        if not entity:
            return False, f"Entity '{entity_name}' not found"
        
        entity_id = entity["id"]
        
        # Extract updated values
        updates = []
        params = []
        
        if "name" in entity_data:
            # Check if new name already exists (but isn't the current entity)
            if entity_data["name"] != entity_name:
                existing = get_entity_by_name(entity_data["name"])
                if existing and existing["id"] != entity_id:
                    return False, f"Entity with name '{entity_data['name']}' already exists"
            
            updates.append("name = %s")
            params.append(entity_data["name"])
        
        if "type" in entity_data:
            updates.append("type = %s")
            params.append(entity_data["type"])
        
        if "category" in entity_data:
            updates.append("category = %s")
            params.append(entity_data["category"])
        
        if "subcategory" in entity_data:
            updates.append("subcategory = %s")
            params.append(entity_data["subcategory"])
        
        if "metadata" in entity_data:
            updates.append("metadata = %s")
            params.append(json.dumps(entity_data["metadata"]))
        
        # Add updated_at timestamp
        updates.append("updated_at = CURRENT_TIMESTAMP")
        
        if not updates:
            return False, "No updates provided"
        
        # Build the query
        query = f"UPDATE entities SET {', '.join(updates)} WHERE id = %s"
        params.append(entity_id)
        
        # Execute the update
        execute_query(query, params, fetch=False)
        
        # Update entity_name references in historical_metrics if name changed
        if "name" in entity_data and entity_data["name"] != entity_name:
            # Update references in historical_metrics
            execute_query(
                "UPDATE historical_metrics SET entity_id = %s WHERE entity_id = %s",
                (entity_id, entity_id),
                fetch=False
            )
            
            # Update references in current_metrics
            execute_query(
                "UPDATE current_metrics SET entity_id = %s WHERE entity_id = %s",
                (entity_id, entity_id),
                fetch=False
            )
        
        return True, f"Entity updated successfully"
    except Exception as e:
        logger.error(f"Error updating entity: {e}")
        return False, str(e)

def delete_entity(entity_name):
    """
    Delete an entity.
    
    Args:
        entity_name: Name of entity to delete
    
    Returns:
        (success, message) tuple
    """
    try:
        # Get the entity ID
        entity = get_entity_by_name(entity_name)
        if not entity:
            return False, f"Entity '{entity_name}' not found"
        
        entity_id = entity["id"]
        
        # Start transaction
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Delete from historical_metrics
                cursor.execute("DELETE FROM historical_metrics WHERE entity_id = %s", (entity_id,))
                
                # Delete from current_metrics
                cursor.execute("DELETE FROM current_metrics WHERE entity_id = %s", (entity_id,))
                
                # Delete the entity
                cursor.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
                
                conn.commit()
                return True, f"Entity '{entity_name}' deleted successfully"
        except Exception as e:
            conn.rollback()
            raise e
    except Exception as e:
        logger.error(f"Error deleting entity: {e}")
        return False, str(e)

# Metric operations
def save_metric(entity_id, metric_type, value, time_period=None, is_historical=False, schema_name=None):
    """
    Save a metric value.
    
    Args:
        entity_id: Entity ID
        metric_type: Type of metric (e.g., 'hype_score', 'rodmn_score')
        value: Metric value
        time_period: Optional time period (e.g., 'last_30_days')
        is_historical: Whether to save as historical data
        schema_name: Optional schema to save to
    
    Returns:
        Success boolean
    """
    try:
        if is_historical:
            # Save to historical_metrics
            query = """
                INSERT INTO historical_metrics (entity_id, metric_type, value, time_period)
                VALUES (%s, %s, %s, %s)
            """
            execute_query(query, (entity_id, metric_type, value, time_period), fetch=False, schema_name=schema_name)
        else:
            # Save to current_metrics (upsert)
            query = """
                INSERT INTO current_metrics (entity_id, metric_type, value, time_period)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (entity_id, metric_type) DO UPDATE
                SET value = EXCLUDED.value,
                    time_period = EXCLUDED.time_period,
                    timestamp = CURRENT_TIMESTAMP
            """
            execute_query(query, (entity_id, metric_type, value, time_period), fetch=False, schema_name=schema_name)
        
        return True
    except Exception as e:
        logger.error(f"Error saving metric: {e}")
        return False

def get_current_metrics(entity_id=None, metric_types=None, schema_name=None):
    """
    Get current metrics, optionally filtered by entity and metric types.
    
    Args:
        entity_id: Optional entity ID filter
        metric_types: Optional list of metric types
        schema_name: Optional schema to query
    
    Returns:
        List of metric dictionaries
    """
    query = """
        SELECT cm.id, cm.entity_id, e.name as entity_name, cm.metric_type, cm.value, 
               cm.timestamp, cm.time_period
        FROM current_metrics cm
        JOIN entities e ON cm.entity_id = e.id
    """
    params = []
    
    # Add filters if provided
    conditions = []
    if entity_id:
        conditions.append("cm.entity_id = %s")
        params.append(entity_id)
    
    if metric_types:
        placeholders = ', '.join(['%s'] * len(metric_types))
        conditions.append(f"cm.metric_type IN ({placeholders})")
        params.extend(metric_types)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    return execute_query(query, params, schema_name=schema_name)

def get_historical_metrics(entity_id, metric_type, limit=30, start_date=None, end_date=None):
    """
    Get historical metrics for an entity.
    
    Args:
        entity_id: Entity ID
        metric_type: Metric type (e.g., 'hype_score')
        limit: Maximum number of records to return
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
    
    Returns:
        List of historical metric dictionaries
    """
    query = """
        SELECT hm.id, hm.entity_id, e.name as entity_name, hm.metric_type, 
               hm.value, hm.timestamp, hm.time_period
        FROM historical_metrics hm
        JOIN entities e ON hm.entity_id = e.id
        WHERE hm.entity_id = %s AND hm.metric_type = %s
    """
    params = [entity_id, metric_type]
    
    # Add date filters if provided
    if start_date:
        query += " AND hm.timestamp >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND hm.timestamp <= %s"
        params.append(end_date)
    
    query += " ORDER BY hm.timestamp DESC"
    
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    
    return execute_query(query, params)

def get_trending_entities(metric_type="hype_score", limit=10, time_period=None, category=None, subcategory=None):
    """
    Get trending entities based on recent metric changes.
    
    Args:
        metric_type: Metric to analyze
        limit: Maximum number of entities to return
        time_period: Filter by time period
        category: Filter by entity category
        subcategory: Filter by entity subcategory
    
    Returns:
        List of trending entities with change data
    """
    # This is a complex query that compares recent metrics with previous ones
    query = """
        WITH latest AS (
            SELECT 
                e.id as entity_id,
                e.name as entity_name,
                cm.value as current_value,
                cm.timestamp as current_timestamp
            FROM 
                entities e
            JOIN 
                current_metrics cm ON e.id = cm.entity_id
            WHERE 
                cm.metric_type = %s
    """
    params = [metric_type]
    
    # Add filters
    if time_period:
        query += " AND cm.time_period = %s"
        params.append(time_period)
    
    if category:
        query += " AND e.category = %s"
        params.append(category)
    
    if subcategory:
        query += " AND e.subcategory = %s"
        params.append(subcategory)
    
    # Complete the query
    query += """
        ),
        previous AS (
            SELECT 
                e.id as entity_id,
                MAX(hm.timestamp) as prev_timestamp
            FROM 
                entities e
            JOIN 
                historical_metrics hm ON e.id = hm.entity_id
            JOIN 
                latest l ON e.id = l.entity_id AND hm.timestamp < l.current_timestamp
            WHERE 
                hm.metric_type = %s
    """
    params.append(metric_type)
    
    # Add same filters for consistency
    if time_period:
        query += " AND hm.time_period = %s"
        params.append(time_period)
    
    # Complete the query
    query += """
            GROUP BY e.id
        )
        SELECT 
            l.entity_name,
            l.current_value,
            l.current_timestamp,
            hm.value as previous_value,
            hm.timestamp as previous_timestamp,
            CASE 
                WHEN hm.value = 0 THEN 100
                ELSE ((l.current_value - hm.value) / ABS(hm.value)) * 100 
            END as percent_change
        FROM 
            latest l
        JOIN 
            previous p ON l.entity_id = p.entity_id
        JOIN 
            historical_metrics hm ON p.entity_id = hm.entity_id 
                                 AND p.prev_timestamp = hm.timestamp
                                 AND hm.metric_type = %s
        ORDER BY 
            percent_change DESC
        LIMIT %s
    """
    params.extend([metric_type, limit])
    
    return execute_query(query, params)

# Comprehensive data operations
def save_all_data(data):
    """
    Save complete data set including all metrics.
    
    Args:
        data: Dictionary containing all metrics
    
    Returns:
        (success, message) tuple
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN")
        
        try:
            # Process hype scores
            if "hype_scores" in data:
                for entity_name, score in data["hype_scores"].items():
                    # Get or create entity
                    entity = get_entity_by_name(entity_name)
                    if not entity:
                        # Create new entity
                        entity_type = "non-person" if entity_name.upper() == entity_name else "person"
                        cursor.execute(
                            """
                            INSERT INTO entities (name, type, category, subcategory)
                            VALUES (%s, %s, %s, %s)
                            RETURNING id
                            """,
                            (entity_name, entity_type, "Sports", "Unrivaled")
                        )
                        entity_id = cursor.fetchone()[0]
                    else:
                        entity_id = entity["id"]
                    
                    # Save current hype score
                    cursor.execute(
                        """
                        INSERT INTO current_metrics (entity_id, metric_type, value, time_period)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (entity_id, metric_type) DO UPDATE
                        SET value = EXCLUDED.value,
                            time_period = EXCLUDED.time_period,
                            timestamp = CURRENT_TIMESTAMP
                        """,
                        (entity_id, "hype_score", score, "last_30_days")
                    )
                    
                    # Save historical hype score
                    cursor.execute(
                        """
                        INSERT INTO historical_metrics (entity_id, metric_type, value, time_period)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (entity_id, "hype_score", score, "last_30_days")
                    )
            
            # Process all other metrics
            metric_mappings = {
                "rodmn_scores": "rodmn_score",
                "mention_counts": "mentions",
                "talk_time_counts": "talk_time",
                "wikipedia_views": "wikipedia_views",
                "reddit_mentions": "reddit_mentions",
                "google_trends": "google_trends",
                "google_news_mentions": "google_news_mentions"
            }
            
            # Process each metric type
            for data_key, metric_type in metric_mappings.items():
                if data_key in data:
                    for entity_name, value in data[data_key].items():
                        # Get entity ID
                        entity = get_entity_by_name(entity_name)
                        if not entity:
                            continue  # Skip if entity doesn't exist
                        
                        entity_id = entity["id"]
                        
                        # Save current metric
                        cursor.execute(
                            """
                            INSERT INTO current_metrics (entity_id, metric_type, value, time_period)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (entity_id, metric_type) DO UPDATE
                            SET value = EXCLUDED.value,
                                time_period = EXCLUDED.time_period,
                                timestamp = CURRENT_TIMESTAMP
                            """,
                            (entity_id, metric_type, value, "last_30_days")
                        )
                        
                        # Save historical metric
                        cursor.execute(
                            """
                            INSERT INTO historical_metrics (entity_id, metric_type, value, time_period)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (entity_id, metric_type, value, "last_30_days")
                        )
            
            # Commit the transaction
            conn.commit()
            return True, "Data saved successfully"
            
        except Exception as e:
            # Rollback on error
            conn.rollback()
            raise e
            
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return False, str(e)
        
    finally:
        # Close cursor and connection
        if 'cursor' in locals():
            cursor.close()

def load_latest_data():
    """
    Load the latest data, including all metrics.
    
    Returns:
        Dictionary with all metrics
    """
    try:
        # Get all entities
        entities = get_entities()
        
        # Initialize result structure
        result = {
            "hype_scores": {},
            "rodmn_scores": {},
            "mention_counts": {},
            "talk_time_counts": {},
            "wikipedia_views": {},
            "reddit_mentions": {},
            "google_trends": {},
            "google_news_mentions": {},
            "player_sentiment_scores": {}  # This one is special, handled below
        }
        
        # Get all current metrics
        metrics = get_current_metrics()
        
        # Organize metrics by type
        metric_mappings = {
            "hype_score": "hype_scores",
            "rodmn_score": "rodmn_scores",
            "mentions": "mention_counts",
            "talk_time": "talk_time_counts",
            "wikipedia_views": "wikipedia_views",
            "reddit_mentions": "reddit_mentions",
            "google_trends": "google_trends",
            "google_news_mentions": "google_news_mentions"
        }
        
        for metric in metrics:
            data_key = metric_mappings.get(metric["metric_type"])
            if data_key:
                result[data_key][metric["entity_name"]] = metric["value"]
        
        # Special handling for sentiment scores (they're lists)
        # This is a placeholder - we should get actual sentiment scores if available
        for entity in entities:
            result["player_sentiment_scores"][entity["name"]] = []
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return {}

# API key and token management
def create_api_key(client_name, expires_in_days=None):
    """
    Create a new API key.
    
    Args:
        client_name: Name of the client
        expires_in_days: Number of days until key expires (None = never expires)
    
    Returns:
        Dictionary with API key information
    """
    try:
        import secrets
        import hashlib
        
        # Generate a random API key
        api_key = secrets.token_hex(32)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Calculate expiration date
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Insert the key
        query = """
            INSERT INTO api_keys (key_hash, client_name, is_active, expires_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id, client_name, is_active, created_at, expires_at
        """
        result = execute_query(query, (key_hash, client_name, True, expires_at))
        
        if not result:
            return None
        
        # Format result
        key_info = dict(result[0])
        key_info["api_key"] = api_key  # Add the raw key
        
        return key_info
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        return None
    
def validate_api_key(api_key):
    """
    Validate an API key.
    
    Args:
        api_key: The API key to validate
    
    Returns:
        Dictionary with key info if valid, None if invalid
    """
    try:
        import hashlib
        
        # Hash the key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Look up the key
        query = """
            SELECT id, client_name, is_active, created_at, expires_at, token_balance
            FROM api_keys
            WHERE key_hash = %s
        """
        result = execute_query(query, (key_hash,))
        
        if not result:
            return None
        
        key_info = dict(result[0])
        
        # Check if key is active
        if not key_info["is_active"]:
            return None
        
        # Check if key is expired
        if key_info["expires_at"] and key_info["expires_at"] < datetime.now():
            return None
        
        return key_info
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return None

def revoke_api_key(key_id):
    """
    Revoke an API key.
    
    Args:
        key_id: ID of the key to revoke
    
    Returns:
        Success boolean
    """
    try:
        query = """
            UPDATE api_keys
            SET is_active = FALSE
            WHERE id = %s
            RETURNING id
        """
        result = execute_query(query, (key_id,))
        
        return bool(result)
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        return False

def get_api_keys():
    """
    Get all API keys.
    
    Returns:
        List of API key dictionaries
    """
    try:
        query = """
            SELECT id, client_name, is_active, created_at, expires_at, token_balance
            FROM api_keys
            ORDER BY created_at DESC
        """
        return execute_query(query)
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        return []

def add_tokens(key_id, amount, description="Token purchase"):
    """
    Add tokens to an API key.
    
    Args:
        key_id: API key ID
        amount: Number of tokens to add
        description: Transaction description
    
    Returns:
        (success, message) tuple
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN")
        
        try:
            # Update token balance
            cursor.execute(
                """
                UPDATE api_keys
                SET token_balance = token_balance + %s
                WHERE id = %s
                RETURNING id
                """,
                (amount, key_id)
            )
            
            if not cursor.fetchone():
                conn.rollback()
                return False, f"API key with ID {key_id} not found"
            
            # Record transaction
            cursor.execute(
                """
                INSERT INTO token_transactions
                    (api_key_id, amount, transaction_type, description)
                VALUES
                    (%s, %s, %s, %s)
                """,
                (key_id, amount, "purchase", description)
            )
            
            conn.commit()
            return True, f"Added {amount} tokens"
        except Exception as e:
            conn.rollback()
            raise e
    except Exception as e:
        logger.error(f"Error adding tokens: {e}")
        return False, str(e)

def check_token_balance(api_key_id):
    """
    Get the current token balance for an API key.
    
    Args:
        api_key_id: The ID of the API key
    
    Returns:
        Current token balance
    """
    try:
        query = "SELECT token_balance FROM api_keys WHERE id = %s"
        result = execute_query(query, (api_key_id,))
        
        if result:
            return result[0]["token_balance"]
        return 0
    except Exception as e:
        logger.error(f"Error checking token balance: {e}")
        return 0

def deduct_tokens(api_key_id, amount, endpoint, request_id, client_ip, metadata=None):
    """
    Deduct tokens from an API key's balance.
    
    Args:
        api_key_id: The ID of the API key
        amount: Number of tokens to deduct
        endpoint: The API endpoint being called
        request_id: Unique request identifier
        client_ip: Client IP address
        metadata: Additional metadata to store
    
    Returns:
        (success, message) tuple
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN")
        
        try:
            # Check if balance is sufficient
            cursor.execute(
                "SELECT token_balance FROM api_keys WHERE id = %s",
                (api_key_id,)
            )
            
            result = cursor.fetchone()
            if not result:
                conn.rollback()
                return False, "API key not found"
            
            balance = result[0]
            
            if balance < amount:
                conn.rollback()
                return False, "Insufficient token balance"
            
            # Update balance
            cursor.execute(
                """
                UPDATE api_keys
                SET token_balance = token_balance - %s
                WHERE id = %s
                """,
                (amount, api_key_id)
            )
            
            # Record transaction
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute(
                """
                INSERT INTO token_transactions
                    (api_key_id, amount, transaction_type, endpoint, request_id, client_ip, metadata)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """,
                (api_key_id, -amount, "usage", endpoint, request_id, client_ip, metadata_json)
            )
            
            conn.commit()
            return True, f"Deducted {amount} tokens"
        except Exception as e:
            conn.rollback()
            raise e
    except Exception as e:
        logger.error(f"Error deducting tokens: {e}")
        return False, str(e)

def get_token_usage(api_key_id, days=30):
    """
    Get token usage statistics for an API key.
    
    Args:
        api_key_id: The ID of the API key
        days: Number of days of history to include
    
    Returns:
        Dictionary with usage statistics
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get current balance
        cursor.execute(
            "SELECT token_balance, tokens_purchased FROM api_keys WHERE id = %s",
            (api_key_id,)
        )
        
        key_info = cursor.fetchone()
        if not key_info:
            return {"error": "API key not found"}
        
        # Get recent transactions
        cursor.execute(
            """
            SELECT 
                transaction_type, 
                SUM(amount) as total_amount,
                COUNT(*) as transaction_count,
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM token_transactions
            WHERE api_key_id = %s
            AND created_at >= NOW() - INTERVAL '%s days'
            GROUP BY transaction_type
            """,
            (api_key_id, days)
        )
        
        transactions = cursor.fetchall()
        
        # Get usage by endpoint
        cursor.execute(
            """
            SELECT 
                endpoint, 
                SUM(ABS(amount)) as tokens_used,
                COUNT(*) as call_count
            FROM token_transactions
            WHERE api_key_id = %s
            AND transaction_type = 'usage'
            AND created_at >= NOW() - INTERVAL '%s days'
            GROUP BY endpoint
            ORDER BY tokens_used DESC
            """,
            (api_key_id, days)
        )
        
        endpoint_usage = cursor.fetchall()
        
        # Get daily usage pattern
        cursor.execute(
            """
            SELECT 
                DATE_TRUNC('day', created_at) as date,
                SUM(ABS(amount)) as tokens_used,
                COUNT(*) as call_count
            FROM token_transactions
            WHERE api_key_id = %s
            AND transaction_type = 'usage'
            AND created_at >= NOW() - INTERVAL '%s days'
            GROUP BY DATE_TRUNC('day', created_at)
            ORDER BY date
            """,
            (api_key_id, days)
        )
        
        daily_usage = cursor.fetchall()
        
        # Format response
        return {
            "current_balance": key_info["token_balance"],
            "total_purchased": key_info["tokens_purchased"] or 0,
            "total_used": (key_info["tokens_purchased"] or 0) - key_info["token_balance"],
            "transaction_summary": transactions,
            "endpoint_usage": endpoint_usage,
            "daily_usage": daily_usage,
            "period_days": days
        }
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        return {"error": str(e)}

def calculate_token_cost(endpoint, parameters=None):
    """
    Calculate the token cost for a specific API call.
    
    Args:
        endpoint: The API endpoint being called
        parameters: Optional request parameters that might affect cost
    
    Returns:
        The number of tokens to charge
    """
    # Default cost for any endpoint
    DEFAULT_TOKEN_COST = 1
    
    # Define endpoint-specific costs
    ENDPOINT_COSTS = {
        "/api/v1/bulk": 3,         # Bulk queries cost more
        "/api/v1/compare": 2,      # Entity comparison costs more
        "/api/upload_json": 5,     # Data uploads cost more
    }
    
    # Get base cost from the endpoint mapping
    base_cost = ENDPOINT_COSTS.get(endpoint, DEFAULT_TOKEN_COST)
    
    # Factor in query complexity if parameters are provided
    if parameters:
        # If bulk operation, scale by number of entities
        if endpoint == "/api/v1/bulk" and "entities" in parameters:
            entities_count = len(parameters["entities"].split(","))
            return base_cost * max(1, entities_count // 5)  # Charge more for larger batches
        
        # If comparing entities, scale by count
        if endpoint == "/api/v1/compare" and "entities" in parameters:
            entities_count = len(parameters["entities"].split(","))
            return base_cost * entities_count
    
    return base_cost

# System settings
def get_settings():
    """
    Get application settings.
    
    Returns:
        Dictionary with settings
    """
    try:
        query = "SELECT * FROM system_settings WHERE id = 1"
        result = execute_query(query)
        
        if result:
            return dict(result[0])
        
        # Return default settings if none found
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
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return {}

def save_settings(settings):
    """
    Save application settings.
    
    Args:
        settings: Dictionary with settings
    
    Returns:
        Success boolean
    """
    try:
        # Check if settings already exist
        existing = execute_query("SELECT id FROM system_settings WHERE id = 1")
        
        if existing:
            # Update existing settings
            query = """
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
            """
        else:
            # Insert new settings
            query = """
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
            """
        
        params = (
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
        )
        
        execute_query(query, params, fetch=False)
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

# Export and import functions
def export_entities_to_json():
    """
    Export all entities to entities.json file.
    
    Returns:
        Success boolean
    """
    try:
        # Get all entities
        entities = get_entities()
        
        # Create structure for entities.json
        entities_data = {}
        
        # Group entities by category and subcategory
        for entity in entities:
            category = entity["category"]
            subcategory = entity["subcategory"]
            
            if category not in entities_data:
                entities_data[category] = {}
            
            if subcategory not in entities_data[category]:
                entities_data[category][subcategory] = []
            
            # Create entity object
            entity_obj = {
                "name": entity["name"],
                "type": entity["type"],
                "gender": "female" if entity["type"] == "person" else "neutral",
                "aliases": [],
                "related_entities": [subcategory]
            }
            
            # Add to appropriate list
            entities_data[category][subcategory].append(entity_obj)
        
        # Save to file
        with open("entities.json", "w") as f:
            json.dump(entities_data, f, indent=4)
        
        logger.info("Exported entities to entities.json")
        return True
    except Exception as e:
        logger.error(f"Error exporting entities: {e}")
        return False

def import_entities_from_json():
    """
    Import entities from entities.json file.
    
    Returns:
        (success, message) tuple
    """
    try:
        # Check if file exists
        if not os.path.exists("entities.json"):
            return False, "entities.json not found"
        
        # Load entities from JSON
        with open("entities.json", "r") as f:
            entities_data = json.load(f)
        
        # Track stats
        entities_imported = 0
        
        # Process each entity
        for category, subcategories in entities_data.items():
            for subcategory, entity_list in subcategories.items():
                for entity_obj in entity_list:
                    name = entity_obj.get("name")
                    if not name:
                        continue
                    
                    # Check if entity already exists
                    existing = get_entity_by_name(name)
                    if existing:
                        continue
                    
                    # Create entity
                    entity_type = entity_obj.get("type", "person")
                    result = create_entity({
                        "name": name,
                        "type": entity_type,
                        "category": category,
                        "subcategory": subcategory
                    })
                    
                    if result[0]:  # If success
                        entities_imported += 1
        
        return True, f"Imported {entities_imported} entities from entities.json"
    except Exception as e:
        logger.error(f"Error importing entities: {e}")
        return False, str(e)

# Customer-specific functions
def get_entities_from_schema(schema_name):
    """Get entities from a specific schema."""
    return get_entities(schema_name=schema_name)

def get_current_metrics_from_schema(schema_name):
    """Get current metrics from a specific schema.""" 
    return get_current_metrics(schema_name=schema_name)

def get_entity_by_name_and_schema(entity_name, schema_name):
    """Get entity by name from a specific schema."""
    return get_entity_by_name(entity_name, schema_name=schema_name)

def create_entity_in_schema(entity_data, schema_name):
    """Create entity in a specific schema."""
    return create_entity(entity_data, schema_name=schema_name)

def save_metric_to_schema(entity_id, metric_type, value, time_period=None, is_historical=False, schema_name=None):
    """Save metric to a specific schema."""
    return save_metric(entity_id, metric_type, value, time_period, is_historical, schema_name)

def execute_transaction_in_schema(queries, schema_name):
    """Execute multiple queries in a transaction within a specific schema."""
    conn = None
    try:
        conn = get_connection(schema_name)
        with conn.cursor() as cursor:
            # Set search path for schema
            cursor.execute(f"SET search_path TO {schema_name};")
            
            for query, params in queries:
                if params is None:
                    cursor.execute(query)
                else:
                    cursor.execute(query, params or ())
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Transaction error in schema {schema_name}: {e}")
        if conn:
            conn.rollback()
        return False

def bulk_import_entities_to_schema(entities_data, schema_name):
    """
    Bulk import entities to a specific schema.
    
    Args:
        entities_data: List of entity dictionaries
        schema_name: Target schema name
    
    Returns:
        (success_count, error_count) tuple
    """
    success_count = 0
    error_count = 0
    
    for entity_data in entities_data:
        try:
            success, message = create_entity_in_schema(entity_data, schema_name)
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.warning(f"Failed to import entity {entity_data.get('name')}: {message}")
        except Exception as e:
            error_count += 1
            logger.error(f"Error importing entity {entity_data.get('name')}: {e}")
    
    return success_count, error_count

def get_schema_summary(schema_name):
    """Get a summary of data in a schema."""
    try:
        entities = get_entities_from_schema(schema_name)
        current_metrics = get_current_metrics_from_schema(schema_name)
        
        # Group metrics by type
        metrics_by_type = {}
        for metric in current_metrics:
            metric_type = metric['metric_type']
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric)
        
        return {
            'schema_name': schema_name,
            'entity_count': len(entities),
            'total_metrics': len(current_metrics),
            'entities': entities,
            'metrics_by_type': metrics_by_type,
            'available_metric_types': list(metrics_by_type.keys())
        }
    except Exception as e:
        logger.error(f"Error getting schema summary: {e}")
        return {
            'schema_name': schema_name,
            'error': str(e)
        }

# Initialize the module
if __name__ == "__main__":
    # If this file is run directly, initialize the database
    initialize_database()    