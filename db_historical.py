# db_historical.py
import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Import functions from db.py
from db import get_connection, execute_query

def store_hype_data(data, time_period):
    """
    Store HYPE scores and component metrics in the database.
    
    Args:
        data (dict): Dictionary containing HYPE scores and component metrics
        time_period (str): Time period for this data (e.g., "last_7_days")
    
    Returns:
        bool: Success status
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # First, create tables if they don't exist
        print("Creating tables if they don't exist...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            type TEXT,
            category TEXT,
            subcategory TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS hype_scores (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id),
            score FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            time_period TEXT,
            algorithm_version TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS component_metrics (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id),
            metric_type TEXT NOT NULL,
            value FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            time_period TEXT
        )
        """)
        
        conn.commit()
        print("Tables created successfully.")
        
        timestamp = datetime.now()
        
        # Process each entity
        for entity_name, hype_score in data.get("hype_scores", {}).items():
            # Get or create entity
            cursor.execute(
                "SELECT id FROM entities WHERE name = %s",
                (entity_name,)
            )
            result = cursor.fetchone()
            
            if result:
                entity_id = result[0]
            else:
                # Determine entity type (person or non-person)
                entity_type = "non-person" if entity_name.upper() == entity_name else "person"
                
                cursor.execute(
                    """
                    INSERT INTO entities (name, type, category, subcategory)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (entity_name, entity_type, "Sports", "Unrivaled")  # Default category/subcategory
                )
                entity_id = cursor.fetchone()[0]
            
            # Store HYPE score
            cursor.execute(
                """
                INSERT INTO hype_scores (entity_id, score, timestamp, time_period, algorithm_version)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (entity_id, hype_score, timestamp, time_period, "1.0")
            )
            
            # Store RODMN score if available
            if "rodmn_scores" in data and entity_name in data["rodmn_scores"]:
                rodmn_score = data["rodmn_scores"][entity_name]
                cursor.execute(
                    """
                    INSERT INTO component_metrics (entity_id, metric_type, value, timestamp, time_period)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (entity_id, "rodmn_score", rodmn_score, timestamp, time_period)
                )

            # Store component metrics
            for metric_type, metric_dict in {
                "talk_time_counts": data.get("talk_time_counts", {}),
                "mention_counts": data.get("mention_counts", {}),
                "wikipedia_views": data.get("wikipedia_views", {}),
                "reddit_mentions": data.get("reddit_mentions", {}),
                "google_trends": data.get("google_trends", {}),
                "google_news_mentions": data.get("google_news_mentions", {})
            }.items():
                if entity_name in metric_dict:
                    value = metric_dict[entity_name]
                    cursor.execute(
                        """
                        INSERT INTO component_metrics (entity_id, metric_type, value, timestamp, time_period)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (entity_id, metric_type, value, timestamp, time_period)
                    )
        
        conn.commit()
        conn.close()
        print(f"✅ Successfully stored HYPE data for {len(data.get('hype_scores', {}))} entities")
        return True
    except Exception as e:
        print(f"❌ Error storing HYPE data: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def get_entity_history(entity_name, limit=30, start_date=None, end_date=None):
    """
    Get historical HYPE scores for a specific entity.
    
    Args:
        entity_name (str): Entity name
        limit (int): Maximum number of records to return
        start_date (str): Start date in format 'YYYY-MM-DD' (optional)
        end_date (str): End date in format 'YYYY-MM-DD' (optional)
    
    Returns:
        list: Historical HYPE score data
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build the query with optional date filtering
        query = """
        SELECT h.score, h.timestamp, h.time_period
        FROM hype_scores h
        JOIN entities e ON h.entity_id = e.id
        WHERE e.name = %s
        """
        params = [entity_name]
        
        if start_date:
            query += " AND h.timestamp >= %s"
            params.append(start_date + " 00:00:00")
        
        if end_date:
            query += " AND h.timestamp <= %s"
            params.append(end_date + " 23:59:59")
        
        query += " ORDER BY h.timestamp DESC"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        # Execute the query
        cursor.execute(query, params)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ Error retrieving entity history: {e}")
        if 'conn' in locals():
            conn.close()
        return []

def get_entity_metrics_history(entity_name, metric_type, limit=30, start_date=None, end_date=None):
    """
    Get historical metric values for a specific entity and metric type.
    
    Args:
        entity_name (str): Entity name
        metric_type (str): Metric type (e.g., "talk_time_counts")
        limit (int): Maximum number of records to return
        start_date (str): Start date in format 'YYYY-MM-DD' (optional)
        end_date (str): End date in format 'YYYY-MM-DD' (optional)
    
    Returns:
        list: Historical metric data
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build the query with optional date filtering
        query = """
        SELECT cm.value, cm.timestamp, cm.time_period
        FROM component_metrics cm
        JOIN entities e ON cm.entity_id = e.id
        WHERE e.name = %s AND cm.metric_type = %s
        """
        params = [entity_name, metric_type]
        
        if start_date:
            query += " AND cm.timestamp >= %s"
            params.append(start_date + " 00:00:00")
        
        if end_date:
            query += " AND cm.timestamp <= %s"
            params.append(end_date + " 23:59:59")
        
        query += " ORDER BY cm.timestamp DESC"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        # Execute the query
        cursor.execute(query, params)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ Error retrieving entity metrics history: {e}")
        if 'conn' in locals():
            conn.close()
        return []

def get_trending_entities(metric_type="hype_scores", limit=10, time_period=None, category=None, subcategory=None):
    """
    Get trending entities based on recent metric changes.
    
    Args:
        metric_type (str): Metric to analyze ("hype_scores" or component metric name)
        limit (int): Maximum number of entities to return
        time_period (str): Filter by time period (e.g., "last_7_days")
        category (str): Filter by entity category
        subcategory (str): Filter by entity subcategory
    
    Returns:
        list: Trending entities with change data
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # This query identifies entities with the biggest percent change
        # between their most recent measurement and the one before that
        if metric_type == "hype_scores":
            base_table = "hype_scores"
            value_field = "score"
        else:
            base_table = "component_metrics"
            value_field = "value"
        
        # Build the query
        query = f"""
        WITH latest AS (
            SELECT 
                e.name as entity_name,
                e.id as entity_id,
                MAX(t.timestamp) as latest_timestamp
            FROM 
                entities e
            JOIN 
                {base_table} t ON e.id = t.entity_id
            WHERE 1=1
        """
        
        params = []
        
        # Add filters to the CTE
        if metric_type != "hype_scores" and base_table == "component_metrics":
            query += " AND t.metric_type = %s"
            params.append(metric_type)
            
        if time_period:
            query += " AND t.time_period = %s"
            params.append(time_period)
            
        if category:
            query += " AND e.category = %s"
            params.append(category)
            
        if subcategory:
            query += " AND e.subcategory = %s"
            params.append(subcategory)
            
        # Complete the CTE
        query += """
            GROUP BY e.id, e.name
        ), 
        previous AS (
            SELECT 
                e.name as entity_name,
                e.id as entity_id,
                MAX(t.timestamp) as prev_timestamp
            FROM 
                entities e
            JOIN 
                {0} t ON e.id = t.entity_id
            JOIN 
                latest l ON e.id = l.entity_id AND t.timestamp < l.latest_timestamp
            WHERE 1=1
        """.format(base_table)
        
        # Add the same filters to the previous CTE
        if metric_type != "hype_scores" and base_table == "component_metrics":
            query += " AND t.metric_type = %s"
            params.append(metric_type)
            
        if time_period:
            query += " AND t.time_period = %s"
            params.append(time_period)
        
        # Complete the query
        query += """
            GROUP BY e.id, e.name
        )
        SELECT 
            l.entity_name,
            curr.{0} as current_value,
            l.latest_timestamp as current_timestamp,
            prev.{0} as previous_value,
            p.prev_timestamp as previous_timestamp,
            CASE 
                WHEN prev.{0} = 0 THEN 100
                ELSE ((curr.{0} - prev.{0}) / ABS(prev.{0})) * 100 
            END as percent_change
        FROM 
            latest l
        JOIN 
            {1} curr ON l.entity_id = curr.entity_id AND l.latest_timestamp = curr.timestamp
        JOIN 
            previous p ON l.entity_id = p.entity_id
        JOIN 
            {1} prev ON p.entity_id = prev.entity_id AND p.prev_timestamp = prev.timestamp
        """.format(value_field, base_table)
        
        # Add additional filter for component_metrics
        if metric_type != "hype_scores" and base_table == "component_metrics":
            query += " AND curr.metric_type = %s AND prev.metric_type = %s"
            params.extend([metric_type, metric_type])
        
        # Order by change and limit results
        query += """
        ORDER BY 
            percent_change DESC
        LIMIT %s
        """
        
        params.append(limit)
        
        # Execute the query
        cursor.execute(query, params)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ Error retrieving trending entities: {e}")
        if 'conn' in locals():
            conn.close()
        return []