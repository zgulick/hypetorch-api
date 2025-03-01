# db_operations.py
"""Database operations for HypeTorch API"""

from db_wrapper import DB_AVAILABLE, execute_query
from datetime import datetime

# If database is not available, provide dummy functions
if not DB_AVAILABLE:
    def dummy_function(*args, **kwargs):
        return {"status": "skipped", "reason": "database not available"}
    
    save_hype_scores = dummy_function
    save_metrics = dummy_function
    get_entity_history = dummy_function
    get_metric_history = dummy_function
    get_top_entities = dummy_function
    save_all_current_data = dummy_function
    
else:
    # Real database functions
    def ensure_entity_exists(entity_name, entity_type="person", category="Sports", subcategory="Unrivaled"):
        """Make sure an entity exists in the database, return its ID"""
        # Check if entity already exists
        query = "SELECT id FROM entities WHERE name = %s"
        result = execute_query(query, (entity_name,))
        
        if result:
            return result[0]['id']
        
        # If not, create it
        query = """
        INSERT INTO entities (name, type, category, subcategory) 
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        result = execute_query(query, (entity_name, entity_type, category, subcategory))
        if result:
            return result[0]['id']
        return None
    
    def save_hype_scores(hype_scores, algorithm_version="1.0"):
        """Save current HYPE scores to the database"""
        timestamp = datetime.now()
        count = 0
        
        for entity_name, score in hype_scores.items():
            entity_id = ensure_entity_exists(entity_name)
            if not entity_id:
                continue
                
            query = """
            INSERT INTO hype_scores (entity_id, score, timestamp, algorithm_version)
            VALUES (%s, %s, %s, %s)
            """
            execute_query(query, (entity_id, score, timestamp, algorithm_version), fetch=False)
            count += 1
        
        return count
    
    def save_metrics(metrics_dict, metric_type):
        """Save individual metrics to the database"""
        timestamp = datetime.now()
        count = 0
        
        for entity_name, value in metrics_dict.items():
            entity_id = ensure_entity_exists(entity_name)
            if not entity_id:
                continue
                
            query = """
            INSERT INTO metrics (entity_id, metric_type, value, timestamp)
            VALUES (%s, %s, %s, %s)
            """
            execute_query(query, (entity_id, metric_type, value, timestamp), fetch=False)
            count += 1
        
        return count
    
    def get_entity_history(entity_name, days=30):
        """Get historical HYPE scores for an entity"""
        query = """
        SELECT h.score, h.timestamp, h.algorithm_version
        FROM hype_scores h
        JOIN entities e ON h.entity_id = e.id
        WHERE e.name = %s
        AND h.timestamp > NOW() - INTERVAL '%s days'
        ORDER BY h.timestamp
        """
        return execute_query(query, (entity_name, days))
    
    def get_metric_history(entity_name, metric_type, days=30):
        """Get historical metrics for an entity"""
        query = """
        SELECT m.value, m.timestamp
        FROM metrics m
        JOIN entities e ON m.entity_id = e.id
        WHERE e.name = %s
        AND m.metric_type = %s
        AND m.timestamp > NOW() - INTERVAL '%s days'
        ORDER BY m.timestamp
        """
        return execute_query(query, (entity_name, metric_type, days))
    
    def get_top_entities(limit=10):
        """Get the top entities by latest HYPE score"""
        query = """
        WITH latest_scores AS (
            SELECT DISTINCT ON (entity_id) 
                entity_id, 
                score, 
                timestamp
            FROM hype_scores
            ORDER BY entity_id, timestamp DESC
        )
        SELECT e.name, ls.score, ls.timestamp
        FROM latest_scores ls
        JOIN entities e ON ls.entity_id = e.id
        ORDER BY ls.score DESC
        LIMIT %s
        """
        return execute_query(query, (limit,))
    
    def save_all_current_data(data_dict):
        """Save all current data to the database"""
        # Save HYPE scores
        hype_scores_saved = save_hype_scores(data_dict.get("hype_scores", {}))
        
        # Save individual metrics
        metrics_saved = 0
        metrics_saved += save_metrics(data_dict.get("talk_time_counts", {}), "talk_time")
        metrics_saved += save_metrics(data_dict.get("mention_counts", {}), "mentions")
        metrics_saved += save_metrics(data_dict.get("wikipedia_views", {}), "wikipedia_views")
        metrics_saved += save_metrics(data_dict.get("reddit_mentions", {}), "reddit_mentions")
        metrics_saved += save_metrics(data_dict.get("google_news_mentions", {}), "google_news_mentions")
        metrics_saved += save_metrics(data_dict.get("google_trends", {}), "google_trends")
        
        return {
            "hype_scores_saved": hype_scores_saved,
            "metrics_saved": metrics_saved
        }