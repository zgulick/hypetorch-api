# token_manager.py
import time
import uuid
import json
from typing import Dict, Optional, Tuple, Any
import logging
from db_pool import DatabaseConnection, execute_query
import psycopg2
from psycopg2.extras import RealDictCursor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('token_manager')

# Token cost configuration
DEFAULT_TOKEN_COST = 1  # Default cost for any API call

# Endpoint-specific costs (more complex endpoints cost more)
ENDPOINT_COSTS = {
    "/api/v1/bulk": 3,         # Bulk queries cost more
    "/api/v1/compare": 2,      # Entity comparison costs more
    "/api/upload_json": 5,     # Data uploads cost more
}

# Import token plans configuration
from token_plans import get_endpoint_cost

def calculate_token_cost(endpoint: str, parameters: Dict = None) -> int:
    """
    Calculate the token cost for a specific API call.
    
    Args:
        endpoint: The API endpoint being called
        parameters: Optional request parameters that might affect cost
        
    Returns:
        int: The number of tokens to charge
    """
    # Get base cost from configuration
    base_cost = get_endpoint_cost(endpoint)
    
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

def check_token_balance(api_key_id: int) -> int:
    """
    Get the current token balance for an API key.
    
    Args:
        api_key_id: The ID of the API key
        
    Returns:
        int: Current token balance
    """
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                "SELECT token_balance FROM api_keys WHERE id = %s",
                (api_key_id,)
            )
            
            result = cursor.fetchone()
            if result:
                return result["token_balance"]
            return 0
    except Exception as e:
        logger.error(f"Error checking token balance: {e}")
        return 0

def deduct_tokens(
    api_key_id: int, 
    amount: int, 
    endpoint: str,
    request_id: str,
    client_ip: str,
    metadata: Dict = None
) -> Tuple[bool, str]:
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
        tuple: (success, message)
    """
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # Check if balance is sufficient
            cursor.execute(
                "SELECT check_token_balance(%s, %s) as has_tokens",
                (api_key_id, amount)
            )
            
            result = cursor.fetchone()
            has_tokens = result[0] if result else False
            
            if not has_tokens:
                return False, "Insufficient token balance"
            
            # Record the transaction
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
        logger.error(f"Error deducting tokens: {e}")
        return False, f"Error: {str(e)}"

def add_tokens(api_key_id: int, amount: int, description: str = "Token purchase") -> Tuple[bool, str]:
    """
    Add tokens to an API key's balance.
    
    Args:
        api_key_id: The ID of the API key
        amount: Number of tokens to add
        description: Description of the transaction
        
    Returns:
        tuple: (success, message)
    """
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # Call the add_tokens function
            cursor.execute(
                "SELECT add_tokens(%s, %s, %s)",
                (api_key_id, amount, description)
            )
            
            conn.commit()
            return True, f"Added {amount} tokens"
    except Exception as e:
        logger.error(f"Error adding tokens: {e}")
        return False, f"Error: {str(e)}"

def get_token_usage(api_key_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Get token usage statistics for an API key.
    
    Args:
        api_key_id: The ID of the API key
        days: Number of days of history to include
        
    Returns:
        dict: Usage statistics
    """
    try:
        with DatabaseConnection(psycopg2.extras.RealDictCursor) as conn:
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
                "total_purchased": key_info["tokens_purchased"],
                "total_used": key_info["tokens_purchased"] - key_info["token_balance"],
                "transaction_summary": [dict(t) for t in transactions],
                "endpoint_usage": [dict(e) for e in endpoint_usage],
                "daily_usage": [dict(d) for d in daily_usage],
                "period_days": days
            }
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        return {"error": str(e)}