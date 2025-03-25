# token_plans.py
"""
Configuration for token plans and pricing.
"""

# Token Plans
TOKEN_PLANS = {
    "free": {
        "name": "Free Tier",
        "tokens": 1000,
        "price_usd": 0,
        "description": "Free tier with limited tokens",
        "features": ["Basic API access", "Rate limited to 60 calls/minute"],
        "max_tokens_per_month": 1000
    },
    "starter": {
        "name": "Starter",
        "tokens": 10000,
        "price_usd": 49,
        "description": "Starter plan for small projects",
        "features": ["Full API access", "Rate limited to 300 calls/minute"],
        "max_tokens_per_month": 10000
    },
    "professional": {
        "name": "Professional",
        "tokens": 100000,
        "price_usd": 299,
        "description": "Professional plan for business use",
        "features": ["Full API access", "Rate limited to 1000 calls/minute", "Priority support"],
        "max_tokens_per_month": 100000
    },
    "enterprise": {
        "name": "Enterprise",
        "tokens": 1000000,
        "price_usd": 999,
        "description": "Enterprise plan for high-volume use",
        "features": ["Full API access", "Unlimited rate", "Priority support", "Custom integrations", "SLA guarantee"],
        "max_tokens_per_month": 1000000
    }
}

# Token Cost Configuration
DEFAULT_TOKEN_COST = 1  # Default cost for simple API calls

# Endpoint specific costs
ENDPOINT_COSTS = {
    # Entity endpoints
    "/api/v1/entities": 1,
    "/api/v1/entities/{entity_id}": 1,
    "/api/v1/entities/{entity_id}/metrics": 1,
    "/api/v1/entities/{entity_id}/trending": 1,
    "/api/v1/entities/{entity_id}/history": 2,
    "/api/v1/entities/{entity_id}/related": 3,
    "/api/v1/entities/{entity_id}/relationships": 2,
    
    # Bulk operations
    "/api/v1/bulk": 3,
    "/api/v1/compare": 3,
    
    # Data submission
    "/api/upload_json": 5,
    "/api/store_historical": 5,
    
    # Advanced analytics
    "/api/trending": 3,
    "/api/controversial": 3
}

def get_token_plan(plan_id: str):
    """Get details for a specific token plan."""
    return TOKEN_PLANS.get(plan_id, None)

def get_all_token_plans():
    """Get a list of all available token plans."""
    return TOKEN_PLANS

def get_endpoint_cost(endpoint: str):
    """Get the token cost for a specific endpoint."""
    # Try exact match first
    if endpoint in ENDPOINT_COSTS:
        return ENDPOINT_COSTS[endpoint]
    
    # Try pattern matching for parameterized endpoints
    for pattern, cost in ENDPOINT_COSTS.items():
        if "{" in pattern:
            # Convert pattern with parameters to regex pattern
            import re
            regex_pattern = pattern.replace("{", "\\{").replace("}", "\\}")
            regex_pattern = regex_pattern.replace("\\{entity_id\\}", "[^/]+")
            
            if re.match(regex_pattern, endpoint):
                return cost
    
    # Default cost if no match found
    return DEFAULT_TOKEN_COST