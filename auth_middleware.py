# auth_middleware.py
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
from datetime import datetime
import os
from api_key_manager import validate_api_key

# Define the API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(API_KEY_HEADER)) -> Optional[dict]:
    """
    Dependency to extract and validate the API key from the request header
    
    Args:
        api_key: The API key from the X-API-Key header
        
    Returns:
        dict: Key information if valid
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    # Special case: development mode or using dev key
    dev_mode = os.environ.get("DEVELOPMENT_MODE", "false").lower() == "true"
    dev_key = os.environ.get("DEV_API_KEY", "")
    
    if dev_mode or api_key == dev_key:
        # In development mode or with dev key, return a dummy key info
        return {
            "id": 0,
            "client_name": "Development",
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "token_balance": 1000000
        }
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing. Please provide a valid API key in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    key_info = validate_api_key(api_key)
    
    if not key_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return key_info

# Simplified version that can be used as a decorator
def api_key_required(request: Request):
    """
    Middleware-style function to check if API key is valid
    
    Args:
        request: The FastAPI request object
        
    Returns:
        dict: Key information if valid
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    # Extract API key from header
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing. Please provide a valid API key in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    key_info = validate_api_key(api_key)
    
    if not key_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return key_info