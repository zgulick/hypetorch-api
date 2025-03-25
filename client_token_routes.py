# client_token_routes.py
from fastapi import APIRouter, Depends
from auth_middleware import get_api_key
from token_manager import check_token_balance, get_token_usage
from api_utils import StandardResponse
import time

# Create router
router = APIRouter(prefix="/api/v1/tokens", tags=["Client Tokens"])

@router.get("/balance")
def get_client_token_balance(key_info: dict = Depends(get_api_key)):
    """Get token balance for the authenticated client."""
    # The key_info from auth middleware contains the client's API key info
    api_key_id = key_info.get("id")
    
    if not api_key_id:
        return StandardResponse.error(
            message="Invalid API key information",
            status_code=400
        )
    
    balance = check_token_balance(api_key_id)
    
    return StandardResponse.success(
        data={
            "balance": balance
        },
        metadata={
            "timestamp": time.time(),
            "client": key_info.get("client_name")
        }
    )

from token_plans import get_all_token_plans

@router.get("/plans")
def get_available_token_plans():
    """Get a list of all available token plans."""
    plans = get_all_token_plans()
    
    return StandardResponse.success(
        data={
            "plans": plans
        },
        metadata={
            "timestamp": time.time()
        }
    )

@router.get("/usage")
def get_client_token_usage(days: int = 30, key_info: dict = Depends(get_api_key)):
    """Get token usage statistics for the authenticated client."""
    api_key_id = key_info.get("id")
    
    if not api_key_id:
        return StandardResponse.error(
            message="Invalid API key information",
            status_code=400
        )
    
    usage_data = get_token_usage(api_key_id, days)
    
    if "error" in usage_data:
        return StandardResponse.error(
            message=usage_data["error"],
            status_code=400
        )
    
    return StandardResponse.success(
        data=usage_data,
        metadata={
            "timestamp": time.time(),
            "client": key_info.get("client_name"),
            "days": days
        }
    )