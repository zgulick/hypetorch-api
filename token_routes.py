# token_routes.py
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from auth_middleware import get_api_key
from token_manager import add_tokens, get_token_usage, check_token_balance
from api_utils import StandardResponse

# Create router
router = APIRouter(prefix="/api/admin/tokens", tags=["Tokens"])

# Models
class TokenPurchase(BaseModel):
    api_key_id: int
    amount: int
    description: Optional[str] = "Token purchase"

# Endpoints
@router.post("/purchase")
def purchase_tokens(purchase: TokenPurchase, key_info: dict = Depends(get_api_key)):
    """Add tokens to an API key (admin only)."""
    success, message = add_tokens(purchase.api_key_id, purchase.amount, purchase.description)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return StandardResponse.success(
        data={
            "api_key_id": purchase.api_key_id,
            "amount": purchase.amount,
            "message": message
        },
        metadata={
            "timestamp": import_time()
        }
    )

@router.get("/usage/{api_key_id}")
def get_token_usage_endpoint(
    api_key_id: int,
    days: int = Query(30, ge=1, le=365),
    key_info: dict = Depends(get_api_key)
):
    """Get token usage statistics for an API key (admin only)."""
    usage_data = get_token_usage(api_key_id, days)
    
    if "error" in usage_data:
        raise HTTPException(status_code=400, detail=usage_data["error"])
    
    return StandardResponse.success(
        data=usage_data,
        metadata={
            "timestamp": import_time(),
            "api_key_id": api_key_id,
            "days": days
        }
    )

@router.get("/balance/{api_key_id}")
def get_balance(api_key_id: int, key_info: dict = Depends(get_api_key)):
    """Get current token balance for an API key (admin only)."""
    balance = check_token_balance(api_key_id)
    
    return StandardResponse.success(
        data={
            "api_key_id": api_key_id,
            "balance": balance
        },
        metadata={
            "timestamp": import_time()
        }
    )

# Helper for timestamp
def import_time():
    import time
    return time.time()