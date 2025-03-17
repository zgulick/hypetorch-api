# api_key_routes.py
import os
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List

from api_key_manager import create_api_key, get_api_keys, revoke_api_key

# This will need to be replaced with your actual admin authentication
# For now, we'll use a simple secret
ADMIN_SECRET = "admin-secret-replace-this"

router = APIRouter(prefix="/api/admin/keys", tags=["API Keys"])

class ApiKeyCreate(BaseModel):
    client_name: str
    expires_in_days: Optional[int] = None

class ApiKeyInfo(BaseModel):
    id: int
    client_name: str
    is_active: bool
    created_at: str  # This expects a string
    expires_at: Optional[str] = None
    
class ApiKeyResponse(BaseModel):
    api_key: Optional[str] = None
    info: ApiKeyInfo

def verify_admin(admin_key: str = Query(None)):
    """Simple admin verification using environment variable"""
    # Get from environment variable
    ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "default-unsafe-secret")
    NEXT_PUBLIC_ADMIN_SECRET = os.environ.get("NEXT_PUBLIC_ADMIN_SECRET", "")
    
    print(f"Received admin key: {admin_key}")
    print(f"Expected ADMIN_SECRET: {ADMIN_SECRET}")
    print(f"Expected NEXT_PUBLIC_ADMIN_SECRET: {NEXT_PUBLIC_ADMIN_SECRET}")
    
    # Check against both possible keys
    if admin_key and (admin_key == ADMIN_SECRET or admin_key == NEXT_PUBLIC_ADMIN_SECRET):
        return True
    
    raise HTTPException(status_code=403, detail="Unauthorized access to admin functions")

@router.post("", response_model=ApiKeyResponse)
def admin_create_key(key_data: ApiKeyCreate, _: bool = Depends(verify_admin)):
    """Admin endpoint to create a new API key"""
    try:
        key_info = create_api_key(key_data.client_name, key_data.expires_in_days)
        
        # Extract the raw key before returning
        api_key = key_info.pop('api_key', None)
        
        return {
            "api_key": api_key,
            "info": key_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating API key: {str(e)}")

@router.get("", response_model=List[ApiKeyInfo])
def admin_list_keys(_: bool = Depends(verify_admin)):
    print("🔍 Admin Keys Route Hit!")  # Add this debug print
    try:
        return get_api_keys()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing API keys: {str(e)}")

@router.delete("/{key_id}")
def admin_revoke_key(key_id: int, _: bool = Depends(verify_admin)):
    """Admin endpoint to revoke an API key"""
    try:
        success = revoke_api_key(key_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"API key {key_id} not found")
        return {"message": f"API key {key_id} revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error revoking API key: {str(e)}")