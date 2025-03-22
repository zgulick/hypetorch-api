# api_utils.py
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Union
import time

class StandardResponse:
    """
    Create standardized API responses with consistent structure.
    """
    
    @staticmethod
    def success(data: Any, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a success response.
        
        Args:
            data: The response data
            metadata: Optional metadata
            
        Returns:
            dict: Standardized response object
        """
        return {
            "status": "success",
            "data": data,
            "metadata": metadata or {
                "timestamp": time.time(),
                "processing_time_ms": 0  # Should be set by the caller
            }
        }
    
    @staticmethod
    def error(message: str, status_code: int = 400, details: Optional[Any] = None) -> Dict:
        """
        Create an error response.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
            
        Returns:
            dict: Standardized error object
        """
        error_response = {
            "status": "error",
            "error": {
                "message": message,
                "code": status_code,
                "details": details
            },
            "metadata": {
                "timestamp": time.time()
            }
        }
        
        # Raises an HTTPException that FastAPI will handle
        raise HTTPException(
            status_code=status_code,
            detail=error_response
        )
    
    @staticmethod
    def paginate(
        items: List[Any], 
        page: int, 
        page_size: int, 
        total_items: int,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a paginated response.
        
        Args:
            items: List of items for the current page
            page: Current page number (1-based)
            page_size: Items per page
            total_items: Total number of items
            metadata: Additional metadata
            
        Returns:
            dict: Standardized paginated response
        """
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        response_metadata = metadata or {}
        
        # Add pagination metadata
        response_metadata.update({
            "timestamp": time.time(),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        })
        
        return {
            "status": "success",
            "data": items,
            "metadata": response_metadata
        }