# rate_limiter.py
import time
from collections import defaultdict
import threading
from typing import Dict, Tuple, Optional, List

class RateLimiter:
    """
    A simple in-memory rate limiter using the token bucket algorithm.
    This limits how many requests a client can make in a given time period.
    """
    
    def __init__(self):
        # Structure: {client_id: {endpoint: (tokens, last_refill_time)}}
        self._buckets: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
        self._lock = threading.Lock()
        
        # Default rate limits (can be overridden per endpoint)
        self.default_rate_limits = {
            "requests_per_minute": 60,  # 60 requests per minute
            "requests_per_hour": 1000,  # 1000 requests per hour
            "requests_per_day": 10000,  # 10000 requests per day
        }
        
        # Endpoint-specific rate limits
        self.endpoint_rate_limits = {
            # Format: "endpoint_path": {"requests_per_minute": X, "requests_per_hour": Y}
            "/api/bulk": {"requests_per_minute": 30, "requests_per_hour": 500},
            "/api/upload_json": {"requests_per_minute": 10, "requests_per_hour": 100},
        }
    
    def check_rate_limit(self, client_id: str, endpoint: str, current_time: Optional[float] = None) -> Tuple[bool, Dict]:
        """
        Check if a request from client_id to endpoint should be rate limited.
        
        Args:
            client_id: Identifier for the client (e.g., API key)
            endpoint: The endpoint being accessed
            current_time: Current time (for testing)
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        if current_time is None:
            current_time = time.time()
            
        # Get rate limits for this endpoint (or defaults)
        rate_limits = self.endpoint_rate_limits.get(endpoint, self.default_rate_limits)
        
        # Check minute limit
        minute_allowed = self._check_bucket(
            client_id, f"{endpoint}:minute", 
            rate_limits["requests_per_minute"], 60, 
            current_time
        )
        
        # Check hour limit
        hour_allowed = self._check_bucket(
            client_id, f"{endpoint}:hour", 
            rate_limits["requests_per_hour"], 3600, 
            current_time
        )
        
        # Check day limit
        day_allowed = self._check_bucket(
            client_id, f"{endpoint}:day", 
            rate_limits.get("requests_per_day", self.default_rate_limits["requests_per_day"]), 86400,
            current_time
        )
        
        # Create rate limit info for headers
        minute_bucket = self._buckets[client_id].get(f"{endpoint}:minute", (0, 0))
        hour_bucket = self._buckets[client_id].get(f"{endpoint}:hour", (0, 0))
        day_bucket = self._buckets[client_id].get(f"{endpoint}:day", (0, 0))
        
        rate_limit_info = {
            "X-RateLimit-Limit-Minute": str(rate_limits["requests_per_minute"]),
            "X-RateLimit-Remaining-Minute": str(max(0, int(minute_bucket[0]))),
            "X-RateLimit-Limit-Hour": str(rate_limits["requests_per_hour"]),
            "X-RateLimit-Remaining-Hour": str(max(0, int(hour_bucket[0]))),
            "X-RateLimit-Reset": str(int(current_time + 60 - (current_time % 60))),
        }
        
        # Request is allowed only if all limits are satisfied
        allowed = minute_allowed and hour_allowed and day_allowed
        
        return allowed, rate_limit_info
    
    def _check_bucket(self, client_id: str, bucket_key: str, limit: int, 
                     refill_time: int, current_time: float) -> bool:
        """
        Check and update a token bucket.
        
        Args:
            client_id: Client identifier
            bucket_key: Unique key for this bucket
            limit: Maximum number of tokens
            refill_time: Time in seconds to refill tokens
            current_time: Current time
            
        Returns:
            bool: True if request is allowed, False if rate limited
        """
        with self._lock:
            # Get or create the bucket
            if bucket_key not in self._buckets[client_id]:
                # New bucket starts full
                self._buckets[client_id][bucket_key] = (float(limit), current_time)
                return True
                
            tokens, last_refill = self._buckets[client_id][bucket_key]
            
            # Calculate token refill
            time_passed = current_time - last_refill
            token_increment = time_passed * (limit / refill_time)
            
            # Refill tokens (up to the limit)
            new_tokens = min(limit, tokens + token_increment)
            
            # Check if we have enough tokens
            if new_tokens < 1:
                # Not enough tokens, request should be rate limited
                self._buckets[client_id][bucket_key] = (new_tokens, current_time)
                return False
                
            # Use 1 token and update the bucket
            self._buckets[client_id][bucket_key] = (new_tokens - 1, current_time)
            return True
    
    def get_client_usage(self, client_id: str) -> Dict:
        """Get usage statistics for a client"""
        usage = {}
        
        with self._lock:
            if client_id in self._buckets:
                for bucket_key, (tokens, _) in self._buckets[client_id].items():
                    endpoint, period = bucket_key.split(":")
                    if endpoint not in usage:
                        usage[endpoint] = {}
                    
                    # Calculate limits
                    if period == "minute":
                        limit = self.endpoint_rate_limits.get(
                            endpoint, self.default_rate_limits
                        )["requests_per_minute"]
                    elif period == "hour":
                        limit = self.endpoint_rate_limits.get(
                            endpoint, self.default_rate_limits
                        )["requests_per_hour"]
                    else:  # day
                        limit = self.endpoint_rate_limits.get(
                            endpoint, self.default_rate_limits
                        ).get("requests_per_day", self.default_rate_limits["requests_per_day"])
                    
                    usage[endpoint][f"{period}_used"] = max(0, limit - int(tokens))
                    usage[endpoint][f"{period}_limit"] = limit
        
        return usage

# Global instance
rate_limiter = RateLimiter()