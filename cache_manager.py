# cache_manager.py
import time
import json
import hashlib
import threading
from typing import Any, Dict, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta

class CacheManager:
    """
    A simple in-memory cache manager with time-based expiration.
    This can be expanded later to use Redis or another distributed cache.
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the cache manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the cache store and config."""
        # Only execute if this is the first instance
        if CacheManager._instance is None:
            self._cache: Dict[str, Tuple[Any, float]] = {}  # {key: (value, expiry_time)}
            self._stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "invalidations": 0
            }
            
            # Default cache settings
            self.default_ttl = 300  # 5 minutes in seconds
            self.max_items = 1000
            
            # Start the cleanup thread
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start a thread to periodically clean up expired items."""
        cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Periodically remove expired items from the cache."""
        while True:
            time.sleep(60)  # Check every minute
            try:
                now = time.time()
                with self._lock:
                    # Get keys to remove (items that have expired)
                    keys_to_remove = [
                        key for key, (_, expiry) in self._cache.items()
                        if expiry < now
                    ]
                    
                    # Remove expired items
                    for key in keys_to_remove:
                        del self._cache[key]
                    
                    # If still over max items, remove oldest
                    if len(self._cache) > self.max_items:
                        sorted_items = sorted(
                            self._cache.items(), 
                            key=lambda x: x[1][1]  # Sort by expiry time
                        )
                        # Keep only the newest max_items
                        self._cache = dict(sorted_items[-self.max_items:])
            except Exception as e:
                print(f"Error in cache cleanup: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    self._stats["hits"] += 1
                    return value
                # Remove if expired
                del self._cache[key]
            
            self._stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None uses default)
        """
        with self._lock:
            expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
            self._cache[key] = (value, expiry)
            self._stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["invalidations"] += 1
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: String pattern to match (simple contains check)
            
        Returns:
            Number of keys invalidated
        """
        with self._lock:
            keys_to_remove = [key for key in self._cache if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]
            
            self._stats["invalidations"] += len(keys_to_remove)
            return len(keys_to_remove)
    
    def clear(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats["invalidations"] += count
            return count
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["size"] = len(self._cache)
            stats["hit_ratio"] = (
                stats["hits"] / (stats["hits"] + stats["misses"]) * 100 
                if stats["hits"] + stats["misses"] > 0 else 0
            )
            return stats

# Helper functions for common caching patterns

def generate_cache_key(prefix: str, params: Dict[str, Any]) -> str:
    """
    Generate a deterministic cache key from parameters.
    
    Args:
        prefix: Key prefix
        params: Dictionary of parameters
        
    Returns:
        Cache key string
    """
    # Sort to ensure deterministic key generation
    param_str = json.dumps(params, sort_keys=True)
    # Use hash to keep keys reasonable length
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"{prefix}:{param_hash}"

def cached_query(ttl: int = 300):
    """
    Decorator for caching database query results.
    
    Args:
        ttl: Cache time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_prefix = f"query:{func.__name__}"
            
            # Extract non-function arguments for cache key
            cache_params = {}
            for i, arg in enumerate(args):
                if not callable(arg):
                    cache_params[f"arg{i}"] = str(arg)
            
            # Add keyword arguments
            for key, value in kwargs.items():
                if not callable(value):
                    cache_params[key] = str(value)
            
            cache_key = generate_cache_key(cache_prefix, cache_params)
            
            # Try to get from cache
            cache_manager = CacheManager.get_instance()
            cached_result = cache_manager.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Cache miss, execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def invalidate_entity_cache(entity_id: Union[int, str]):
    """
    Invalidate all cache entries related to an entity.
    
    Args:
        entity_id: Entity ID or name
    """
    cache_manager = CacheManager.get_instance()
    # Invalidate by ID
    cache_manager.invalidate_pattern(f"entity:{entity_id}")
    # If string name provided, invalidate by name too
    if isinstance(entity_id, str):
        cache_manager.invalidate_pattern(f"entity:{entity_id}")

# Initialize the cache manager instance on module import
cache_manager = CacheManager.get_instance()