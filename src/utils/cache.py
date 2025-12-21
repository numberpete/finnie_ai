from typing import Dict, Any, Optional
from datetime import datetime
import time
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging

# Setup tracing and logging
setup_tracing("ttl-cache", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.INFO)


class TTLCache:
    """Simple in-memory cache with TTL (time-to-live) support."""
    
    def __init__(self, default_ttl_seconds: int = 1800, name: str = "ttl-cache"):  # 30 minutes default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
        self.name = name
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired."""
        expiry_time = entry.get("expires_at", 0)
        return time.time() > expiry_time
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        if self._is_expired(entry):
            LOGGER.debug(f"Cache {self.name} EXPIRED: {key}")
            del self.cache[key]
            return None
        
        LOGGER.info(f"Cache  {self.name} HIT: {key}")
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store value in cache with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "cached_at": datetime.now().isoformat()
        }
        LOGGER.debug(f"Cache {self.name} SET: {key} (TTL: {ttl}s)")
    
    def remove(self, key: str) -> bool:
        """
        Remove a specific entry from cache.
        
        Args:
            key: The cache key to remove
            
        Returns:
            True if the key existed and was removed, False otherwise
        """
        if key in self.cache:
            del self.cache[key]
            LOGGER.debug(f"Cache {self.name} REMOVE: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        LOGGER.info(f"Cache {self.name} cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = len(self.cache)
        expired = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        return {
            "cache": self.name,
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired
        }