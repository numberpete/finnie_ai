# src/mcp/yfinance_mcp.py

import yfinance as yf
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastmcp import FastMCP
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging

# Setup tracing and logging
setup_tracing("mcp-server-yfinance", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.INFO)

# Initialize FastMCP
mcp = FastMCP("yFinance Market Data Server")

# ============================================================================
# CACHE IMPLEMENTATION
# ============================================================================

class MarketDataCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl_seconds: int = 1800):  # 30 minutes default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
    
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
            LOGGER.debug(f"Cache EXPIRED: {key}")
            del self.cache[key]
            return None
        
        LOGGER.info(f"Cache HIT: {key}")
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
        LOGGER.debug(f"Cache SET: {key} (TTL: {ttl}s)")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        LOGGER.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = len(self.cache)
        expired = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired
        }


# Global cache instance
market_cache = MarketDataCache(default_ttl_seconds=1800)  # 30 minutes


# ============================================================================
# RETRY MECHANISM WITH EXPONENTIAL BACKOFF
# ============================================================================

def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    *args,
    **kwargs
):
    """
    Execute function with exponential backoff retry logic.
    
    Args:
        func: Function to execute (must be synchronous)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result from func or raises last exception
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            LOGGER.debug(f"Attempt {attempt + 1}/{max_retries}")
            # Call the function directly (it should be a callable that returns immediately)
            result = func(*args, **kwargs)
            return result
        
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                LOGGER.error(f"All {max_retries} attempts failed: {str(e)}")
    
    raise last_exception


# ============================================================================
# MOCK DATA FOR FALLBACK
# ============================================================================

MOCK_DATA = {
    "AAPL": {
        "price": 185.50,
        "change": 2.30,
        "percent_change": 1.25,
        "volume": 50000000,
        "market_cap": 2900000000000,
        "pe_ratio": 29.5,
        "52week_high": 199.62,
        "52week_low": 164.08
    },
    "SPY": {
        "price": 450.25,
        "change": 5.75,
        "percent_change": 1.29,
        "volume": 75000000,
        "market_cap": None,
        "pe_ratio": None,
        "52week_high": 475.00,
        "52week_low": 380.00
    }
}


def get_mock_data(symbol: str) -> Dict[str, Any]:
    """Return mock data for a symbol."""
    if symbol.upper() in MOCK_DATA:
        data = MOCK_DATA[symbol.upper()].copy()
        data["_mock"] = True
        data["_timestamp"] = datetime.now().isoformat()
        return data
    
    # Generic mock data for unknown symbols
    return {
        "price": 100.00,
        "change": 0.50,
        "percent_change": 0.50,
        "volume": 1000000,
        "_mock": True,
        "_timestamp": datetime.now().isoformat()
    }


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
def get_stock_price(symbol: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Get current stock price and basic information.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dictionary with stock price and info, or mock data if API fails
    """
    LOGGER.info(f"get_stock_price called: symbol={symbol}, use_cache={use_cache}")
    
    # Create cache key
    cache_key = f"stock_price:{symbol.upper()}"
    
    # Check cache first
    if use_cache:
        cached_data = market_cache.get(cache_key)
        if cached_data is not None:
            return cached_data
    
    # Try to fetch from yFinance with retry
    try:
        def fetch_stock_data():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key data
            data = {
                "symbol": symbol.upper(),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "change": info.get("regularMarketChange"),
                "percent_change": info.get("regularMarketChangePercent"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52week_high": info.get("fiftyTwoWeekHigh"),
                "52week_low": info.get("fiftyTwoWeekLow"),
                "company_name": info.get("longName", info.get("shortName")),
                "currency": info.get("currency", "USD"),
                "timestamp": datetime.now().isoformat(),
                "_mock": False
            }
            
            return data
        
        # Execute with retry logic
        result = retry_with_backoff(
            fetch_stock_data,
            max_retries=3,
            base_delay=1.0
        )
        
        # Cache the result
        market_cache.set(cache_key, result, ttl_seconds=1800)
        
        LOGGER.info(f"Successfully fetched data for {symbol}")
        return result
    
    except Exception as e:
        LOGGER.error(f"Failed to fetch {symbol} after retries: {str(e)}")
        
        # Return mock data as fallback
        LOGGER.warning(f"Returning mock data for {symbol}")
        mock_data = get_mock_data(symbol)
        mock_data["symbol"] = symbol.upper()
        mock_data["error"] = str(e)
        
        return mock_data


@mcp.tool()
def get_stock_history(
    symbol: str,
    period: str = "1mo",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Get historical stock price data.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max')
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dictionary with historical data or error message
    """
    LOGGER.info(f"get_stock_history called: symbol={symbol}, period={period}")
    
    # Create cache key
    cache_key = f"stock_history:{symbol.upper()}:{period}"
    
    # Check cache
    if use_cache:
        cached_data = market_cache.get(cache_key)
        if cached_data is not None:
            return cached_data
    
    try:
        def fetch_history():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            # Convert to serializable format
            data = {
                "symbol": symbol.upper(),
                "period": period,
                "data": [
                    {
                        "date": str(date.date()),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"])
                    }
                    for date, row in hist.iterrows()
                ],
                "timestamp": datetime.now().isoformat(),
                "_mock": False
            }
            
            return data
        
        result = retry_with_backoff(fetch_history, max_retries=3)
        
        # Cache with 30-minute TTL
        market_cache.set(cache_key, result, ttl_seconds=1800)
        
        LOGGER.info(f"Successfully fetched history for {symbol}")
        return result
    
    except Exception as e:
        LOGGER.error(f"Failed to fetch history for {symbol}: {str(e)}")
        return {
            "symbol": symbol.upper(),
            "period": period,
            "error": str(e),
            "_mock": True,
            "message": "Historical data unavailable. Please try again later."
        }

@mcp.tool()
def get_ticker(company_name: str) -> Dict[str, Any]:
    """
    Get stock ticker symbol from company name.
    
    Args:
        company_name: Full or partial company name (e.g., 'Apple Inc')
    Returns:
        Dictionary with ticker symbol or error message
    """
    LOGGER.info(f"get_ticker called: company_name={company_name}")
    
    try:
        def fetch_ticker():
            search_results = yf.Search(company_name, max_results=1)
            if not search_results:
                raise ValueError("No matching ticker found")
            
            # Return the first matching ticker
            ticker_info = search_results.quotes[0]
            return {
                "company_name": company_name,
                "ticker": ticker_info["symbol"],
                "short_name": ticker_info["shortname"],
                "exchange": ticker_info.get("exchange"),
                "timestamp": datetime.now().isoformat(),
                "_mock": False
            }
        
        result = retry_with_backoff(fetch_ticker, max_retries=3)
        
        LOGGER.info(f"Successfully found ticker for {company_name}")
        return result
    
    except Exception as e:
        LOGGER.error(f"Failed to find ticker for {company_name}: {str(e)}")
        return {
            "company_name": company_name,
            "error": str(e),
            "_mock": True,
            "message": "Ticker lookup failed. Please try again later."
        }
    

@mcp.tool()
def get_market_summary(use_cache: bool = True) -> Dict[str, Any]:
    """
    Get summary of major market indices.
    
    Args:
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dictionary with major market indices data
    """
    LOGGER.info("get_market_summary called")
    
    cache_key = "market_summary"
    
    # Check cache
    if use_cache:
        cached_data = market_cache.get(cache_key)
        if cached_data is not None:
            return cached_data
    
    indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
    results = {}
    
    for symbol in indices:
        try:
            data = get_stock_price(symbol, use_cache=False)
            results[symbol] = data
        except Exception as e:
            LOGGER.error(f"Failed to fetch {symbol}: {str(e)}")
            results[symbol] = {"error": str(e), "_mock": True}
    
    summary = {
        "indices": results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache for 30 minutes
    market_cache.set(cache_key, summary, ttl_seconds=1800)
    
    return summary


@mcp.tool()
def clear_cache() -> Dict[str, str]:
    """Clear all cached market data."""
    LOGGER.info("Clearing cache")
    market_cache.clear()
    return {"status": "success", "message": "Cache cleared"}


@mcp.tool()
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    stats = market_cache.get_stats()
    LOGGER.info(f"Cache stats: {stats}")
    return stats


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    LOGGER.info("Starting yFinance MCP Server on port 8002")
    mcp.run(transport="sse", port=8002)