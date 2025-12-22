# src/mcp/yfinance_mcp.py

import yfinance as yf
import time
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any
from fastmcp import FastMCP
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging
from src.utils.cache import TTLCache

# Setup tracing and logging
setup_tracing("mcp-server-yfinance", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.INFO)


# Initialize FastMCP
mcp = FastMCP("yFinance Market Data Server")

# Global cache instance
market_cache = TTLCache(default_ttl_seconds=1800, name="yfinance-mcp-cache")  # 30 minutes


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

KNOWN_COMMODITIES_FUNDS = {
    # Broad Commodities
    "DBC", "PDBC", "GSG", "USCI", "DJP", "BCI", "COMT", "COMB",
    
    # Precious Metals
    "GLD", "IAU", "GLDM", "SGOL", "BAR", "AAAU",  # Gold
    "SLV", "PSLV", "SIVR",  # Silver
    "PPLT", "PALL",  # Platinum/Palladium
    "GLTR",  # Multi-metal
    
    # Energy
    "USO", "BNO", "UCO", "SCO", "USL", "OILK",  # Oil
    "UNG", "BOIL", "KOLD", "GAZ",  # Natural Gas
    "UGA",  # Gasoline
    
    # Agricultural
    "DBA", "WEAT", "CORN", "SOYB", "CANE", "NIB", "COW", "JO",
    
    # Industrial Metals
    "CPER", "JJC", "COPX", "JJU", "JJN", "LD",
}

KNOWN_CRYPTO = {
    "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "SOL-USD", "TRX-USD", "DOT-USD",
    "MATIC-USD", "LTC-USD", "SHIB-USD", "AVAX-USD", "UNI-USD",
    "LINK-USD", "XLM-USD", "ATOM-USD", "ETC-USD", "XMR-USD",
    # Add more as needed
}

def is_commodities_fund(ticker) -> bool:
    """Check if a ticker is a commodities fund."""
    info = ticker.info
    
    # Check 1: Name/category keywords
    fund_name = info.get("longName", "").lower()
    category = info.get("category", "").lower()
    
    commodities_keywords = ["commodities", "commodity", "gold", "silver", 
                           "precious metals", "natural resources"]
    
    if any(keyword in fund_name or keyword in category 
           for keyword in commodities_keywords):
        return True
    
    # Check 2: Known commodities tickers
    if info.get("symbol").upper() in KNOWN_COMMODITIES_FUNDS:
        return True
    
    # Check 3: Morningstar category
    if "commodit" in category.lower() or "precious metal" in category.lower():
        return True
    
    return False

def is_crypto(ticker) -> bool:
    """
    Determine if a ticker symbol is cryptocurrency.
    
    Args:
        symbol: Ticker symbol to check
        
    Returns:
        True if crypto, False otherwise
    """
    try:
        info = ticker.info
        
        # Check 1: Quote type (most reliable)
        quote_type = info.get("quoteType", "")
        if quote_type == "CRYPTOCURRENCY":
            return True
        
        # Check 2: Exchange
        exchange = info.get("exchange", "")
        if exchange in ["CCC", "CCY", "CryptoCurrency"]:
            return True
        
        # Check 3: Symbol suffix pattern
        crypto_suffixes = ["-USD", "-EUR", "-GBP", "-USDT", "-BUSD"]
        if any(info.get("symbol").upper().endswith(suffix) for suffix in crypto_suffixes):
            return True
        
        return False
        
    except Exception as e:
        # Fallback: check suffix pattern
        crypto_suffixes = ["-USD", "-EUR", "-GBP", "-USDT", "-BUSD"]
        return any(info.get("symbol").upper().endswith(suffix) for suffix in crypto_suffixes)
    
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
def get_asset_classes(symbol: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Look up the asset class breakdown for a ticker.
    
    Args:
        symbol: symbol for the ticker (e.g, 'VFIAX','VFORX')
        use_cache:  Whether to use cached data (default: True)

    Returns: 
        Dictionary with asset classes mapped to the ratio (0 to 1) of the fund they comprise,
        or mock data if API call fails
    """
    LOGGER.info(f"get_asset_classes called: symbol={symbol}, use_cache={use_cache}")

    cache_key = f"asset_ticker:{symbol.upper()}"

    # Check cache first
    if use_cache:
        cached_data = market_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

    try:
        def fetch_allocation():
            data = {
                "symbol": symbol.upper(),
                "Equities": 0.0,
                "Fixed_Income": 0.0,
                "Cash": 0.0,
                "Real_Estate": 0.0,
                "Commodities": 0.0,
                "Crypto": 0.0,
                "_mock": False
            }

            ticker = yf.Ticker(symbol)
            if is_crypto(ticker):
                data["Crypto"] = 1.0
            elif is_commodities_fund(ticker):
                data["Commodities"] = 1.0
            elif ticker.funds_data is not None and ticker.funds_data.asset_classes:
                funds_data = ticker.funds_data
                """
                yfinance asset class: our asset class
                * cashPosition: Cash
                * stockPosition: Equities
                * bondPosition: Fixed_Income
                * preferredPosition: Fixed_Income
                * convertiblePosition: 50% Equities/50% Fixed Income
                * otherPosition: 50% Commodities/50% Real_Estate
                """

                if funds_data.asset_classes:
                    for asset_class, ratio in funds_data.asset_classes.items():
                        if asset_class == "cashPosition":
                            data["Cash"] += ratio
                        elif asset_class == "stockPosition":
                            data["Equities"] += ratio
                        elif asset_class == "bondPosition":
                            data["Fixed_Income"] += ratio
                        elif asset_class == "preferredPosition":
                            data["Fixed_Income"] += ratio
                        elif asset_class == "convertiblePosition":
                            data["Equities"] += ratio/2
                            data["Fixed_Income"] += ratio/2
                        elif asset_class == "otherPosition":
                            data["Real_Estate"] += ratio/2
                            data["Commodities"] += ratio/2
            else:
                sector = ticker.info.get("sector")
                industry = ticker.info.get("industry")
                if sector == "Real Estate" and industry and industry.startswith("REIT -"):
                    data["Real_Estate"] = 1.0
                else:
                    data["Equities"] = 1.0

            # Normalization Step
            total_ratio = data["Equities"] + data["Fixed_Income"] + data["Cash"] + \
                        data["Real_Estate"] + data["Commodities"] + data["Crypto"]

            if 0 < total_ratio != 1.0:
                for key in ["Equities", "Fixed_Income", "Cash", "Real_Estate", "Commodities", "Crypto"]:
                    data[key] /= total_ratio

            return data
        
        # Execute with retry logic
        result = retry_with_backoff(
            fetch_allocation,
            max_retries=3,
            base_delay=1.0
        )
        
        # Cache the result
        market_cache.set(cache_key, result, ttl_seconds=1800)
        
        LOGGER.info(f"Successfully fetched allocations for {symbol}")
        LOGGER.debug("Fetched data: {result}")
        return result
    
    except Exception as e:
        LOGGER.error(f"Failed to fetch {symbol} after retries: {str(e)}")
        
        # Return mock data as fallback
        LOGGER.warning(f"Returning mock allocation for {symbol}")
        mock_data =  {
                "symbol": symbol.upper(),
                "Equities": 0.6,
                "Fixed_Income": 0.4,
                "Cash": 0.0,
                "Real_Estate": 0.0,
                "Commodities": 0.0,
                "Crypto": 0.0,
                "error": str(e),
                "message": "allocation data unavailable",
                "_mock": True
        }
        mock_data["error"] = str(e)
        
        return mock_data

@mcp.tool()
def get_ticker_quote(symbol: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Get current stock price and basic information.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dictionary with stock price and info, or mock data if API fails
    """
    LOGGER.info(f"get_ticker_quote called: symbol={symbol}, use_cache={use_cache}")
    
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
                "all_time_high": info.get("allTimeHigh"),
                "all_time_low": info.get("allTimeLow"),
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
        LOGGER.debug("Fetched data: {result}")
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
def get_ticker_history(
    symbol: str,
    period: str = "1mo",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Get historical ticker price data.
    
    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'MSFT', '^DJI')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max')
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dictionary with historical data or error message
    """
    LOGGER.info(f"get__history called: symbol={symbol}, period={period}")
    
    # Create cache key
    cache_key = f"ticker_history:{symbol.upper()}:{period}"
    
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
                "period_start_date": hist.index[0].isoformat(),
                "period_end_date": hist.index[-1].isoformat(),
                "period_open": float(hist["Open"].iloc[0]),
                "period_close": float(hist["Close"].iloc[-1]),
                "period_volume": int(hist["Volume"].sum()),
                "period_high": float(hist["High"].max()),
                "period_low": float(hist["Low"].min()),
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
            data = get_ticker_quote(symbol, use_cache=False)
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