# src/mcp/portfolio_mcp.py (PATCHED VERSION with data validation)

import os
import json  # ✅ ADD THIS - needed for get_portfolio_summary
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging
from src.utils.cache import TTLCache
from mcp.types import TextContent


# Setup tracing and logging
setup_tracing("mcp-server-portfolio", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, service_name="mcp-server-portfolio")
ASSET_KEYS = ["Equities", "Fixed_Income", "Real_Estate", "Commodities", "Crypto", "Cash"]

# Initialize FastMCP
mcp = FastMCP("Portfolio Server")

@mcp.tool()
def get_new_portfolio() -> Dict[str, Any]:
    """
    Returns a 6-asset portfolio with all assets at 0.
    Returns:
    - a new portfolio
    """
    LOGGER.info(f"Getting new portfolio")
    
    return {
        "Equities": 0.0,
        "Fixed_Income": 0.0,
        "Cash": 0.0,
        "Real_Estate": 0.0,
        "Commodities": 0.0,
        "Crypto": 0.0
    }

@mcp.tool()
def add_to_portfolio_with_allocation(
    amount: float,  # ✅ ADD TYPE HINTS
    portfolio: Dict[str, float], 
    asset_allocation: Dict[str, float]
) -> Dict[str, float]:  # ✅ FIX RETURN TYPE
    """
    Adds the amount given to the portfolio provided according to the asset allocation provided.
    Returns the updated portfolio.

    Args:
        amount: amount to add to the portfolio in dollars (can be negative to subtract)
        portfolio: the portfolio with amounts broken out into 6 asset classes
        asset_allocation: dictionary with the asset allocation with ratios from 0 to 1.

    Returns:
        Updated portfolio

    Example:
        add_to_portfolio_with_allocation(
            amount=400000, 
            portfolio={"Equities":0.0, "Fixed_Income":0.0, "Cash":0.0, "Real_Estate":0.0, "Commodities":0.0, "Crypto":0.0},
            asset_allocation={"Equities":0.7, "Fixed_Income":0.2, "Real_Estate":0.1}    
        ) -> {"Equities":280000.0, "Fixed_Income":80000.0, "Cash":0.0, "Real_Estate":40000.0, "Commodities":0.0, "Crypto":0.0}
    """
    LOGGER.info(f"Adding {amount} to portfolio with allocation: {asset_allocation}")

    updated_portfolio = {
        key: value + (amount * asset_allocation.get(key, 0))
        for key, value in portfolio.items()
    }
    
    LOGGER.info(f"Updated portfolio total: ${sum(updated_portfolio.values()):,.2f}")
    
    return updated_portfolio


@mcp.tool()
def get_portfolio_summary(portfolio: Dict[str, float]) -> Dict[str, Any]:  # ✅ CHANGE RETURN TYPE
    """
    Get summary statistics for portfolio.
    
    Args:
        portfolio: Portfolio dict
    
    Returns:
        Summary with total value and percentage allocation
    """
    total = sum(portfolio.values())
    
    if total == 0:
        percentages = {k: 0 for k in portfolio.keys()}
    else:
        percentages = {k: (v / total) * 100 for k, v in portfolio.items()}
    
    # ✅ RETURN DICT DIRECTLY (FastMCP handles serialization)
    return {
        "success": True,
        "total_value": total,
        "asset_values": portfolio,
        "asset_percentages": percentages,
        "asset_count": len([v for v in portfolio.values() if v > 0])
    }


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    LOGGER.info("Starting Portfolio MCP Server on port 8005")
    mcp.run(transport="sse", port=8005)