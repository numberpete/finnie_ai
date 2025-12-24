# src/mcp/portfolio_mcp.py (PATCHED VERSION with data validation)

from typing import  Dict, Any
from fastmcp import FastMCP
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
from mcp.types import TextContent


# Setup tracing and logging
setup_tracing("mcp-server-portfolio", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, service_name="mcp-server-portfolio")
ASSET_KEYS = ["Equities", "Fixed_Income", "Real_Estate", "Commodities", "Crypto", "Cash"]

# Initialize FastMCP
mcp = FastMCP("Portfolio Server")

@mcp.tool()
def get_new_portfolio() -> Dict[str, float]:
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
def add_to_portfolio(portfolio: Dict[str,float], additions: Dict[str,float]) -> Dict[str,float]:
    """
    Adds the asset amounts provided (additions) to the portfolio provided.  Both arguments are portfolios, the first
    being the current_portfolio, the second containing amounts to add to the portfolio.

    Can also be to subtract from portfolio assets by passing in negative values in the additions.

    Args:
        portfolio: the portfolio to have amounts added to
        additions: a "portfolio" with amounts to add to the portfolio
    Returns:
        a portfolio with the asset class amounts from the two arguments added together
    """
    LOGGER.info(f"Adding ${sum(additions.values()):,.2f} to portfolio${sum(portfolio.values()):,.2f}.")

    updated_portfolio = {
        key: value + additions.get(key, 0)
        for key, value in portfolio.items()
    }
    
    LOGGER.info(f"Updated portfolio total: ${sum(updated_portfolio.values()):,.2f}")
    
    return updated_portfolio

@mcp.tool()
def add_to_portfolio_asset_class(asset_class_key: str, amount: float, portfolio: Dict[str,float]) -> Dict[str, float]:
    """
    Add a specific amount for a single asset class to the portfolio.

    Args:
        asset_class_key: the asset class to add the value to
        amount: the amount to add to the asset class
        portfolio: the portfolio which contains the asset class being added to
    """
    LOGGER.info(f"Adding ${amount:,.2f} to {asset_class_key}.")

    try:
        asset_class_key = asset_class_key.replace(" ","_")
        portfolio[asset_class_key] += amount
    except KeyError:
        raise ValueError(f"{asset_class_key} is not an asset class in the portfolio")        

    LOGGER.info(f"Updated portfolio total: ${sum(portfolio.values()):,.2f}")

    return portfolio

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

@mcp.tool()
def assess_risk_tolerance(portfolio):
    """
    Calculates the weighted volatility of a 6-asset portfolio and 
    maps it to a risk tolerance tier.
    """
    # 1. Standard Volatility Assumptions (Annual Sigma)
    # Higher sigma = more risk/swing
    vol_map = {
        "Equities": 0.18,       # 18% annual swing
        "Fixed_Income": 0.06,   # 6% annual swing
        "Real_Estate": 0.12,    # 12% annual swing
        "Commodities": 0.15,    # 15% annual swing
        "Crypto": 0.70,         # 70% annual swing
        "Cash": 0.01            # 1% annual swing
    }

    total_val = sum(portfolio.values())
    if total_val == 0:
        return "Empty Portfolio", 0.0

    # 2. Calculate Weighted Portfolio Volatility
    # This is a simplified linear weight for a quick assessment tool
    p_vol = sum((val / total_val) * vol_map[ac] for ac, val in portfolio.items() if ac in vol_map)

    # 3. Map Volatility to Risk Tier
    if p_vol < 0.04:
        tier = "Conservative (Preservation focused)"
    elif p_vol < 0.09:
        tier = "Moderate-Conservative (Income focused)"
    elif p_vol < 0.14:
        tier = "Moderate (Balanced Growth)"
    elif p_vol < 0.20:
        tier = "Aggressive (Growth focused)"
    else:
        tier = "Very Aggressive (Speculative/High Growth)"

    return {
        "weighted_volatility": f"{p_vol:.2%}",
        "risk_tolerance_tier": tier,
        "primary_risk_driver": max(portfolio, key=portfolio.get)
    }

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    LOGGER.info("Starting Portfolio MCP Server on port 8005")
    mcp.run(transport="sse", port=8005)