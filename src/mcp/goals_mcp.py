# src/mcp/charts_mcp.py (PATCHED VERSION with data validation)

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging

# Setup tracing and logging
setup_tracing("mcp-server-goals", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, service_name="mcp-server-goals")
ASSET_KEYS = ["Equities", "Fixed_Income", "Real_Estate", "Commodities", "Crypto", "Cash"]

# Initialize FastMCP
mcp = FastMCP("Goals Server")

def analyze_correlated_portfolio(portfolio, target_goal=500000, years=10, sims=1000) -> Dict[str, Any]:
    """
    Simulates a 6-asset portfolio with correlated returns and goal-seeking.
    
    Returns:
    - Success Probability
    - Portfolio breakdown for Median, Top 10%, and Bottom 10% cases.
    """
    
    # 1. Market Assumptions (Mu and Sigma)
    mu = np.array([0.085, 0.040, 0.060, 0.050, 0.200, 0.025])
    sigma = np.array([0.19, 0.06, 0.15, 0.16, 0.75, 0.01])
    
    # 2. 6x6 Correlation Matrix 
    # (Captures how Gold/Commodities and Bonds hedge against Stocks)
    corr_matrix = np.array([
        [1.0,  0.1,  0.6,  0.2,  0.5,  0.0],  # Equities
        [0.1,  1.0,  0.1, -0.1, -0.1,  0.2],  # Fixed_Income
        [0.6,  0.1,  1.0,  0.3,  0.2,  0.0],  # Real_Estate
        [0.2, -0.1,  0.3,  1.0,  0.2,  0.0],  # Commodities
        [0.5, -0.1,  0.2,  0.2,  1.0,  0.0],  # Crypto
        [0.0,  0.2,  0.0,  0.0,  0.0,  1.0]   # Cash
    ])

    # 3. Cholesky Decomposition
    L = np.linalg.cholesky(corr_matrix)
    
    total_start = sum(portfolio.values())
    initial_weights = np.array([portfolio.get(k, 0) for k in ASSET_KEYS])
    
    # 4. Simulation Engine
    paths = np.zeros((sims, years + 1))
    comp_paths = np.zeros((sims, years + 1, 6))
    comp_paths[:, 0, :] = initial_weights
    paths[:, 0] = total_start

    for t in range(1, years + 1):
        z = np.random.standard_normal((sims, 6)) @ L.T
        growth = np.exp((mu - 0.5 * sigma**2) + sigma * z)
        comp_paths[:, t, :] = comp_paths[:, t-1, :] * growth
        paths[:, t] = comp_paths[:, t, :].sum(axis=1)

    # 5. Extract Final Statistics
    final_vals = paths[:, -1]
    prob_success = (np.sum(final_vals >= target_goal) / sims) * 100
    
    # Determine indices for specific scenarios
    idx_med = (np.abs(final_vals - np.percentile(final_vals, 50))).argmin()
    idx_bot = (np.abs(final_vals - np.percentile(final_vals, 10))).argmin()
    idx_top = (np.abs(final_vals - np.percentile(final_vals, 90))).argmin()

    def get_breakdown(idx):
        return {ASSET_KEYS[i]: round(comp_paths[idx, -1, i], 2) for i in range(6)}

    return {
        "goal_analysis": {
            "target": target_goal,
            "success_probability": f"{prob_success:.2f}%"
        },
        "median_scenario": {
            "total": round(final_vals[idx_med], 2),
            "portfolio": get_breakdown(idx_med)
        },
        "bottom_10_percent_scenario": {
            "total": round(final_vals[idx_bot], 2),
            "portfolio": get_breakdown(idx_bot)
        },
        "top_10_percent_scenario": {
            "total": round(final_vals[idx_top], 2),
            "portfolio": get_breakdown(idx_top)
        }
    }


# ============================================================================
# TOOLS
# ============================================================================

@mcp.tool()
def simple_monte_carlo_simulation(
    portfolio: Dict[str, Any], 
    target_goal=500000, 
    years=10, 
    sims=1000) -> Dict[str, Any]:
    """
    Simulates a 6-asset portfolio with correlated returns and goal-seeking.
    Args:
    - portfolio: portfolio containing 6 asset classes and amounts for each asset class
    - target_goal: total amount we'd like to have portfolio grow to after the specified number of years
    - years: number of years to cover in our simulation
    - sims: number of simulations to conduct 
    Returns:
    - Success Probability of hitting the target goal
    - Portfolio breakdown for Median, Top 10%, and Bottom 10% cases.
    """
    LOGGER.info(f"Doing simple simulation on portfolio")
    
    # Normalize portfolio keys: replace spaces with underscores
    normalized_portfolio = {
        key.replace(" ", "_"): value 
        for key, value in portfolio.items()
    }

    return analyze_correlated_portfolio(normalized_portfolio, target_goal, years, sims)

@mcp.tool()
def get_asset_classes() -> List[str]:
    """
    Returns the 6 asset classes we use for all portfolio analysis.

    Returns:
        Portfolio asset class keys
    """
    LOGGER.info(f"Calling get_asset_classes")

    return ASSET_KEYS


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
    LOGGER.info("Starting Goals MCP Server on port 8004")
    mcp.run(transport="sse", port=8004)