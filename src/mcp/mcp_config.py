# src/mcp/mcp_config.py
# Helper to get MCP server URLs from environment variables (for Docker support)

import os

def get_mcp_url(service_name: str) -> str:
    """
    Get MCP server URL from environment variable or fall back to localhost.
    
    In Docker, set environment variables like:
        QANDA_MCP_HOST=qanda-mcp
        YFINANCE_MCP_HOST=yfinance-mcp
        CHARTS_MCP_HOST=charts-mcp
        GOALS_MCP_HOST=goals-mcp
        PORTFOLIO_MCP_HOST=portfolio-mcp
    
    For local development, these default to localhost.
    """
    
    service_map = {
        "qanda": ("QANDA_MCP_HOST", "localhost", 8001),
        "finance_qanda": ("QANDA_MCP_HOST", "localhost", 8001),
        "yfinance": ("YFINANCE_MCP_HOST", "localhost", 8002),
        "charts": ("CHARTS_MCP_HOST", "localhost", 8003),
        "goals": ("GOALS_MCP_HOST", "localhost", 8004),
        "portfolio": ("PORTFOLIO_MCP_HOST", "localhost", 8005),
    }
    
    if service_name not in service_map:
        raise ValueError(f"Unknown MCP service: {service_name}")
    
    env_var, default_host, port = service_map[service_name]
    host = os.environ.get(env_var, default_host)
    
    return f"http://{host}:{port}/sse"


# Pre-built configurations for each agent
def get_finance_qanda_mcp_servers():
    return {
        "finance_qanda_tool": {
            "url": get_mcp_url("qanda"),
            "description": "Financial Q&A vector lookup"
        }
    }

def get_finance_market_mcp_servers():
    return {
        "yfinance_mcp": {
            "url": get_mcp_url("yfinance"),
            "description": "Market data from yFinance"
        },
        "charts_mcp": {
            "url": get_mcp_url("charts"),
            "description": "Chart generation tools"
        }
    }

def get_finance_portfolio_mcp_servers():
    return {
        "charts_mcp": {
            "url": get_mcp_url("charts"),
            "description": "Chart generation tools"
        },
        "portfolio_mcp": {
            "url": get_mcp_url("portfolio"),
            "description": "Portfolio building and assessment tools"
        },
        "yfinance_mcp": {
            "url": get_mcp_url("yfinance"),
            "description": "yFinance tools to look up stock/company/fund symbols and get asset allocations for tickers"
        }
    }

def get_finance_goals_mcp_servers():
    return {
        "charts_mcp": {
            "url": get_mcp_url("charts"),
            "description": "Chart generation tools"
        },
        "goals_mcp": {
            "url": get_mcp_url("goals"),
            "description": "Goals tools"
        }
    }