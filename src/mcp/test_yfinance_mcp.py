# test_yfinance_mcp.py
# Quick test script for yFinance tools

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_yfinance_tools():
    """Test the yFinance MCP tools."""
    
    # Configure MCP client
    sse_config = {
        "yfinance_tools": {
            "transport": "sse",
            "url": "http://localhost:8002/sse"
        }
    }
    
    client = MultiServerMCPClient(sse_config)
    tools = await client.get_tools()
    
    print(f"\n✅ Connected! Found {len(tools)} tools:")
    for tool in tools:
        print(f"   • {tool.name}: {tool.description[:60]}...")
    
    # Test 1: Get stock price (should miss cache)
    print("\n" + "="*70)
    print("TEST 1: Get AAPL stock price (cache miss expected)")
    print("="*70)
    
    price_tool = next(t for t in tools if t.name == "get_ticker_quote")
    result1 = await price_tool.ainvoke({"symbol": "AAPL"})
    print(f"Result: {result1}")
    
    # Test 2: Get same stock price (should hit cache)
    print("\n" + "="*70)
    print("TEST 2: Get AAPL stock price again (cache hit expected)")
    print("="*70)
    
    result2 = await price_tool.ainvoke({"symbol": "AAPL"})
    print(f"Result: {result2}")
    
    # Test 3: Get cache stats
    print("\n" + "="*70)
    print("TEST 3: Get cache statistics")
    print("="*70)
    
    stats_tool = next(t for t in tools if t.name == "get_cache_stats")
    stats = await stats_tool.ainvoke({})
    print(f"Cache stats: {stats}")
    
    # Test 4: Get historical data
    print("\n" + "="*70)
    print("TEST 4: Get AAPL historical data (1 month)")
    print("="*70)
    
    history_tool = next(t for t in tools if t.name == "get_ticker_history")
    history = await history_tool.ainvoke({"symbol": "AAPL", "period": "1mo"})
    
    # Handle both dict and list responses
    if isinstance(history, dict):
        data_points = history.get('data', [])
        print(f"Found {len(data_points)} data points")
        if data_points:
            print(f"First data point: {data_points[0]}")
    elif isinstance(history, list):
        print(f"Found {len(history)} data points")
        if history:
            print(f"First data point: {history[0]}")
    else:
        print(f"Unexpected response type: {type(history)}")
        print(f"Response: {history}")
    
    # Test 5: Test fallback with invalid symbol
    print("\n" + "="*70)
    print("TEST 5: Get invalid symbol (fallback expected)")
    print("="*70)
    
    result_invalid = await price_tool.ainvoke({"symbol": "INVALID_SYMBOL_XYZ"})
    print(f"Result: {result_invalid}")
    
    # Handle both dict and other response types
    if isinstance(result_invalid, dict):
        print(f"Is mock data: {result_invalid.get('_mock', False)}")
    else:
        print(f"Response type: {type(result_invalid)}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_yfinance_tools())