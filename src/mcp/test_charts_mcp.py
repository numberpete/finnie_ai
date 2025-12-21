# test_charts_mcp.py
# Test script for Chart Tools MCP server

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_chart_tools():
    """Test the Chart Tools MCP server."""
    
    print("\n" + "="*70)
    print("CHART TOOLS MCP SERVER TEST")
    print("="*70 + "\n")
    
    # Configure MCP client
    sse_config = {
        "chart_tools": {
            "transport": "sse",
            "url": "http://localhost:8003/sse"
        }
    }
    
    try:
        client = MultiServerMCPClient(sse_config)
        tools = await client.get_tools()
        
        print(f"✅ Connected! Found {len(tools)} tools:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description[:60]}...")
        
        print("\n" + "="*70)
        print("TEST 1: Create Pie Chart")
        print("="*70)
        
        pie_tool = next(t for t in tools if t.name == "create_pie_chart")
        result1 = await pie_tool.ainvoke({
            "labels": ["Stocks", "Bonds", "Cash"],
            "values": [60, 30, 10],
            "title": "Portfolio Allocation"
        })
        
        # Parse the result - it comes as a list with text content
        import json
        if isinstance(result1, list) and len(result1) > 0:
            result1_data = json.loads(result1[0]['text'])
        else:
            result1_data = result1
            
        print(f"✅ Created: {result1_data}")
        print(f"   View at: http://localhost:8010/chart/{result1_data['filename']}")
        
        print("\n" + "="*70)
        print("TEST 2: Create Bar Chart")
        print("="*70)
        
        bar_tool = next(t for t in tools if t.name == "create_bar_chart")
        result2 = await bar_tool.ainvoke({
            "categories": ["2020", "2021", "2022", "2023"],
            "values": [50000, 55000, 62000, 70000],
            "title": "Annual Savings",
            "ylabel": "Amount ($)"
        })
        
        if isinstance(result2, list) and len(result2) > 0:
            result2_data = json.loads(result2[0]['text'])
        else:
            result2_data = result2
            
        print(f"✅ Created: {result2_data}")
        print(f"   View at: http://localhost:8010/chart/{result2_data['filename']}")
        
        print("\n" + "="*70)
        print("TEST 3: Create Line Chart")
        print("="*70)
        
        line_tool = next(t for t in tools if t.name == "create_line_chart")
        result3 = await line_tool.ainvoke({
            "x_values": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "y_values": [10000, 12000, 11500, 13000, 14500, 15000],
            "title": "Monthly Portfolio Value",
            "ylabel": "Value ($)"
        })
        
        if isinstance(result3, list) and len(result3) > 0:
            result3_data = json.loads(result3[0]['text'])
        else:
            result3_data = result3
            
        print(f"✅ Created: {result3_data}")
        print(f"   View at: http://localhost:8010/chart/{result3_data['filename']}")
        
        print("\n" + "="*70)
        print("TEST 4: Create Multi-Line Chart")
        print("="*70)
        
        multi_line_tool = next(t for t in tools if t.name == "create_multi_line_chart")
        result4 = await multi_line_tool.ainvoke({
            "x_values": ["2020", "2021", "2022", "2023"],
            "y_series": {
                "Stocks": [40000, 45000, 50000, 60000],
                "Bonds": [20000, 22000, 25000, 28000],
                "Cash": [5000, 5500, 6000, 7000]
            },
            "title": "Investment Growth by Asset Class",
            "ylabel": "Value ($)"
        })
        
        if isinstance(result4, list) and len(result4) > 0:
            result4_data = json.loads(result4[0]['text'])
        else:
            result4_data = result4
            
        print(f"✅ Created: {result4_data}")
        print(f"   View at: http://localhost:8010/chart/{result4_data['filename']}")
        
        print("\n" + "="*70)
        print("TEST 5: Create Goal Projection Chart")
        print("="*70)
        
        goal_tool = next(t for t in tools if t.name == "create_goal_projection_chart")
        result5 = await goal_tool.ainvoke({
            "current_value": 50000,
            "goal_value": 1000000,
            "years": 30,
            "monthly_contribution": 1000,
            "annual_return_rate": 0.07,
            "title": "Retirement Goal Projection"
        })
        
        if isinstance(result5, list) and len(result5) > 0:
            result5_data = json.loads(result5[0]['text'])
        else:
            result5_data = result5
            
        print(f"✅ Created: {result5_data}")
        print(f"   Goal reached: {result5_data.get('goal_reached', 'Unknown')}")
        print(f"   Final value: {result5_data.get('final_value', 'N/A')}")
        print(f"   View at: http://localhost:8010/chart/{result5_data['filename']}")
        
        print("\n" + "="*70)
        print("TEST 6: List All Charts")
        print("="*70)
        
        list_tool = next(t for t in tools if t.name == "list_generated_charts")
        charts_result = await list_tool.ainvoke({})
        
        if isinstance(charts_result, list) and len(charts_result) > 0:
            charts = json.loads(charts_result[0]['text'])
        else:
            charts = charts_result
            
        print(f"✅ Found {charts['chart_count']} charts:")
        for i, chart_name in enumerate(charts['charts'][:5], 1):
            print(f"   {i}. {chart_name}")
        if charts['chart_count'] > 5:
            print(f"   ... and {charts['chart_count'] - 5} more")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nTo view all charts, visit:")
        print("  • Image Server: http://localhost:8010/charts")
        print("  • Chart Directory:", charts['chart_directory'])
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("  1. Chart MCP server is running: python -m src.mcp.chart_tools_mcp")
        print("  2. Image server is running: python -m src.servers.image_server")
        print("  3. Port 8003 is available")

if __name__ == "__main__":
    asyncio.run(test_chart_tools())