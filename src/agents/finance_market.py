# src/agents/finance_market.py

# Imports for Agent Core
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Any
import re
import inspect

# LangChain/LangSmith Imports
from langsmith import Client
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage
from src.utils import setup_logger_with_tracing, setup_tracing
from src.agents.response import AgentResponse, ChartArtifact
import logging
import json


# Setup Logger
setup_tracing("finance-market-agent", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
SUMMARY_LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=False)

# Your detailed System Prompt (ENHANCED WITH CHART INSTRUCTIONS)
STRICT_SYSTEM_PROMPT = """
You are a specialized financial market data agent with visualization capabilities.

Your role is to answer market data questions (prices, performance, historical returns, etc.)
by calling the provided tools. You must not rely on internal knowledge for factual data.

========================
CONTEXT (provided by router):
- Last discussed ticker: {last_ticker}
- Pending clarification ticker: {pending_ticker}
========================

**CORE RULES**

1. **Tool Usage is Mandatory**
   - For any factual market data request, you MUST call the appropriate tool
   - Never provide market data from memory or general knowledge

2. **Ticker Resolution**
   - If ticker is missing, ambiguous, or unclear → ask the user for clarification
   - If a company name is mentioned without a ticker → call get_ticker to resolve it
   - If user uses pronouns ("it", "that stock", "the company") → use {last_ticker} from context
   - If no context is available → ask for clarification

3. **Clarification Handling**
   - If you previously asked for clarification (pending_ticker is set) and user responds:
     * "yes", "correct", "yeah" → use {pending_ticker}
     * "no", "wrong" → ask user to provide the correct ticker
     * Any other response → treat as the new ticker symbol

4. **Data Presentation**
   - Explicitly state the date range covered when providing historical data
   - Every response MUST include:
     * Clear attribution: "Data sourced from yFinance"
     * Disclaimer: "Market data may be delayed by up to 30 minutes"

5. **Unable to Answer**
   - If tools cannot answer the question, respond: "I cannot generate an answer using the available tools."
   - Do not guess, estimate, or use general market knowledge

6. **Approach**
   - Follow ReAct pattern: THINK → ACT (tool call) → OBSERVE → RESPOND

**CHART GENERATION RULES**

7. **When to ALWAYS Generate Charts (Automatic)**
   
   **MUST create chart for these scenarios:**
   - Historical price data (any time range) → create_line_chart
     * Example: "Show AAPL over last year" → get_stock_history + create_line_chart
   - Multiple stocks comparison → create_multi_line_chart
     * Example: "Compare AAPL and MSFT" → get multiple histories + create_multi_line_chart
   - Performance over time periods → create_line_chart
     * Example: "How did TSLA do in 2023?" → get_stock_history + create_line_chart
   - Year-over-year returns → create_bar_chart
     * Example: "Show yearly returns for SPY" → get_stock_history + create_bar_chart
   
   **Pipeline: Data Tool → Chart Tool → Response**
   - Step 1: Call the data retrieval tool (get_stock_history, etc.)
   - Step 2: Immediately call the appropriate chart tool with the data
   - Step 3: Respond with summary and chart reference

8. **When NOT to Generate Charts**
   - Single current price queries (e.g., "What's AAPL trading at?")
   - Company information lookups (e.g., "Tell me about Apple")
   - Asset class breakdown/composition queries
   - General market questions without time-series data

9. **Valid Time Periods**
   - Supported: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
   - If user specifies invalid period → use closest valid period and inform them

10. **Data Formatting Requirements**
    - x_values and y_values must be lists with matching lengths
    - Convert date objects to strings (ISO format preferred: "2024-01-15")
    - Ensure numeric values are float or int (not strings)
    - Clean data: remove NaN, None, or invalid values before charting

11. **Chart References**
    - Chart titles must include ticker and explicit date ranges
      * Example: "AAPL Stock Price (Jan 1, 2024 - Dec 21, 2024)"
    - Reference charts naturally in your response:
      * "I've generated a line chart showing AAPL's performance over the last year."
      * "See the chart for the complete price history."
    - Do NOT embed chart images - just mention them

12. **Asset Class Lookups & Normalization**
    - If a user asks for "breakdown," "composition," or "asset class" of a ticker:
      * Call get_asset_classes with the resolved ticker symbol
      * Normalization: Convert ratios (0-1) to percentages (0-100%) in your text response
      * Report exact percentages for the 6 core asset classes
      * DO NOT generate a chart for asset class lookups (text summary only)

**EXAMPLE WORKFLOWS**

Example 1 - Historical Data (MUST CREATE CHART):
```
User: "How has Apple performed over the last 6 months?"
Step 1: Call get_stock_history(ticker="AAPL", period="6mo")
Step 2: Call create_line_chart(
    x_values=[list of dates],
    y_values=[list of prices],
    title="AAPL Stock Price (Jun 21, 2024 - Dec 21, 2024)",
    xlabel="Date",
    ylabel="Price ($)"
)
Step 3: Respond with summary and chart reference
```

Example 2 - Current Price (NO CHART):
```
User: "What's the current price of Tesla?"
Step 1: Call get_current_price(ticker="TSLA")
Step 2: Respond with price (no chart needed)
```

Example 3 - Asset Breakdown (NO CHART):
```
User: "What's the asset breakdown of VOO?"
Step 1: Call get_asset_classes(ticker="VOO")
Step 2: Convert ratios to percentages
Step 3: Respond with text summary (no chart)
```

========================
{agent_scratchpad}
========================
"""


# MCP Server Configuration - NOW WITH TWO SERVERS
MCP_SERVERS = {
    "yfinance_mcp": {
        "url": "http://localhost:8002/sse",
        "description": "Market data from yFinance"
    },
    "charts_mcp": {
        "url": "http://localhost:8003/sse", 
        "description": "Chart generation tools"
    }
}


# --- ASYNC TOOL LOADER (ENHANCED FOR MULTIPLE SERVERS) ---

async def a_load_all_mcp_tools() -> tuple[List[Any], MultiServerMCPClient]:
    """Initializes the MCP client for ALL configured servers using HTTP/SSE transport."""
    
    LOGGER.info("🔌 Initializing Multi-Server MCP Client...")
    
    # Build the configuration for all servers
    sse_config = {}
    for server_name, server_info in MCP_SERVERS.items():
        LOGGER.info(f"   Configuring: {server_name} ({server_info['description']})")
        LOGGER.info(f"   URL: {server_info['url']}")
        sse_config[server_name] = {
            "transport": "sse",
            "url": server_info["url"]
        }
    
    client = MultiServerMCPClient(sse_config)
    tools = await client.get_tools()
    
    LOGGER.info("✅ MCP Client initialized successfully.")
    LOGGER.info(f"📊 Total tools loaded: {len(tools)}")
    
    for tool in tools:
       LOGGER.info(f"{tool.name}: {tool.description[:60]}...")

    return tools, client


# --- THE ENHANCED AGENT CLASS ---

class FinanceMarketAgent:
    """
    Enhanced LangChain ReAct agent for financial market data analysis WITH chart generation.
    """
    
    _invocation_count = 0  # Class variable to track invocations
    
    def __init__(self):
        self.mcp_client: MultiServerMCPClient | None = None
        self.tools: List[Any] = []
        self.instance_id = id(self)  # Unique instance identifier

        # Initialize MCP tools from ALL servers
        try:
            self.tools, self.mcp_client = asyncio.run(a_load_all_mcp_tools())
            LOGGER.info(f"✅ Successfully loaded {len(self.tools)} tools from all MCP servers")
        except Exception as e:
            LOGGER.error(f"FATAL ERROR: Could not connect to MCP servers: {e}")
            self.tools = []

        # Create the core ReAct agent chain
        self.core_agent = create_agent(
            model=SUMMARY_LLM,
            tools=self.tools,
            system_prompt=STRICT_SYSTEM_PROMPT,
            debug=False,
        )
        
        LOGGER.debug(f"✅ FinanceMarketAgent initialized with {len(self.tools)} tools (Instance ID: {self.instance_id})")

    async def run_query(self, history: List[BaseMessage], session_id: str) -> AgentResponse:
        """
        Runs the agent against the conversation history and returns the response.
        NOW INCLUDES CHART DETECTION AND ARTIFACT POPULATION.
        
        :param history: The full conversation history (including the new user message)
        :param session_id: The session identifier (for logging/debugging)
        :return: AgentResponse with message and any generated charts
        """
        LOGGER.info(f"Processing query: {history[-1].content[:50]}...")

        # Increment and track invocations
        FinanceMarketAgent._invocation_count += 1
        current_invocation = FinanceMarketAgent._invocation_count
        
        LOGGER.debug(f"🤖 FINANCE MARKET AGENT - Query #{current_invocation}")
        LOGGER.debug(f"🆔 Instance ID: {self.instance_id}")
        LOGGER.debug(f"📝 Session ID: {session_id}")
        LOGGER.debug(f"💬 User Query: {history[-1].content[:100]}...")
        LOGGER.debug(f"📊 History Length: {len(history)} messages")
        
        # Log the last few messages for context
        LOGGER.debug(f"📜 Message History:")
        for i, msg in enumerate(history[-3:], start=max(0, len(history)-3)):
            msg_type = "👤 USER" if isinstance(msg, HumanMessage) else "🤖 AI"
            content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            LOGGER.debug(f"   [{i}] {msg_type}: {content_preview}")
        
        LOGGER.debug(f"🔧 Available tools: {len(self.tools)}")
        
        LOGGER.info(f"⚙️  Invoking Agent (Call #{current_invocation})...")
        
        tool_call_count = 0
        tool_call_details = []
        generated_charts = []  # Track generated charts
        
        try:
            # Invoke the agent with the full message history
            response = await self.core_agent.ainvoke(
                {"messages": history}
            )
            
            LOGGER.debug(f"✅ AGENT INVOCATION COMPLETE")

            # Extract charts from tool calls and responses
            if isinstance(response, dict) and "messages" in response:
                LOGGER.debug(f"📬 Analyzing response messages for charts...")
                LOGGER.debug(f"🔢 Total messages received: {len(response['messages'])}")
                
                for i, msg in enumerate(response["messages"]):
                    msg_type = type(msg).__name__
                    
                    # Check for chart generation tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_call_count += len(msg.tool_calls)
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', 'unknown')
                            tool_args = tc.get('args', {})
                            tool_call_details.append(f"{tool_name}({tool_args})")
                            
                    
                    # Check for chart generation results in ToolMessage
                    # There must be a better way, but for the sake of time, here we are!
                    if msg_type == "ToolMessage":
                        tool_name = getattr(msg, 'name', 'unknown')
                        LOGGER.debug("Tool name: " + tool_name)
                        # If this is a chart tool response, extract the chart info
                        LOGGER.debug(vars(msg))
                        #We got a dependency here on the tool name, this is a little fragile, but oh well, time keeps on ticking...
                        if 'chart' in tool_name.lower() and hasattr(msg, 'content'):
                            try:
                                # Parse the tool result (it should be a dict with chart info)
                                for chart_data in msg.content:
                                    if chart_data['type'] == 'text':
                                        chart = json.loads(chart_data['text'])
                                        chart_artifact = ChartArtifact(
                                            title=chart['title'],
                                            filename=f"{chart['filename']}"
                                        )
                                        generated_charts.append(chart_artifact)
                                        LOGGER.info(f"📊 Chart generated: {chart_artifact.title} -> {chart_artifact.path}")
                            except Exception as e:
                                LOGGER.warning(f"⚠️  Could not parse chart data from tool response: {e}")
                
                LOGGER.debug(f"\n🔧 Total tool calls made: {tool_call_count}")
                LOGGER.debug(f"📊 Total charts generated: {len(generated_charts)}")
                
                if tool_call_details:
                    LOGGER.debug(f"\n🔨 Tool calls with arguments:")
                    for i, detail in enumerate(tool_call_details, 1):
                        LOGGER.debug(f"   {i}. {detail}")
                
                # Extract the final message
                last_message: BaseMessage = response["messages"][-1]
                
                if hasattr(last_message, 'content'):
                    response_preview = last_message.content[:150] + "..." if len(last_message.content) > 150 else last_message.content
                    LOGGER.debug(f"💬 Response Preview: {response_preview}")
                    LOGGER.debug(f"✅ Response length: {len(last_message.content)} characters")
                    
                    # Return AgentResponse with charts
                    return AgentResponse(
                        agent="FinanceMarketAgent",
                        message=last_message.content,
                        charts=generated_charts
                    )
                else:
                    return AgentResponse(
                        agent="FinanceMarketAgent",
                        message=str(last_message),
                        charts=generated_charts
                    )
            
            # Fallback for unexpected response structure
            LOGGER.debug(f"⚠️  Unexpected response structure: {type(response)}")
            return AgentResponse(
                agent="FinanceMarketAgent",
                message=str(response),
                charts=[]
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            LOGGER.error(f"❌ Error: {error_msg}")
            LOGGER.debug(f"Tool calls before error: {tool_call_count}")
            return AgentResponse(
                agent="FinanceMarketAgent",
                message=f"I apologize, but I encountered an error while processing your request: {error_msg}",
                charts=[]
            )

    async def cleanup(self):
        """Cleanup method to properly close the MCP client connection."""
        if self.mcp_client:
            try:
                # Note: Check if your MCP client has a close/cleanup method
                # await self.mcp_client.close()
                LOGGER.info("🧹 Cleanup complete")
            except Exception as e:
                LOGGER.error(f"Error during cleanup: {e}")


# Example usage with chart generation
async def main():
    """Test the enhanced agent with chart generation"""
    agent = FinanceMarketAgent()
    
    # Test 1: Simple price query (might generate a line chart)
    print("\n" + "="*60)
    print("TEST 1: Price history query")
    print("="*60)
    history = [HumanMessage(content="Show me Apple's stock price over the last 6 months")]
    response = await agent.run_query(history, session_id="test123")
    print(f"\nAgent: {response.agent}")
    print(f"Message: {response.message}")
    print(f"Charts: {len(response.charts)} generated")
    for chart in response.charts:
        print(f"  - {chart.title}: {chart.path}")
    
    # Test 2: Comparison query (might generate multi-line chart)
    print("\n" + "="*60)
    print("TEST 2: Comparison query")
    print("="*60)
    history = [HumanMessage(content="Compare the performance of AAPL and MSFT over the last year")]
    response = await agent.run_query(history, session_id="test456")
    print(f"\nAgent: {response.agent}")
    print(f"Message: {response.message}")
    print(f"Charts: {len(response.charts)} generated")
    for chart in response.charts:
        print(f"  - {chart.title}: {chart.path}")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())