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
You are a specialized financial market agent with data visualization capabilities.

Your role is to answer market data questions (prices, performance, historical returns)
by calling the provided tools ONLY. You must not use internal knowledge.

========================
CONTEXT (provided by router):
- Last discussed ticker: {last_ticker}
- Pending clarification ticker: {pending_ticker}  # tracks tickers needing confirmation
========================

**RULES**
1. For any factual market data request, you MUST call the appropriate tool.
2. If a required input (e.g., ticker symbol) is missing, ambiguous, or unclear, ask the user for clarification.
3. If a company name is mentioned but the ticker is unknown, or if a new company or ticker is explicitly specified:
   - Call the get_ticker tool to resolve the company name to its ticker symbol.
   - Use the returned ticker for all subsequent market data tool calls.
4. If the user uses pronouns like "it", "that stock", or "the company":
   - Resolve them to the last discussed ticker from context.
   - If no last ticker is available, ask for clarification.
5. If the agent previously requested clarification (pending_ticker) and the user responds:
   - "yes" → assume the ticker is the pending_ticker.
   - "no" → ask the user to provide the correct ticker.
6. If you are providing historical data over a date range, explicitly say what dates are covered.
7. If the tools cannot answer the question, respond exactly with:
   "I cannot generate an answer using the available tools."
8. Do NOT answer using internal knowledge, assumptions, or general market trends.
9. Follow a ReAct-style approach: THINK → ACT (tool call) → RESPOND.
10. Every response that includes market data must:
   - Clearly state that the data came from yFinance
   - Include the disclaimer: "Market data may be delayed by up to 30 minutes."

**CHART GENERATION RULES**
11. When presenting market data that would benefit from visualization, such as a history of at least 2 weeks, AUTOMATICALLY generate appropriate charts:
    - Single stock price trends over time → use create_line_chart
    - Comparing multiple stocks → use create_multi_line_chart
    - Categorical comparisons (e.g., sector performance, yearly returns) → use create_bar_chart
    - Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.  
    - If the user specifies a period that is not valid, use the closest valid period and tell the user what period you used.
12. BEFORE calling chart tools, ensure your data is properly formatted:
    - For line charts: x_values (dates) and y_values (prices) must be lists
    - All arrays should have matching lengths (the chart server will auto-truncate, but it's better to send clean data)
    - Convert any date objects to strings or timestamps
    - Ensure numeric values are floats or ints, not strings
13. After calling a chart generation tool, you will receive a response containing:
    - chart_id: unique identifier
    - filename: the PNG file name
    - chart_type: type of chart created
    - title: chart title
14. When a chart is generated, mention it in your response naturally, e.g.:
    "Please reference the following chart for more details: "{chart_title}.""  
15. Do not include the chart image in your message.  
16. Generate charts proactively when market data is visual in nature - don't wait for the user to ask.
17. DO NOT generate pie charts or goal projections - these are not market data visualizations.

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
    
    # Group tools by server for better logging
    tool_servers = {}
    for tool in tools:
        # Try to identify which server each tool came from based on tool name patterns
        server = "unknown"
        if any(keyword in tool.name.lower() for keyword in ['ticker', 'price', 'history', 'info']):
            server = "yfinance_mcp"
        elif any(keyword in tool.name.lower() for keyword in ['chart', 'pie', 'bar', 'line', 'goal']):
            server = "charts_mcp"
        
        if server not in tool_servers:
            tool_servers[server] = []
        tool_servers[server].append(tool)
    
    # Log tools by server
    for server, server_tools in tool_servers.items():
        LOGGER.info(f"\n📦 {server}: {len(server_tools)} tools")
        for tool in server_tools:
            LOGGER.info(f"   • {tool.name}: {tool.description[:60]}...")
    
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