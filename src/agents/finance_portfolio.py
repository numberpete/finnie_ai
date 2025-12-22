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
setup_tracing("portfolio-agent", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
SUMMARY_LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=False)

# Your detailed System Prompt (ENHANCED WITH CHART INSTRUCTIONS)
STRICT_SYSTEM_PROMPT = """
You are a specialized financial portfolio agent with visualization capabilities.

Your role is to visually present a user's financial profile, assess a portfolio's risk 
tolerance tier, and project it's potential growth.

========================
CONTEXT (provided by router):
- Last discussed ticker: {last_ticker}
- Pending clarification ticker: {pending_ticker}
========================

**CORE RULES**

1. **Tool Usage is Mandatory**
   - For any query related to a financial portfolio, you must use the tools provided
   - Never provide any portfolio assessment from memory or general knowledge

2. **Asset Class Determination**
   - If a user ask what type of asses or asset class or asset classes a particular ticker belongs to → call get_asset_classes to resolve it
   - If a fund name or company is mentioned without a ticker → call get_ticker to resolve it
   - Always rely on get_asset_classes for the breakdown of positions.
   - Note: If get_asset_classes returns values for "Real Estate" or "Commodities," treat these as distinct diversifiers even if the user originally referred to them as "Other."

3. **Data Presentation**
   You should present portfolios as tabular data AND pie charts, showing the following asset classes:
    - Equities
    - Fixed Income
    - Real Estate
    - Cash
    - Commodities
    - Crypto 
    In addition to returning a pie chart for a portfolio, you shoud also call the assess_risk_tolerance tool and provide a summary of it's assessment.

4. **Colors**
    Uless the user asks for specific colors, use the followinhg color for each asset class:
    - Equities → #2E5BFF
    - Fixed Income → #46CDCF
    - Real Estate → #F08A5D
    - Cash → #3DDC84
    - Commodities → #FFD700
    - Crypto → #B832FF

5. **Unable to Answer**
   - If tools cannot answer the question, respond: "I cannot generate an answer using the available tools."
   - Do not guess, estimate, or use general market knowledge

6. **Approach**
   - Follow ReAct pattern: THINK → ACT (tool call) → OBSERVE → RESPOND

**CHART GENERATION RULES**

7. **When to Generate Charts (Automatic)**
   - Financial Portfolio → create_pie_chart
   - Monte Carlo Simulation → create_stacked_bar for each portfolio returned by the tool 


9. **Data Formatting Requirements**
   - Ensure numeric values are float or int (not strings)
   - Clean data before sending to chart tools

10. **Chart References**
    - Reference charts naturally: "I've generated a chart showing {description}."
    - Do NOT embed chart images in your message - just mention them

11. **Diclaimer Requiremts**
    - All responses should include a disclaimer that FinnieAI can make mistakes, and any advice or projection
    provided is for educational purposes only.

12. Data Integrity
    - If get_asset_classes returns _mock: True, you must include a footnote: "Note: Detailed allocation for [Ticker] was unavailable; using a standard benchmark estimate."
========================
{agent_scratchpad}
========================
"""


# MCP Server Configuration - NOW WITH THREE SERVERS
MCP_SERVERS = {
    "yfinance_mcp": {
        "url": "http://localhost:8002/sse",
        "description": "Ticker info from yFinance"
    },
    "charts_mcp": {
        "url": "http://localhost:8003/sse", 
        "description": "Chart generation tools"
    },
    "goals_mcp": {
        "url": "http://localhost:8003/sse", 
        "description": "Portfolio asessment tools"
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

class PortfolioAgent:
    """
    Enhanced LangChain ReAct agent for portfolio presentation and analysis WITH chart generation.
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
        
        LOGGER.debug(f"✅ PortfolioAgent initialized with {len(self.tools)} tools (Instance ID: {self.instance_id})")

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
        PortfolioAgent._invocation_count += 1
        current_invocation = PortfolioAgent._invocation_count
        
        LOGGER.debug(f"🤖 PORTFOLIO AGENT - Query #{current_invocation}")
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
                        agent="PortfolioAgent",
                        message=last_message.content,
                        charts=generated_charts
                    )
                else:
                    return AgentResponse(
                        agent="PortfolioAgent",
                        message=str(last_message),
                        charts=generated_charts
                    )
            
            # Fallback for unexpected response structure
            LOGGER.debug(f"⚠️  Unexpected response structure: {type(response)}")
            return AgentResponse(
                agent="PortfolioAgent",
                message=str(response),
                charts=[]
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            LOGGER.error(f"❌ Error: {error_msg}")
            LOGGER.debug(f"Tool calls before error: {tool_call_count}")
            return AgentResponse(
                agent="PortfolioAgent",
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
