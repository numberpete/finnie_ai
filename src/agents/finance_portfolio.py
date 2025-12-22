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
# ROLE
You are the Portfolio Risk & Simulation Engine. You analyze portfolios and create visualizations.

# ASSET CLASSES
The portfolio uses these 6 asset classes:
- Equities
- Fixed Income
- Real Estate
- Cash
- Commodities
- Crypto

When creating charts, use these exact colors (unless user requests different colors):
- Equities: #2E5BFF
- Fixed Income: #46CDCF
- Real Estate: #F08A5D
- Cash: #3DDC84
- Commodities: #FFD700
- Crypto: #B832FF

# CORE RULES
1. **Extract portfolio from user message** - Look for JSON or key-value pairs in the current message FIRST
2. **Call each tool only ONCE** - Check {agent_scratchpad}. If a tool has an "Observation:", don't call it again.
3. **Execute only what's requested** - Don't do extra analysis unless asked.
4. **CRITICAL: Never call create_stacked_bar_chart without simulation results** - You MUST call simple_monte_carlo_simulation FIRST and receive its results BEFORE calling create_stacked_bar_chart

# EXECUTION PIPELINE

**Determine what the user is asking for:**

**IF** user asked for simulation/projection/future view:
  - Skip to Step 3 (simulation only)

**ELSE IF** user asked for portfolio analysis/risk assessment:
  - Execute Steps 1-2 (risk and pie chart)

**Step 1**: Call `assess_risk_tolerance` with portfolio
**Step 2**: Call `create_pie_chart` with the SAME portfolio

**Step 3 (simulation)**: 
  - Call `simple_monte_carlo_simulation` with:
    - **portfolio**: REQUIRED - The portfolio dictionary from the user's message
    - **years**: extracted from user request (default: 10)
    - **target_goal**: ONLY if user explicitly mentioned a financial goal/target
  - Call `create_stacked_bar_chart` with simulation results

**Step 4**: Provide text summary and STOP

# EXTRACTING PORTFOLIO FROM USER MESSAGE

**Look for these patterns in the user's message:**

1. **JSON format**: `{"Equities": 1000000, "Fixed_Income": 600000, ...}`
   - Extract the entire JSON object
   - This is your portfolio parameter

2. **Key-value format**: "Equities: 60%, Bonds: 40%"
   - Parse each asset class and percentage/amount
   - Build dictionary: `{"Equities": 60, "Bonds": 40}`

3. **Natural language**: "60% stocks, 40% bonds"
   - Map "stocks" → "Equities", "bonds" → "Fixed Income"
   - Build dictionary: `{"Equities": 60, "Fixed Income": 40}`

**CRITICAL**: When you call `simple_monte_carlo_simulation`, you MUST include the `portfolio` parameter with the dictionary you extracted.

**Example**:
```
User: "How will this look in 10 years: {"Equities": 1000000, "Cash": 400000}"
YOU MUST CALL: simple_monte_carlo_simulation(portfolio={"Equities": 1000000, "Cash": 400000}, years=10)
```

# IF PORTFOLIO IS MISSING

Only if the current message does NOT contain portfolio data:
1. Check conversation history for portfolio data
2. Check previous tool results (assess_risk_tolerance, create_pie_chart)
3. If still not found, ask the user to provide their portfolio allocation

# RESPONSE FORMAT
- Summarize key findings from tool results (risk tier, projections, etc.)
- **Reference charts by title only** - Do NOT attempt to embed, display, or recreate the charts in your response
- If target_goal was NOT provided by the user, do NOT mention it in your summary
- End with: "FinnieAI can make mistakes, and answers are for educational purposes only."

========================
{agent_scratchpad}
========================
"""


# MCP Server Configuration - NOW WITH THREE SERVERS
MCP_SERVERS = {
    "charts_mcp": {
        "url": "http://localhost:8003/sse", 
        "description": "Chart generation tools"
    },
    "goals_mcp": {
        "url": "http://localhost:8004/sse", 
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
        Handles multiple sequential or parallel tool calls.
        """
        LOGGER.info(f"Processing query: {history[-1].content[:50]}...")

        PortfolioAgent._invocation_count += 1
        current_invocation = PortfolioAgent._invocation_count
        
        LOGGER.debug(f"🤖 PORTFOLIO AGENT - Query #{current_invocation}")
        LOGGER.debug(f"🆔 Instance ID: {self.instance_id}")
        LOGGER.debug(f"📝 Session ID: {session_id}")
        LOGGER.debug(f"💬 User Query: {history[-1].content[:100]}...")
        LOGGER.debug(f"📊 History Length: {len(history)} messages")
        
        tool_call_count = 0
        tool_call_details = []
        generated_charts = []
        
        try:
            working_history = list(history)
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                LOGGER.debug(f"\n{'='*60}")
                LOGGER.debug(f"🔄 ITERATION {iteration}")
                LOGGER.debug(f"{'='*60}")
                LOGGER.debug(f"📝 Working history length: {len(working_history)} messages")
                
                # Invoke the agent
                response = await self.core_agent.ainvoke(
                    {"messages": working_history}
                )
                
                if not isinstance(response, dict) or "messages" not in response:
                    LOGGER.warning(f"⚠️  Unexpected response structure: {type(response)}")
                    break
                
                new_messages = response["messages"]
                LOGGER.debug(f"📬 Received {len(new_messages)} new message(s)")
                
                # Log each new message
                for i, msg in enumerate(new_messages):
                    msg_type = type(msg).__name__
                    LOGGER.debug(f"  [{i}] {msg_type}")
                    
                    if msg_type == "AIMessage" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        LOGGER.debug(f"      🔧 Contains {len(msg.tool_calls)} tool call(s):")
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', 'unknown')
                            LOGGER.debug(f"         - {tool_name}")
                            tool_call_details.append(tool_name)
                            tool_call_count += 1
                    
                    elif msg_type == "ToolMessage":
                        tool_name = getattr(msg, 'name', 'unknown')
                        LOGGER.debug(f"      ✅ Tool result for: {tool_name}")
                        
                        # Extract charts if applicable
                        if 'chart' in tool_name.lower() and hasattr(msg, 'content'):
                            try:
                                for chart_data in msg.content:
                                    if isinstance(chart_data, dict) and chart_data.get('type') == 'text':
                                        chart = json.loads(chart_data['text'])
                                        chart_artifact = ChartArtifact(
                                            title=chart['title'],
                                            filename=f"{chart['filename']}"
                                        )
                                        generated_charts.append(chart_artifact)
                                        LOGGER.info(f"         📊 Chart captured: {chart_artifact.title}")
                            except Exception as e:
                                LOGGER.warning(f"         ⚠️  Could not parse chart data: {e}")
                
                # Check if the last message has tool calls
                last_message = new_messages[-1]
                has_tool_calls = (
                    hasattr(last_message, 'tool_calls') and 
                    last_message.tool_calls and 
                    len(last_message.tool_calls) > 0
                )
                
                if has_tool_calls:
                    LOGGER.debug(f"➡️  Agent needs to execute tools, continuing loop...")
                    # Add new messages to history for next iteration
                    working_history.extend(new_messages)
                else:
                    # Agent is done - final response received
                    LOGGER.debug(f"\n{'='*60}")
                    LOGGER.debug(f"✅ AGENT COMPLETE")
                    LOGGER.debug(f"{'='*60}")
                    LOGGER.debug(f"🔧 Total tool calls: {tool_call_count}")
                    LOGGER.debug(f"📊 Charts generated: {len(generated_charts)}")
                    
                    if tool_call_details:
                        LOGGER.debug(f"🔨 Tool execution sequence:")
                        for i, tool in enumerate(tool_call_details, 1):
                            LOGGER.debug(f"   {i}. {tool}")
                    
                    # Return final response
                    if hasattr(last_message, 'content'):
                        response_preview = last_message.content[:150] + "..." if len(last_message.content) > 150 else last_message.content
                        LOGGER.debug(f"💬 Response Preview: {response_preview}")
                        
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
            
            # Max iterations reached
            LOGGER.warning(f"⚠️  Reached max iterations ({max_iterations})")
            LOGGER.warning(f"Tool calls made: {tool_call_count}, Tools: {tool_call_details}")
            return AgentResponse(
                agent="PortfolioAgent",
                message="I apologize, but I couldn't complete the request within the iteration limit.",
                charts=generated_charts
            )
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            LOGGER.error(f"❌ Error: {error_msg}")
            import traceback
            LOGGER.error(traceback.format_exc())
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
