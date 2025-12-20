# src/agents/finance_market.py

# Imports for Agent Core
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Any

# LangChain/LangSmith Imports
from langsmith import Client
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage
from src.utils import setup_logger_with_tracing, setup_tracing
import logging


# Setup Logger
setup_tracing("finance-market-agent", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
SUMMARY_LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=False)

# Your detailed System Prompt
STRICT_SYSTEM_PROMPT = """
You are a specialized financial market agent.

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
6. If the tools cannot answer the question, respond exactly with:
   "I cannot generate an answer using the available tools."
7. Do NOT answer using internal knowledge, assumptions, or general market trends.
8. Follow a ReAct-style approach: THINK → ACT (tool call) → RESPOND.
9. Every response that includes market data must:
   - Clearly state that the data came from yFinance
   - Include the disclaimer: "Market data may be delayed by up to 30 minutes."

========================
{agent_scratchpad}
========================
"""


# Tool Loading Setup
MCP_SERVER_NAME = "yfinance_mcp"
MCP_SERVER_URL = "http://localhost:8002/sse"


# --- ASYNC TOOL LOADER ---

async def a_load_mcp_tools(server_name: str, server_url: str = "http://localhost:8002/sse") -> tuple[List[Any], MultiServerMCPClient]:
    """Initializes the MCP client using HTTP/SSE transport."""
    
    LOGGER.info("🔌 Initializing MCP Client...")
    LOGGER.info(f"Server name: {server_name}")
    LOGGER.info(f"Server URL: {server_url}")
    LOGGER.info(f"Transport: HTTP/SSE")
    
    sse_config = {
        server_name: {
            "transport": "sse",
            "url": server_url
        }
    }
    
    client = MultiServerMCPClient(sse_config)
    tools = await client.get_tools()
    
    LOGGER.info("✅ MCP Client initialized successfully.")
    LOGGER.info(f"Tools loaded: {len(tools)}")
    for tool in tools:
        LOGGER.info(f"   • {tool.name}: {tool.description[:60]}...")
    
    return tools, client


# --- THE AGENT CLASS (The Core Logic) ---

class FinanceMarketAgent:
    """Encapsulates the entire LangChain ReAct agent logic for financial market data analysis."""
    
    _invocation_count = 0  # Class variable to track invocations
    
    def __init__(self):
        self.mcp_client: MultiServerMCPClient | None = None
        self.tools: List[Any] = []
        self.instance_id = id(self)  # Unique instance identifier

        # Initialize MCP tools using HTTP transport
        try:
            self.tools, self.mcp_client = asyncio.run(
                a_load_mcp_tools(MCP_SERVER_NAME, MCP_SERVER_URL)
            )
        except Exception as e:
            LOGGER.error(f"FATAL ERROR: Could not connect to MCP server at {MCP_SERVER_URL}: {e}")
            self.tools = []

        # Create the core ReAct agent chain
        # Note: The router will handle history, so we don't need RunnableWithMessageHistory
        self.core_agent = create_agent(
            model=SUMMARY_LLM,
            tools=self.tools,
            system_prompt=STRICT_SYSTEM_PROMPT,
            debug=False,
        )
        
        LOGGER.debug(f"✅ FinanceMarketAgent initialized with {len(self.tools)} tools (Instance ID: {self.instance_id})")

    async def run_query(self, history: List[BaseMessage], session_id: str) -> str:
        """
        Runs the agent against the conversation history and returns the response.
        
        :param history: The full conversation history (including the new user message)
        :param session_id: The session identifier (for logging/debugging)
        :return: The agent's response as a string
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
        
        # log the last few messages for context
        LOGGER.debug(f"📜 Message History:")
        for i, msg in enumerate(history[-3:], start=max(0, len(history)-3)):
            msg_type = "👤 USER" if isinstance(msg, HumanMessage) else "🤖 AI"
            content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            LOGGER.debug(f"   [{i}] {msg_type}: {content_preview}")
        
        LOGGER.debug(f"🔧 Available tools: {len(self.tools)}")
        for tool in self.tools:
            LOGGER.debug(f"   • {tool.name}: {tool.description[:60]}...")
        
        LOGGER.info(f"⚙️  Invoking Agent (Call #{current_invocation})...")
        
        tool_call_count = 0
        tool_call_details = []
        
        try:
            # Invoke the agent with the full message history
            response = await self.core_agent.ainvoke(
                {"messages": history}
            )
            
            LOGGER.debug(f"✅ AGENT INVOCATION COMPLETE")

            # Extract the last message from the response
            if isinstance(response, dict) and "messages" in response:
                # Analyze all messages in the response
                LOGGER.debug(f"📬 Analyzing response messages...")
                LOGGER.debug(f"🔢 Total messages received: {len(response['messages'])}")
                LOGGER.debug(f"\n📋 Message breakdown:")
                
                for i, msg in enumerate(response["messages"]):
                    msg_type = type(msg).__name__
                    LOGGER.debug(f"   [{i}] {msg_type}")
                    
                    # Check for tool calls in AIMessage
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_call_count += len(msg.tool_calls)
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', 'unknown')
                            tool_args = tc.get('args', {})
                            tool_call_details.append(f"{tool_name}({tool_args})")
                            LOGGER.debug(f" -> Tool: {tool_name}")
                            # Show abbreviated arguments
                            if tool_args:
                                args_preview = str(tool_args)[:100]
                                if len(str(tool_args)) > 100:
                                    args_preview += "..."
                                LOGGER.debug(f" | Args: {args_preview}")
                    
                    # Check for tool results in ToolMessage
                    if msg_type == "ToolMessage":
                        tool_name = getattr(msg, 'name', 'unknown')
                        result_preview = str(msg.content)[:80] if hasattr(msg, 'content') else ""
                        LOGGER.debug(f" -> Result from: {tool_name} | {result_preview}...")
                    
                    # Show content preview for HumanMessage and AIMessage
                    if hasattr(msg, 'content') and msg.content and isinstance(msg.content, str) and msg_type in ["HumanMessage", "AIMessage"]:
                        preview = msg.content[:50].replace('\n', ' ')
                        LOGGER.debug(f" | '{preview}...'")
                    
                
                LOGGER.debug(f"\n🔧 Total tool calls made: {tool_call_count}")
                if tool_call_details:
                    LOGGER.debug(f"\n🔨 Tool calls with arguments:")
                    for i, detail in enumerate(tool_call_details, 1):
                        LOGGER.debug(f"   {i}. {detail}")
                
                last_message: BaseMessage = response["messages"][-1]
                
                # Return the content of the last message
                if hasattr(last_message, 'content'):
                    response_preview = last_message.content[:150] + "..." if len(last_message.content) > 150 else last_message.content
                    LOGGER.debug(f"💬 Response Preview:  {response_preview}")
                    LOGGER.debug(f"✅ Response length: {len(last_message.content)} characters")
                    return last_message.content
                else:
                    return str(last_message)
            
            # Fallback for unexpected response structure
            LOGGER.debug(f"⚠️  Unexpected response structure: {type(response)}")
            return str(response)
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            LOGGER.error(f"Error: {error_msg}")
            LOGGER.debug(f"Tool calls before error: {tool_call_count}")
            return f"I apologize, but I encountered an error while processing your request: {error_msg}"

    async def cleanup(self):
        """Cleanup method to properly close the MCP client connection."""
        if self.mcp_client:
            try:
                # Note: Check if your MCP client has a close/cleanup method
                # await self.mcp_client.close()
                pass
            except Exception as e:
                LOGGER.error(f"Error during cleanup: {e}")


# Example of how to use this class (for testing):
# async def main():
#     agent = FinanceMarketAgent()
#     
#     # Simulate a conversation
#     history = [HumanMessage(content="What is a 401k?")]
#     response = await agent.run_query(history, session_id="test123")
#     print(response)
#     
#     # Follow-up question
#     history.append(AIMessage(content=response))
#     history.append(HumanMessage(content="How much can I contribute?"))
#     response2 = await agent.run_query(history, session_id="test123")
#     print(response2)
#     
#     await agent.cleanup()
#
# if __name__ == "__main__":
#     asyncio.run(main())