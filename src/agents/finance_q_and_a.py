# src/agents/finance_q_and_a.py

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
from src.agents.response import AgentResponse
import logging


# Setup Logger
setup_tracing("finance-q-and-a-agent", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
SUMMARY_LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=False)

# Your detailed System Prompt
STRICT_SYSTEM_PROMPT = """
You are a specialized financial analysis assistant. Your role is STRICTLY to answer user questions 
by utilizing the provided tools. Your tone must be professional, clear, and easy-to-understand 
for a general audience.

**RULES:**
1.  For any factual inquiry or request for information, you MUST use the available tools.
2.  If the available tools CANNOT provide an answer, or if the user asks a question 
    that is outside the scope of the tools, you MUST respond with the exact phrase: "I cannot generate an answer using the available tools."
3.  Do NOT attempt to answer questions from your internal knowledge base.
4.  Follow the ReAct framework to reason and take actions.
5.  **Tool Priority:** The advanced_query tool is only to be used if more detail is needed than can be provided by the basic_query tool, 
    and ONLY after basic_query has been used first, UNLESS the user specifically requests advanced information.
6.  **Category Use:** If a definition or simple concept is requested, prioritize using the basic_query tool with a precise category (e.g., 'Glossary', 'Tax'). If a query spans multiple domains (e.g., 'Retirement' and 'Tax'), pass a list of categories to the tool. If no specific category is clear, run the basic_query tool without a category.
7.  If a query with category='Glossary' is made, always use the basic_query tool with that category to provide definitions of financial terms.  If it fails to return a response, try again with no category specified.
8.  When a user's question clearly indicates a specific financial domain (e.g., 'IRA' or 'Estate Planning'), you should first consider using the list_categories tool to confirm the exact spelling of the category before attempting to call basic_query.
9..  **Citing Sources:** When providing a final answer based on a tool's output, you MUST clearly 
    list the source(s) at the end of your response, including the full URL(s) if the tool output 
    provides them. Structure your answer clearly with a "Sources:" section at the very end.
---
{agent_scratchpad}
"""

# Tool Loading Setup
MCP_SERVER_NAME = "finance_qanda_tool"
MCP_SERVER_URL = "http://localhost:8001/sse"


# --- ASYNC TOOL LOADER ---

async def a_load_mcp_tools(server_name: str, server_url: str = "http://localhost:8001/sse") -> tuple[List[Any], MultiServerMCPClient]:
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

class FinanceQandAAgent:
    """Encapsulates the entire LangChain ReAct agent logic for financial Q&A."""
    
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
        # Note: The supervisor will handle history, so we don't need RunnableWithMessageHistory
        self.core_agent = create_agent(
            model=SUMMARY_LLM,
            tools=self.tools,
            system_prompt=STRICT_SYSTEM_PROMPT,
            debug=False,
        )
        
        LOGGER.debug(f"✅ FinanceQandAAgent initialized with {len(self.tools)} tools (Instance ID: {self.instance_id})")

    async def run_query(self, history: List[BaseMessage], session_id: str) -> AgentResponse:
        """
        Runs the agent against the conversation history and returns the response.
        
        :param history: The full conversation history (including the new user message)
        :param session_id: The session identifier (for logging/debugging)
        :return: The agent's response as a string
        """
        LOGGER.info(f"Processing query: {history[-1].content[:50]}...")


        # Increment and track invocations
        FinanceQandAAgent._invocation_count += 1
        current_invocation = FinanceQandAAgent._invocation_count
        
        LOGGER.debug(f"🤖 FINANCE Q&A AGENT - Query #{current_invocation}")
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
                    return AgentResponse(agent="FinanceQandAAgent", message=last_message.content)
                else:
                    return AgentResponse(agent="FinanceQandAAgent", message=str(last_message))
            
            # Fallback for unexpected response structure
            LOGGER.debug(f"⚠️  Unexpected response structure: {type(response)}")
            return AgentResponse(agent="FinanceQandAAgent", message=str(response))
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            LOGGER.error(f"Error: {error_msg}")
            LOGGER.debug(f"Tool calls before error: {tool_call_count}")
            return AgentResponse(agent="FinanceQandAAgent", message=f"I apologize, but I encountered an error while processing your request: {error_msg}")

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
#     agent = FinanceQandAAgent()
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