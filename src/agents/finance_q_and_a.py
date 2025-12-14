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
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
SUMMARY_LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=False)

# Your detailed System Prompt (copied from the original file)
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
9.  **Citing Sources:** When providing a final answer based on a tool's output, you MUST clearly 
    list the source(s) at the end of your response, including the full URL(s) if the tool output 
    provides them. Structure your answer clearly with a "Sources:" section at the very end.
10.  after the sources, identify which tools you called.
---
{agent_scratchpad}
"""

# Tool Loading Setup
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
# Assuming the mcp file is at finnie_ai/mcp/finance_q_and_a_mcp.py
SERVER_SCRIPT_PATH = THIS_SCRIPT_DIR.parent / "mcp" / "finance_q_and_a_mcp.py"
MCP_SERVER_NAME = "finance_qanda_tool"


# --- ASYNC TOOL LOADER ---

async def a_load_mcp_tools(server_script_path: str, server_name: str) -> List[Any]:
    """Initializes the MCP client and asynchronously loads tools using STDIO."""
    PYTHON_EXECUTABLE = sys.executable 
    
    stdio_config = {
        server_name: {
            "transport": "stdio",
            "command": PYTHON_EXECUTABLE,
            "args": [str(server_script_path)],
            "env": os.environ.copy() 
        }
    }
    
    client = MultiServerMCPClient(stdio_config)
    tools = await client.get_tools()
    print(f"✅ Successfully loaded {len(tools)} tool(s) from STDIO process: {server_script_path}")
    return tools


# --- SESSION MANAGEMENT ---

# Simple in-memory storage for session history
STORE = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a session history object."""
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]


# --- THE AGENT CLASS (The Core Logic) ---

class FinanceQandAAgent:
    """Encapsulates the entire LangChain ReAct agent and history logic."""
    def __init__(self):
        try:
            self.tools = asyncio.run(a_load_mcp_tools(SERVER_SCRIPT_PATH, MCP_SERVER_NAME))
        except Exception as e:
            print(f"❌ FATAL ERROR: Could not start or connect to MCP server: {e}")
            self.tools = []

        # Create the core ReAct agent chain
        self.core_agent = create_agent(
            model=SUMMARY_LLM,
            tools=self.tools,
            system_prompt=STRICT_SYSTEM_PROMPT,
            debug=True,
        )

        # Wrap the core agent with history management
        self.agent_with_history = RunnableWithMessageHistory(
            self.core_agent,
            get_session_history,
            input_messages_key="input"
        )

    async def run_query(self, user_input: str, session_id: str) -> str:
        """
        Runs the agent against user input, updates history, and returns the response.
        
        This is the main public method the UI will call.
        """

        session_history = get_session_history(session_id)
        session_history.add_user_message(user_input)
        
        response = await self.agent_with_history.ainvoke(
            {"input": user_input, "messages": session_history.messages},
            config={"configurable": {"session_id": session_id}}
        )

        # The RunnableWithMessageHistory returns the full message list, so extract the last message
        if isinstance(response, dict) and "messages" in response:
            last_message: BaseMessage = response["messages"][-1]
            return last_message.content
        
        # Fallback for unexpected response structure
        return str(response)