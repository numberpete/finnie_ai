# Necessary Imports
import csv
import pandas as pd
import math
import numpy as np
import os
import asyncio
from typing import List, Any
import sys
from pathlib import Path
from langsmith import Client, uuid7
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


import os
async def a_load_mcp_tools(server_script_path: str, server_name: str) -> List[Any]:
    """Initializes the MCP client and asynchronously loads tools using STDIO."""
    
    # We use sys.executable to get the path to the current Python interpreter 
    # (which should be the one in your .venv)
    PYTHON_EXECUTABLE = sys.executable 
    
    # Configuration tells the client how to START the server as a subprocess
    stdio_config = {
        server_name: {
            "transport": "stdio",
            # The executable command is the python interpreter
            "command": PYTHON_EXECUTABLE,
            # The arguments are the path to your server script
            "args": [str(server_script_path)],
            # If your server script relies on environment variables (which it does), 
            # you must pass the current environment to the subprocess.
            "env": os.environ.copy() 
        }
    }
    
    # Initialize the MultiServerMCPClient with the stdio configuration
    client = MultiServerMCPClient(stdio_config)
    
    # The client will now start the server subprocess, connect via stdio,
    # discover the tools, and return the LangChain tool objects.
    tools = await client.get_tools()
    print(f"✅ Successfully loaded {len(tools)} tool(s) from STDIO process: {server_script_path}")
    return tools

## Set the OpenAI API key and model name
MODEL="gpt-4o-mini"
summary_llm = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=False)


# hwchase17/react is a prompt template designed for ReAct-style
# conversational agents.
client = Client()
#prompt = client.pull_prompt("hwchase17/react", include_model=True) # pull "hwchase17/react" prompt from langchain hub

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
7.  When a user's question clearly indicates a specific financial domain (e.g., 'IRA' or 'Estate Planning'), you should first consider using the list_categories tool to confirm the exact spelling of the category before attempting to call basic_query.
8.  **Citing Sources:** When providing a final answer based on a tool's output, you MUST clearly 
    list the source(s) at the end of your response, including the full URL(s) if the tool output 
    provides them. Structure your answer clearly with a "Sources:" section at the very end.
---
{agent_scratchpad}
"""

# Tools
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT_PATH = THIS_SCRIPT_DIR.parent / "mcp" / "finance_q_and_a_mcp.py"
MCP_SERVER_NAME = "finance_qanda_tool"


## Create a list of tools: retriever_tool and search_tool
try:
    # Use asyncio.run to execute the async function and get the tools list
    # This automatically starts and manages the lifecycle of the stdio server process!
    tools = asyncio.run(a_load_mcp_tools(SERVER_SCRIPT_PATH, MCP_SERVER_NAME))
except Exception as e:
    print(f"❌ FATAL ERROR: Could not start or connect to MCP server via STDIO: {SERVER_SCRIPT_PATH}")
    print(e)
    tools = []


# Create a ReAct agent
# The agent will reason and take actions based on retrieved tools and memory.


summary_react_agent = create_agent(
    model=summary_llm,
    tools=tools,  # Pass your list of tools here
    system_prompt=STRICT_SYSTEM_PROMPT,
    debug=True,
)


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    summary_react_agent,
    lambda session_id: get_session_history(session_id),
    input_messages_key="input"
)


# Building an UI for the chatbot with agents
import gradio as gr

# Define function for Gradio interface
async def chat_with_agent(user_input, session_id):
    """Processes user input and maintains session-based chat history."""
    
    get_session_history(session_id).messages.append(HumanMessage(content=user_input))
    response = await agent_with_history.ainvoke(
        {"input":user_input, "messages":get_session_history(session_id).messages},
        config={"configurable": {"session_id": session_id}}
    )

    # Extract only the 'output' field from the response
    if isinstance(response, dict) and "messages" in response:
        return response["messages"][-1].content  # Return clean text response
    else:
        return response

# Create Gradio app interface
with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")
    session_state = gr.State(value=str(uuid7()))  # Unique session ID for user

    submit_button.click(chat_with_agent, inputs=[input_box, session_state], outputs=output_box)

# Launch the Gradio app
app.launch(debug=True, share=True,server_name="0.0.0.0", server_port=7860)
