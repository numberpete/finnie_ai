# src/agents/finance_q_and_a.py

from langchain_openai import ChatOpenAI
from src.utils import setup_logger_with_tracing, setup_tracing
from src.agents.base_agent import BaseAgent
import logging


# Setup Logger
setup_tracing("finance-q-and-a-agent", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=True)

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
MCP_SERVERS = {
    "finance_qanda_tool": {
        "url": "http://localhost:8001/sse", 
        "description": "Financial Q&A vector lookup"
    }
}


class FinanceQandAAgent(BaseAgent):
    """
    Enhanced LangChain ReAct agent for financial Q and A.
    """
    def __init__(self):
        super().__init__(
            agent_name="FinanceQandAAgent",
            llm=LLM,
            system_prompt=STRICT_SYSTEM_PROMPT,
            logger=LOGGER,
            mcp_servers=MCP_SERVERS
        )



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