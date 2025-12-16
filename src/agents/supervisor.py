# src/agents/supervisor_agent.py

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Import your existing worker agent
from src.agents.finance_q_and_a import FinanceQandAAgent 
from src.utils import setup_logger_with_tracing, setup_tracing,  get_tracer
import logging


# Setup Logger
setup_tracing("supervisor", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)
TRACER = get_tracer(__name__)

# --- CONFIGURATION ---
SUPERVISOR_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
AGENT_NAMES = ["FinanceQandAAgent"] # Add "OtherAgent" when ready

# Define the state structure for the graph
class AgentState(TypedDict):
    """Represents the state of our multi-agent conversation."""
    messages: List[BaseMessage]
    next: str # The name of the next agent/node to run
    session_id: str


class SupervisorAgent:
    """
    Manages the multi-agent system using LangGraph for routing.
    """
    def __init__(self):
        # 1. Initialize worker agents
        self.workers = {
            "FinanceQandAAgent": FinanceQandAAgent(),
            # "OtherAgent": OtherAgent(), # Initialize other agents here
        }
        self.checkpointer = InMemorySaver()
        # 2. Build the graph components
        self.supervisor_chain = self._create_supervisor_chain()
        self.graph = self._build_supervisor_graph()

        LOGGER.debug("✅ SupervisorAgent initialized and LangGraph built.")

    # --- Worker Node Runner ---
    async def _run_worker_agent(self, state: AgentState) -> AgentState:
        """
        Runs the worker agent specified in state['next'] and updates the state.
        This function handles all worker agent calls.
        """
        agent_name = state["next"]
        worker_instance = self.workers.get(agent_name)
        full_message_history: List[BaseMessage] = state["messages"]
        session_id = state["session_id"]
        
        # Get the last user message for logging
        last_user_msg = next((msg.content for msg in reversed(full_message_history) 
                              if isinstance(msg, HumanMessage)), "")
        
        LOGGER.debug(f"🔧 WORKER NODE: {agent_name}")
        LOGGER.debug(f"📝 Session: {session_id}")
        LOGGER.debug(f"💬 Query: {last_user_msg[:60]}...")
        LOGGER.debug(f"📊 Current history length: {len(full_message_history)} messages")
        
        # Call the worker agent's core function (must be async)
        response_text = await worker_instance.run_query(full_message_history, session_id)
        
        # Create the response message
        ai_response = AIMessage(content=response_text, name=agent_name)
        
        LOGGER.debug(f"✅ {agent_name} completed. Response length: {len(response_text)} chars\n")
        
        # Return the new state with the agent's response appended
        return {
            "messages": state["messages"] + [ai_response],
            "next": "",  # Clear next to allow supervisor to route again
            "session_id": session_id
        }

    # --- Supervisor Chain Builder ---
    def _create_supervisor_chain(self):
        """Builds the LLM chain responsible for routing."""
        
        tool_names = AGENT_NAMES + ["FINISH"]
        
        SUPERVISOR_SYSTEM_PROMPT = f"""You are a financial services supervisor. Your role is to determine which specialist agent should handle the user's request.

CRITICAL ROUTING RULES (READ CAREFULLY):
1. Look at the LAST message in the conversation:
   - If it's a HumanMessage (from user), route to the appropriate agent
   - If it's an AIMessage (from an agent like FinanceQandAAgent), ALWAYS select 'FINISH'
   
2. NEVER route to an agent after that agent has just responded. This creates an infinite loop.

3. Specific routing:
   - User asks financial question → 'FinanceQandAAgent'
   - Agent just responded → 'FINISH'
   - Simple greeting/thanks → 'FINISH'

AVAILABLE AGENTS: {tool_names}

REMEMBER: If the last message has name='FinanceQandAAgent', you MUST return 'FINISH'.
"""

        function_schema = {
            "name": "route_to_agent",
            "description": "Routes the conversation to the appropriate specialist agent or signals completion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "next": {
                        "type": "string",
                        "enum": tool_names,
                        "description": "The name of the next agent to route the message to, or FINISH."
                    }
                },
                "required": ["next"]
            }
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            # LangGraph will automatically pass the full list of messages here
            ("placeholder", "{messages}"), 
        ])
        
        supervisor_chain = (
            prompt
            | SUPERVISOR_LLM.bind(functions=[function_schema], function_call={"name": "route_to_agent"})
            | JsonOutputFunctionsParser()
        )
        
        return supervisor_chain

    # --- Graph Builder ---
    def _build_supervisor_graph(self):
        """Assembles the LangGraph StateGraph."""
        
        workflow = StateGraph(AgentState)

        # 1. Add the Supervisor node (uses the chain built above)
        workflow.add_node("Supervisor", self.supervisor_chain)

        # 2. Add worker nodes
        for name in AGENT_NAMES:
            # All worker agents use the same execution node: _run_worker_agent
            workflow.add_node(name, self._run_worker_agent) 
            
        workflow.set_entry_point("Supervisor")

        # 3. Define the conditional routing edge
        def router(state: AgentState):
            """Routes based on the supervisor's decision with safeguards."""
            # SAFEGUARD: If the last message is from an agent, force FINISH
            last_msg = state["messages"][-1] if state["messages"] else None
            if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'name') and last_msg.name in AGENT_NAMES:
                LOGGER.debug(f"🛑 Safeguard triggered: Last message from {last_msg.name}, forcing FINISH")
                return "FINISH"
            
            next_route = state["next"]
            LOGGER.debug(f"🔀 Supervisor routing to: {next_route}")
            return next_route 
        
        workflow.add_conditional_edges(
            "Supervisor",
            router,
            # Map the string output from the router to a node name
            {name: name for name in AGENT_NAMES} | {"FINISH": END}
        )

        # 4. Define the cycle: Workers always return to the Supervisor
        for name in AGENT_NAMES:
            workflow.add_edge(name, "Supervisor")

        return workflow.compile(checkpointer=self.checkpointer)

    # --- Public API for UI Integration (Async) ---
    async def run_query(self, user_input: str, session_id: str) -> str:
        """
        The public method called by the client/UI.
        
        :param user_input: The user's query string.
        :param session_id: The session identifier for maintaining conversation history.
        :return: The final response string from the agent system.
        """
        with TRACER.start_as_current_span("supervisor_run_query") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("user_input", user_input[:50])  # Log first 50 chars

            LOGGER.info(f"Processing query: {user_input[:50]}...")

            # 1. Define the LangGraph/LangChain configuration
            config: RunnableConfig = {
                "configurable": {
                    # This is the key LangGraph uses to retrieve/save the state
                    "thread_id": session_id 
                },
                "recursion_limit": 10  # Prevent infinite loops - max 10 steps
            }

            # 2. Prepare the input
            # Only pass the new user message - the checkpointer will merge with history
            input_data = {
                "messages": [HumanMessage(content=user_input)],
                "next": "",
                "session_id": session_id
            }

            try:
                # 3. Invoke the graph with recursion limit
                final_state = await self.graph.ainvoke(input_data, config=config)
                
                # 4. Extract the final AI response (last message in the list)
                # Find the last AIMessage in the conversation
                for message in reversed(final_state["messages"]):
                    if isinstance(message, AIMessage):
                        return message.content
                
                # Fallback: if no AI message found, return a default message
                return "I apologize, but I was unable to generate a response."
                
            except Exception as e:
                error_msg = f"Error in supervisor: {str(e)}"
                LOGGER.error(f"{error_msg}")
                return f"I apologize, but I encountered an error: {error_msg}"


# Example of how to use this class:
# async def main():
#     supervisor = SupervisorAgent()
#     
#     # First query
#     response1 = await supervisor.run_query("What is a 401k?", session_id="user123")
#     print(response1)
#     
#     # Follow-up query in same session
#     response2 = await supervisor.run_query("How much can I contribute?", session_id="user123")
#     print(response2)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())