from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import Literal   

# Import your existing worker agent
from src.agents.finance_q_and_a import FinanceQandAAgent 
from src.agents.finance_market import FinanceMarketAgent
from src.utils import setup_logger_with_tracing, setup_tracing,  get_tracer
import logging


# Setup Logger
setup_tracing("supervisor", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)
TRACER = get_tracer(__name__)

# --- CONFIGURATION ---
ROUTER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
AGENT_NAMES = ["FinanceQandAAgent","FinanceMarketAgent"] # Add "OtherAgent" when ready

# Define the state structure for the graph
class AgentState(TypedDict):
    """Represents the state of our multi-agent conversation."""
    messages: Annotated[list, add_messages]
    next: str # The name of the next agent/node to run
    session_id: str


class RouterAgent:
    """A router agent that directs queries to specialized financial agents."""
    
    def __init__(self):
        # Build the state graph
        router_builder = StateGraph(AgentState)
        router_builder.add_node("router_node", router_node)
        router_builder.add_node("FinanceQandAAgent", finance_q_and_a_node)
        router_builder.add_node("FinanceMarketAgent", finance_market_node)

        router_builder.add_edge(START, "router_node")
        router_builder.add_conditional_edges(
            "router_node", 
            self.route_next,
            {
                "FinanceQandAAgent": "FinanceQandAAgent",
                "FinanceMarketAgent": "FinanceMarketAgent",
            },
        )
        router_builder.add_edge("FinanceQandAAgent", END)
        router_builder.add_edge("FinanceMarketAgent", END)
        self.workflow = router_builder.compile()

    def route_next(state: AgentState) -> str:
        """Determines the next node based on the state's 'next' field."""
        return state["next"]

    async def run_query(self, user_query: str, session_id: str) -> str:
        """Runs the router agent while maintaining conversation history."""
        
        # 1. Map session_id to thread_id in the config
        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {"tracer": TRACER} # Metadata for tracing
        }
        
        # 2. Only pass the NEW message. 
        # LangGraph will automatically merge this with existing state for this thread_id.
        input_data = {"messages": [HumanMessage(content=user_query)]}
        
        # 3. Invoke the graph with the config
        final_state = await self.workflow.ainvoke(
            input_data,
            config=config
        )
        
        if final_state and "messages" in final_state:
            return final_state["messages"][-1].content
        
        return "No response generated."   



def router_node(state: AgentState) -> AgentState:   
    """Node to route the query to the appropriate specialized agent."""
    LOGGER.info("Routing node invoked")
    
    # Prepare the prompt for the router LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a routing agent that decides which specialized financial agent "
            "should handle the user's query based on its content. "
            "Available agents: " + ", ".join(AGENT_NAMES) + 
            ". Respond with the name of the most appropriate agent."
        )),
        HumanMessage(content=(
            "User query: {user_query}\n"
            "Available agents: " + ", ".join(AGENT_NAMES) + 
            "\nWhich agent should handle this query? Respond with only the agent name."
        ))
    ])

    # Get the latest user message
    user_message = state["messages"][-1]
    
    # Format the prompt with the user query
    formatted_prompt = prompt.format_prompt(user_query=user_message.content)
    
    # Call the router LLM
    response = ROUTER_LLM(formatted_prompt.to_messages())
    chosen_agent = response.content.strip()
    
    LOGGER.info(f"Router selected agent: {chosen_agent}")
    
    # Update state to indicate next node
    if chosen_agent in AGENT_NAMES:
        state["next"] = chosen_agent
    else:
        LOGGER.warning(f"Unknown agent selected: {chosen_agent}. Defaulting to FinanceQandAAgent.")
        state["next"] = "FinanceQandAAgent"  # Default fallback
    
    return state
    
#Nodes for each specialized agent
def finance_q_and_a_node(state: AgentState) -> AgentState:
    """Node to handle Finance Q&A Agent."""
    LOGGER.info("Routing to FinanceQandAAgent")
    agent = FinanceQandAAgent()
    user_message = state["messages"][-1]  # Get the latest user message
    response = agent.run_query(user_message.content, state["session_id"])
    state["messages"].append(AIMessage(content=response))
    state["next"] = "end"  # Next node
    return state

def finance_market_node(state: AgentState) -> AgentState:
    """Node to handle Finance Market Agent."""
    LOGGER.info("Routing to FinanceMarketAgent")
    agent = FinanceMarketAgent()
    user_message = state["messages"][-1]  # Get the latest user message
    response = agent.run_query(user_message.content, state["session_id"])
    state["messages"].append(AIMessage(content=response))
    state["next"] = "end"  # Next node
    return state