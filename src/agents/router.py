from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import Literal   

# Import your existing worker agent
from src.agents.finance_q_and_a import FinanceQandAAgent 
from src.agents.finance_market import FinanceMarketAgent
#from src.agents.finance_portfolio import PortfolioAgent
from src.agents.response import AgentResponse, ChartArtifact
from src.utils import setup_logger_with_tracing, setup_tracing,  get_tracer
import logging


# Setup Logger
setup_tracing("router", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)
TRACER = get_tracer(__name__)

_FINANCE_QA_AGENT = FinanceQandAAgent()
_FINANCE_MARKET_AGENT = FinanceMarketAgent()
#_PORTFOLIO_AGENT = PortfolioAgent()

# --- CONFIGURATION ---
ROUTER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
AGENT_NAMES = ["FinanceQandAAgent","FinanceMarketAgent"] # Add "OtherAgent" when ready

# Define the state structure for the graph
class AgentState(TypedDict):
    """Represents the state of our multi-agent conversation."""
    messages: Annotated[List[BaseMessage], add_messages]
    next: str # The name of the next agent/node to run
    session_id: str
    response: Optional[AgentResponse]


class RouterAgent:
    """A router agent that directs queries to specialized financial agents."""
    
    def __init__(self):
        #async initialization of agents will be handled in run_query
        self.finance_qa_agent = _FINANCE_QA_AGENT
        self.finance_market_agent = _FINANCE_MARKET_AGENT
        #self.portfolio_agent = _PORTFOLIO_AGENT
    
        self.saver = InMemorySaver()
        # Build the state graph
        router_builder = StateGraph(AgentState)
        router_builder.add_node("router_node", self.router_node)
        router_builder.add_node("FinanceQandAAgent", self.finance_q_and_a_node)
        router_builder.add_node("FinanceMarketAgent", self.finance_market_node)
#        router_builder.add_node("PortfolioAgent", self.portfolio_node)

        router_builder.add_edge(START, "router_node")
        router_builder.add_conditional_edges(
            "router_node", 
            self.route_next,
            {
                "FinanceQandAAgent": "FinanceQandAAgent",
                "FinanceMarketAgent": "FinanceMarketAgent",
#                "PortfolioAgent": "PortfolioAgent"
            },
        )
        router_builder.add_edge("FinanceQandAAgent", END)
        router_builder.add_edge("FinanceMarketAgent", END)
        #router_builder.add_edge("PortfolioAgent", END)
        self.workflow = router_builder.compile(checkpointer=self.saver)



    def route_next(self,state: AgentState) -> str:
        """Determines the next node based on the state's 'next' field."""
        return state["next"]

    async def run_query(self, user_query: str, session_id: str) -> AgentResponse:
        """Runs the router agent while maintaining conversation history."""
        with TRACER.start_as_current_span("router_run_query") as span:       
            # 1. Map session_id to thread_id in the config
            config = {
                "configurable": {"thread_id": session_id},
                "metadata": {"tracer": span} # Metadata for tracing
            }

            current_state = await self.workflow.aget_state(config)
            LOGGER.debug(f"History has {len(current_state.values.get('messages', []))} messages")
            
            # 2. Only pass the NEW message. 
            # LangGraph will automatically merge this with existing state for this thread_id.
            input_data = {
                "messages": [HumanMessage(content=user_query)],
                "session_id": session_id
            }
            
            # 3. Invoke the graph with the config
            final_state: AgentState = await self.workflow.ainvoke(
                input_data,
                config=config
            )
            
            if final_state.get("response"):
                return final_state["response"]

            return AgentResponse(
                agent="Router",
                message="No response generated.",
                charts=[]
            )

    async def finance_q_and_a_node(self, state: AgentState) -> AgentState:
        LOGGER.info("Routing to FinanceQandAAgent")
        try:
            agent_response: AgentResponse = await self.finance_qa_agent.run_query(
                state["messages"],
                state["session_id"]
            )

            # 1. Preserve conversation continuity
            state["messages"].append(
                AIMessage(content=agent_response.message)
            )

            # 2. Store structured output
            state["response"] = agent_response
            state["next"] = "end"

        except Exception as e:
            LOGGER.error(f"Error in FinanceQandAAgent: {e}")
            state["messages"].append(
                AIMessage(content="An error occurred while processing your request.")
            )
            state["response"] = AgentResponse(
                agent="FinanceQandAAgent",
                message="An error occurred while processing your request.",
                charts=[]
            )
            state["next"] = "end"

        return state

    async def finance_market_node(self, state: AgentState) -> AgentState:
        LOGGER.info("Routing to FinanceMarketAgent")
        try:
            agent_response: AgentResponse = await self.finance_market_agent.run_query(
                state["messages"],
                state["session_id"]
            )

            state["messages"].append(
                AIMessage(content=agent_response.message)
            )

            state["response"] = agent_response
            state["next"] = "end"

        except Exception as e:
            LOGGER.error(f"Error in FinanceMarketAgent: {e}")
            state["messages"].append(
                AIMessage(content="An error occurred while processing your request.")
            )
            state["response"] = AgentResponse(
                agent="FinanceMarketAgent",
                message="An error occurred while processing your request.",
                charts=[]
            )
            state["next"] = "end"

        return state

    async def portfolio_node(self, state: AgentState) -> AgentState:
        LOGGER.info("Routing to PortfolioAgent")
        try:
            agent_response: AgentResponse = await self.portfolio_agent.run_query(
                state["messages"],
                state["session_id"]
            )

            state["messages"].append(
                AIMessage(content=agent_response.message)
            )

            state["response"] = agent_response
            state["next"] = "end"

        except Exception as e:
            LOGGER.error(f"Error in PortfolioAgent: {e}")
            state["messages"].append(
                AIMessage(content="An error occurred while processing your request.")
            )
            state["response"] = AgentResponse(
                agent="PortfolioAgent",
                message="An error occurred while processing your request.",
                charts=[]
            )
            state["next"] = "end"

        return state


    async def router_node(self, state: AgentState) -> AgentState:   
        """Node to route the query to the appropriate specialized agent."""
        LOGGER.info("Routing node invoked")
        
        # Prepare the prompt for the router LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""
# ROLE
You are a highly efficient Intent Routing Agent. Your sole purpose is to analyze the user's "Query" and determine which specialized expert agent from the "AGENTS" list is best equipped to handle the request.

# AGENTS
- FinanceQandAAgent: For financial concepts, definitions, or general educational questions.
- FinanceMarketAgent: For real-time prices, historical data, specific market indices, or ticker-related questions.

# RULES & CONSTRAINTS
1. Analyze the user's "Query" carefully to determine the underlying intent.
2. If the user is asking for a specific price, ticker, or historical performance, route to the FinanceQandAAgent.
3. If the user is asking "what is" or "how does" a concept work, route to the FinanceQandAAgent.
4. If a query is ambiguous, use your best judgment to route to the most likely relevant agent.
5. If a query contains multiple intents, route to the agent responsible for the primary or first mentioned task.
6. STICK TO THE LIST: You must only output the name of the agent in the format specified below. Do not invent new agent names.

# OUTPUT FORMAT
Your output must be a single word (the Agent Name) in uppercase with no other text, explanation, or punctuation.
Example: FinanceMarketAgent

# EXAMPLES
Query: What is a P/E ratio?
Output: FinanceQandAAgent

Query: What is the current price of SPY?
Output: FinanceMarketAgent
"""           ),
            ("human",
                "User query: {user_query}\n"
                "Available agents: " + ", ".join(AGENT_NAMES) + 
                "\nWhich agent should handle this query? Respond with only the agent name."
            )
        ])

        # Get the latest user message
        user_message = state["messages"][-1]
        
        # Format the prompt with the user query
        formatted_prompt = prompt.format_prompt(user_query=user_message.content)
        
        # Call the router LLM
        response = await ROUTER_LLM.ainvoke(formatted_prompt.to_messages())
        chosen_agent = response.content.strip()
        
        LOGGER.info(f"Router selected agent: {chosen_agent}")
        
        # Update state to indicate next node
        if chosen_agent in AGENT_NAMES:
            state["next"] = chosen_agent
        else:
            LOGGER.warning(f"Unknown agent selected: {chosen_agent}. Defaulting to FinanceQandAAgent.")
            state["next"] = "FinanceQandAAgent"  # Default fallback
        
        return state
    
