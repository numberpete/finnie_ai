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
from src.agents.finance_q_and_a import FinanceQandAAgent 
from src.agents.finance_market import FinanceMarketAgent
from src.agents.finance_portfolio import PortfolioAgent
from src.agents.response import AgentResponse, ChartArtifact
from src.utils import setup_logger_with_tracing, setup_tracing,  get_tracer
import logging


# Setup Logger
setup_tracing("router", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)
TRACER = get_tracer(__name__)

_FINANCE_QA_AGENT = FinanceQandAAgent()
_FINANCE_MARKET_AGENT = FinanceMarketAgent()
_PORTFOLIO_AGENT = PortfolioAgent()

# --- CONFIGURATION ---
ROUTER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
AGENT_NAMES = ["FinanceQandAAgent","FinanceMarketAgent","PortfolioAgent"] # Add "OtherAgent" when ready

# Define the state structure for the graph
class AgentState(TypedDict):
    """Represents the state of our multi-agent conversation."""
    messages: Annotated[List[BaseMessage], add_messages]
    next: str # The name of the next agent/node to run
    session_id: str
    last_agent_used: Optional[str]
    response: List[AgentResponse]


class RouterAgent:
    """A router agent that directs queries to specialized financial agents."""
    
    def __init__(self):
        #async initialization of agents will be handled in run_query
        self.finance_qa_agent = _FINANCE_QA_AGENT
        self.finance_market_agent = _FINANCE_MARKET_AGENT
        self.portfolio_agent = _PORTFOLIO_AGENT
    
        self.saver = InMemorySaver()
        # Build the state graph
        router_builder = StateGraph(AgentState)
        router_builder.add_node("router_node", self.router_node)
        router_builder.add_node("FinanceQandAAgent", self.finance_q_and_a_node)
        router_builder.add_node("FinanceMarketAgent", self.finance_market_node)
        router_builder.add_node("PortfolioAgent", self.portfolio_node)

        router_builder.add_edge(START, "router_node")
        router_builder.add_conditional_edges(
            "router_node", 
            self.route_next,
            {
                "FinanceQandAAgent": "FinanceQandAAgent",
                "FinanceMarketAgent": "FinanceMarketAgent",
                "PortfolioAgent": "PortfolioAgent",
            },
        )
        #STRETCH: if we have time, have these conect with router, to allow multi-agent response
        router_builder.add_edge("FinanceQandAAgent", END)
        router_builder.add_edge("FinanceMarketAgent", END)
        router_builder.add_edge("PortfolioAgent", END)
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
You are a high-precision Financial Intent Router. Your sole purpose is to select the correct NODE for the user's request.

# ROUTING TABLE (MATCH TO NODE NAMES)
1. **FinanceMarketAgent**: 
    - PRIMARY INTENT: Real-time data, price quotes, ticker/company lookups, and asset breakdowns.
    - TRIGGER: Mention of a company name (Oracle, Apple) or ticker (ORCL).
    - TRIGGER: Questions about asset classes, holdings, composition, or breakdown of a specific fund/stock/ETF
    - EXAMPLES: "What are the asset classes for VOO?", "Show me the breakdown of Vanguard 2040", "What does AAPL hold?"
2. **PortfolioAgent**: 
    - PRIMARY INTENT: Personal financial portfolio and growth simulations.
    - TRIGGER: User provides dollar/percentage totals for asset classes (e.g., "Equities: 100k") OR provides data in a JSON/List format containing "Equities", "Fixed Income", "Cash", etc.
    - TRIGGER: Questions about portfolio simulations, portfolio projections, or portfolio growth (e.g., "how will my portfolio do in 10 years?")
    - RULE: If you see a list of asset classes with associated numbers, it is ALWAYS a `PortfolioAgent` intent.
3. **FinanceQandAAgent**: 
    - PRIMARY INTENT: General financial theory, definitions, concepts, and educational "how-to" advice.
    - TRIGGER: "What is", "How does", "Explain", "Define" WITHOUT a specific company/ticker/fund name
    - EXAMPLES: "What is a 401k?", "How does compound interest work?", "Explain diversification"

# CONTEXT-AWARE ROUTING
**Check conversation history before routing:**
- If the previous message was handled by `PortfolioAgent` AND the current query is a follow-up question (e.g., "how will it do?", "what about 10 years?", "run a simulation"), route to `PortfolioAgent`
- If the previous message was handled by `FinanceMarketAgent` AND the current query is a follow-up (e.g., "what's the price?", "how's it performing?", "how about Apple?"), route to `FinanceMarketAgent`
- Context clues: pronouns (it, that, this), time references without entities (10 years, next month), or vague queries indicate a follow-up

# SPECIAL RULE: MIXED INTENT (FIRST MENTIONED)
If a user query contains multiple intents, you MUST route based on the **FIRST MENTIONED** or **PRIMARY** request.
- Example: "What is the price of Oracle and how does a 401k work?" -> Route to `FinanceMarketAgent` (Price was mentioned first).
- Example: "How do I save for college and what's the price of Bitcoin?" -> Route to `FinanceQandAAgent` (General advice was mentioned first).

# RULES & OVERRIDES
- **Context Priority**: If the query is clearly a follow-up to the previous conversation, route to the same agent that handled the previous query
- **Entity Priority**: Any mention of a specific company (Oracle, Nvidia) is ALWAYS a `FinanceMarketAgent` intent. NEVER send specific company queries to `FinanceQandAAgent`.
- **Output Format**: Exactly one string in lowercase: `FinanceMarketAgent`, `PortfolioAgent`, or `FinanceQandAAgent`.

# MANDATORY FOOTER (Direct Responses Only)
"FinnieAI can make mistakes, and answers are for educational purposes only."
"""           ),
            ("human",
                "User query: {user_query}\n"
                "Available agents: " + ", ".join(AGENT_NAMES) + 
                "\nWhich agent should handle this query? Respond with only the agent name."
            )
        ])

        # Get the last agent used from state (if available)
        last_agent = state.get("last_agent_used", None)
        if last_agent:
            LOGGER.info(f"📝 Last agent used: {last_agent}")

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
            state["last_agent_used"] = chosen_agent
        else:
            LOGGER.warning(f"Unknown agent selected: {chosen_agent}. Defaulting to FinanceQandAAgent.")
            state["next"] = "FinanceQandAAgent"  # Default fallback
            state["last_agent_used"] = "FinanceQandAAgent"
        
        return state
    
