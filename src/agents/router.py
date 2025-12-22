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
                "PortfolioAgent": "PortfolioAgent"
            },
        )
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
You are a routing supervisor that coordinates specialist agents to answer financial questions.

**AVAILABLE AGENTS:**
- FinanceQandAAgent: Financial concepts, definitions, investment strategies, retirement planning, general financial education
- FinanceMarketAgent: Real-time market data, stock prices, historical performance, ticker lookup, market indices, chart generation
- PortfolioAgent: Portfolio construction, analysis, risk assessment, asset allocation, performance projections and simulations

**YOUR ROLE:**
After each user message, decide whether to:
1. Route to an agent to gather information
2. FINISH (if the question has been fully answered or no agent is needed)

**ROUTING RULES:**
- Financial concepts/theory/education → FinanceQandAAgent
- Market data/prices/charts/historical performance → FinanceMarketAgent
- Portfolio building/analysis/risk assessment/projections → PortfolioAgent
- Follow-up questions → Route to appropriate agent (context is maintained)
- Greetings/thanks/chitchat → FINISH immediately
- Question already fully answered → FINISH immediately

**CONTEXT TRACKING:**
- last_ticker: {last_ticker} - Use for pronoun resolution ("it", "that stock")
- pending_ticker: {pending_ticker} - Tracks ticker awaiting user confirmation

**MULTI-PART QUESTIONS:**
For questions requiring multiple agents:
- Route to the FIRST agent needed
- Agent will return with partial answer
- You'll route to the NEXT agent needed
- Continue until all information is gathered
- Then FINISH

Example flow:
User: "Explain DCA and show AAPL's performance"
→ Route to FinanceQandAAgent (explains DCA)
→ [Agent returns]
→ Route to FinanceMarketAgent (shows AAPL data)
→ [Agent returns]
→ FINISH (question fully answered)

**ROUTING EXAMPLES:**
✓ "What is dollar-cost averaging?" → FinanceQandAAgent
✓ "What's the current price of AAPL?" → FinanceMarketAgent
✓ "What's AAPL?" [agent responds] "Now show MSFT" → FinanceMarketAgent
✓ "Analyze my portfolio: 60% stocks, 30% bonds, 10% cash" → PortfolioAgent
✓ "Thanks!" → FINISH
✓ "What was AAPL's price?" [agent responds] "What about it?" → FinanceMarketAgent (resolve "it" to AAPL)

**IMPORTANT CONSTRAINTS:**
- **No Direct Answers:** Your job is routing only - let agents answer questions
- **No Hallucination:** If unsure which agent, pick the most logical one
- **Max Routing Depth:** Track agent calls - if you've routed 3+ times for the same question, FINISH
- **Efficiency:** Don't re-route if the question has already been answered

**WHEN TO FINISH:**
- User's question has been fully answered by agent(s)
- User sends greeting/thanks/casual remark
- Answer already exists in conversation history
- Maximum routing depth reached (3 agent calls)
- User's intent is unclear (respond asking for clarification)
"""
            ),
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
    
