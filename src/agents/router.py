from typing import TypedDict, List, Dict, Annotated, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.agents.finance_q_and_a import FinanceQandAAgent 
from src.agents.finance_market import FinanceMarketAgent
from src.agents.finance_portfolio import PortfolioAgent
from src.agents.finance_goals import GoalsAgent
from src.agents.response import AgentResponse
from src.utils import setup_logger_with_tracing, setup_tracing,  get_tracer
from functools import partial

# Setup Logger
setup_tracing("router", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, service_name="router")
TRACER = get_tracer(__name__)

_FINANCE_QA_AGENT = FinanceQandAAgent()
_FINANCE_MARKET_AGENT = FinanceMarketAgent()
_PORTFOLIO_AGENT = PortfolioAgent()
_GOALS_AGENT = GoalsAgent()

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
    current_portfolio: Dict[str,float]
    response: List[AgentResponse]

def get_empty_portfolio() -> Dict[str, float]:
    """Returns a new portfolio with all zeros."""
    return {
        "Equities": 0.0,
        "Fixed_Income": 0.0,
        "Real_Estate": 0.0,
        "Commodities": 0.0,
        "Crypto": 0.0,
        "Cash": 0.0
    }


class RouterAgent:
    """A router agent that directs queries to specialized financial agents."""
    
    def __init__(self):
        #async initialization of agents will be handled in run_query
        self.finance_qa_agent = _FINANCE_QA_AGENT
        self.finance_market_agent = _FINANCE_MARKET_AGENT
        self.portfolio_agent = _PORTFOLIO_AGENT
        self.goals_agent = _GOALS_AGENT
    
        self.saver = InMemorySaver()
        # Build the state graph
        router_builder = StateGraph(AgentState)
        router_builder.add_node("router_node", self.router_node)
        router_builder.add_node(partial(self._run_agent_logic, agent_instance=self.finance_qa_agent), self.finance_q_and_a_node)
        router_builder.add_node(partial(self._run_agent_logic, agent_instance=self.finance_market_agent), self.finance_market_node)
        router_builder.add_node(partial(self._run_agent_logic, agent_instance=self.finance_market_agent), self.portfolio_node)
        router_builder.add_node(partial(self._run_agent_logic, agent_instance=self.finance_market_agent), self.goals_node)

        router_builder.add_edge(START, "router_node")
        router_builder.add_conditional_edges(
            "router_node", 
            self.route_next,
            {
                "FinanceQandAAgent": "FinanceQandAAgent",
                "FinanceMarketAgent": "FinanceMarketAgent",
                "PortfolioAgent": "PortfolioAgent",
                "GoalsAgent": "GoalsAgent"
            },
        )
        #STRETCH: if we have time, have these conect with router, to allow multi-agent response
        router_builder.add_edge("FinanceQandAAgent", END)
        router_builder.add_edge("FinanceMarketAgent", END)
        router_builder.add_edge("PortfolioAgent", END)
        router_builder.add_edge("GoalsAgent", END)
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

    async def _run_agent_logic(self, state: AgentState, agent_instance) -> AgentState:
        agent_name = agent_instance.__class__.__name__
        LOGGER.info(f"Routing to {agent_name}")
        
        try:
            agent_response: AgentResponse = await agent_instance.run_query(
                state["messages"],
                state["session_id"]
            )

            state["messages"].append(AIMessage(content=agent_response.message))
            state["response"] = agent_response
            state["next"] = "end"

        except Exception as e:
            LOGGER.error(f"Error in {agent_name}: {e}")
            error_msg = "An error occurred while processing your request."
            state["messages"].append(AIMessage(content=error_msg))
            state["response"] = AgentResponse(
                agent=agent_name,
                message=error_msg,
                charts=[]
            )
            state["next"] = "end"

        return state

    async def router_node(self, state: AgentState) -> AgentState:   
        """Node to route the query to the appropriate specialized agent."""
        LOGGER.info("Routing node invoked")

        if "current_portfolio" not in state or state.get("current_portfolio") is None:
            state["current_portfolio"] = get_empty_portfolio()
            LOGGER.info("Initialized empty portfolio for new session")

        # Prepare the prompt for the router LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""
# ROLE
You are a high-precision Financial Intent Router. Your sole purpose is to select the correct NODE for the user's request.

# ROUTING TABLE (MATCH TO NODE NAMES)
1. **FinanceMarketAgent**: 
    - PRIMARY INTENT: Real-time data, price quotes, ticker/company lookups, and asset breakdowns.
    - TRIGGER: Mention of a company name (Oracle, Apple) or ticker (ORCL) WITHOUT portfolio building context
    - TRIGGER: Questions about asset classes, holdings, composition, or breakdown of a specific fund/stock/ETF
    - EXAMPLES: "What are the asset classes for VOO?", "Show me the breakdown of Vanguard 2040", "What's the price of AAPL?"

2. **PortfolioAgent**: 
    - PRIMARY INTENT: Building, modifying, and analyzing personal portfolio allocations.
    - TRIGGER: Portfolio building language: "build", "add", "remove", "create portfolio"
    - TRIGGER: Portfolio modification: "I have $X in Y", "add $X to Z", "remove from", "start over"
    - TRIGGER: Portfolio analysis: "summarize my portfolio", "show my portfolio", "analyze my portfolio", "assess risk"
    - TRIGGER: Hypothetical analysis: "what if", "compare", "how would X look"
    - TRIGGER: User provides dollar/percentage amounts with intent to build/modify portfolio
    - EXAMPLES: "I want to build my portfolio", "Add $100k to Equities", "I have $500k in Vanguard 2040", "Summarize my portfolio", "What if I had 60% stocks?"

3. **GoalsAgent**:
    - PRIMARY INTENT: Future projections, simulations, and goal planning.
    - TRIGGER: Time-based questions: "in 10 years", "future", "projection", "forecast", "simulate"
    - TRIGGER: Goal-oriented: "will I reach", "chances of hitting", "probability", "target"
    - TRIGGER: Simulation requests: "run a simulation", "monte carlo", "project growth"
    - EXAMPLES: "How will my portfolio do in 10 years?", "What are the chances I'll reach $1M?", "Simulate 20 years", "Project my growth"

4. **FinanceQandAAgent**: 
    - PRIMARY INTENT: General financial theory, definitions, concepts, and educational "how-to" advice.
    - TRIGGER: "What is", "How does", "Explain", "Define" WITHOUT a specific company/ticker/fund name
    - TRIGGER: General financial education questions
    - EXAMPLES: "What is a 401k?", "How does compound interest work?", "Explain diversification"

# CONTEXT-AWARE ROUTING
**Check conversation history and context:**
- If the previous message was handled by `PortfolioAgent` AND current query is about future/simulation (e.g., "how will it do in 10 years?"), route to `GoalsAgent`
- If the previous message was handled by `GoalsAgent` AND current query is about modifying portfolio (e.g., "add more to equities"), route to `PortfolioAgent`
- If the previous message was handled by `FinanceMarketAgent` AND current query is a follow-up (e.g., "what's the price?", "how's it performing?"), route to `FinanceMarketAgent`
- Context clues: pronouns (it, that, this, my portfolio) indicate a follow-up to the previous agent

# DISAMBIGUATION RULES

**Portfolio Building vs. Market Lookup:**
- "I have $100k in Apple" + building context → `PortfolioAgent` (adding to portfolio)
- "What's in Apple?" or "Apple's breakdown?" → `FinanceMarketAgent` (just looking up info)

**Portfolio Analysis vs. Simulation:**
- "Show my portfolio" or "Summarize my allocation" → `PortfolioAgent` (current state analysis)
- "How will my portfolio do?" or "Simulate 10 years" → `GoalsAgent` (future projection)

**Current vs. Future:**
- Current state questions → `PortfolioAgent`
- Future/time-based questions → `GoalsAgent`

# SPECIAL RULE: MIXED INTENT (FIRST MENTIONED)
If a user query contains multiple intents, route based on the **FIRST MENTIONED** or **PRIMARY** request:
- "What is the price of Oracle and how does a 401k work?" → `FinanceMarketAgent` (Price first)
- "How do I save for college and what's the price of Bitcoin?" → `FinanceQandAAgent` (General advice first)
- "Add $100k to my portfolio and simulate 10 years" → `PortfolioAgent` (Building first, simulation is follow-up)

# RULES & OVERRIDES
- **Context Priority**: Follow-up questions route to the agent that makes sense given context
- **Entity Priority**: Specific company/ticker mentions → `FinanceMarketAgent` UNLESS clearly about portfolio building
- **Time Priority**: Future-oriented questions → `GoalsAgent`
- **Building Priority**: Portfolio modification/analysis → `PortfolioAgent`
- **Output Format**: Exactly one string: `FinanceMarketAgent`, `PortfolioAgent`, `GoalsAgent`, or `FinanceQandAAgent`

# ROUTING DECISION TREE
```
Does query mention future/time/simulation/goals?
├─ YES → GoalsAgent
└─ NO → Continue

Does query mention building/adding/removing/analyzing portfolio?
├─ YES → PortfolioAgent
└─ NO → Continue

Does query mention specific company/ticker/fund for lookup?
├─ YES → FinanceMarketAgent
└─ NO → Continue

Is query general financial education?
├─ YES → FinanceQandAAgent
└─ NO → Check context and route to last_agent_used
```

# EXAMPLES

**GoalsAgent routing:**
- "How will my portfolio do in 10 years?" → GoalsAgent
- "What are my chances of reaching $5M?" → GoalsAgent
- "Run a 20-year simulation" → GoalsAgent
- "Project growth over next decade" → GoalsAgent

**PortfolioAgent routing:**
- "I want to build my portfolio" → PortfolioAgent
- "Add $100k to Equities" → PortfolioAgent
- "I have $500k in Vanguard 2040" → PortfolioAgent
- "Show me my portfolio" → PortfolioAgent
- "What if I had 70% stocks?" → PortfolioAgent
- "Assess my risk" → PortfolioAgent

**FinanceMarketAgent routing:**
- "What's Apple's stock price?" → FinanceMarketAgent
- "Show me VOO's asset breakdown" → FinanceMarketAgent
- "What are the holdings in QQQ?" → FinanceMarketAgent

**FinanceQandAAgent routing:**
- "What is diversification?" → FinanceQandAAgent
- "How does a Roth IRA work?" → FinanceQandAAgent
- "Explain dollar cost averaging" → FinanceQandAAgent

**Context-aware routing:**
- Previous: User built portfolio with PortfolioAgent
- Current: "How will it do in 10 years?"
- Route: GoalsAgent (simulation on existing portfolio)

- Previous: User ran simulation with GoalsAgent  
- Current: "Add $50k to bonds"
- Route: PortfolioAgent (modifying portfolio)

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
    
