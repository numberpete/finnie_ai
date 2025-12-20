from typing import TypedDict, List, Dict, Any, Annotated
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
from src.utils import setup_logger_with_tracing, setup_tracing,  get_tracer
import logging


# Setup Logger
setup_tracing("router", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.DEBUG)
TRACER = get_tracer(__name__)

_FINANCE_QA_AGENT = FinanceQandAAgent()
_FINANCE_MARKET_AGENT = FinanceMarketAgent()

# --- CONFIGURATION ---
ROUTER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
AGENT_NAMES = ["FinanceQandAAgent","FinanceMarketAgent"] # Add "OtherAgent" when ready

# Define the state structure for the graph
class AgentState(TypedDict):
    """Represents the state of our multi-agent conversation."""
    messages: Annotated[List[BaseMessage], add_messages]
    next: str # The name of the next agent/node to run
    session_id: str


class RouterAgent:
    """A router agent that directs queries to specialized financial agents."""
    
    def __init__(self):
        #async initialization of agents will be handled in run_query
        self.finance_qa_agent = _FINANCE_QA_AGENT
        self.finance_market_agent = _FINANCE_MARKET_AGENT
    
        self.saver = InMemorySaver()
        # Build the state graph
        router_builder = StateGraph(AgentState)
        router_builder.add_node("router_node", self.router_node)
        router_builder.add_node("FinanceQandAAgent", self.finance_q_and_a_node)
        router_builder.add_node("FinanceMarketAgent", self.finance_market_node)

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
        self.workflow = router_builder.compile(checkpointer=self.saver)



    def route_next(self,state: AgentState) -> str:
        """Determines the next node based on the state's 'next' field."""
        return state["next"]

    async def run_query(self, user_query: str, session_id: str) -> str:
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
            final_state = await self.workflow.ainvoke(
                input_data,
                config=config
            )
            
            if final_state and "messages" in final_state:
                return final_state["messages"][-1].content
            
            return "No response generated."   

    async def finance_q_and_a_node(self, state: AgentState) -> AgentState:
        """Node to handle Finance Q&A Agent."""
        LOGGER.info("Routing to FinanceQandAAgent")
        try:
            response = await self.finance_qa_agent.run_query(state["messages"], state["session_id"])
            state["messages"].append(AIMessage(content=response))
            state["next"] = "end"
        except Exception as e:
            LOGGER.error(f"Error in FinanceQandAAgent: {e}")
            state["messages"].append(AIMessage(content="An error occurred while processing your request."))
            state["next"] = "end"
        return state

    async def finance_market_node(self, state: AgentState) -> AgentState:
        """Node to handle Finance Market Agent."""
        LOGGER.info("Routing to FinanceMarketAgent")
        try:
            response = await self.finance_market_agent.run_query(state["messages"], state["session_id"])
            state["messages"].append(AIMessage(content=response))
            state["next"] = "end"
        except Exception as e:
            LOGGER.error(f"Error in FinanceMarketAgent: {e}")
            state["messages"].append(AIMessage(content="An error occurred while processing your request."))
            state["next"] = "end"
        return state


    async def router_node(self, state: AgentState) -> AgentState:   
        """Node to route the query to the appropriate specialized agent."""
        LOGGER.info("Routing node invoked")
        
        # Prepare the prompt for the router LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""
You are a supervisor coordinating specialist agents to answer financial questions.

**AGENTS:**
- FinanceQandAAgent: Financial concepts, definitions, investment strategies, retirement planning
- FinanceMarketAgent: Real-time prices, historical data, market indices

**APPROACH:**
1. Analyze what information is needed to answer the user's question
2. Route to appropriate agent(s) to gather that information
3. If question needs multiple agents, route sequentially
4. FINISH when the user's question is fully answered

**ROUTING:**
- Financial concepts/advice → FinanceQandAAgent
- Market data/prices → FinanceMarketAgent  
- Multi-part questions → Route to each agent needed, then FINISH
- Agent responded and question is fully answered → FINISH
- Simple greeting/thanks → FINISH

**EXAMPLES:**
- "What is an IRA?" → FinanceQandAAgent → FINISH
- "What's AAPL's price?" → FinanceMarketAgent → FINISH
- "Explain DCA and show AAPL history" → FinanceQandAAgent → FinanceMarketAgent → FINISH
- "What's AAPL?" [responds] "Now show MSFT" → FinanceMarketAgent (follow-up is ok)
- "Thanks for your help!" → FINISH

**CONSTRAINTS & SAFETY:**
- **No Advice:** Do not provide specific "Buy/Sell" recommendations or personalized financial advice.
- **Max Depth:** Limit to 3 agent calls per turn.
- **State Awareness:** If the answer is already present in the conversation history, do not call an agent; simply summarize.
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
    
