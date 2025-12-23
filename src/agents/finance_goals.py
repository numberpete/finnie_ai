# src/agents/finance_market.py

from langchain_openai import ChatOpenAI
from src.utils import setup_logger_with_tracing, setup_tracing
from src.agents.base_agent import BaseAgent
import logging


# Setup Logger
setup_tracing("finance-goals-agent", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, service_name="finance-goals-agent")


# --- CONFIGURATION & PROMPT ---

# Set the OpenAI API key and model name
MODEL = "gpt-4o-mini"
LLM = ChatOpenAI(model=MODEL, temperature=0, streaming=True, cache=True)

# Your detailed System Prompt (ENHANCED WITH CHART INSTRUCTIONS)
STRICT_SYSTEM_PROMPT = """
# ROLE
You run Monte Carlo simulations on portfolios and create visualizations.

# ASSET CLASS COLORS
Equities: #2E5BFF, Fixed Income: #46CDCF, Real Estate: #F08A5D, Cash: #3DDC84, Commodities: #FFD700, Crypto: #B832FF

# CORE RULES
1. **Each tool called ONCE** - If tool shows "Observation:" in {agent_scratchpad}, don't call again
2. **Never chart before simulation** - simple_monte_carlo_simulation MUST complete before create_stacked_bar_chart
3. **Read-only state** - Use AgentState.current_portfolio, never modify it

# FINDING PORTFOLIO
1. Check current message for portfolio data
2. If not found, use AgentState.current_portfolio (if not empty/all zeros)
3. If still not found, check conversation history
4. If still not found, ask user and STOP

# PIPELINE
**When user asks for simulation:**
1. Get portfolio (see above)
2. Call `simple_monte_carlo_simulation(portfolio={...}, years=10, target_goal=None)` 
   - Extract years from user request (default: 10)
   - Only set target_goal if user mentioned specific dollar amount goal
3. Wait for simulation results
4. Call `create_stacked_bar_chart` with simulation results
5. Summarize findings and reference chart by title
6. End with: "FinnieAI can make mistakes, and answers are for educational purposes only."

# EXAMPLES
```
"How will {"Equities": 1000000, "Cash": 400000} do in 10 years?"
→ Use portfolio from message, years=10

"What are chances my portfolio hits $10M in 20 years?"
→ Use AgentState.current_portfolio, years=20, target_goal=10000000

"Simulate my portfolio for 5 years"
→ If no portfolio found, ask: "Please provide your portfolio allocation across: Equities, Fixed Income, Real Estate, Cash, Commodities, Crypto"
```

========================
{agent_scratchpad}
========================
"""


# MCP Server Configuration - NOW WITH THREE SERVERS
MCP_SERVERS = {
    "charts_mcp": {
        "url": "http://localhost:8003/sse", 
        "description": "Chart generation tools"
    },
    "goals_mcp": {
        "url": "http://localhost:8004/sse", 
        "description": "Goals tools"
    }
}


class GoalsAgent(BaseAgent):
    """
    Enhanced LangChain ReAct agent for financial goal planning WITH chart generation.
    """
    def __init__(self):
        super().__init__(
            agent_name="GoalsAgent",
            llm=LLM,
            system_prompt=STRICT_SYSTEM_PROMPT,
            logger=LOGGER,
            mcp_servers=MCP_SERVERS
        )

