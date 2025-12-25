# Finnie AI - Multi-Agent Financial Assistant

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

**Finnie AI** is a multi-agent financial assistant system that leverages LangChain, LangGraph, and Model Context Protocol (MCP) tooling to provide (artificially) intelligent portfolio management, market analysis, and financial goal planning.  It was built to fulfill the capstone requirements for the Interview Kickstart Agentic AI program for developers.

---
## Quick Start

This is assuming you are runing on a Mac.  For the purpose of this project, it's a local "deployment".

Python 3.11+ is required.


1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/finnie_ai.git
cd finnie_ai
```

2. **Set up environment variables:**

Create a `.envrc` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

3. **run start.sh shell script**

The first run will take a few minutes as it builds up the FAISS indexes.  Subsequent starts should be faster.

```bash
./start.sh
```

And that should be it. If not, you can either dig into the documentation and try to figure it out.  Or contact me.

---
### Tests

```bash
 pytest tests/ --cov=src --cov-report=term-missing
```
---
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
  - [System Architecture](#system-architecture)
  - [Agent Object Model](#agent-object-model)
  - [Routing](#routing)
  - [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Setup Quick Start](#setup-quick-start)
  - [Environment Setup](#environment-setup)
  - [Running the Application](#running-the-application)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Agent System](#agent-system)
  - [RouterAgent](#routeragent)
  - [PortfolioAgent](#portfolioagent)
  - [FinanceMarketAgent](#financemarketagent)
  - [GoalsAgent](#goalsagent)
  - [QandAAgent](#qandaagent)
- [MCP Tools](#mcp-tools)
  - [Portfolio Tools](#portfolio-tools)
  - [Market Data Tools](#market-data-tools)
  - [Chart Generation Tools](#chart-generation-tools)
  - [Goals & Simulation Tools](#goals--simulation-tools)
- [API Documentation](#api-documentation)
  - [BaseAgent API](#baseagent-api)
  - [RouterAgent API](#routeragent-api)
  - [AgentResponse Schema](#agentresponse-schema)
- [User Interface](#user-interface)
  - [Streamlit UI Features](#streamlit-ui-features)
  - [Tab Management](#tab-management)
  - [Response Controls](#response-controls)
- [State Management](#state-management)
  - [AgentState Schema](#agentstate-schema)
  - [Portfolio State Flow](#portfolio-state-flow)
  - [Conversation History](#conversation-history)
- [Prompt Engineering](#prompt-engineering)
  - [Router Prompt](#router-prompt)
  - [Agent-Specific Prompts](#agent-specific-prompts)
  - [Best Practices](#best-practices)
- [Chart Generation](#chart-generation)
  - [Chart Types](#chart-types)
  - [Chart Storage](#chart-storage)
  - [Chart Caching](#chart-caching)
- [Testing](#testing)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [Agent Testing](#agent-testing)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Debugging](#debugging)
  - [Performance Optimization](#performance-optimization)
- [Development](#development)
  - [Adding New Agents](#adding-new-agents)
  - [Adding New Tools](#adding-new-tools)
  - [Extending Functionality](#extending-functionality)
- [Deployment](#deployment)
  - [Production Considerations](#production-considerations)
  - [Scaling](#scaling)
  - [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Finnie AI is a financial assistant that uses a multi-agent architecture to handle financial queries. The system routes user requests to specialized agents that can:

- **Build and manage investment portfolios** across 6 asset classes
- **Analyze market data** and generate visualizations
- **Run Basic Monte Carlo simulations** against portfolios for goal planning
- **Answer general financial questions**

The system is designed for **educational purposes** and provides financial insights while maintaining clear disclaimers about its limitations.

---

## Features

### Core Capabilities

- ✅ **Multi-Agent Architecture** - Intelligent routing to specialized agents
- ✅ **Portfolio Management** - Build, modify, and analyze portfolios
- ✅ **Market Data Integration** - Real-time and historical data via yFinance
- ✅ **Basic Monte Carlo Simulations** - Statistical goal planning and projections
- ✅ **Interactive Visualizations** - Charts for market trends and portfolio allocations
- ✅ **Conversation Memory** - Stateful conversations with context preservation for the session only
- ✅ **Tool Calling** - Extensible MCP-based tool system
- ✅ **Error Handling** - Graceful degradation and retry logic

### User Experience

- 🎨 **Clean Streamlit UI** - Tabbed interface for different agent outputs
- 📊 **Charts** - Auto-generated visualizations
- 💾 **Session Management** - Persistent conversation state
- 🗑️ **Response Management** - Delete, reorder, and organize agent responses
- 🖨️ **PDF Export** - Print-friendly output
- ⚡ **Real-time Updates** - Live badge counts and status indicators

---

## Architecture

### System Architecture
```mermaid
flowchart TB
    UI[Streamlit UI]
    Router[RouterAgent]
    
    PortfolioAgent[PortfolioAgent]
    MarketAgent[MarketAgent]
    GoalsAgent[GoalsAgent]
    QandAAgent[QandAAgent]
    
    QAMCP[Q&A MCP :8001]
    YFinanceMCP[yFinance MCP :8002]
    ChartsMCP[Charts MCP :8003]
    GoalsMCP[Goals MCP :8004]
    PortfolioMCP[Portfolio MCP :8005]
    
    FAISS1[(FAISS: financial_articles)]
    FAISS2[(FAISS: bogleheads)]
    
    YFinanceCache[TTL Cache - 30 min]
    ChartsCache[TTL Cache - 30 min]
    
    ImagesServer[Images Server :8010]
    ChartStorage[charts/ directory]
    
    UI -->|User Query| Router
    
    Router --> PortfolioAgent
    Router --> MarketAgent
    Router --> GoalsAgent
    Router --> QandAAgent
    
    PortfolioAgent --> PortfolioMCP
    PortfolioAgent --> YFinanceMCP
    PortfolioAgent --> ChartsMCP
    
    MarketAgent --> YFinanceMCP
    MarketAgent --> ChartsMCP
    
    GoalsAgent --> GoalsMCP
    GoalsAgent --> ChartsMCP
    
    QandAAgent --> QAMCP
    
    QAMCP --> FAISS1
    QAMCP --> FAISS2
    
    YFinanceMCP --> YFinanceCache
    ChartsMCP --> ChartsCache
    
    ChartsMCP -->|Write PNGs| ChartStorage
   ImagesServer <--> ChartStorage 
    UI <-->|HTTP GET| ImagesServer
    
    PortfolioAgent --> Router
    MarketAgent --> Router
    GoalsAgent --> Router
    QandAAgent --> Router
    
    Router -->|AgentResponse| UI
```



A note onthe chart generation.  The Chart tools are creating basic bitmap charts with matplotlib, and writing them to a folder.  There is the a FastAPI build images server which offer a get to retrieve the chart.  Currently there is no clean up being done (it would make sense to cleanup files older than 30 minutes as that's the chache life for the Chart tools).

The name of the generated file is passed back by the Charts tools to the calling agent.  This can then be used by the UI to construct IMG elements that reference the charts on the image server.  A future enhancement would be to pass back the data used to construct the charts and allow the UI to build more sophisticated charts (e.g., Chartly).

You could also eliminate the Image Server by configuring the Charts tools to write to the Streamlit static folder.  The current architecture demonstrates how various UIs could be used.

A basic im-memory cache layer is in front of the yFinance and Charts tools, with a TTL of 30 minutes.  API calls to yFinance are rate limited.

### Agent Object Model
```mermaid
classDiagram
    class AgentState {
        +List~BaseMessage~ messages
        +str next
        +str session_id
        +str last_agent_used
        +Dict~str,float~ current_portfolio
        +AgentResponse response
    }
    
    class AgentResponse {
        +str agent
        +str message
        +List~ChartArtifact~ charts
        +Dict~str,float~ portfolio
    }
    
    class ChartArtifact {
        +str title
        +str filename
    }
    
    class RouterAgent {
        +run_query(user_input, session_id) AgentResponse
    }
    
    class BaseAgent {
        <<abstract>>
        +run_query(history, session_id, portfolio) AgentResponse
    }
    
    class FinanceQandAAgent {
        +run_query(history, session_id, portfolio) AgentResponse
    }
    
    class FinanceMarketAgent {
        +run_query(history, session_id, portfolio) AgentResponse
    }
    
    class PortfolioAgent {
        +run_query(history, session_id, portfolio) AgentResponse
    }
    
    class GoalsAgent {
        +run_query(history, session_id, portfolio) AgentResponse
    }
    
    AgentState --> AgentResponse : contains
    AgentResponse --> ChartArtifact : contains
    
    RouterAgent --> AgentResponse : returns
    BaseAgent --> AgentResponse : returns
    
    BaseAgent <|-- FinanceQandAAgent : inherits
    BaseAgent <|-- FinanceMarketAgent : inherits
    BaseAgent <|-- PortfolioAgent : inherits
    BaseAgent <|-- GoalsAgent : inherits
    
    RouterAgent ..> FinanceQandAAgent : uses
    RouterAgent ..> FinanceMarketAgent : uses
    RouterAgent ..> PortfolioAgent : uses
    RouterAgent ..> GoalsAgent : uses
```

### Routing
LangGraph was used to establish the routing, with the StateGraph edges defines as

                     START
                       |
                       ▼
           --------RouterNode--------------------
           |           |              |         |
           ▼           ▼              ▼         ▼
         QandANode  MarketNode  PortfolioNode  GoalsNode
            |           |              |           |
            ▼           ▼              ▼           ▼
           END         END            END         END

So, for now, just a simple routing pattern.  

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User Input                                                    │
│    • User types question in Streamlit UI                        │
│    • Input added to session_state.chat_history                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Router Classification                                         │
│    • RouterAgent analyzes query                                 │
│    • Selects appropriate agent (Portfolio/Market/Goals/Q&A)     │
│    • Injects context (current_portfolio, last_agent, etc.)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Agent Processing                                              │
│    • Agent receives query + context                             │
│    • LLM determines required tools                              │
│    • Tools executed via MCP servers                             │
│    • Results collected                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. State Updates                                                 │
│    • Portfolio updates saved to AgentState                      │
│    • Charts extracted and stored                                │
│    • Last agent/ticker recorded                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Response Rendering                                            │
│    • Response added to chat_history                             │
│    • Agent-specific history updated (market/portfolio/goals)    │
│    • Charts displayed in UI                                     │
│    • Badge counts updated                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Framework
- **Python 3.11+** - Primary language
- **LangChain** - LLM orchestration and agent framework
- **LangGraph** - State machine and workflow management
- **LangSmith** - Observability and tracing
- **FastMCP** - for the tools
- **FastAPI** - for the image server

### LLM Integration
- **OpenAI GPT-4o-mini** - Default LLM (configurable)

I went with *gpt-4o-mini* over Gemini 2.0 Flash as I found the latter would often end up in prolonged ReAct cycles due to trying to interpret the strcutured data coming back from the tools.  (Some conversations I had with Claude and Gemini found they agreed with this assessment, and that gpt-4o-mini handled that better).  The trade-off is once in a while we get a skipped step by the agents.  It's a work in progress...

### Data & Tools
- **Model Context Protocol (MCP)** - Tool integration framework built with FastMCP
- **yFinance** - Market data provider
- **Matplotlib** - Chart generation
- **Pandas/NumPy** - Data processing

### User Interface
- **Streamlit** - Web interface
- **asyncio** - Async operations
- **nest_asyncio** - Event loop management

### State Management
- **InMemorySaver** - Conversation checkpointing
- **Session State** - UI state persistence

---

## Project Structure

```
finnie_ai/
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # BaseAgent class with tool loading
│   │   ├── router.py                # RouterAgent for query classification
│   │   ├── finance_portfolio.py       # Portfolio management agent
│   │   ├── finance_market.py          # Market data agent
│   │   ├── finance_goals.py         # Goals & simulation agent
│   │   ├── finance_q_and_a.py           # General Q&A agent
│   │   └── response.py              # AgentResponse and ChartArtifact classes
│   │ 
│   ├── indexer/
│   │   └── build_fais_index.py     # Utility for building FAISS indexes   
│   │
│   ├── mcp/
│   │   ├── portfolio_mcp.py         # Portfolio tools MCP server
│   │   ├── yfinance_mcp.py          # Market data MCP server
│   │   ├── charts_mcp.py            # Chart generation MCP server
│   │   └── goals_mcp.py             # Monte Carlo simulation MCP server
│   │
│   ├── servers/
│   │   └── image_server.py          # FastAPI HTTP server for charts images
│   │
│   ├── ui/
│   │   └── app.py                   # Streamlit application
│   │
│   └── utils/
│       ├── logging.py                # Logging configuration
│       ├── cache.py                  # TTL cache implementation
│       └── config.py                 # Configuration management
│
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_integration.py
│
├── charts/                           # Generated chart storage
│
├── .envrc                            # Environment variables (not checked in)
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project metadata
└── README.md                         # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.11+** installed
- **pip** package manager
- **Virtual environment** (recommended)
- **OpenAI API key** (or alternative LLM provider)

### Setup Quick Start
This is assuming you are runing on a Mac.  For the purpose of this project, it's a local "deployment".

Python 3.11+ is required.


1. **Clone the repository:**

```bash
git clone https://github.com/numberpete/finnie_ai.git
cd finnie_ai
```

2. **Set up environment variables:**

Create a `.envrc` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

3. **run start.sh sheel script**

The first run will take a few minutes as it build sup the FAISS indexes.  Subsequent starts shoudl be faster.

```bash
./start.sh
```

And that should be it. 

### Environment Setup Details
You only really need this if the quick start didn't work.  This is included for troubleshooting.

1. **Create and activate virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.envrc` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Chart Service
CHART_URL=http://localhost:8010/chart/
CHART_PATH="generated_charts"
```

4. **Build the FAIS indexes**
You may need to add a /src/data folder before running the tool to build the indexes.

```bash
# Build the index for the basic queries
python -m src.indexer.build_faiss_index -v

# Build the index for the advanced queries
python -m src.indexer.build_faiss_index -v -a articles_bogleheads_detailed.csv -i bogleheads -l bogleheads_fetch.log -f bogleheads_failures.csv -s bogleheads_success.csv
```


5. **Start MCP and Image servers** (in separate terminals):

```bash
# Terminal 1 - Q & A MCP Server
uvicorn src.mcp_servers.portfolio_mcp:app --port 8001

# Terminal 2 - yFinance MCP Server
uvicorn src.mcp_servers.yfinance_mcp:app --port 8002

# Terminal 3 - Charts MCP Server
uvicorn src.mcp_servers.charts_mcp:app --port 8003

# Terminal 4 - Goals MCP Server
uvicorn src.mcp_servers.goals_mcp:app --port 8004

# Terminal 5 - Portfolio MCP Server
uvicorn src.mcp_servers.goals_mcp:app --port 8005

# Terminal 6 - Images REST Server
uvicorn src.mcp_servers.goals_mcp:app --port 8010
```



### Running the Application

**Start the Streamlit UI:**

```bash
streamlit run src/ui/app.py
```

Access the application at: `http://localhost:8501`

---

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | - | Yes |
| `CHART_URL` | Chart image base URL | `http://localhost:8010/chart/` | No |
| `CHART_PATH` | Path to write images to | `generated_charts` | No |


## Agent System

### RouterAgent

**Purpose:** Intelligent query classification and routing to specialized agents.

**Capabilities:**
- Context-aware routing based on conversation history
- State management (portfolio, last ticker, pending clarifications)
- Response aggregation and formatting

**Routing Logic:**

```python
# Portfolio triggers
if query contains: "build", "add", "remove", "portfolio", "allocate"
    → Route to PortfolioAgent

# Market triggers
if query contains: "price", "stock", "market", "ticker", "chart"
    → Route to FinanceMarketAgent

# Goals triggers
if query contains: "simulate", "years", "goal", "future", "chances"
    → Route to GoalsAgent

# Fallback
else:
    → Route to QandAAgent
```

**Context Injection:**

```python
# Portfolio context
context = f"Current portfolio: {current_portfolio}"

# Market context
context = f"Last discussed ticker: {last_ticker}"

# Clarification context
context = f"Pending clarification ticker: {pending_ticker}"
```

### PortfolioAgent

**Purpose:** Build, modify, and analyze investment portfolios.

**Asset Classes:**
- Equities
- Fixed Income
- Real Estate
- Cash
- Commodities
- Crypto

**Operations:**

1. **Add Single Asset Class:**
```python
# User: "Add $100k to Equities"
add_to_portfolio_asset_class(
    asset_class_key="Equities",
    amount=100000,
    portfolio=current_portfolio
)
```

2. **Add Multiple Asset Classes (Atomic):**
```python
# User: "Add $100k to Equities and $50k to Cash"
add_to_portfolio(
    portfolio=current_portfolio,
    additions={"Equities": 100000, "Cash": 50000}
)
```

3. **Add by Fund/Stock:**
```python
# User: "$500k in Vanguard 2040"
ticker = get_ticker("Vanguard 2040")  # → "VFORX"
allocation = get_asset_classes("VFORX")  # → {"Equities": 0.6, "Fixed_Income": 0.35, ...}
add_to_portfolio_with_allocation(
    amount=500000,
    portfolio=current_portfolio,
    asset_allocation=allocation
)
```

4. **Portfolio Summary:**
```python
# User: "Show my portfolio"
summary = get_portfolio_summary(portfolio=current_portfolio)
risk = assess_risk_tolerance(portfolio=current_portfolio)
create_pie_chart(
    labels=["Equities", "Fixed_Income", ...],
    values=[50, 30, ...],  # Percentages
    title="Portfolio Allocation"
)
```

**Tool Selection Rules:**

| Scenario | Tool | Rationale |
|----------|------|-----------|
| Single asset class | `add_to_portfolio_asset_class` | Simplest operation |
| Multiple asset classes | `add_to_portfolio` | Atomic update (safer) |
| Fund/stock name | `add_to_portfolio_with_allocation` | Requires ticker resolution |

### FinanceMarketAgent

**Purpose:** Retrieve and visualize market data.

**Capabilities:**
- Current stock prices
- Historical price data
- Stock comparisons
- Asset class breakdowns
- Chart generation

**Operations:**

1. **Current Price:**
```python
# User: "What's the price of Apple?"
quote = get_ticker_quote(ticker="AAPL")
# Response: "Apple (AAPL) is trading at $185.43, up $2.15 (+1.17%)"
```

2. **Historical Data with Chart:**
```python
# User: "Show AAPL over the last year"
history = get_stock_history(ticker="AAPL", period="1y")
create_line_chart(
    x_values=[dates],
    y_values=[prices],
    title="AAPL Stock Price (Dec 2023 - Dec 2024)"
)
```

3. **Stock Comparison:**
```python
# User: "Compare SAP to Oracle"
sap_data = get_stock_history(ticker="SAP", period="3mo")
oracle_data = get_stock_history(ticker="ORCL", period="3mo")
create_multi_line_chart(
    series_data={"SAP": sap_data, "ORCL": oracle_data},
    title="SAP vs Oracle (Sep 2024 - Dec 2024)"
)
# ⚠️ DO NOT create individual charts for each stock
```

4. **Asset Breakdown:**
```python
# User: "What's the asset breakdown of VOO?"
breakdown = get_asset_classes(ticker="VOO")
# Response: "VOO consists of 100% Equities"
# ⚠️ NO chart for asset breakdowns (text only)
```

**Chart Generation Rules:**

| Query Type | Chart Type | Example |
|------------|------------|---------|
| Historical price (single) | Line chart | "Show AAPL over 6 months" |
| Comparison (multiple) | Multi-line chart | "Compare AAPL vs MSFT" |
| Year-over-year returns | Bar chart | "Show yearly returns for SPY" |
| Current price | No chart | "What's TSLA trading at?" |
| Asset breakdown | No chart | "Breakdown of VOO" |

### GoalsAgent

**Purpose:** Run Monte Carlo simulations for financial goal planning.

The monte carlo simulation runs against a static portfolio; ie, you can't have it assume you will be regularly investing in equities, etc.  It does utilize a correlation matrix so that when stocks go down, other assets like 
commodities or bonds go up, etc.  It's a basic model; ie, a "work-in-progress."

**Capabilities:**
- Portfolio growth projections
- Goal probability analysis
- Multi-scenario visualization

**Operations:**

1. **Simple Projection:**
```python
# User: "How will my portfolio do in 10 years?"
result = simple_monte_carlo_simulation(
    portfolio=current_portfolio,
    years=10
)
create_stacked_bar_chart(
    categories=["Bottom 10%", "Median", "Top 10%"],
    series_data={
        "Equities": [bottom_eq, median_eq, top_eq],
        "Fixed_Income": [...],
        ...
    },
    title="Portfolio Projection - 10 Year Scenarios"
)
```

2. **Goal-Based Simulation:**
```python
# User: "Can I double my portfolio in 10 years?"
current_total = sum(current_portfolio.values())  # $2.7M
target_goal = current_total * 2  # $5.4M

result = simple_monte_carlo_simulation(
    portfolio=current_portfolio,
    years=10,
    target_goal=5400000
)
# Response: "You have a 36.9% chance of doubling to $5.4M in 10 years"
```

**Simulation Output:**

```python
{
    "goal_analysis": {
        "target": 5400000,
        "success_probability": "36.90%"
    },
    "median_scenario": {
        "total": 4601999.67,
        "portfolio": {"Equities": 4601999.67, ...}
    },
    "bottom_10_percent_scenario": {
        "total": 2935376.78,
        "portfolio": {"Equities": 2935376.78, ...}
    },
    "top_10_percent_scenario": {
        "total": 7738156.07,
        "portfolio": {"Equities": 7738156.07, ...}
    }
}
```

**Chart Data Transformation:**

```python
# Extract portfolio values for each scenario
series_data = {
    "Equities": [
        bottom_10_scenario["portfolio"]["Equities"],    # 2935376.78
        median_scenario["portfolio"]["Equities"],        # 4601999.67
        top_10_scenario["portfolio"]["Equities"]         # 7738156.07
    ],
    "Fixed_Income": [...],
    # ... other asset classes
}
```

### QandAAgent

**Purpose:** Answer general financial questions not requiring tools.

**Capabilities:**
- Financial definitions
- Concept explanations
- General advice (educational only)

**Example Queries:**
- "What is dollar-cost averaging?"
- "Explain the difference between stocks and bonds"
- "What factors affect stock prices?"

---

## MCP Tools

### Portfolio Tools

**Server:** `portfolio_mcp.py` (Port 8005)

#### `get_new_portfolio()`
Creates a new empty portfolio.

**Returns:**
```python
{
    "Equities": 0.0,
    "Fixed_Income": 0.0,
    "Real_Estate": 0.0,
    "Cash": 0.0,
    "Commodities": 0.0,
    "Crypto": 0.0
}
```

#### `add_to_portfolio_asset_class(asset_class_key, amount, portfolio)`
Add amount to a single asset class.

**Parameters:**
- `asset_class_key` (str): Asset class name
- `amount` (float): Amount to add (negative to remove)
- `portfolio` (dict): Current portfolio

**Returns:** Updated portfolio dict

#### `add_to_portfolio(portfolio, additions)`
Add to multiple asset classes atomically.

**Parameters:**
- `portfolio` (dict): Current portfolio
- `additions` (dict): Asset classes and amounts to add

**Returns:** Updated portfolio dict

**Example:**
```python
add_to_portfolio(
    portfolio={"Equities": 100000, ...},
    additions={"Equities": 50000, "Cash": 25000}
)
# Returns: {"Equities": 150000, "Cash": 25000, ...}
```

#### `add_to_portfolio_with_allocation(amount, portfolio, asset_allocation)`
Add amount distributed across asset classes per allocation.

**Parameters:**
- `amount` (float): Total amount to add
- `portfolio` (dict): Current portfolio
- `asset_allocation` (dict): Asset class ratios (0-1)

**Returns:** Updated portfolio dict

**Example:**
```python
add_to_portfolio_with_allocation(
    amount=500000,
    portfolio={...},
    asset_allocation={"Equities": 0.6, "Fixed_Income": 0.4}
)
# Adds $300k to Equities, $200k to Fixed_Income
```

#### `get_portfolio_summary(portfolio)`
Calculate portfolio statistics.

**Returns:**
```python
{
    "total_value": 1000000.0,
    "asset_values": {"Equities": 500000, ...},
    "asset_percentages": {"Equities": 50.0, ...},
    "asset_count": 3
}
```

#### `assess_risk_tolerance(portfolio)`
Calculate portfolio risk metrics.

**Returns:**
```python
{
    "risk_tier": "Moderate",
    "volatility_score": 14.2,
    "explanation": "Balanced portfolio with moderate risk..."
}
```

### Market Data Tools

**Server:** `yfinance_mcp.py` (Port 8002)

#### `get_ticker(company_name)`
Resolve company name to ticker symbol.

**Parameters:**
- `company_name` (str): Company or fund name

**Returns:** Ticker symbol (str)

**Example:**
```python
get_ticker("Apple") → "AAPL"
get_ticker("Vanguard 2040") → "VFORX"
```

#### `get_ticker_quote(ticker)`
Get current quote data.

**Returns:**
```python
{
    "symbol": "AAPL",
    "shortName": "Apple Inc.",
    "regularMarketPrice": 185.43,
    "regularMarketChange": 2.15,
    "regularMarketChangePercent": 1.17,
    "regularMarketDayHigh": 186.50,
    "regularMarketDayLow": 184.20,
    "volume": 52428800,
    ...
}
```

#### `get_stock_history(ticker, period)`
Get historical price data.

**Parameters:**
- `ticker` (str): Stock ticker
- `period` (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

**Returns:**
```python
{
    "dates": ["2024-01-01", "2024-01-02", ...],
    "prices": [150.25, 151.30, ...],
    "metadata": {
        "ticker": "AAPL",
        "period": "1y",
        "data_points": 252
    }
}
```

#### `get_asset_classes(ticker)`
Get asset class breakdown for a fund.

**Returns:**
```python
{
    "Equities": 0.6,
    "Fixed_Income": 0.35,
    "Cash": 0.05,
    "Real_Estate": 0.0,
    "Commodities": 0.0,
    "Crypto": 0.0
}
```

### Chart Generation Tools

**Server:** `charts_mcp.py` (Port 8003)

#### `create_line_chart(x_values, y_values, title, xlabel, ylabel)`
Create a single-line chart.

**Parameters:**
- `x_values` (list): X-axis values (dates/categories)
- `y_values` (list): Y-axis values (prices/amounts)
- `title` (str): Chart title
- `xlabel` (str): X-axis label
- `ylabel` (str): Y-axis label

**Returns:**
```python
{
    "chart_id": "abc123",
    "filename": "abc123.png",
    "title": "AAPL Stock Price (Jan 2024 - Dec 2024)"
}
```

#### `create_multi_line_chart(series_data, title, xlabel, ylabel)`
Create a multi-line comparison chart.

**Parameters:**
- `series_data` (dict): Mapping of series names to data
- `title` (str): Chart title
- `xlabel` (str): X-axis label
- `ylabel` (str): Y-axis label

**Example:**
```python
create_multi_line_chart(
    series_data={
        "AAPL": {"dates": [...], "prices": [...]},
        "MSFT": {"dates": [...], "prices": [...]}
    },
    title="AAPL vs MSFT"
)
```

#### `create_pie_chart(labels, values, title, colors)`
Create a pie chart.

**Parameters:**
- `labels` (list): Slice labels
- `values` (list): Slice percentages
- `title` (str): Chart title
- `colors` (list): Hex color codes

**Example:**
```python
create_pie_chart(
    labels=["Equities", "Fixed_Income", "Cash"],
    values=[50, 30, 20],
    title="Portfolio Allocation",
    colors=["#2E5BFF", "#46CDCF", "#3DDC84"]
)
```

#### `create_stacked_bar_chart(categories, series_data, title, xlabel, ylabel, colors)`
Create a stacked bar chart.

**Parameters:**
- `categories` (list): X-axis categories
- `series_data` (dict): Asset class → values mapping
- `title` (str): Chart title
- `xlabel` (str): X-axis label
- `ylabel` (str): Y-axis label
- `colors` (list): Hex color codes for each series

**Example:**
```python
create_stacked_bar_chart(
    categories=["Bottom 10%", "Median", "Top 10%"],
    series_data={
        "Equities": [2935376, 4601999, 7738156],
        "Fixed_Income": [0, 0, 0],
        ...
    },
    title="Portfolio Projection - 10 Year Scenarios"
)
```

### Goals & Simulation Tools

**Server:** `goals_mcp.py` (Port 8004)

#### `simple_monte_carlo_simulation(portfolio, years, target_goal)`
Run Monte Carlo simulation on portfolio.

**Parameters:**
- `portfolio` (dict): Current portfolio allocation
- `years` (int): Number of years to project
- `target_goal` (float, optional): Target dollar amount

**Returns:**
```python
{
    "goal_analysis": {
        "target": 5400000,
        "success_probability": "36.90%"
    },
    "median_scenario": {
        "total": 4601999.67,
        "portfolio": {...}
    },
    "bottom_10_percent_scenario": {
        "total": 2935376.78,
        "portfolio": {...}
    },
    "top_10_percent_scenario": {
        "total": 7738156.07,
        "portfolio": {...}
    }
}
```

**Assumptions:**
- Equities: 10% annual return, 15% volatility
- Fixed Income: 4% return, 5% volatility
- Real Estate: 6% return, 10% volatility
- Cash: 2% return, 1% volatility
- Commodities: 5% return, 20% volatility
- Crypto: 15% return, 50% volatility

---

## API Documentation

### BaseAgent API

**Class:** `BaseAgent`

**Purpose:** Base class for all agents with tool loading and execution.

**Constructor:**
```python
BaseAgent(
    agent_name: str,
    llm: Any,
    system_prompt: str,
    logger: Logger,
    mcp_servers: Optional[Dict[str, Dict]] = None,
    debug: bool = False
)
```

**Parameters:**
- `agent_name`: Agent identifier (e.g., "PortfolioAgent")
- `llm`: Language model instance
- `system_prompt`: System instructions
- `logger`: Logger instance
- `mcp_servers`: MCP server configurations
- `debug`: Enable debug logging

**Methods:**

#### `async run_query(history, session_id, portfolio=None) -> AgentResponse`

Execute agent query with conversation history.

**Parameters:**
- `history` (List[BaseMessage]): Conversation messages
- `session_id` (str): Session identifier
- `portfolio` (Dict, optional): Current portfolio state

**Returns:** `AgentResponse` object

**Example:**
```python
response = await agent.run_query(
    history=[HumanMessage(content="Build me a portfolio")],
    session_id="session_123"
)
```

#### `async cleanup()`

Clean up resources (close MCP connections).

**Example:**
```python
await agent.cleanup()
```

### RouterAgent API

**Class:** `RouterAgent`

**Purpose:** Route queries to appropriate specialized agents.

**Constructor:**
```python
RouterAgent(checkpointer: InMemorySaver)
```

**Methods:**

#### `async run_query(user_input, session_id) -> AgentResponse`

Route and execute query.

**Parameters:**
- `user_input` (str): User's question
- `session_id` (str): Session identifier

**Returns:** `AgentResponse` from selected agent

**Example:**
```python
router = RouterAgent(checkpointer=InMemorySaver())
response = await router.run_query(
    user_input="What's the price of Apple?",
    session_id="session_123"
)
```

#### `async cleanup()`

Clean up all agents.

### AgentResponse Schema

**Class:** `AgentResponse`

**Fields:**
```python
@dataclass
class AgentResponse:
    agent: str                      # Agent name
    message: str                    # Response text
    charts: List[ChartArtifact]     # Generated charts
    portfolio: Optional[Dict]       # Updated portfolio (if applicable)
```

**Class:** `ChartArtifact`

**Fields:**
```python
@dataclass
class ChartArtifact:
    title: str        # Chart title
    filename: str     # Chart filename (e.g., "abc123.png")
```

---

## User Interface

### Streamlit UI Features

**URL:** `http://localhost:8501`

**Main Components:**

1. **Title Bar**
   - Application title
   - Clear Session button

2. **Tab Navigation**
   - 💬 Chat - Full conversation history
   - 📈 Market (n) - Market analysis history
   - 💼 Portfolio (n) - Portfolio management history
   - 🎯 Goals (n) - Goals & simulation history

3. **Chat Input**
   - Always-visible input bar
   - Placeholder: "Ask a financial question..."

4. **Disclaimer**
   - "FinnieAI can make mistakes, and answers are for educational purposes only."

### Tab Management

**Chat Tab:**
- Shows all user messages and agent responses
- No controls (read-only)
- Chronological conversation flow

**Agent Tabs (Market/Portfolio/Goals):**
- Show only responses from that specific agent
- Badge count shows number of responses
- Editable with action controls

**Badge Behavior:**
```python
# Badge format
f"📈 Market ({count})" if count > 0 else "📈 Market"

# Counts
chat_count = len([m for m in chat_history if m["role"] == "assistant"])
market_count = len(market_history)
portfolio_count = len(portfolio_history)
goals_count = len(goals_history)
```

### Response Controls

**Available in:** Market, Portfolio, Goals tabs only

**Actions (in expandable "⚙️ Actions" menu):**

1. **🗑️ Delete**
   - Removes response from history
   - Immediate rerender

2. **⬆️ Move Up**
   - Swaps with previous response
   - Disabled if first item

3. **⬇️ Move Down**
   - Swaps with next response
   - Disabled if last item

**Implementation:**
```python
with st.expander("⚙️ Actions", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Delete", ...):
            st.session_state[history_key].pop(index)
            st.rerun()
```

**Print-Friendly:**
```css
@media print {
    .stExpander {
        display: none !important;
    }
}
```

---

## State Management

### AgentState Schema

**TypedDict Definition:**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]           # Conversation history
    current_portfolio: Dict[str, float]   # Portfolio state
    last_agent_used: Optional[str]        # Last agent name
    last_ticker: Optional[str]            # Last discussed ticker
    pending_ticker: Optional[str]         # Ticker awaiting clarification
    response: Optional[AgentResponse]     # Last agent response
```

### Portfolio State Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Initial State (Empty Portfolio)                              │
│    current_portfolio = {                                        │
│        "Equities": 0, "Fixed_Income": 0, ...                   │
│    }                                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. User Adds to Portfolio                                       │
│    "Add $100k to Equities"                                      │
│    PortfolioAgent calls add_to_portfolio_asset_class            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Tool Returns Updated Portfolio                               │
│    {"Equities": 100000, "Fixed_Income": 0, ...}                │
│    BaseAgent extracts portfolio from ToolMessage                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. AgentResponse Includes Portfolio                             │
│    AgentResponse(                                               │
│        portfolio={"Equities": 100000, ...}                      │
│    )                                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Router Updates AgentState                                    │
│    state["current_portfolio"] = response.portfolio              │
│    Portfolio persists across conversation                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Portfolio is **read-only** for agents (they don't modify state directly)
- Tools return updated portfolios
- Router extracts portfolio from response and updates state
- Portfolio is injected into agents via context messages

### Conversation History

**Session State Structure:**
```python
st.session_state = {
    "session_id": "uuid_string",
    "chat_history": [
        {"role": "user", "content": "message"},
        {"role": "assistant", "content": AgentResponse}
    ],
    "market_history": [AgentResponse, AgentResponse, ...],
    "portfolio_history": [AgentResponse, AgentResponse, ...],
    "goals_history": [AgentResponse, AgentResponse, ...],
    "checkpointer": InMemorySaver()
}
```

**History Flow:**
```
User Input
    │
    ▼
Added to chat_history (role: "user")
    │
    ▼
Agent processes and returns AgentResponse
    │
    ▼
Added to chat_history (role: "assistant")
    │
    └──> If FinanceMarketAgent → added to market_history
    └──> If PortfolioAgent → added to portfolio_history
    └──> If GoalsAgent → added to goals_history
```

---

## Prompt Engineering

### Router Prompt

**Location:** `src/agents/prompts/router_prompt.py`

**Structure:**
```
# ROLE
You are a financial assistant router

# CONTEXT
- current_portfolio: {...}
- last_agent_used: "PortfolioAgent"
- last_ticker: "AAPL"

# ROUTING RULES
- Portfolio building/modification → PortfolioAgent
- Market data/prices/charts → FinanceMarketAgent
- Future projections/simulations → GoalsAgent
- General questions → QandAAgent

# CONTEXT-AWARE ROUTING
- "How will it do?" after portfolio discussion → GoalsAgent
- "Chart that" after market discussion → FinanceMarketAgent
- Pronouns ("it", "that") → use last_agent_used

# DISAMBIGUATION
- Portfolio building vs market lookup
- Current analysis vs future simulation
```

### Agent-Specific Prompts

**Common Structure:**
```
# CRITICAL WARNINGS
[Mandatory requirements, e.g., "ALWAYS PASS PORTFOLIO PARAMETER"]

# ROLE
[Agent's purpose and capabilities]

# CORE RULES
[Numbered list of fundamental rules]

# FINDING [RESOURCE]
[How to locate necessary data (portfolio, ticker, etc.)]

# EXECUTION PIPELINE
[Step-by-step workflow]

# EXAMPLES
[Concrete examples with inputs/outputs]

# AGENT SCRATCHPAD
{agent_scratchpad}
```

**Example: PortfolioAgent**
```
# ⚠️ CRITICAL: ALWAYS PASS PORTFOLIO PARAMETER
[Warning about common errors]

# ROLE
Build and analyze investment portfolios

# CORE RULES
1. Each tool called ONCE
2. Use actual portfolio dictionary
3. Never reset without explicit request
4. NEVER do math yourself
5. NEVER use add_to_portfolio_with_allocation for direct asset classes

# TOOL SELECTION RULES
[Decision tree for which tool to use]

# PORTFOLIO SUMMARY - MANDATORY 3-STEP PROCESS
[Step 1: get_portfolio_summary]
[Step 2: assess_risk_tolerance]
[Step 3: create_pie_chart]

# EXAMPLES
[5-10 concrete examples]
```

### Best Practices

**1. Explicit Parameter Requirements:**
```
❌ NEVER call: simple_monte_carlo_simulation(years=10)
✅ ALWAYS call: simple_monte_carlo_simulation(portfolio={...}, years=10)
```

**2. Concrete Examples:**
```
User: "Add $100k to Equities"
Tool: add_to_portfolio_asset_class(asset_class_key="Equities", amount=100000, portfolio={...})
Response: "Added $100k to Equities."
```

**3. Error Prevention:**
```
CRITICAL CHECKLIST - Verify before responding:
□ Called simple_monte_carlo_simulation - received scenarios
□ Called create_stacked_bar_chart - received filename
□ Both Observations are present in {agent_scratchpad}
```

**4. Loop Prevention:**
```
Before calling ANY tool, check {agent_scratchpad}:
- Look for: "Action: simple_monte_carlo_simulation" followed by "Observation:"
- If found: Simulation DONE. Extract results and create chart.
- If NOT found: Call it now WITH portfolio parameter.
```

**5. Graceful Degradation:**
```
If create_stacked_bar_chart fails twice:
- Provide text-only response with all scenario values
- Add: "Note: Unable to generate visualization at this time."
- DO NOT mention charts
```

---

## Chart Generation

### Chart Types

**1. Line Chart** - Single time series
```python
create_line_chart(
    x_values=["2024-01-01", "2024-01-02", ...],
    y_values=[150.25, 151.30, ...],
    title="AAPL Stock Price (Jan 2024 - Dec 2024)",
    xlabel="Date",
    ylabel="Price ($)"
)
```

**2. Multi-Line Chart** - Multiple time series comparison
```python
create_multi_line_chart(
    series_data={
        "AAPL": {"dates": [...], "prices": [...]},
        "MSFT": {"dates": [...], "prices": [...]}
    },
    title="AAPL vs MSFT Stock Prices",
    xlabel="Date",
    ylabel="Price ($)"
)
```

**3. Pie Chart** - Portfolio allocation
```python
create_pie_chart(
    labels=["Equities", "Fixed_Income", "Cash"],
    values=[50, 30, 20],  # Percentages
    title="Portfolio Allocation",
    colors=["#2E5BFF", "#46CDCF", "#3DDC84"]
)
```

**4. Stacked Bar Chart** - Scenario comparison
```python
create_stacked_bar_chart(
    categories=["Bottom 10%", "Median", "Top 10%"],
    series_data={
        "Equities": [2935376, 4601999, 7738156],
        "Fixed_Income": [0, 0, 0],
        ...
    },
    title="Portfolio Projection - 10 Year Scenarios",
    xlabel="Scenario",
    ylabel="Portfolio Value ($)",
    colors=["#2E5BFF", "#46CDCF", "#F08A5D", "#3DDC84", "#FFD700", "#B832FF"]
)
```

### Chart Storage

**Directory:** `charts/`

**Naming Convention:** `{chart_id}.png`

**Chart ID Generation:**
```python
import hashlib
import json

def generate_chart_id(chart_type, params):
    data = json.dumps({"type": chart_type, **params}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:12]
```

**File Serving:**
```python
# FastAPI endpoint
@app.get("/chart/{filename}")
async def serve_chart(filename: str):
    chart_path = os.path.join(CHART_STORAGE_DIR, filename)
    return FileResponse(chart_path)
```

**URL Format:** `http://localhost:8010/chart/{chart_id}.png`

### Chart Caching

**Implementation:** TTL-based cache

```python
class TTLCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        timestamp = self._timestamps[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any, ttl_seconds: int = 1800):
        self._cache[key] = value
        self._timestamps[key] = time.time()
```

**Cache TTL:** 30 minutes (1800 seconds)

**Benefits:**
- Reduces redundant chart generation
- Faster response times for repeated queries
- Lower computational overhead

---

## Testing
This is definitely a work-in-progress.  

### Unit Tests

**Location:** `tests/test_agents.py`

**Example:**
```python
import pytest
from src.agents.portfolio_agent import PortfolioAgent

@pytest.mark.asyncio
async def test_portfolio_agent_add_single_asset():
    agent = PortfolioAgent(llm=mock_llm, logger=mock_logger)
    
    response = await agent.run_query(
        history=[HumanMessage(content="Add $100k to Equities")],
        session_id="test_session"
    )
    
    assert response.portfolio["Equities"] == 100000
    assert "Added $100k to Equities" in response.message
```

### Integration Tests

**Location:** `tests/test_integration.py`

**Example:**
```python
@pytest.mark.asyncio
async def test_full_portfolio_workflow():
    router = RouterAgent(checkpointer=InMemorySaver())
    
    # Build portfolio
    response1 = await router.run_query(
        "Build me a portfolio with $100k in Equities",
        "test_session"
    )
    assert response1.portfolio["Equities"] == 100000
    
    # Add to portfolio
    response2 = await router.run_query(
        "Add $50k to Cash",
        "test_session"
    )
    assert response2.portfolio["Cash"] == 50000
    
    # Summarize
    response3 = await router.run_query(
        "Summarize my portfolio",
        "test_session"
    )
    assert len(response3.charts) > 0  # Pie chart
    assert "$150,000" in response3.message  # Total
```

### Agent Testing

**Mock LLM:**
```python
from unittest.mock import Mock

def mock_llm_response(tool_calls=None, content="Mock response"):
    mock_msg = Mock()
    mock_msg.tool_calls = tool_calls or []
    mock_msg.content = content
    return {"messages": [mock_msg]}

llm = Mock()
llm.ainvoke = AsyncMock(return_value=mock_llm_response())
```

**Test Tool Calling:**
```python
@pytest.mark.asyncio
async def test_agent_calls_correct_tool():
    agent = PortfolioAgent(llm=mock_llm, logger=mock_logger)
    
    # Mock tool call
    mock_tool_call = {
        "name": "add_to_portfolio_asset_class",
        "args": {
            "asset_class_key": "Equities",
            "amount": 100000,
            "portfolio": {...}
        }
    }
    
    response = await agent.run_query(...)
    
    # Verify correct tool was called
    assert mock_tool_call in agent.tool_calls
```

---

## Troubleshooting

### Common Issues

#### 1. "Missing required argument: portfolio"

**Cause:** Agent not passing portfolio parameter to tool

**Solution:**
- Check agent prompt includes portfolio parameter examples
- Verify context message includes current portfolio
- Review BaseAgent logs for portfolio extraction

**Debug:**
```python
# In base_agent.py
self.LOGGER.info(f"💼 Current portfolio: {current_portfolio}")
self.LOGGER.info(f"📝 Context message: {context_msg.content}")
```

#### 2. Charts not appearing in UI

**Cause:** Chart extraction failing in BaseAgent

**Solution:**
- Check chart server is running (port 8010)
- Verify CHART_URL in .env
- Check BaseAgent chart extraction logs

**Debug:**
```python
# Check if charts were captured
self.LOGGER.info(f"📊 Charts captured: {len(generated_charts)}")
for chart in generated_charts:
    self.LOGGER.info(f"   - {chart.title}: {chart.filename}")
```

#### 3. Agent stuck in loop (max iterations)

**Cause:** Agent keeps calling tools without completing

**Solution:**
- Check for duplicate tool calls in logs
- Verify agent prompt has loop prevention rules
- Review scratchpad checking logic

**Debug:**
```python
# In base_agent.py
self.LOGGER.debug(f"🔄 ITERATION {iteration}")
self.LOGGER.debug(f"📋 Last message type: {last_msg_type}")
self.LOGGER.debug(f"   - has_tool_calls: {has_tool_calls}")
```

#### 4. Router routing to wrong agent

**Cause:** Ambiguous query or context not being used

**Solution:**
- Add more specific routing triggers to router prompt
- Check context injection (last_agent_used, last_ticker)
- Review router decision logs

**Debug:**
```python
# In router.py
self.LOGGER.info(f"📝 Last agent used: {last_agent_used}")
self.LOGGER.info(f"🎯 Selected agent: {selected_agent}")
```

#### 5. MCP server connection failed

**Cause:** Server not running or wrong port

**Solution:**
```bash
# Check if servers are running
lsof -i :8001  # Portfolio
lsof -i :8002  # yFinance
lsof -i :8010  # Charts
lsof -i :8003  # Goals

# Start missing servers
uvicorn src.mcp_servers.portfolio_mcp:app --port 8001
```

#### 6. Simulation returns strange probabilities

**Cause:** Invalid portfolio or extreme volatility assumptions

**Solution:**
- Verify portfolio has at least one asset > 0
- Check simulation parameters (years, target_goal)
- Review asset class assumptions in goals_mcp.py

**Debug:**
```python
# In goals_mcp.py
LOGGER.info(f"Portfolio: {portfolio}")
LOGGER.info(f"Total value: {sum(portfolio.values())}")
LOGGER.info(f"Years: {years}, Target: {target_goal}")
```

### Debugging

**Enable Debug Mode:**
```python
# In agent initialization
agent = BaseAgent(
    agent_name="PortfolioAgent",
    llm=llm,
    system_prompt=PORTFOLIO_AGENT_PROMPT,
    logger=logger,
    mcp_servers=mcp_servers,
    debug=True  # ✅ Enable debug output
)
```

**View Agent Logs:**
```python
# Logs show:
# - Each iteration
# - Tool calls
# - Tool results
# - Decision points

[portfolio-agent] 🔄 ITERATION 1
[portfolio-agent]   [0] AIMessage
[portfolio-agent]       🔧 Contains 1 tool call(s):
[portfolio-agent]          - add_to_portfolio_asset_class
[portfolio-agent]   [1] ToolMessage
[portfolio-agent]       ✅ Tool result for: add_to_portfolio_asset_class
[portfolio-agent]       💼 Portfolio updated: Total $100,000
```

**LangSmith Tracing:**
```bash
# Enable in .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=finnie-ai
```

Access traces at: `https://smith.langchain.com`

### Performance Optimization

#### 1. Reduce Tool Calls

**Problem:** Agent making redundant tool calls

**Solution:**
- Update prompt to check scratchpad before calling tools
- Implement loop prevention rules
- Use atomic operations (e.g., add_to_portfolio vs multiple add_to_portfolio_asset_class)

#### 2. Cache Chart Generation

**Already Implemented:** TTL cache (30 min)

**Monitor Cache Hit Rate:**
```python
# In charts_mcp.py
cache_hits = 0
cache_misses = 0

if use_cache:
    cached_data = charts_cache.get(chart_id)
    if cached_data:
        cache_hits += 1
        return cached_data
    cache_misses += 1

LOGGER.info(f"Cache hit rate: {cache_hits / (cache_hits + cache_misses):.2%}")
```

#### 3. Optimize LLM Token Usage

**Strategies:**
- Use concise prompts (avoid repetition)
- Limit conversation history length
- Use cheaper models for simple queries (GPT-4o-mini vs GPT-4)

**Monitor Token Usage:**
```python
# With LangSmith callback
from langsmith import LangSmithCallback

callback = LangSmithCallback()
response = await llm.ainvoke(..., callbacks=[callback])

# View token counts in LangSmith dashboard
```

#### 4. Parallel MCP Server Calls

**Current:** Sequential tool execution

**Future Optimization:**
```python
# Execute multiple independent tools in parallel
import asyncio

async def parallel_tool_execution(tools):
    tasks = [execute_tool(tool) for tool in tools]
    results = await asyncio.gather(*tasks)
    return results
```

---

## Development

### Adding New Agents

**1. Create Agent File:**

```python
# src/agents/new_agent.py
from src.agents.base_agent import BaseAgent
from src.agents.response import AgentResponse

class NewAgent(BaseAgent):
    def __init__(self, llm, logger, mcp_servers=None, debug=False):
        super().__init__(
            agent_name="NewAgent",
            llm=llm,
            system_prompt=NEW_AGENT_PROMPT,
            logger=logger,
            mcp_servers=mcp_servers,
            debug=debug
        )
```

**2. Create Prompt File:**

```python
# src/agents/prompts/new_agent_prompt.py
NEW_AGENT_PROMPT = """
# ROLE
You are a [specific purpose] agent.

# CORE RULES
1. [Rule 1]
2. [Rule 2]

# EXECUTION PIPELINE
[Steps]

# EXAMPLES
[Examples]

{agent_scratchpad}
"""
```

**3. Add to Router:**

```python
# In router.py
from src.agents.new_agent import NewAgent

class RouterAgent:
    def __init__(self, checkpointer):
        # ... existing agents ...
        self.new_agent = NewAgent(llm, LOGGER, mcp_servers)
    
    async def route_node(self, state: AgentState):
        # Add routing logic
        if "new_trigger" in user_message:
            return await self.new_node(state)
```

**4. Add Routing Node:**

```python
async def new_node(self, state: AgentState) -> AgentState:
    """New agent node"""
    response = await self.new_agent.run_query(
        history=state["messages"],
        session_id=state.get("session_id", "default")
    )
    state["response"] = response
    state["last_agent_used"] = "NewAgent"
    return state
```

### Adding New Tools

**1. Create Tool in MCP Server:**

```python
# In appropriate MCP server file
@mcp.tool()
def new_tool(param1: str, param2: int) -> Dict:
    """
    Tool description.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Result dictionary
    """
    result = do_something(param1, param2)
    return {"success": True, "result": result}
```

**2. Update Agent Prompt:**

```python
# In agent prompt
# NEW TOOL USAGE
new_tool(param1="value", param2=123)

# Example:
User: "Do something new"
Tool: new_tool(param1="value", param2=123)
Result: {"success": True, "result": ...}
```

**3. Test Tool:**

```python
# tests/test_tools.py
def test_new_tool():
    result = new_tool(param1="test", param2=42)
    assert result["success"] == True
    assert "result" in result
```

### Extending Functionality
What are some ideal extensions for this.

#### Multiple portfolio support

#### Save user state beyond the session

#### Have AgentResponse include chart data to allow UI to construct dynamic charting (e.g., Chartly)

#### Add additional calculator to the Goals Tools, such as
- Interest Calculator
- Amortization Calculator
- etc.

#### More Dynamic StageGraph.  
Have edges between the specialty Agents to allow the LLM to construct more elaborate responses to questions.



---

## Deployment

### Production Considerations
It's not there yet.  Below outlines areas where we need to advance the code, further.


**1. Environment Variables:**
```bash
# Use production API keys
OPENAI_API_KEY=prod_key_here

# Disable debug logging, NYI
DEBUG=false

# Use production MCP URLs, NYI
PORTFOLIO_MCP_URL=https://api.yoursite.com/portfolio/sse
YFINANCE_MCP_URL=https://api.yoursite.com/market/sse
# ... etc
```

**2. Error Handling:**
- All tool calls wrapped in try/except
- Graceful degradation on failures
- User-friendly error messages

**3. Rate Limiting:**
```python
# Implement rate limiting for API calls, this is implemented for calls to yFinance, already
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute
async def call_llm(prompt):
    return await llm.ainvoke(prompt)
```

**4. Monitoring:**
- Enable LangSmith tracing
- Log all errors to centralized logging
- Monitor API usage and costs

**5. Security:**
- Don't commit .envrc files
- Use secrets management (AWS Secrets Manager, etc.) NYI
- Sanitize user inputs
- Implement authentication for UI NYI

### Scaling

**Horizontal Scaling:**

```
┌──────────────┐
│  Load        │
│  Balancer    │
└──────┬───────┘
       │
   ────┼────────────────
   │   │              │
   ▼   ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│Streamlit│  │Streamlit│  │Streamlit│
│Instance1│  │Instance2│  │Instance3│
└────┬────┘  └────┬────┘  └────┬────┘
     └───────────┬┴────────────┘
                 │
                 ▼
      ┌──────────────────┐
      │  Shared MCP      │
      │  Servers         │
      └──────────────────┘
```

**Database for State:**

Replace `InMemorySaver` with persistent storage (NYI):

```python
from langchain.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(
    connection_string="postgresql://user:pass@host:5432/db"
)
```

**Caching Layer:**

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379)

# Cache portfolio state
redis_client.set(f"portfolio:{session_id}", json.dumps(portfolio))

# Retrieve
portfolio = json.loads(redis_client.get(f"portfolio:{session_id}"))
```

### Monitoring NYI

**1. LangSmith Dashboard:**
- Track all LLM calls
- Monitor token usage
- View traces and errors
- Analyze latency

**2. Application Metrics:**
```python
from prometheus_client import Counter, Histogram

# Request counter
request_counter = Counter('finnie_requests_total', 'Total requests')

# Response time histogram
response_time = Histogram('finnie_response_seconds', 'Response time')

@response_time.time()
async def run_query(...):
    request_counter.inc()
    # ... agent logic
```

**3. Error Tracking:**
```python
import sentry_sdk

sentry_sdk.init(
    dsn="your_sentry_dsn",
    traces_sample_rate=1.0
)

try:
    result = await agent.run_query(...)
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise
```

**4. Cost Tracking:**
```python
# Track LLM costs
from langsmith import Client

client = Client()
runs = client.list_runs(project_name="finnie-ai")

total_cost = sum(run.total_cost for run in runs)
print(f"Total LLM cost: ${total_cost:.2f}")
```

---

## Contributing

**This is a capstone project for Interview Kickstart Agentic AI program.  You are welcome to fork, but I don't anticipate I'll be maintaining this repo very rigorously.**

### Guidelines

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Write tests** for new functionality
4. **Follow code style:**
   - PEP 8 for Python
   - Type hints for all functions
   - Docstrings for all classes/methods
5. **Update documentation** (this README)
6. **Commit with clear messages:** `git commit -m 'Add amazing feature'`
7. **Push to branch:** `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Style

```python
# Good example
async def run_query(
    self,
    history: List[BaseMessage],
    session_id: str,
    portfolio: Optional[Dict[str, float]] = None
) -> AgentResponse:
    """
    Execute agent query with conversation history.
    
    Args:
        history: List of conversation messages
        session_id: Unique session identifier
        portfolio: Optional current portfolio state
    
    Returns:
        AgentResponse with message, charts, and portfolio
    
    Raises:
        ValueError: If history is empty
    """
    if not history:
        raise ValueError("History cannot be empty")
    
    # Implementation...
```

### Testing Requirements

- Unit tests for all new functions
- Integration tests for workflows
- Minimum 80% code coverage

---


## Disclaimer

**Finnie AI is for educational purposes only.**

- Not financial advice
- Not a substitute for professional financial planning
- Market data may be delayed
- Simulations are based on historical assumptions
- Past performance does not guarantee future results

**Always consult a licensed financial advisor before making investment decisions.**

---

## Support

**Issues:** https://github.com/yourusername/finnie_ai/issues

**Documentation:** This README

**Contact:** peter.hanus@gmail.com

---

**Built with ❤️, and the occasional curse word, using LangChain, LangGraph, and Streamlit**