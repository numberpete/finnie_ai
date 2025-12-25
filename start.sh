#!/bin/bash

# start_app.sh - Starts MCP servers and streamlit app

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Finnie AI Application...${NC}\n"

# --- 1. VIRTUAL ENVIRONMENT CHECK & AUTO-CREATE ---
if [ -d ".venv" ]; then
    echo -e "${BLUE}📦 Activating virtual environment (.venv)...${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "${BLUE}📦 Activating virtual environment (venv)...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠️  No virtual environment found. Creating one with Python 3.11...${NC}"
    
    # Explicitly try to find the 3.11 executable
    if command -v python3.11 &> /dev/null; then
        PYTHON_EXE="python3.11"
    elif [[ $(python --version 2>&1) == *"3.11"* ]]; then
        PYTHON_EXE="python"
    else
        echo -e "${RED}❌ Python 3.11 not found. Please install it first.${NC}"
        exit 1
    fi

    echo -e "${BLUE}🛠️  Using: $($PYTHON_EXE --version)${NC}"
    $PYTHON_EXE -m venv .venv
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment (.venv) created successfully.${NC}"
        source .venv/bin/activate
    else
        echo -e "${RED}❌ Failed to create virtual environment.${NC}"
        exit 1
    fi
fi

# --- 2. DEPENDENCY INSTALLATION & DATA INITIALIZATION ---
echo -e "${BLUE}🛠️  Checking dependencies (pip install)...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${YELLOW}⚠️  requirements.txt not found. Skipping install.${NC}"
fi

if [ ! -d "src/data" ]; then
    echo -e "${YELLOW}📂 src/data not found. Initializing FAISS indices...${NC}"
    
    echo -e "${BLUE}🔨 Building default index...${NC}"
    python -m src.indexer.build_faiss_index -v
    
    echo -e "${BLUE}🔨 Building Bogleheads detailed index...${NC}"
    python -m src.indexer.build_faiss_index -v -a articles_bogleheads_detailed.csv -i bogleheads -l bogleheads_fetch.log -f bogleheads_failures.csv -s bogleheads_success.csv
    
    echo -e "${GREEN}✅ Data initialization complete.${NC}"
else
    echo -e "${GREEN}✅ src/data directory exists. Skipping index build.${NC}"
fi

# --- 3. DIRENV / ENV VARS LOAD ---
if command -v direnv &> /dev/null; then
    echo -e "${BLUE}🔑 Loading environment variables via direnv...${NC}"
    direnv allow .
    eval "$(direnv export bash)"
else
    echo -e "${YELLOW}⚠️  direnv not found. Skipping auto-load. Ensure .env is loaded manually.${NC}"
fi

# --- 4. ENVIRONMENT VARIABLE VALIDATION ---
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY is not set.${NC}"
    echo -e "${YELLOW}👉 Please add 'export OPENAI_API_KEY=your_key_here' to your .envrc file.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ OPENAI_API_KEY detected.${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}🛑 Shutting down services...${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $MCP_GOALS_PID $MCP_PORTFOLIO_PID $IMAGE_SERVER_PID $STREAMLIT_PID 2>/dev/null
    wait $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $MCP_GOALS_PID $MCP_PORTFOLIO_PID $IMAGE_SERVER_PID $STREAMLIT_PID 2>/dev/null
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
}

# Trap CTRL+C and other termination signals
trap cleanup SIGINT SIGTERM

# Kill any existing instances
echo -e "${YELLOW}🧹 Checking for existing processes...${NC}"
pkill -f "src.mcp.finance_q_and_a_mcp" 2>/dev/null
pkill -f "src.mcp.yfinance_mcp" 2>/dev/null
pkill -f "src.mcp.charts_mcp" 2>/dev/null
pkill -f "src.mcp.goals_mcp" 2>/dev/null
pkill -f "src.mcp.portfolio_mcp" 2>/dev/null
pkill -f "src.servers.image_server" 2>/dev/null
pkill -f "src.ui.app" 2>/dev/null

# Check if ports are in use and kill them
for port in 8001 8002 8003 8004 8005 8010 8501; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠️  Port $port is in use, killing process...${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null
        sleep 1
    fi
done

echo -e "${GREEN}✅ Cleaned up existing processes${NC}\n"

# Start Finance Q&A MCP Server in background
echo -e "${BLUE}📡 Starting Finance Q&A MCP Server (port 8001)...${NC}"
python -m src.mcp.finance_q_and_a_mcp &
MCP_FINANCE_PID=$!
sleep 4

# Start yFinance MCP Server in background
echo -e "${BLUE}📊 Starting yFinance MCP Server (port 8002)...${NC}"
python -m src.mcp.yfinance_mcp &
MCP_YFINANCE_PID=$!
sleep 4

# Start Chart MCP Server in background
echo -e "${BLUE}📈 Starting Chart MCP Server (port 8003)...${NC}"
python -m src.mcp.charts_mcp &
MCP_CHARTS_PID=$!
sleep 4

# Start Goals MCP Server in background
echo -e "${BLUE}🎯 Starting Goals MCP Server (port 8004)...${NC}"
python -m src.mcp.goals_mcp &
MCP_GOALS_PID=$!
sleep 4

# Start Portfolio MCP Server in background
echo -e "${BLUE}💼 Starting Portfolio MCP Server (port 8005)...${NC}"
python -m src.mcp.portfolio_mcp &
MCP_PORTFOLIO_PID=$!
sleep 4

# Start Image Server in background
echo -e "${BLUE}🖼️  Starting Image Server (port 8010)...${NC}"
python -m src.servers.image_server &
IMAGE_SERVER_PID=$!
sleep 4

# Start Streamlit App in background
echo -e "${BLUE}🌐 Starting Streamlit App...${NC}"
PYTHONPATH="${PWD}:${PYTHONPATH}" streamlit run src/ui/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo -e "${GREEN}✅ Streamlit App started (PID: $STREAMLIT_PID)${NC}\n"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Application is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Wait for all processes
wait $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $MCP_GOALS_PID $MCP_PORTFOLIO_PID $IMAGE_SERVER_PID $STREAMLIT_PID