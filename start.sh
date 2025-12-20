#!/bin/bash

# start_app.sh - Starts MCP servers and streamlit app

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Finnie AI Application...${NC}\n"

# --- 1. VIRTUAL ENVIRONMENT CHECK ---
if [ -d ".venv" ]; then
    echo -e "${BLUE}📦 Activating virtual environment (.venv)...${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "${BLUE}📦 Activating virtual environment (venv)...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}❌ No virtual environment found (.venv or venv). Please create one first.${NC}"
    exit 1
fi

# --- 2. DIRENV / ENV VARS LOAD ---
if command -v direnv &> /dev/null; then
    echo -e "${BLUE}🔑 Loading environment variables via direnv...${NC}"
    direnv allow .
    eval "$(direnv export bash)"
else
    echo -e "${YELLOW}⚠️  direnv not found. Skipping auto-load. Ensure .env is loaded manually.${NC}"
fi

# --- 3. ENVIRONMENT VARIABLE VALIDATION ---
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY is not set.${NC}"
    echo -e "${YELLOW}👉 Please add 'export OPENAI_API_KEY=your_key_here' to your .envrc file.${NC}"
    echo -e "${YELLOW}👉 Then run 'direnv allow' or restart this script.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ OPENAI_API_KEY detected.${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}🛑 Shutting down services...${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $IMAGE_SERVER_PID $STREAMLIT_PID 2>/dev/null
    wait $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $IMAGE_SERVER_PID $STREAMLIT_PID 2>/dev/null
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
pkill -f "src.servers.image_server" 2>/dev/null
pkill -f "src.ui.app_streamlit" 2>/dev/null

# Check if ports 8001 and 8002 are in use and kill them
for port in 8001 8002 8003 8010 8501; do
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

# Wait a moment for it to start
sleep 4

# Check if Finance MCP server is still running
if ! kill -0 $MCP_FINANCE_PID 2>/dev/null; then
    echo -e "${RED}❌ Finance MCP server failed to start${NC}"
    exit 1
fi

# Verify port 8001 is listening
if ! lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}❌ Finance MCP server is not listening on port 8001${NC}"
    kill $MCP_FINANCE_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Finance Q&A MCP Server started (PID: $MCP_FINANCE_PID)${NC}\n"

# Start yFinance MCP Server in background
echo -e "${BLUE}📊 Starting yFinance MCP Server (port 8002)...${NC}"
python -m src.mcp.yfinance_mcp &
MCP_YFINANCE_PID=$!

# Wait a moment for it to start
sleep 4

# Check if yFinance MCP server is still running
if ! kill -0 $MCP_YFINANCE_PID 2>/dev/null; then
    echo -e "${RED}❌ yFinance MCP server failed to start${NC}"
    kill $MCP_FINANCE_PID 2>/dev/null
    exit 1
fi

# Verify port 8002 is listening
if ! lsof -Pi :8002 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}❌ yFinance MCP server is not listening on port 8002${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ yFinance MCP Server started (PID: $MCP_YFINANCE_PID)${NC}\n"

# Start Chart MCP Server in background
echo -e "${BLUE}📈 Starting Chart  MCP Server (port 8003)...${NC}"
python -m src.mcp.charts_mcp &
MCP_CHARTS_PID=$!

# Wait a moment for it to start
sleep 4

# Check if Chart MCP server is still running
if ! kill -0 $MCP_CHARTS_PID 2>/dev/null; then
    echo -e "${RED}❌ Chart  MCP server failed to start${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID 2>/dev/null
    exit 1
fi

# Verify port 8003 is listening
if ! lsof -Pi :8003 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}❌ Chart  MCP server is not listening on port 8003${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Chart MCP Server started (PID: $MCP_CHARTS_PID)${NC}\n"

# Start Image Server in background
echo -e "${BLUE}🖼️  Starting Image Server (port 8010)...${NC}"
python -m src.servers.image_server &
IMAGE_SERVER_PID=$!

# Wait a moment for it to start
sleep 4

# Check if Image Server is still running
if ! kill -0 $IMAGE_SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Image Server failed to start${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID 2>/dev/null
    exit 1
fi

# Verify port 8010 is listening
if ! lsof -Pi :8010 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}❌ Image Server is not listening on port 8010${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $IMAGE_SERVER_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Image Server started (PID: $IMAGE_SERVER_PID)${NC}\n"

# Start Streamlit App in background
echo -e "${BLUE}🌐 Starting Streamlit App...${NC}"
PYTHONPATH="${PWD}:${PYTHONPATH}" streamlit run src/ui/app_streamlit.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo -e "${GREEN}✅ Streamlit App started (PID: $STREAMLIT_PID)${NC}\n"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Application is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Finance MCP Server PID:${NC} $MCP_FINANCE_PID"
echo -e "${BLUE}yFinance MCP Server PID:${NC} $MCP_YFINANCE_PID"
echo -e "${BLUE}Chart MCP Server PID:${NC} $MCP_CHARTS_PID"
echo -e "${BLUE}Image Server PID:${NC} $IMAGE_SERVER_PID"
echo -e "${BLUE}Streamlit App PID:${NC} $STREAMLIT_PID"
echo -e "\n${BLUE}Press CTRL+C to stop all services${NC}\n"

# Wait for all processes
wait $MCP_FINANCE_PID $MCP_YFINANCE_PID $MCP_CHARTS_PID $IMAGE_SERVER_PID $STREAMLIT_PID