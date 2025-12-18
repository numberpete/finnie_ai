#!/bin/bash

# start_app.sh - Starts MCP servers and Gradio app

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Finnie AI Application...${NC}\n"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}🛑 Shutting down services...${NC}"
    kill $MCP_FINANCE_PID $MCP_YFINANCE_PID $GRADIO_PID 2>/dev/null
    wait $MCP_FINANCE_PID $MCP_YFINANCE_PID $GRADIO_PID 2>/dev/null
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
}

# Trap CTRL+C and other termination signals
trap cleanup SIGINT SIGTERM

# Kill any existing instances
echo -e "${YELLOW}🧹 Checking for existing processes...${NC}"
pkill -f "src.mcp.finance_q_and_a_mcp" 2>/dev/null
pkill -f "src.mcp.yfinance_mcp" 2>/dev/null
pkill -f "src.ui.app_chatbot" 2>/dev/null

# Check if ports 8001 and 8002 are in use and kill them
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8001 is in use, killing process...${NC}"
    lsof -ti:8001 | xargs kill -9 2>/dev/null
    sleep 1
fi

if lsof -Pi :8002 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8002 is in use, killing process...${NC}"
    lsof -ti:8002 | xargs kill -9 2>/dev/null
    sleep 1
fi

echo -e "${GREEN}✅ Cleaned up existing processes${NC}\n"

# Start Finance Q&A MCP Server in background
echo -e "${BLUE}📡 Starting Finance Q&A MCP Server (port 8001)...${NC}"
python -m src.mcp.finance_q_and_a_mcp &
MCP_FINANCE_PID=$!

# Wait a moment for it to start
sleep 2

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
sleep 2

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

# Start Gradio App in background
echo -e "${BLUE}🌐 Starting Gradio App...${NC}"
python -m src.ui.app_chatbot &
GRADIO_PID=$!

echo -e "${GREEN}✅ Gradio App started (PID: $GRADIO_PID)${NC}\n"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Application is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Finance MCP Server PID:${NC} $MCP_FINANCE_PID"
echo -e "${BLUE}yFinance MCP Server PID:${NC} $MCP_YFINANCE_PID"
echo -e "${BLUE}Gradio App PID:${NC} $GRADIO_PID"
echo -e "${BLUE}Finance MCP URL:${NC} http://localhost:8001"
echo -e "${BLUE}yFinance MCP URL:${NC} http://localhost:8002"
echo -e "\n${BLUE}Press CTRL+C to stop all services${NC}\n"

# Wait for all processes
wait $MCP_FINANCE_PID $MCP_YFINANCE_PID $GRADIO_PID