#!/bin/bash

# start_app.sh - Starts both MCP server and Gradio app

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Finnie AI Application...${NC}\n"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}🛑 Shutting down services...${NC}"
    kill $MCP_PID $GRADIO_PID 2>/dev/null
    wait $MCP_PID $GRADIO_PID 2>/dev/null
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
}

# Trap CTRL+C and other termination signals
trap cleanup SIGINT SIGTERM

# Start MCP Server in background
echo -e "${BLUE}📡 Starting MCP Server...${NC}"
python -m src.mcp.finance_q_and_a_mcp &
MCP_PID=$!

# Wait for MCP server to be ready
echo -e "${BLUE}⏳ Waiting for MCP server to start...${NC}"
sleep 3

# Check if MCP server is still running
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo -e "${RED}❌ MCP server failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✅ MCP Server started (PID: $MCP_PID)${NC}\n"

# Start Gradio App in background
echo -e "${BLUE}🌐 Starting Gradio App...${NC}"
python -m src.ui.app_chatbot &
GRADIO_PID=$!

echo -e "${GREEN}✅ Gradio App started (PID: $GRADIO_PID)${NC}\n"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Application is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}MCP Server PID:${NC} $MCP_PID"
echo -e "${BLUE}Gradio App PID:${NC} $GRADIO_PID"
echo -e "\n${BLUE}Press CTRL+C to stop both services${NC}\n"

# Wait for both processes
wait $MCP_PID $GRADIO_PID