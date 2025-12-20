#!/bin/bash
# test_charts.sh

echo "Starting servers..."
python -m src.mcp.charts_mcp > /dev/null 2>&1 &
MCP_PID=$!

python -m src.servers.image_server > /dev/null 2>&1 &
IMG_PID=$!

sleep 3
echo "Servers started (MCP: $MCP_PID, Image: $IMG_PID)"
echo ""

# Run tests
python src/mcp/test_charts_mcp.py

# Cleanup
echo ""
echo "Cleaning up..."
kill $MCP_PID $IMG_PID 2>/dev/null
echo "Done!"