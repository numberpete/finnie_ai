#!/bin/bash

# install.sh - Sets up the Finnie AI development environment

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🛠️  Starting Finnie AI Installation...${NC}\n"

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install it and try again.${NC}"
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}📦 Creating virtual environment (.venv)...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created.${NC}"
else
    echo -e "${YELLOW}ℹ️  Virtual environment already exists. Skipping creation.${NC}"
fi

# 3. Activate Virtual Environment
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
echo -e "${BLUE}🚀 Installing/Updating dependencies...${NC}"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed.${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found. Installing core packages manually...${NC}"
    pip install langchain langgraph gradio openai aiohttp
fi

# 5. Setup direnv (.envrc)
echo -e "${BLUE}🔑 Configuring environment variables...${NC}"
if [ ! -f ".envrc" ]; then
    echo "export OPENAI_API_KEY=your_key_here" > .envrc
    echo "export LANGSMITH_API_KEY=your_key_here" >> .envrc
    echo -e "${GREEN}✅ Created template .envrc file.${NC}"
else
    echo -e "${YELLOW}ℹ️  .envrc already exists. Skipping creation.${NC}"
fi

# Check if direnv is installed
if command -v direnv &> /dev/null; then
    echo -e "${BLUE}🔓 Authorizing direnv...${NC}"
    direnv allow .
else
    echo -e "${YELLOW}⚠️  direnv is not installed. You should install it to manage env vars easily.${NC}"
fi

# 6. Create necessary directories
echo -e "${BLUE}📁 Ensuring project structure...${NC}"
mkdir -p src/agents src/mcp src/ui
touch src/agents/__init__.py src/mcp/__init__.py src/ui/__init__.py

# 7. Final Instructions
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Installation Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "1. Edit your ${BLUE}.envrc${NC} file and add your actual API keys."
echo -e "2. Run ${BLUE}direnv allow${NC} (if using direnv)."
echo -e "3. Start the application using: ${GREEN}./start_app.sh${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"