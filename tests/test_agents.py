# tests/test_real_agents.py
"""
Real tests for agent classes.
These tests import and test the actual agent implementations.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.response import AgentResponse, ChartArtifact
from src.agents.base_agent import BaseAgent


# ============================================================================
# BaseAgent Tests
# ============================================================================

class TestBaseAgent:
    """Test BaseAgent class"""
    
    @pytest.mark.asyncio
    async def test_base_agent_initialization_no_mcp(self):
        """Test BaseAgent initializes without MCP servers"""
        mock_llm = Mock()
        mock_logger = Mock()
        
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test prompt",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize the counter since we're not using a subclass
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        assert agent.agent_name == "TestAgent"
        assert agent.LOGGER == mock_logger
        assert agent.tools == []
        assert agent.mcp_client is None
    
    @pytest.mark.asyncio
    async def test_base_agent_run_query_simple_response(self):
        """Test BaseAgent handles simple text response"""
        mock_llm = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        mock_logger.warning = Mock()
        
        # Create agent
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize counter
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        # Mock the core_agent to return a simple response
        mock_response_msg = Mock()
        mock_response_msg.content = "This is a test response"
        mock_response_msg.tool_calls = []
        
        agent.core_agent = Mock()
        agent.core_agent.ainvoke = AsyncMock(
            return_value={"messages": [mock_response_msg]}
        )
        
        # Run query
        history = [HumanMessage(content="Test question")]
        response = await agent.run_query(history, "test_session")
        
        # Assertions
        assert isinstance(response, AgentResponse)
        assert response.agent == "TestAgent"
        assert response.message == "This is a test response"
        assert response.charts == []
        assert response.portfolio is None
    
    @pytest.mark.asyncio
    async def test_base_agent_extracts_charts(self):
        """Test BaseAgent extracts chart artifacts from tool messages"""
        mock_llm = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize counter
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        # Mock AI message with tool call
        ai_msg_with_tool = Mock()
        ai_msg_with_tool.__class__.__name__ = "AIMessage"
        ai_msg_with_tool.tool_calls = [{"name": "create_line_chart"}]
        ai_msg_with_tool.content = ""
        
        # Mock tool message with chart data - IMPORTANT: set the type name
        tool_msg = Mock()
        tool_msg.__class__.__name__ = "ToolMessage"
        tool_msg.name = "create_line_chart"
        tool_msg.content = [{
            'type': 'text',
            'text': '{"title": "Test Chart", "filename": "test.png", "chart_type": "line"}'
        }]
        
        # Mock final AI message
        final_msg = Mock()
        final_msg.__class__.__name__ = "AIMessage"
        final_msg.content = "Here is your chart"
        final_msg.tool_calls = []
        
        # First call returns AI message with tool call
        # Second call returns both tool message AND final message together
        agent.core_agent = Mock()
        agent.core_agent.ainvoke = AsyncMock(
            side_effect=[
                {"messages": [ai_msg_with_tool]},
                {"messages": [tool_msg, final_msg]}
            ]
        )
        
        history = [HumanMessage(content="Create a chart")]
        response = await agent.run_query(history, "test_session")
        
        # Should have extracted the chart
        assert len(response.charts) == 1
        assert response.charts[0].title == "Test Chart"
        assert response.charts[0].filename == "test.png"
    
    @pytest.mark.asyncio
    async def test_base_agent_max_iterations(self):
        """Test BaseAgent respects max iterations"""
        mock_llm = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        mock_logger.warning = Mock()
        
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize counter
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        # Mock infinite loop - always return tool calls
        tool_call_msg = Mock()
        tool_call_msg.content = ""
        tool_call_msg.tool_calls = [{"name": "some_tool"}]
        
        agent.core_agent = Mock()
        agent.core_agent.ainvoke = AsyncMock(
            return_value={"messages": [tool_call_msg]}
        )
        
        history = [HumanMessage(content="Test")]
        response = await agent.run_query(history, "test_session")
        
        # Should stop at max iterations
        assert "couldn't complete" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_base_agent_error_handling(self):
        """Test BaseAgent handles errors gracefully"""
        mock_llm = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        mock_logger.error = Mock()
        
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize counter
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        # Mock error
        agent.core_agent = Mock()
        agent.core_agent.ainvoke = AsyncMock(
            side_effect=Exception("Test error")
        )
        
        history = [HumanMessage(content="Test")]
        response = await agent.run_query(history, "test_session")
        
        # Should return error response
        assert isinstance(response, AgentResponse)
        assert "error" in response.message.lower() or "encountered" in response.message.lower()


# ============================================================================
# AgentResponse Tests
# ============================================================================

class TestAgentResponse:
    """Test AgentResponse dataclass"""
    
    def test_agent_response_creation(self):
        """Test creating AgentResponse"""
        response = AgentResponse(
            agent="TestAgent",
            message="Test message",
            charts=[],
            portfolio=None
        )
        
        assert response.agent == "TestAgent"
        assert response.message == "Test message"
        assert response.charts == []
        assert response.portfolio is None
    
    def test_agent_response_with_charts(self):
        """Test AgentResponse with charts"""
        chart = ChartArtifact(
            title="Test Chart",
            filename="test.png"
        )
        
        response = AgentResponse(
            agent="TestAgent",
            message="Here's your chart",
            charts=[chart]
        )
        
        assert len(response.charts) == 1
        assert response.charts[0].title == "Test Chart"
        assert response.charts[0].filename == "test.png"
    
    def test_agent_response_with_portfolio(self):
        """Test AgentResponse with portfolio"""
        portfolio = {
            "Equities": 100000,
            "Fixed_Income": 50000,
            "Real_Estate": 0,
            "Cash": 25000,
            "Commodities": 0,
            "Crypto": 0
        }
        
        response = AgentResponse(
            agent="PortfolioAgent",
            message="Portfolio updated",
            charts=[],
            portfolio=portfolio
        )
        
        assert response.portfolio is not None
        assert response.portfolio["Equities"] == 100000
        assert sum(response.portfolio.values()) == 175000


# ============================================================================
# Specialized Agent Tests (without actual MCP servers)
# ============================================================================

class TestPortfolioAgent:
    """Test PortfolioAgent initialization and structure"""
    
    @patch('src.agents.finance_portfolio.BaseAgent.__init__')
    def test_portfolio_agent_initialization(self, mock_base_init):
        """Test PortfolioAgent initializes correctly"""
        from src.agents.finance_portfolio import PortfolioAgent
        
        mock_base_init.return_value = None
        agent = PortfolioAgent()
        
        # Should have called BaseAgent.__init__ with correct parameters
        mock_base_init.assert_called_once()
        call_kwargs = mock_base_init.call_args[1]
        
        assert call_kwargs['agent_name'] == "PortfolioAgent"
        assert 'mcp_servers' in call_kwargs
        assert 'portfolio_mcp' in call_kwargs['mcp_servers']
        assert 'charts_mcp' in call_kwargs['mcp_servers']
        assert 'yfinance_mcp' in call_kwargs['mcp_servers']


class TestMarketAgent:
    """Test FinanceMarketAgent initialization and structure"""
    
    @patch('src.agents.finance_market.BaseAgent.__init__')
    def test_market_agent_initialization(self, mock_base_init):
        """Test FinanceMarketAgent initializes correctly"""
        from src.agents.finance_market import FinanceMarketAgent
        
        mock_base_init.return_value = None
        agent = FinanceMarketAgent()
        
        mock_base_init.assert_called_once()
        call_kwargs = mock_base_init.call_args[1]
        
        assert call_kwargs['agent_name'] == "FinanceMarketAgent"
        assert 'mcp_servers' in call_kwargs
        assert 'yfinance_mcp' in call_kwargs['mcp_servers']
        assert 'charts_mcp' in call_kwargs['mcp_servers']


class TestGoalsAgent:
    """Test GoalsAgent initialization and structure"""
    
    @patch('src.agents.finance_goals.BaseAgent.__init__')
    def test_goals_agent_initialization(self, mock_base_init):
        """Test GoalsAgent initializes correctly"""
        from src.agents.finance_goals import GoalsAgent
        
        mock_base_init.return_value = None
        agent = GoalsAgent()
        
        mock_base_init.assert_called_once()
        call_kwargs = mock_base_init.call_args[1]
        
        assert call_kwargs['agent_name'] == "GoalsAgent"
        assert 'mcp_servers' in call_kwargs
        assert 'charts_mcp' in call_kwargs['mcp_servers']
        assert 'goals_mcp' in call_kwargs['mcp_servers']


class TestQandAAgent:
    """Test FinanceQandAAgent initialization and structure"""
    
    @patch('src.agents.finance_q_and_a.BaseAgent.__init__')
    def test_qanda_agent_initialization(self, mock_base_init):
        """Test FinanceQandAAgent initializes correctly"""
        from src.agents.finance_q_and_a import FinanceQandAAgent
        
        mock_base_init.return_value = None
        agent = FinanceQandAAgent()
        
        mock_base_init.assert_called_once()
        call_kwargs = mock_base_init.call_args[1]
        
        assert call_kwargs['agent_name'] == "FinanceQandAAgent"
        assert 'mcp_servers' in call_kwargs
        assert 'finance_qanda_tool' in call_kwargs['mcp_servers']


# ============================================================================
# RouterAgent Tests
# ============================================================================

class TestRouterAgent:
    """Test RouterAgent routing logic"""
    
    def test_router_agent_initialization(self):
        """Test RouterAgent initializes with all agents"""
        from src.agents.router import RouterAgent
        from langgraph.checkpoint.memory import InMemorySaver
        
        # Patch the BaseAgent.__init__ to avoid MCP initialization
        with patch('src.agents.base_agent.BaseAgent.__init__', return_value=None):
            checkpointer = InMemorySaver()
            router = RouterAgent(checkpointer=checkpointer)
            
            assert router.finance_qa_agent is not None
            assert router.finance_market_agent is not None
            assert router.portfolio_agent is not None
            assert router.goals_agent is not None
            assert router.workflow is not None
    
    def test_get_empty_portfolio(self):
        """Test get_empty_portfolio helper function"""
        from src.agents.router import get_empty_portfolio
        
        portfolio = get_empty_portfolio()
        
        assert isinstance(portfolio, dict)
        assert len(portfolio) == 6
        assert all(v == 0.0 for v in portfolio.values())
        assert "Equities" in portfolio
        assert "Fixed_Income" in portfolio
    
    @pytest.mark.asyncio
    async def test_router_route_next(self):
        """Test route_next method"""
        from src.agents.router import RouterAgent, AgentState
        from langgraph.checkpoint.memory import InMemorySaver
        
        # Patch BaseAgent to avoid MCP initialization
        with patch('src.agents.base_agent.BaseAgent.__init__', return_value=None):
            router = RouterAgent(checkpointer=InMemorySaver())
            
            state = {
                "messages": [],
                "next": "PortfolioAgent",
                "session_id": "test",
                "last_agent_used": None,
                "current_portfolio": {},
                "response": []
            }
            
            result = router.route_next(state)
            assert result == "PortfolioAgent"


# ============================================================================
# Integration-style Tests (mocked MCP)
# ============================================================================

class TestAgentIntegration:
    """Test agent workflows with mocked dependencies"""
    
    @pytest.mark.asyncio
    async def test_portfolio_agent_workflow(self):
        """Test complete portfolio agent workflow"""
        mock_llm = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        
        agent = BaseAgent(
            agent_name="PortfolioAgent",
            llm=mock_llm,
            system_prompt="Test",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize counter
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        # Mock successful portfolio addition
        final_msg = Mock()
        final_msg.content = "Added $100k to Equities. Portfolio total: $100k"
        final_msg.tool_calls = []
        
        agent.core_agent = Mock()
        agent.core_agent.ainvoke = AsyncMock(
            return_value={"messages": [final_msg]}
        )
        
        history = [HumanMessage(content="Add $100k to Equities")]
        response = await agent.run_query(history, "test_session")
        
        assert "100k" in response.message
        assert "Equities" in response.message
    
    @pytest.mark.asyncio
    async def test_market_agent_chart_generation(self):
        """Test market agent generates charts"""
        mock_llm = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        
        agent = BaseAgent(
            agent_name="FinanceMarketAgent",
            llm=mock_llm,
            system_prompt="Test",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Initialize counter
        if not hasattr(BaseAgent, '_invocation_count'):
            BaseAgent._invocation_count = 0
        
        # Mock AI message with tool call
        ai_msg_with_tool = Mock()
        ai_msg_with_tool.__class__.__name__ = "AIMessage"
        ai_msg_with_tool.tool_calls = [{"name": "create_line_chart"}]
        ai_msg_with_tool.content = ""
        
        # Mock chart generation workflow
        tool_msg = Mock()
        tool_msg.__class__.__name__ = "ToolMessage"
        tool_msg.name = "create_line_chart"
        tool_msg.content = [{
            'type': 'text',
            'text': '{"title": "AAPL Stock Price", "filename": "aapl.png", "chart_type": "line"}'
        }]
        
        final_msg = Mock()
        final_msg.__class__.__name__ = "AIMessage"
        final_msg.content = "Here's the chart for Apple stock"
        final_msg.tool_calls = []
        
        agent.core_agent = Mock()
        agent.core_agent.ainvoke = AsyncMock(
            side_effect=[
                {"messages": [ai_msg_with_tool]},
                {"messages": [tool_msg, final_msg]}
            ]
        )
        
        history = [HumanMessage(content="Show me AAPL chart")]
        response = await agent.run_query(history, "test_session")
        
        assert len(response.charts) == 1
        assert "AAPL" in response.charts[0].title


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])