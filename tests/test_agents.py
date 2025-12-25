# tests/test_agents.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.market_agent import FinanceMarketAgent
from src.agents.goals_agent import GoalsAgent
from src.agents.qanda_agent import QandAAgent
from src.agents.router import RouterAgent
from src.agents.response import AgentResponse, ChartArtifact


# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def mock_llm():
    """Mock LLM that returns configurable responses"""
    llm = Mock()
    
    def create_response(content="Mock response", tool_calls=None):
        msg = Mock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        return {"messages": [msg]}
    
    llm.ainvoke = AsyncMock(side_effect=lambda x, **kwargs: create_response())
    llm.create_response = create_response
    return llm


@pytest.fixture
def empty_portfolio():
    """Empty portfolio fixture"""
    return {
        "Equities": 0.0,
        "Fixed_Income": 0.0,
        "Real_Estate": 0.0,
        "Cash": 0.0,
        "Commodities": 0.0,
        "Crypto": 0.0
    }


@pytest.fixture
def sample_portfolio():
    """Sample portfolio with some assets"""
    return {
        "Equities": 100000.0,
        "Fixed_Income": 50000.0,
        "Real_Estate": 0.0,
        "Cash": 25000.0,
        "Commodities": 0.0,
        "Crypto": 0.0
    }


# --- BaseAgent Tests ---

class TestBaseAgent:
    """Test BaseAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_base_agent_initialization(self, mock_llm, mock_logger):
        """Test BaseAgent initializes correctly"""
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test prompt",
            logger=mock_logger,
            mcp_servers=None,
            debug=False
        )
        
        assert agent.agent_name == "TestAgent"
        assert agent.LOGGER == mock_logger
        assert agent.tools == []
        assert agent.mcp_client is None
    
    @pytest.mark.asyncio
    async def test_base_agent_run_query_simple(self, mock_llm, mock_logger):
        """Test BaseAgent can run a simple query"""
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test prompt",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Mock LLM to return a simple response (no tool calls)
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(content="Test response")
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Test question")],
            session_id="test_session"
        )
        
        assert isinstance(response, AgentResponse)
        assert response.agent == "TestAgent"
        assert response.message == "Test response"
        assert response.charts == []
        assert response.portfolio is None
    
    @pytest.mark.asyncio
    async def test_base_agent_max_iterations(self, mock_llm, mock_logger):
        """Test BaseAgent respects max iterations"""
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test prompt",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Mock LLM to always return tool calls (infinite loop scenario)
        mock_tool_call = Mock()
        mock_tool_call.get = Mock(return_value="mock_tool")
        
        mock_llm.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    Mock(content="", tool_calls=[mock_tool_call])
                ]
            }
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Test")],
            session_id="test"
        )
        
        assert "couldn't complete the request" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_base_agent_error_handling(self, mock_llm, mock_logger):
        """Test BaseAgent handles errors gracefully"""
        agent = BaseAgent(
            agent_name="TestAgent",
            llm=mock_llm,
            system_prompt="Test prompt",
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Mock LLM to raise an exception
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        response = await agent.run_query(
            history=[HumanMessage(content="Test")],
            session_id="test"
        )
        
        assert isinstance(response, AgentResponse)
        assert "error" in response.message.lower()
        assert response.charts == []


# --- PortfolioAgent Tests ---

class TestPortfolioAgent:
    """Test PortfolioAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_portfolio_agent_initialization(self, mock_llm, mock_logger):
        """Test PortfolioAgent initializes correctly"""
        agent = PortfolioAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        assert agent.agent_name == "PortfolioAgent"
    
    @pytest.mark.asyncio
    async def test_portfolio_agent_add_single_asset(self, mock_llm, mock_logger, empty_portfolio):
        """Test adding to a single asset class"""
        agent = PortfolioAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Mock successful portfolio update
        updated_portfolio = empty_portfolio.copy()
        updated_portfolio["Equities"] = 100000.0
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="Added $100k to Equities"
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Add $100k to Equities")],
            session_id="test",
            portfolio=empty_portfolio
        )
        
        assert isinstance(response, AgentResponse)
        assert "equities" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_portfolio_agent_requires_portfolio(self, mock_llm, mock_logger):
        """Test agent handles missing portfolio gracefully"""
        agent = PortfolioAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="I need a portfolio to work with"
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Add $100k to Equities")],
            session_id="test"
        )
        
        # Agent should handle gracefully
        assert isinstance(response, AgentResponse)


# --- FinanceMarketAgent Tests ---

class TestFinanceMarketAgent:
    """Test FinanceMarketAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_market_agent_initialization(self, mock_llm, mock_logger):
        """Test FinanceMarketAgent initializes correctly"""
        agent = FinanceMarketAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        assert agent.agent_name == "FinanceMarketAgent"
    
    @pytest.mark.asyncio
    async def test_market_agent_stock_price(self, mock_llm, mock_logger):
        """Test getting stock price"""
        agent = FinanceMarketAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="Apple (AAPL) is trading at $185.43"
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="What's the price of Apple?")],
            session_id="test"
        )
        
        assert isinstance(response, AgentResponse)
        assert "aapl" in response.message.lower() or "apple" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_market_agent_chart_generation(self, mock_llm, mock_logger):
        """Test chart generation"""
        agent = FinanceMarketAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Mock response with chart
        mock_chart = ChartArtifact(
            title="AAPL Stock Price",
            filename="test_chart.png"
        )
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="Here's the chart for Apple stock"
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Show me AAPL chart")],
            session_id="test"
        )
        
        assert isinstance(response, AgentResponse)


# --- GoalsAgent Tests ---

class TestGoalsAgent:
    """Test GoalsAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_goals_agent_initialization(self, mock_llm, mock_logger):
        """Test GoalsAgent initializes correctly"""
        agent = GoalsAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        assert agent.agent_name == "GoalsAgent"
    
    @pytest.mark.asyncio
    async def test_goals_agent_simulation(self, mock_llm, mock_logger, sample_portfolio):
        """Test running a simulation"""
        agent = GoalsAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="Based on simulation over 10 years..."
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="How will my portfolio do in 10 years?")],
            session_id="test",
            portfolio=sample_portfolio
        )
        
        assert isinstance(response, AgentResponse)
        assert "simulation" in response.message.lower() or "years" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_goals_agent_requires_portfolio(self, mock_llm, mock_logger):
        """Test agent handles missing portfolio"""
        agent = GoalsAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="I need your portfolio to run a simulation"
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Simulate my portfolio")],
            session_id="test"
        )
        
        assert isinstance(response, AgentResponse)
        assert "portfolio" in response.message.lower()


# --- QandAAgent Tests ---

class TestQandAAgent:
    """Test QandAAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_qanda_agent_initialization(self, mock_llm, mock_logger):
        """Test QandAAgent initializes correctly"""
        agent = QandAAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        assert agent.agent_name == "QandAAgent"
    
    @pytest.mark.asyncio
    async def test_qanda_agent_general_question(self, mock_llm, mock_logger):
        """Test answering general questions"""
        agent = QandAAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        mock_llm.ainvoke = AsyncMock(
            return_value=mock_llm.create_response(
                content="Dollar-cost averaging is an investment strategy..."
            )
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="What is dollar-cost averaging?")],
            session_id="test"
        )
        
        assert isinstance(response, AgentResponse)
        assert len(response.message) > 0
        assert response.charts == []  # Q&A shouldn't generate charts


# --- RouterAgent Tests ---

class TestRouterAgent:
    """Test RouterAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_router_initialization(self):
        """Test RouterAgent initializes correctly"""
        from langgraph.checkpoint.memory import InMemorySaver
        
        checkpointer = InMemorySaver()
        router = RouterAgent(checkpointer=checkpointer)
        
        assert router.portfolio_agent is not None
        assert router.market_agent is not None
        assert router.goals_agent is not None
        assert router.qanda_agent is not None
    
    @pytest.mark.asyncio
    async def test_router_portfolio_routing(self):
        """Test router routes to PortfolioAgent"""
        from langgraph.checkpoint.memory import InMemorySaver
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio:
            # Setup mock
            mock_agent = Mock()
            mock_agent.run_query = AsyncMock(
                return_value=AgentResponse(
                    agent="PortfolioAgent",
                    message="Portfolio created",
                    charts=[],
                    portfolio={"Equities": 100000.0, "Fixed_Income": 0.0, "Real_Estate": 0.0, 
                              "Cash": 0.0, "Commodities": 0.0, "Crypto": 0.0}
                )
            )
            MockPortfolio.return_value = mock_agent
            
            checkpointer = InMemorySaver()
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = mock_agent
            
            response = await router.run_query(
                user_input="Build me a portfolio with $100k in Equities",
                session_id="test"
            )
            
            assert isinstance(response, AgentResponse)
            assert response.agent == "PortfolioAgent"
    
    @pytest.mark.asyncio
    async def test_router_market_routing(self):
        """Test router routes to FinanceMarketAgent"""
        from langgraph.checkpoint.memory import InMemorySaver
        
        with patch('src.agents.market_agent.FinanceMarketAgent') as MockMarket:
            # Setup mock
            mock_agent = Mock()
            mock_agent.run_query = AsyncMock(
                return_value=AgentResponse(
                    agent="FinanceMarketAgent",
                    message="Apple is trading at $185.43",
                    charts=[],
                    portfolio=None
                )
            )
            MockMarket.return_value = mock_agent
            
            checkpointer = InMemorySaver()
            router = RouterAgent(checkpointer=checkpointer)
            router.market_agent = mock_agent
            
            response = await router.run_query(
                user_input="What's the price of Apple?",
                session_id="test"
            )
            
            assert isinstance(response, AgentResponse)
            assert response.agent == "FinanceMarketAgent"
    
    @pytest.mark.asyncio
    async def test_router_goals_routing(self):
        """Test router routes to GoalsAgent"""
        from langgraph.checkpoint.memory import InMemorySaver
        
        with patch('src.agents.goals_agent.GoalsAgent') as MockGoals:
            # Setup mock
            mock_agent = Mock()
            mock_agent.run_query = AsyncMock(
                return_value=AgentResponse(
                    agent="GoalsAgent",
                    message="Simulation results...",
                    charts=[],
                    portfolio=None
                )
            )
            MockGoals.return_value = mock_agent
            
            checkpointer = InMemorySaver()
            router = RouterAgent(checkpointer=checkpointer)
            router.goals_agent = mock_agent
            
            response = await router.run_query(
                user_input="How will my portfolio do in 10 years?",
                session_id="test"
            )
            
            assert isinstance(response, AgentResponse)
            assert response.agent == "GoalsAgent"


# --- AgentResponse Tests ---

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
            charts=[chart],
            portfolio=None
        )
        
        assert len(response.charts) == 1
        assert response.charts[0].title == "Test Chart"
        assert response.charts[0].filename == "test.png"
    
    def test_agent_response_with_portfolio(self):
        """Test AgentResponse with portfolio"""
        portfolio = {
            "Equities": 100000.0,
            "Fixed_Income": 50000.0,
            "Real_Estate": 0.0,
            "Cash": 25000.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        response = AgentResponse(
            agent="PortfolioAgent",
            message="Portfolio updated",
            charts=[],
            portfolio=portfolio
        )
        
        assert response.portfolio is not None
        assert response.portfolio["Equities"] == 100000.0
        assert sum(response.portfolio.values()) == 175000.0


# --- Integration Tests with Mocked Tools ---

class TestAgentToolIntegration:
    """Test agents with mocked tool calls"""
    
    @pytest.mark.asyncio
    async def test_portfolio_agent_tool_call_flow(self, mock_llm, mock_logger, empty_portfolio):
        """Test PortfolioAgent makes correct tool calls"""
        agent = PortfolioAgent(
            llm=mock_llm,
            logger=mock_logger,
            mcp_servers=None
        )
        
        # Mock tool call followed by final response
        tool_call_msg = Mock()
        tool_call_msg.content = ""
        tool_call_msg.tool_calls = [
            {"name": "add_to_portfolio_asset_class", "args": {"asset_class_key": "Equities", "amount": 100000}}
        ]
        
        final_msg = Mock()
        final_msg.content = "Added $100k to Equities"
        final_msg.tool_calls = []
        
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                {"messages": [tool_call_msg]},
                {"messages": [final_msg]}
            ]
        )
        
        response = await agent.run_query(
            history=[HumanMessage(content="Add $100k to Equities")],
            session_id="test",
            portfolio=empty_portfolio
        )
        
        assert isinstance(response, AgentResponse)
        # Should have made tool call
        assert mock_llm.ainvoke.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
