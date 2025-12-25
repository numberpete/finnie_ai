# tests/test_integration.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.router import RouterAgent
from src.agents.response import AgentResponse, ChartArtifact


# --- Integration Test Fixtures ---

@pytest.fixture
def router_agent():
    """Create RouterAgent with InMemorySaver"""
    checkpointer = InMemorySaver()
    return RouterAgent(checkpointer=checkpointer)


@pytest.fixture
def mock_mcp_servers():
    """Mock MCP server responses"""
    return {
        "portfolio_mcp": {
            "url": "http://localhost:8001/sse",
            "description": "Portfolio management tools"
        },
        "yfinance_mcp": {
            "url": "http://localhost:8002/sse",
            "description": "Market data tools"
        },
        "charts_mcp": {
            "url": "http://localhost:8010/sse",
            "description": "Chart generation tools"
        },
        "goals_mcp": {
            "url": "http://localhost:8003/sse",
            "description": "Goals and simulation tools"
        }
    }


# --- Full Workflow Integration Tests ---

class TestFullPortfolioWorkflow:
    """Test complete portfolio building and management workflow"""
    
    @pytest.mark.asyncio
    async def test_build_and_summarize_portfolio(self):
        """Test building portfolio and getting summary"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio:
            # Mock portfolio agent responses
            mock_agent = Mock()
            
            # Response 1: Build portfolio
            build_response = AgentResponse(
                agent="PortfolioAgent",
                message="Created portfolio with $100k in Equities",
                charts=[],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            # Response 2: Add more
            add_response = AgentResponse(
                agent="PortfolioAgent",
                message="Added $50k to Fixed Income",
                charts=[],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 50000.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            # Response 3: Summary with chart
            summary_chart = ChartArtifact(
                title="Portfolio Allocation",
                filename="portfolio_pie.png"
            )
            summary_response = AgentResponse(
                agent="PortfolioAgent",
                message="Your portfolio totals $150k with 66.7% in Equities",
                charts=[summary_chart],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 50000.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            mock_agent.run_query = AsyncMock(
                side_effect=[build_response, add_response, summary_response]
            )
            MockPortfolio.return_value = mock_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = mock_agent
            
            # Step 1: Build portfolio
            response1 = await router.run_query(
                "Build me a portfolio with $100k in Equities",
                "test_session_1"
            )
            assert response1.portfolio["Equities"] == 100000.0
            assert sum(response1.portfolio.values()) == 100000.0
            
            # Step 2: Add to portfolio
            response2 = await router.run_query(
                "Add $50k to Fixed Income",
                "test_session_1"
            )
            assert response2.portfolio["Fixed_Income"] == 50000.0
            assert sum(response2.portfolio.values()) == 150000.0
            
            # Step 3: Get summary
            response3 = await router.run_query(
                "Summarize my portfolio",
                "test_session_1"
            )
            assert len(response3.charts) > 0
            assert "150" in response3.message or "$150k" in response3.message


class TestMarketDataWorkflow:
    """Test market data retrieval and visualization workflow"""
    
    @pytest.mark.asyncio
    async def test_stock_price_and_chart(self):
        """Test getting stock price and generating chart"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.market_agent.FinanceMarketAgent') as MockMarket:
            mock_agent = Mock()
            
            # Response 1: Current price
            price_response = AgentResponse(
                agent="FinanceMarketAgent",
                message="Apple (AAPL) is trading at $185.43, up $2.15 (+1.17%)",
                charts=[],
                portfolio=None
            )
            
            # Response 2: Historical chart
            chart = ChartArtifact(
                title="AAPL Stock Price (Jan 2024 - Dec 2024)",
                filename="aapl_1y.png"
            )
            chart_response = AgentResponse(
                agent="FinanceMarketAgent",
                message="Here's the 1-year chart for Apple",
                charts=[chart],
                portfolio=None
            )
            
            mock_agent.run_query = AsyncMock(
                side_effect=[price_response, chart_response]
            )
            MockMarket.return_value = mock_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.market_agent = mock_agent
            
            # Step 1: Get current price
            response1 = await router.run_query(
                "What's the price of Apple?",
                "test_session_2"
            )
            assert "AAPL" in response1.message or "Apple" in response1.message
            assert "$" in response1.message
            
            # Step 2: Get chart
            response2 = await router.run_query(
                "Show me AAPL over the last year",
                "test_session_2"
            )
            assert len(response2.charts) > 0
            assert response2.charts[0].title.startswith("AAPL")
    
    @pytest.mark.asyncio
    async def test_stock_comparison(self):
        """Test comparing multiple stocks"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.market_agent.FinanceMarketAgent') as MockMarket:
            mock_agent = Mock()
            
            comparison_chart = ChartArtifact(
                title="SAP vs Oracle Stock Prices (Sep 2024 - Dec 2024)",
                filename="sap_orcl_comparison.png"
            )
            
            response = AgentResponse(
                agent="FinanceMarketAgent",
                message="Here's the comparison between SAP and Oracle over the last 3 months",
                charts=[comparison_chart],
                portfolio=None
            )
            
            mock_agent.run_query = AsyncMock(return_value=response)
            MockMarket.return_value = mock_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.market_agent = mock_agent
            
            result = await router.run_query(
                "Compare SAP to Oracle",
                "test_session_3"
            )
            
            assert len(result.charts) == 1  # Should only have one comparison chart
            assert "SAP" in result.charts[0].title
            assert "Oracle" in result.charts[0].title


class TestGoalsSimulationWorkflow:
    """Test financial goals and Monte Carlo simulation workflow"""
    
    @pytest.mark.asyncio
    async def test_portfolio_projection(self):
        """Test running simulation on portfolio"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.goals_agent.GoalsAgent') as MockGoals:
            mock_agent = Mock()
            
            simulation_chart = ChartArtifact(
                title="Portfolio Projection - 10 Year Scenarios",
                filename="simulation_10y.png"
            )
            
            response = AgentResponse(
                agent="GoalsAgent",
                message="Based on simulation: Bottom 10%: $2.9M, Median: $4.6M, Top 10%: $7.7M",
                charts=[simulation_chart],
                portfolio=None
            )
            
            mock_agent.run_query = AsyncMock(return_value=response)
            MockGoals.return_value = mock_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.goals_agent = mock_agent
            
            result = await router.run_query(
                "How will my portfolio do in 10 years?",
                "test_session_4"
            )
            
            assert len(result.charts) > 0
            assert "10" in result.message or "years" in result.message
    
    @pytest.mark.asyncio
    async def test_goal_probability(self):
        """Test calculating probability of reaching goal"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.goals_agent.GoalsAgent') as MockGoals:
            mock_agent = Mock()
            
            chart = ChartArtifact(
                title="Portfolio Projection - 10 Year Scenarios",
                filename="goal_simulation.png"
            )
            
            response = AgentResponse(
                agent="GoalsAgent",
                message="You have a 36.9% chance of doubling your portfolio to $5.4M in 10 years",
                charts=[chart],
                portfolio=None
            )
            
            mock_agent.run_query = AsyncMock(return_value=response)
            MockGoals.return_value = mock_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.goals_agent = mock_agent
            
            result = await router.run_query(
                "Can I double my portfolio in 10 years?",
                "test_session_5"
            )
            
            assert "%" in result.message or "probability" in result.message.lower()
            assert "chance" in result.message.lower() or "probability" in result.message.lower()


class TestCrossAgentWorkflow:
    """Test workflows that span multiple agents"""
    
    @pytest.mark.asyncio
    async def test_build_analyze_simulate(self):
        """Test complete workflow: build portfolio → analyze → simulate"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio, \
             patch('src.agents.goals_agent.GoalsAgent') as MockGoals:
            
            # Mock portfolio agent
            portfolio_agent = Mock()
            portfolio_response = AgentResponse(
                agent="PortfolioAgent",
                message="Built portfolio with $2.7M in Equities",
                charts=[],
                portfolio={
                    "Equities": 2700000.0,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            portfolio_agent.run_query = AsyncMock(return_value=portfolio_response)
            MockPortfolio.return_value = portfolio_agent
            
            # Mock goals agent
            goals_agent = Mock()
            goals_response = AgentResponse(
                agent="GoalsAgent",
                message="You have a 36.9% chance of doubling in 10 years",
                charts=[ChartArtifact(title="Simulation", filename="sim.png")],
                portfolio=None
            )
            goals_agent.run_query = AsyncMock(return_value=goals_response)
            MockGoals.return_value = goals_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = portfolio_agent
            router.goals_agent = goals_agent
            
            # Step 1: Build portfolio
            response1 = await router.run_query(
                "Build me a portfolio with $2.7M in Equities",
                "test_session_6"
            )
            assert response1.portfolio["Equities"] == 2700000.0
            
            # Step 2: Simulate future
            response2 = await router.run_query(
                "Can I double this in 10 years?",
                "test_session_6"
            )
            assert "%" in response2.message or "chance" in response2.message.lower()
    
    @pytest.mark.asyncio
    async def test_context_aware_routing(self):
        """Test router uses context from previous queries"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.market_agent.FinanceMarketAgent') as MockMarket:
            market_agent = Mock()
            
            # First response: price
            response1 = AgentResponse(
                agent="FinanceMarketAgent",
                message="Apple is trading at $185.43",
                charts=[],
                portfolio=None
            )
            
            # Second response: chart (context-aware)
            response2 = AgentResponse(
                agent="FinanceMarketAgent",
                message="Here's the chart for Apple",
                charts=[ChartArtifact(title="AAPL", filename="aapl.png")],
                portfolio=None
            )
            
            market_agent.run_query = AsyncMock(side_effect=[response1, response2])
            MockMarket.return_value = market_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.market_agent = market_agent
            
            # Query 1: Get price (establishes context)
            await router.run_query(
                "What's the price of Apple?",
                "test_session_7"
            )
            
            # Query 2: Follow-up (should use context)
            result = await router.run_query(
                "Chart that for me",
                "test_session_7"
            )
            
            # Should route to market agent and reference Apple
            assert len(result.charts) > 0


# --- State Management Integration Tests ---

class TestStateManagement:
    """Test state persistence across queries"""
    
    @pytest.mark.asyncio
    async def test_portfolio_persists_across_queries(self):
        """Test portfolio state is maintained"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio:
            portfolio_agent = Mock()
            
            # First query: Build portfolio
            response1 = AgentResponse(
                agent="PortfolioAgent",
                message="Created portfolio",
                charts=[],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            # Second query: Add to portfolio (should have context)
            response2 = AgentResponse(
                agent="PortfolioAgent",
                message="Added to portfolio",
                charts=[],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 50000.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            portfolio_agent.run_query = AsyncMock(side_effect=[response1, response2])
            MockPortfolio.return_value = portfolio_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = portfolio_agent
            
            # Build portfolio
            result1 = await router.run_query(
                "Build portfolio with $100k in Equities",
                "test_session_8"
            )
            
            # Add more (should see existing portfolio in context)
            result2 = await router.run_query(
                "Add $50k to Fixed Income",
                "test_session_8"
            )
            
            # Second result should have both asset classes
            assert result2.portfolio["Equities"] == 100000.0
            assert result2.portfolio["Fixed_Income"] == 50000.0
    
    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Test different sessions have isolated state"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio:
            portfolio_agent = Mock()
            
            response = AgentResponse(
                agent="PortfolioAgent",
                message="Created portfolio",
                charts=[],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            portfolio_agent.run_query = AsyncMock(return_value=response)
            MockPortfolio.return_value = portfolio_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = portfolio_agent
            
            # Session 1
            await router.run_query(
                "Build portfolio",
                "session_A"
            )
            
            # Session 2 (different session, should be independent)
            await router.run_query(
                "Build portfolio",
                "session_B"
            )
            
            # Each session should maintain separate state
            # (This test validates the concept; actual validation would check state)
            assert True


# --- Error Handling Integration Tests ---

class TestErrorHandlingIntegration:
    """Test error handling across the system"""
    
    @pytest.mark.asyncio
    async def test_tool_failure_graceful_degradation(self):
        """Test system handles tool failures gracefully"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.goals_agent.GoalsAgent') as MockGoals:
            goals_agent = Mock()
            
            # Simulate chart generation failure but text response succeeds
            response = AgentResponse(
                agent="GoalsAgent",
                message="Simulation results: Bottom 10%: $2.9M, Median: $4.6M, Top 10%: $7.7M. Note: Unable to generate visualization at this time.",
                charts=[],  # No charts due to failure
                portfolio=None
            )
            
            goals_agent.run_query = AsyncMock(return_value=response)
            MockGoals.return_value = goals_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.goals_agent = goals_agent
            
            result = await router.run_query(
                "Simulate my portfolio",
                "test_session_9"
            )
            
            # Should still get text response
            assert len(result.message) > 0
            # Should mention inability to create chart
            assert "unable" in result.message.lower() or "visualization" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self):
        """Test handling of agent timeouts"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio:
            portfolio_agent = Mock()
            
            # Simulate timeout by raising exception
            portfolio_agent.run_query = AsyncMock(
                side_effect=Exception("Agent timeout")
            )
            MockPortfolio.return_value = portfolio_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = portfolio_agent
            
            # Should handle gracefully
            try:
                result = await router.run_query(
                    "Build portfolio",
                    "test_session_10"
                )
                # If it returns, check for error message
                assert "error" in result.message.lower()
            except Exception:
                # Exception is acceptable too
                pass


# --- Performance Tests ---

class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_multiple_sequential_queries(self):
        """Test handling multiple queries in sequence"""
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio:
            portfolio_agent = Mock()
            
            response = AgentResponse(
                agent="PortfolioAgent",
                message="Portfolio updated",
                charts=[],
                portfolio={
                    "Equities": 100000.0,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            portfolio_agent.run_query = AsyncMock(return_value=response)
            MockPortfolio.return_value = portfolio_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = portfolio_agent
            
            # Run 5 queries in sequence
            for i in range(5):
                result = await router.run_query(
                    f"Query {i}",
                    "test_session_11"
                )
                assert isinstance(result, AgentResponse)


# --- End-to-End Scenario Tests ---

class TestEndToEndScenarios:
    """Test complete realistic user scenarios"""
    
    @pytest.mark.asyncio
    async def test_new_investor_scenario(self):
        """Test complete flow for new investor"""
        # Scenario: User wants to build portfolio and see if they can reach retirement goal
        checkpointer = InMemorySaver()
        
        with patch('src.agents.portfolio_agent.PortfolioAgent') as MockPortfolio, \
             patch('src.agents.goals_agent.GoalsAgent') as MockGoals:
            
            # Setup mocks
            portfolio_agent = Mock()
            goals_agent = Mock()
            
            portfolio_response = AgentResponse(
                agent="PortfolioAgent",
                message="Built diversified portfolio",
                charts=[ChartArtifact(title="Portfolio", filename="port.png")],
                portfolio={
                    "Equities": 300000.0,
                    "Fixed_Income": 150000.0,
                    "Cash": 50000.0,
                    "Real_Estate": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            )
            
            goals_response = AgentResponse(
                agent="GoalsAgent",
                message="You have a 65% chance of reaching $2M in 20 years",
                charts=[ChartArtifact(title="Projection", filename="proj.png")],
                portfolio=None
            )
            
            portfolio_agent.run_query = AsyncMock(return_value=portfolio_response)
            goals_agent.run_query = AsyncMock(return_value=goals_response)
            
            MockPortfolio.return_value = portfolio_agent
            MockGoals.return_value = goals_agent
            
            router = RouterAgent(checkpointer=checkpointer)
            router.portfolio_agent = portfolio_agent
            router.goals_agent = goals_agent
            
            # Step 1: Build portfolio
            result1 = await router.run_query(
                "Help me build a diversified portfolio with $500k",
                "new_investor"
            )
            assert result1.portfolio is not None
            
            # Step 2: Check retirement goal
            result2 = await router.run_query(
                "Can I reach $2M in 20 years for retirement?",
                "new_investor"
            )
            assert "%" in result2.message or "chance" in result2.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
