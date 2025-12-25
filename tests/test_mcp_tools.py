# tests/test_mcp_tools.py
"""
Real tests for MCP server tool functions.
These tests import the actual tool functions and test their logic directly.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path so we can import MCP modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Portfolio MCP Tests
# ============================================================================

class TestPortfolioMCP:
    """Test portfolio_mcp.py tool functions"""
    
    def test_get_new_portfolio(self):
        """Test get_new_portfolio returns empty portfolio"""
        # Import the MCP module to get the wrapped function
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        # Access the underlying function from the FunctionTool
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        
        portfolio = get_new_portfolio()
        
        assert isinstance(portfolio, dict)
        assert len(portfolio) == 6
        assert all(v == 0.0 for v in portfolio.values())
        assert "Equities" in portfolio
        assert "Fixed_Income" in portfolio
        assert "Cash" in portfolio
        assert "Real_Estate" in portfolio
        assert "Commodities" in portfolio
        assert "Crypto" in portfolio
    
    def test_add_to_portfolio_asset_class(self):
        """Test adding to a single asset class"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        add_to_portfolio_asset_class = portfolio_mcp.add_to_portfolio_asset_class.fn
        
        portfolio = get_new_portfolio()
        
        # Add to Equities
        result = add_to_portfolio_asset_class("Equities", 100000, portfolio)
        
        assert result["Equities"] == 100000.0
        assert result["Fixed_Income"] == 0.0
        assert sum(result.values()) == 100000.0
    
    def test_add_to_portfolio_asset_class_negative(self):
        """Test removing from portfolio (negative amount)"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        add_to_portfolio_asset_class = portfolio_mcp.add_to_portfolio_asset_class.fn
        
        portfolio = get_new_portfolio()
        portfolio["Equities"] = 100000.0
        
        # Remove half
        result = add_to_portfolio_asset_class("Equities", -50000, portfolio)
        
        assert result["Equities"] == 50000.0
    
    def test_add_to_portfolio_asset_class_invalid_key(self):
        """Test adding to invalid asset class raises error"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        add_to_portfolio_asset_class = portfolio_mcp.add_to_portfolio_asset_class.fn
        
        portfolio = get_new_portfolio()
        
        with pytest.raises(ValueError):
            add_to_portfolio_asset_class("InvalidAsset", 100000, portfolio)
    
    def test_add_to_portfolio(self):
        """Test adding multiple asset classes atomically"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        add_to_portfolio = portfolio_mcp.add_to_portfolio.fn
        
        portfolio = get_new_portfolio()
        additions = {
            "Equities": 100000,
            "Fixed_Income": 50000,
            "Cash": 25000
        }
        
        result = add_to_portfolio(portfolio, additions)
        
        assert result["Equities"] == 100000
        assert result["Fixed_Income"] == 50000
        assert result["Cash"] == 25000
        assert result["Real_Estate"] == 0
        assert sum(result.values()) == 175000
    
    def test_add_to_portfolio_with_allocation(self):
        """Test adding with asset allocation"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        add_to_portfolio_with_allocation = portfolio_mcp.add_to_portfolio_with_allocation.fn
        
        portfolio = get_new_portfolio()
        allocation = {
            "Equities": 0.6,
            "Fixed_Income": 0.35,
            "Cash": 0.05
        }
        
        result = add_to_portfolio_with_allocation(500000, portfolio, allocation)
        
        assert result["Equities"] == 300000
        assert result["Fixed_Income"] == 175000
        assert result["Cash"] == 25000
        assert abs(sum(result.values()) - 500000) < 0.01  # Floating point tolerance
    
    def test_get_portfolio_summary(self):
        """Test portfolio summary calculation"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_portfolio_summary = portfolio_mcp.get_portfolio_summary.fn
        
        portfolio = {
            "Equities": 500000,
            "Fixed_Income": 300000,
            "Real_Estate": 0,
            "Cash": 200000,
            "Commodities": 0,
            "Crypto": 0
        }
        
        summary = get_portfolio_summary(portfolio)
        
        assert summary["success"] is True
        assert summary["total_value"] == 1000000
        assert summary["asset_count"] == 3
        assert summary["asset_percentages"]["Equities"] == 50.0
        assert summary["asset_percentages"]["Fixed_Income"] == 30.0
        assert summary["asset_percentages"]["Cash"] == 20.0
    
    def test_get_portfolio_summary_empty(self):
        """Test summary for empty portfolio"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        get_portfolio_summary = portfolio_mcp.get_portfolio_summary.fn
        
        portfolio = get_new_portfolio()
        summary = get_portfolio_summary(portfolio)
        
        assert summary["total_value"] == 0
        assert summary["asset_count"] == 0
        assert all(v == 0 for v in summary["asset_percentages"].values())
    
    def test_assess_risk_tolerance_conservative(self):
        """Test risk assessment for conservative portfolio"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        assess_risk_tolerance = portfolio_mcp.assess_risk_tolerance.fn
        
        # Conservative: mostly bonds and cash
        portfolio = {
            "Equities": 20000,
            "Fixed_Income": 60000,
            "Real_Estate": 0,
            "Cash": 20000,
            "Commodities": 0,
            "Crypto": 0
        }
        
        risk = assess_risk_tolerance(portfolio)
        
        assert "Conservative" in risk["risk_tolerance_tier"]
        assert "weighted_volatility" in risk
    
    def test_assess_risk_tolerance_aggressive(self):
        """Test risk assessment for aggressive portfolio"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        assess_risk_tolerance = portfolio_mcp.assess_risk_tolerance.fn
        
        # Aggressive: mostly equities and crypto
        portfolio = {
            "Equities": 60000,
            "Fixed_Income": 10000,
            "Real_Estate": 10000,
            "Cash": 0,
            "Commodities": 0,
            "Crypto": 20000
        }
        
        risk = assess_risk_tolerance(portfolio)
        
        assert "Aggressive" in risk["risk_tolerance_tier"]
        assert risk["primary_risk_driver"] == "Equities"


# ============================================================================
# Charts MCP Tests
# ============================================================================

class TestChartsMCP:
    """Test charts_mcp.py tool functions"""
    
    def test_generate_chart_id_deterministic(self):
        """Test chart ID generation is deterministic"""
        from src.mcp.charts_mcp import generate_chart_id
        
        data1 = {"labels": ["A", "B"], "values": [1, 2]}
        data2 = {"labels": ["A", "B"], "values": [1, 2]}
        data3 = {"labels": ["A", "B"], "values": [2, 3]}
        
        id1 = generate_chart_id("pie", data1)
        id2 = generate_chart_id("pie", data2)
        id3 = generate_chart_id("pie", data3)
        
        assert id1 == id2  # Same data should produce same ID
        assert id1 != id3  # Different data should produce different ID
        assert len(id1) == 12  # ID should be 12 characters
    
    def test_validate_data_lengths_matching(self):
        """Test validate_data_lengths with matching arrays"""
        from src.mcp.charts_mcp import validate_data_lengths
        
        arr1 = [1, 2, 3]
        arr2 = [4, 5, 6]
        
        result1, result2 = validate_data_lengths(arr1, arr2)
        
        assert result1 == arr1
        assert result2 == arr2
    
    def test_validate_data_lengths_mismatched(self):
        """Test validate_data_lengths truncates to shortest"""
        from src.mcp.charts_mcp import validate_data_lengths
        
        arr1 = [1, 2, 3, 4, 5]
        arr2 = [6, 7, 8]
        
        result1, result2 = validate_data_lengths(arr1, arr2)
        
        assert len(result1) == 3
        assert len(result2) == 3
        assert result1 == [1, 2, 3]
        assert result2 == [6, 7, 8]
    
    @patch('src.mcp.charts_mcp.plt')
    def test_create_pie_chart(self, mock_plt):
        """Test pie chart creation (mocking matplotlib)"""
        import src.mcp.charts_mcp as charts_mcp
        
        create_pie_chart = charts_mcp.create_pie_chart.fn
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.pie.return_value = ([], [], [])
        
        labels = ["Equities", "Fixed Income", "Cash"]
        values = [50, 30, 20]
        
        result = create_pie_chart(
            labels=labels,
            values=values,
            title="Test Portfolio",
            use_cache=False
        )
        
        assert "chart_id" in result
        assert "filename" in result
        assert result["chart_type"] == "pie"
        assert result["title"] == "Test Portfolio"
    
    def test_create_pie_chart_filters_zeros(self):
        """Test pie chart filters out zero values"""
        import src.mcp.charts_mcp as charts_mcp
        
        create_pie_chart = charts_mcp.create_pie_chart.fn
        
        labels = ["Equities", "Fixed Income", "Cash", "Crypto"]
        values = [50, 30, 0, 20]
        
        with patch('src.mcp.charts_mcp.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_ax.pie.return_value = ([], [], [])
            
            result = create_pie_chart(labels, values, use_cache=False)
            
            # Should only have 3 non-zero values
            call_args = mock_ax.pie.call_args
            assert len(call_args[1]['labels']) == 3


# ============================================================================
# Goals MCP Tests
# ============================================================================

class TestGoalsMCP:
    """Test goals_mcp.py tool functions"""
    
    def test_simple_monte_carlo_simulation_structure(self):
        """Test Monte Carlo simulation returns correct structure"""
        import src.mcp.goals_mcp as goals_mcp
        
        simple_monte_carlo_simulation = goals_mcp.simple_monte_carlo_simulation.fn
        
        portfolio = {
            "Equities": 100000,
            "Fixed_Income": 0,
            "Real_Estate": 0,
            "Cash": 0,
            "Commodities": 0,
            "Crypto": 0
        }
        
        result = simple_monte_carlo_simulation(
            portfolio=portfolio,
            target_goal=200000,
            years=10,
            sims=100  # Use fewer sims for faster test
        )
        
        # Check structure
        assert "goal_analysis" in result
        assert "median_scenario" in result
        assert "bottom_10_percent_scenario" in result
        assert "top_10_percent_scenario" in result
        
        # Check goal analysis
        assert result["goal_analysis"]["target"] == 200000
        assert "%" in result["goal_analysis"]["success_probability"]
        
        # Check scenarios have totals and portfolios
        for scenario_key in ["median_scenario", "bottom_10_percent_scenario", "top_10_percent_scenario"]:
            assert "total" in result[scenario_key]
            assert "portfolio" in result[scenario_key]
            assert isinstance(result[scenario_key]["portfolio"], dict)
    
    def test_simple_monte_carlo_portfolio_conservation(self):
        """Test that portfolio asset classes are maintained"""
        import src.mcp.goals_mcp as goals_mcp
        
        simple_monte_carlo_simulation = goals_mcp.simple_monte_carlo_simulation.fn
        
        portfolio = {
            "Equities": 100000,
            "Fixed_Income": 50000,
            "Real_Estate": 0,
            "Cash": 25000,
            "Commodities": 0,
            "Crypto": 0
        }
        
        result = simple_monte_carlo_simulation(
            portfolio=portfolio,
            years=5,
            sims=100
        )
        
        # Check that all asset classes are in the result portfolios
        median = result["median_scenario"]["portfolio"]
        assert set(median.keys()) == set(portfolio.keys())
    
    def test_get_asset_classes(self):
        """Test get_asset_classes returns correct keys"""
        import src.mcp.goals_mcp as goals_mcp
        
        get_asset_classes = goals_mcp.get_asset_classes.fn
        
        asset_classes = get_asset_classes()
        
        assert isinstance(asset_classes, list)
        assert len(asset_classes) == 6
        assert "Equities" in asset_classes
        assert "Fixed_Income" in asset_classes
        assert "Cash" in asset_classes
        assert "Real_Estate" in asset_classes
        assert "Commodities" in asset_classes
        assert "Crypto" in asset_classes


# ============================================================================
# Q&A MCP Tests (without loading actual FAISS indexes)
# ============================================================================

class TestQAMCP:
    """Test qa_mcp.py tool functions (mocking FAISS)"""
    
    def test_list_categories(self):
        """Test list_categories returns expected categories"""
        import src.mcp.finance_q_and_a_mcp as qanda_mcp
        
        list_categories = qanda_mcp.list_categories.fn
        
        categories = list_categories()
        
        assert isinstance(categories, list)
        assert "Investing" in categories
        assert "Retirement" in categories
        assert "Tax" in categories
    
    def test_list_advanced_categories(self):
        """Test list_advanced_categories"""
        import src.mcp.finance_q_and_a_mcp as qanda_mcp
        
        list_advanced_categories = qanda_mcp.list_advanced_categories.fn
        
        categories = list_advanced_categories()
        
        assert isinstance(categories, list)
        assert "Investing" in categories
        assert "Retirement Planning" in categories


# ============================================================================
# yFinance MCP Tests (with mocking)
# ============================================================================

class TestYFinanceMCP:
    """Test yfinance_mcp.py tool functions"""
    
    def test_normalize_time_period_valid(self):
        """Test normalize_time_period with valid periods"""
        from src.mcp.yfinance_mcp import normalize_time_period
        
        assert normalize_time_period("1mo") == "1mo"
        assert normalize_time_period("1y") == "1y"
        assert normalize_time_period("max") == "max"
    
    def test_normalize_time_period_conversion(self):
        """Test normalize_time_period converts to nearest valid period"""
        from src.mcp.yfinance_mcp import normalize_time_period
        
        assert normalize_time_period("2mo") == "3mo"
        assert normalize_time_period("10y") == "max"
        assert normalize_time_period("3d") == "5d"
        assert normalize_time_period("2y") == "5y"
        assert normalize_time_period("15 days") == "1mo"
    
    def test_normalize_time_period_weeks(self):
        """Test normalize_time_period with weeks"""
        from src.mcp.yfinance_mcp import normalize_time_period
        
        assert normalize_time_period("2w") == "1mo"
        assert normalize_time_period("52 weeks") == "1y"
    
    def test_get_mock_data(self):
        """Test get_mock_data returns valid structure"""
        from src.mcp.yfinance_mcp import get_mock_data
        
        data = get_mock_data("AAPL")
        
        assert data["_mock"] is True
        assert "price" in data
        assert "change" in data
        assert "percent_change" in data
        assert "_timestamp" in data
    
    @patch('src.mcp.yfinance_mcp.yf.Ticker')
    def test_get_ticker_quote_success(self, mock_ticker_class):
        """Test get_ticker_quote with successful API call"""
        import src.mcp.yfinance_mcp as yfinance_mcp
        
        get_ticker_quote = yfinance_mcp.get_ticker_quote.fn
        
        # Mock ticker info
        mock_ticker = Mock()
        mock_ticker.info = {
            "symbol": "AAPL",
            "currentPrice": 185.50,
            "regularMarketPrice": 185.50,
            "regularMarketChange": 2.30,
            "regularMarketChangePercent": 1.25,
            "volume": 50000000,
            "marketCap": 2900000000000,
            "trailingPE": 29.5,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 164.08,
            "longName": "Apple Inc.",
            "currency": "USD"
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = get_ticker_quote("AAPL", use_cache=False)
        
        assert result["symbol"] == "AAPL"
        assert result["price"] == 185.50
        assert result["_mock"] is False
    
    @patch('src.mcp.yfinance_mcp.yf.Ticker')
    def test_get_ticker_quote_failure_returns_mock(self, mock_ticker_class):
        """Test get_ticker_quote returns mock data on failure"""
        import src.mcp.yfinance_mcp as yfinance_mcp
        
        get_ticker_quote = yfinance_mcp.get_ticker_quote.fn
        
        # Mock ticker to raise exception
        mock_ticker_class.side_effect = Exception("API Error")
        
        result = get_ticker_quote("INVALID", use_cache=False)
        
        assert result["_mock"] is True
        assert "error" in result


# ============================================================================
# Integration-style tests
# ============================================================================

class TestMCPIntegration:
    """Test interactions between MCP tools"""
    
    def test_portfolio_workflow(self):
        """Test complete portfolio building workflow"""
        import src.mcp.portfolio_mcp as portfolio_mcp
        
        get_new_portfolio = portfolio_mcp.get_new_portfolio.fn
        add_to_portfolio_asset_class = portfolio_mcp.add_to_portfolio_asset_class.fn
        get_portfolio_summary = portfolio_mcp.get_portfolio_summary.fn
        assess_risk_tolerance = portfolio_mcp.assess_risk_tolerance.fn
        
        # Step 1: Create portfolio
        portfolio = get_new_portfolio()
        assert sum(portfolio.values()) == 0
        
        # Step 2: Add assets
        portfolio = add_to_portfolio_asset_class("Equities", 100000, portfolio)
        portfolio = add_to_portfolio_asset_class("Fixed_Income", 50000, portfolio)
        
        # Step 3: Get summary
        summary = get_portfolio_summary(portfolio)
        assert summary["total_value"] == 150000
        assert summary["asset_count"] == 2
        
        # Step 4: Assess risk
        risk = assess_risk_tolerance(portfolio)
        assert "risk_tolerance_tier" in risk
    
    def test_chart_data_validation(self):
        """Test chart tools validate data correctly"""
        from src.mcp.charts_mcp import validate_data_lengths
        
        # Test with mismatched lengths
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30]
        
        x_valid, y_valid = validate_data_lengths(x, y)
        
        # Should truncate to shortest
        assert len(x_valid) == 3
        assert len(y_valid) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
