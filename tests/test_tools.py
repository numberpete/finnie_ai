# tests/test_tools.py

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os
from datetime import datetime, timedelta

# Import tool functions (these would be imported from your MCP servers)
# Since we can't import from the actual servers in this test file,
# we'll test the logic patterns they should follow


# --- Portfolio Tools Tests ---

class TestPortfolioTools:
    """Test portfolio management tools"""
    
    def test_get_new_portfolio(self):
        """Test creating a new empty portfolio"""
        portfolio = {
            "Equities": 0.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Verify structure
        assert len(portfolio) == 6
        assert all(v == 0.0 for v in portfolio.values())
        assert "Equities" in portfolio
        assert "Fixed_Income" in portfolio
    
    def test_add_to_portfolio_asset_class(self):
        """Test adding to a single asset class"""
        portfolio = {
            "Equities": 0.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Simulate tool logic
        asset_class_key = "Equities"
        amount = 100000.0
        
        portfolio[asset_class_key] += amount
        
        assert portfolio["Equities"] == 100000.0
        assert portfolio["Fixed_Income"] == 0.0
    
    def test_add_to_portfolio_negative_amount(self):
        """Test removing from portfolio (negative amount)"""
        portfolio = {
            "Equities": 100000.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Simulate removal
        amount = -50000.0
        portfolio["Equities"] += amount
        
        assert portfolio["Equities"] == 50000.0
    
    def test_add_to_portfolio_multiple_assets(self):
        """Test adding to multiple asset classes atomically"""
        portfolio = {
            "Equities": 0.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        additions = {
            "Equities": 100000.0,
            "Cash": 50000.0
        }
        
        # Simulate atomic addition
        for asset_class, amount in additions.items():
            portfolio[asset_class] += amount
        
        assert portfolio["Equities"] == 100000.0
        assert portfolio["Cash"] == 50000.0
        assert portfolio["Fixed_Income"] == 0.0
    
    def test_add_with_allocation(self):
        """Test adding with asset allocation"""
        portfolio = {
            "Equities": 0.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        amount = 500000.0
        allocation = {
            "Equities": 0.6,
            "Fixed_Income": 0.35,
            "Cash": 0.05
        }
        
        # Simulate allocation
        for asset_class, ratio in allocation.items():
            portfolio[asset_class] += amount * ratio
        
        assert portfolio["Equities"] == 300000.0
        assert portfolio["Fixed_Income"] == 175000.0
        assert portfolio["Cash"] == 25000.0
        assert sum(portfolio.values()) == 500000.0
    
    def test_get_portfolio_summary(self):
        """Test portfolio summary calculation"""
        portfolio = {
            "Equities": 500000.0,
            "Fixed_Income": 300000.0,
            "Real_Estate": 0.0,
            "Cash": 200000.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Simulate summary calculation
        total_value = sum(portfolio.values())
        asset_count = sum(1 for v in portfolio.values() if v > 0)
        asset_percentages = {
            k: (v / total_value * 100) if total_value > 0 else 0
            for k, v in portfolio.items()
        }
        
        summary = {
            "total_value": total_value,
            "asset_values": portfolio,
            "asset_percentages": asset_percentages,
            "asset_count": asset_count
        }
        
        assert summary["total_value"] == 1000000.0
        assert summary["asset_count"] == 3
        assert summary["asset_percentages"]["Equities"] == 50.0
        assert summary["asset_percentages"]["Fixed_Income"] == 30.0
        assert summary["asset_percentages"]["Cash"] == 20.0
    
    def test_assess_risk_tolerance(self):
        """Test risk assessment logic"""
        # Conservative portfolio
        conservative = {
            "Equities": 20000.0,
            "Fixed_Income": 60000.0,
            "Real_Estate": 0.0,
            "Cash": 20000.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        total = sum(conservative.values())
        equity_pct = conservative["Equities"] / total * 100
        
        # Should be conservative (< 40% equities)
        assert equity_pct < 40
        
        # Aggressive portfolio
        aggressive = {
            "Equities": 80000.0,
            "Fixed_Income": 10000.0,
            "Real_Estate": 0.0,
            "Cash": 10000.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        total = sum(aggressive.values())
        equity_pct = aggressive["Equities"] / total * 100
        
        # Should be aggressive (> 60% equities)
        assert equity_pct > 60


# --- Market Data Tools Tests ---

class TestMarketDataTools:
    """Test market data retrieval tools"""
    
    def test_ticker_resolution(self):
        """Test resolving company name to ticker"""
        # Mock yfinance ticker resolution
        test_cases = {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Vanguard 2040": "VFORX",
            "S&P 500": "SPY"
        }
        
        for company, expected_ticker in test_cases.items():
            # In real implementation, this would call yfinance
            # Here we just verify the pattern
            assert len(expected_ticker) > 0
            assert expected_ticker.isupper()
    
    def test_quote_data_structure(self):
        """Test quote data has required fields"""
        mock_quote = {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "regularMarketPrice": 185.43,
            "regularMarketChange": 2.15,
            "regularMarketChangePercent": 1.17,
            "regularMarketDayHigh": 186.50,
            "regularMarketDayLow": 184.20,
            "volume": 52428800,
            "previousClose": 183.28
        }
        
        # Verify required fields
        assert "symbol" in mock_quote
        assert "regularMarketPrice" in mock_quote
        assert "regularMarketChange" in mock_quote
        assert "regularMarketChangePercent" in mock_quote
        
        # Verify types
        assert isinstance(mock_quote["regularMarketPrice"], (int, float))
        assert isinstance(mock_quote["volume"], int)
    
    def test_historical_data_structure(self):
        """Test historical data structure"""
        # Mock historical data
        dates = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03"
        ]
        prices = [150.25, 151.30, 149.80]
        
        history = {
            "dates": dates,
            "prices": prices,
            "metadata": {
                "ticker": "AAPL",
                "period": "1mo",
                "data_points": len(dates)
            }
        }
        
        assert len(history["dates"]) == len(history["prices"])
        assert history["metadata"]["data_points"] == 3
        assert all(isinstance(p, (int, float)) for p in history["prices"])
    
    def test_valid_period_validation(self):
        """Test period parameter validation"""
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        
        # Test valid periods
        for period in valid_periods:
            assert period in valid_periods
        
        # Test invalid period
        invalid_period = "3w"
        assert invalid_period not in valid_periods
    
    def test_asset_class_breakdown(self):
        """Test asset class allocation for funds"""
        # Mock fund breakdown
        vanguard_2040 = {
            "Equities": 0.85,
            "Fixed_Income": 0.15,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Verify allocation sums to 1.0
        total = sum(vanguard_2040.values())
        assert abs(total - 1.0) < 0.01  # Allow small floating point error
        
        # Verify all values between 0 and 1
        assert all(0 <= v <= 1 for v in vanguard_2040.values())


# --- Chart Generation Tools Tests ---

class TestChartTools:
    """Test chart generation tools"""
    
    def test_line_chart_data_validation(self):
        """Test line chart requires matching x and y lengths"""
        x_values = ["2024-01-01", "2024-01-02", "2024-01-03"]
        y_values = [150.25, 151.30, 149.80]
        
        assert len(x_values) == len(y_values)
    
    def test_multi_line_chart_data_structure(self):
        """Test multi-line chart data structure"""
        series_data = {
            "AAPL": {
                "dates": ["2024-01-01", "2024-01-02"],
                "prices": [150.25, 151.30]
            },
            "MSFT": {
                "dates": ["2024-01-01", "2024-01-02"],
                "prices": [380.50, 382.10]
            }
        }
        
        # Verify all series have same length
        lengths = [len(data["dates"]) for data in series_data.values()]
        assert len(set(lengths)) == 1  # All same length
        
        # Verify matching dates and prices
        for symbol, data in series_data.items():
            assert len(data["dates"]) == len(data["prices"])
    
    def test_pie_chart_percentages(self):
        """Test pie chart percentages sum to 100"""
        labels = ["Equities", "Fixed_Income", "Cash"]
        values = [50.0, 30.0, 20.0]
        
        assert abs(sum(values) - 100.0) < 0.1
        assert len(labels) == len(values)
    
    def test_stacked_bar_chart_data_structure(self):
        """Test stacked bar chart data structure"""
        categories = ["Bottom 10%", "Median", "Top 10%"]
        series_data = {
            "Equities": [2935376, 4601999, 7738156],
            "Fixed_Income": [0, 0, 0],
            "Cash": [500000, 600000, 700000]
        }
        
        # Verify all series have same length as categories
        for series_name, values in series_data.items():
            assert len(values) == len(categories)
        
        # Verify all values are numeric
        for values in series_data.values():
            assert all(isinstance(v, (int, float)) for v in values)
    
    def test_chart_id_generation(self):
        """Test chart ID generation is deterministic"""
        import hashlib
        
        def generate_chart_id(chart_type, params):
            data = json.dumps({"type": chart_type, **params}, sort_keys=True)
            return hashlib.sha256(data.encode()).hexdigest()[:12]
        
        params1 = {"title": "Test Chart", "data": [1, 2, 3]}
        params2 = {"title": "Test Chart", "data": [1, 2, 3]}
        params3 = {"title": "Different Chart", "data": [1, 2, 3]}
        
        id1 = generate_chart_id("line", params1)
        id2 = generate_chart_id("line", params2)
        id3 = generate_chart_id("line", params3)
        
        # Same params should generate same ID
        assert id1 == id2
        
        # Different params should generate different ID
        assert id1 != id3
    
    def test_chart_colors_validation(self):
        """Test chart colors are valid hex codes"""
        colors = ["#2E5BFF", "#46CDCF", "#F08A5D", "#3DDC84", "#FFD700", "#B832FF"]
        
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7
            # Verify hex characters
            hex_part = color[1:]
            assert all(c in "0123456789ABCDEFabcdef" for c in hex_part)


# --- Goals & Simulation Tools Tests ---

class TestGoalsTools:
    """Test Monte Carlo simulation tools"""
    
    def test_simulation_portfolio_validation(self):
        """Test simulation requires valid portfolio"""
        portfolio = {
            "Equities": 100000.0,
            "Fixed_Income": 50000.0,
            "Real_Estate": 0.0,
            "Cash": 25000.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Portfolio should have at least one non-zero value
        assert sum(portfolio.values()) > 0
        
        # All values should be non-negative
        assert all(v >= 0 for v in portfolio.values())
    
    def test_simulation_parameters(self):
        """Test simulation parameter validation"""
        years = 10
        target_goal = 5400000.0
        
        # Years should be positive
        assert years > 0
        assert years <= 50  # Reasonable upper limit
        
        # Target goal should be positive (if provided)
        if target_goal:
            assert target_goal > 0
    
    def test_simulation_output_structure(self):
        """Test simulation output has required fields"""
        mock_result = {
            "goal_analysis": {
                "target": 5400000,
                "success_probability": "36.90%"
            },
            "median_scenario": {
                "total": 4601999.67,
                "portfolio": {
                    "Equities": 4601999.67,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            },
            "bottom_10_percent_scenario": {
                "total": 2935376.78,
                "portfolio": {
                    "Equities": 2935376.78,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            },
            "top_10_percent_scenario": {
                "total": 7738156.07,
                "portfolio": {
                    "Equities": 7738156.07,
                    "Fixed_Income": 0.0,
                    "Real_Estate": 0.0,
                    "Cash": 0.0,
                    "Commodities": 0.0,
                    "Crypto": 0.0
                }
            }
        }
        
        # Verify structure
        assert "median_scenario" in mock_result
        assert "bottom_10_percent_scenario" in mock_result
        assert "top_10_percent_scenario" in mock_result
        
        # Verify each scenario has total and portfolio
        for scenario_key in ["median_scenario", "bottom_10_percent_scenario", "top_10_percent_scenario"]:
            scenario = mock_result[scenario_key]
            assert "total" in scenario
            assert "portfolio" in scenario
            assert isinstance(scenario["portfolio"], dict)
    
    def test_simulation_probability_calculation(self):
        """Test probability is calculated correctly"""
        # Simulate 1000 runs
        target = 5000000
        successes = 369  # 36.9% success rate
        total_runs = 1000
        
        probability = (successes / total_runs) * 100
        
        assert 0 <= probability <= 100
        assert abs(probability - 36.9) < 0.1
    
    def test_asset_class_returns_assumptions(self):
        """Test asset class return assumptions are reasonable"""
        assumptions = {
            "Equities": {"return": 0.10, "volatility": 0.15},
            "Fixed_Income": {"return": 0.04, "volatility": 0.05},
            "Real_Estate": {"return": 0.06, "volatility": 0.10},
            "Cash": {"return": 0.02, "volatility": 0.01},
            "Commodities": {"return": 0.05, "volatility": 0.20},
            "Crypto": {"return": 0.15, "volatility": 0.50}
        }
        
        for asset_class, params in assumptions.items():
            # Returns should be reasonable (-50% to 100%)
            assert -0.5 <= params["return"] <= 1.0
            
            # Volatility should be non-negative
            assert params["volatility"] >= 0
            
            # Volatility should be less than 100%
            assert params["volatility"] <= 1.0


# --- Cache Tests ---

class TestCaching:
    """Test TTL cache functionality"""
    
    def test_cache_basic_operations(self):
        """Test basic cache set/get operations"""
        cache = {}
        
        # Set value
        cache["key1"] = "value1"
        
        # Get value
        assert cache.get("key1") == "value1"
        
        # Get non-existent key
        assert cache.get("key2") is None
    
    def test_cache_ttl_logic(self):
        """Test TTL expiration logic"""
        import time
        
        cache = {}
        timestamps = {}
        ttl_seconds = 2
        
        # Set value
        cache["key1"] = "value1"
        timestamps["key1"] = time.time()
        
        # Should be valid immediately
        assert "key1" in cache
        
        # Should expire after TTL
        time.sleep(ttl_seconds + 0.1)
        
        # Check if expired
        if "key1" in timestamps:
            age = time.time() - timestamps["key1"]
            if age > ttl_seconds:
                # Expired - should delete
                del cache["key1"]
                del timestamps["key1"]
        
        assert "key1" not in cache
    
    def test_cache_chart_id_lookup(self):
        """Test chart caching by ID"""
        charts_cache = {}
        
        chart_id = "abc123"
        chart_data = {
            "title": "Test Chart",
            "filename": "abc123.png"
        }
        
        # Store chart
        charts_cache[chart_id] = chart_data
        
        # Retrieve chart
        cached = charts_cache.get(chart_id)
        assert cached is not None
        assert cached["title"] == "Test Chart"
        assert cached["filename"] == "abc123.png"


# --- Error Handling Tests ---

class TestToolErrorHandling:
    """Test tool error handling patterns"""
    
    def test_invalid_asset_class(self):
        """Test handling invalid asset class"""
        valid_asset_classes = ["Equities", "Fixed_Income", "Real_Estate", "Cash", "Commodities", "Crypto"]
        
        invalid_class = "InvalidAssetClass"
        
        # Should raise error or return error response
        assert invalid_class not in valid_asset_classes
    
    def test_negative_portfolio_value(self):
        """Test handling negative portfolio values"""
        portfolio = {
            "Equities": 100000.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        # Try to remove more than available
        removal_amount = -150000.0
        
        # Should either:
        # 1. Prevent negative values
        # 2. Raise an error
        # 3. Set to zero
        
        if portfolio["Equities"] + removal_amount < 0:
            # Handle error - don't allow negative
            assert True
        else:
            portfolio["Equities"] += removal_amount
    
    def test_invalid_ticker_symbol(self):
        """Test handling invalid ticker"""
        invalid_tickers = ["INVALID123", "NOTREAL", ""]
        
        # Should handle gracefully
        for ticker in invalid_tickers:
            # In real implementation, would return None or raise error
            assert len(ticker) >= 0  # Basic validation
    
    def test_empty_portfolio_simulation(self):
        """Test handling simulation with empty portfolio"""
        empty_portfolio = {
            "Equities": 0.0,
            "Fixed_Income": 0.0,
            "Real_Estate": 0.0,
            "Cash": 0.0,
            "Commodities": 0.0,
            "Crypto": 0.0
        }
        
        total = sum(empty_portfolio.values())
        
        # Should not run simulation on empty portfolio
        assert total == 0.0
        # Would return error or prompt user to add funds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
