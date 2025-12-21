# src/mcp/charts_mcp.py (PATCHED VERSION with data validation)

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime
from fastmcp import FastMCP
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging

# Setup tracing and logging
setup_tracing("mcp-server-charts", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, logging.INFO)

# Initialize FastMCP
mcp = FastMCP("Chart Generation Server")

# Chart storage directory
CHART_DIR = Path(os.getenv("CHART_PATH", "generated_charts"))
CHART_DIR.mkdir(exist_ok=True)

LOGGER.info(f"Charts will be saved to: {CHART_DIR.absolute()}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_chart_id(chart_type: str, data: Any) -> str:
    """Generate a unique ID for a chart based on its data."""
    data_str = json.dumps({"type": chart_type, "data": data}, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()[:12]


def save_chart(fig: plt.Figure, chart_id: str) -> str:
    """Save a matplotlib figure and return the filename."""
    filename = f"{chart_id}.png"
    filepath = CHART_DIR / filename
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    LOGGER.info(f"Chart saved: {filename}")
    return filename


def validate_data_lengths(*arrays) -> tuple:
    """
    Validate that all arrays have the same length. If not, truncate to shortest.
    Returns the validated (possibly truncated) arrays.
    """
    if not arrays:
        return arrays
    
    lengths = [len(arr) for arr in arrays if arr is not None]
    
    if not lengths:
        return arrays
    
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        LOGGER.warning(f"⚠️  Mismatched array lengths: {lengths}. Truncating all to {min_len}")
        return tuple(arr[:min_len] if arr is not None else None for arr in arrays)
    
    return arrays


# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

@mcp.tool()
def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str = "Pie Chart",
    colors: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create a pie chart.
    
    Args:
        labels: List of slice labels
        values: List of values for each slice
        title: Chart title
        colors: Optional list of colors (hex or named colors)
    
    Returns:
        Dictionary with chart_id and filename
    
    Example:
        create_pie_chart(
            labels=["Stocks", "Bonds", "Cash"],
            values=[60, 30, 10],
            title="Portfolio Allocation"
        )
    """
    LOGGER.info(f"Creating pie chart: {title}")
    
    # Validate data
    labels, values = validate_data_lengths(labels, values)
    
    chart_id = generate_chart_id("pie", {"labels": labels, "values": values, "title": title})
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    filename = save_chart(fig, chart_id)
    
    return {
        "chart_id": chart_id,
        "filename": filename,
        "chart_type": "pie",
        "title": title
    }


@mcp.tool()
def create_bar_chart(
    categories: List[str],
    values: List[float],
    title: str = "Bar Chart",
    xlabel: str = "",
    ylabel: str = "Value",
    color: str = "#3498db"
) -> Dict[str, str]:
    """
    Create a bar chart.
    
    Args:
        categories: List of category names (x-axis)
        values: List of values (y-axis)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color (hex or named color)
    
    Returns:
        Dictionary with chart_id and filename
    
    Example:
        create_bar_chart(
            categories=["2020", "2021", "2022", "2023"],
            values=[50000, 55000, 62000, 70000],
            title="Annual Savings",
            ylabel="Amount ($)"
        )
    """
    LOGGER.info(f"Creating bar chart: {title}")
    
    # Validate data
    categories, values = validate_data_lengths(categories, values)
    
    chart_id = generate_chart_id("bar", {"categories": categories, "values": values, "title": title})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(categories, values, color=color, alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'${height:,.0f}' if height > 1000 else f'{height:.1f}',
            ha='center', va='bottom', fontsize=10
        )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    
    filename = save_chart(fig, chart_id)
    
    return {
        "chart_id": chart_id,
        "filename": filename,
        "chart_type": "bar",
        "title": title
    }


@mcp.tool()
def create_line_chart(
    x_values: List[Any],
    y_values: List[float],
    title: str = "Line Chart",
    xlabel: str = "",
    ylabel: str = "Value",
    color: str = "#2ecc71",
    marker: str = "o"
) -> Dict[str, str]:
    """
    Create a single-line chart.
    
    Args:
        x_values: List of x-axis values (dates, categories, numbers)
        y_values: List of y-axis values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
        marker: Marker style ('o', 's', '^', 'D', etc.)
    
    Returns:
        Dictionary with chart_id and filename
    
    Example:
        create_line_chart(
            x_values=["Jan", "Feb", "Mar", "Apr"],
            y_values=[10000, 12000, 11500, 13000],
            title="Monthly Portfolio Value",
            ylabel="Value ($)"
        )
    
    IMPORTANT: If x_values and y_values have different lengths, they will be
    automatically truncated to match the shorter length.
    """
    LOGGER.info(f"Creating line chart: {title}")
    
    # Validate and align data lengths
    x_values, y_values = validate_data_lengths(x_values, y_values)
    
    if len(x_values) == 0 or len(y_values) == 0:
        LOGGER.error("Cannot create chart with empty data")
        return {"error": "Cannot create chart with empty data", "chart_type": "line"}
    
    # Adaptive rendering based on number of data points
    num_points = len(x_values)
    
    if num_points > 200:
        use_marker = None
        markersize = 0
        markevery = None
        linewidth = 1.5
        max_ticks = 8  # Fewer ticks for very dense data
    elif num_points > 100:
        use_marker = marker
        markersize = 4
        markevery = 10
        linewidth = 2
        max_ticks = 10
    elif num_points > 50:
        use_marker = marker
        markersize = 6
        markevery = 3
        linewidth = 2
        max_ticks = 12
    else:
        use_marker = marker
        markersize = 8
        markevery = 1
        linewidth = 2
        max_ticks = 15
    
    chart_id = generate_chart_id("line", {"x": x_values, "y": y_values, "title": title})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with adaptive parameters
    if use_marker is None:
        ax.plot(x_values, y_values, color=color, linewidth=linewidth)
    else:
        ax.plot(x_values, y_values, color=color, marker=use_marker,
                linewidth=linewidth, markersize=markersize, markevery=markevery)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Smart x-axis decluttering
    if num_points > 20:
        # Limit the number of x-axis ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        
        # If x_values look like dates, try to parse and format them
        try:
            from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
            import pandas as pd
            
            # Try to detect if these are date strings
            if isinstance(x_values[0], str) and len(x_values[0]) >= 8:
                # Convert to datetime if they're date strings
                dates = pd.to_datetime(x_values)
                ax.clear()  # Clear and replot with proper dates
                
                if use_marker is None:
                    ax.plot(dates, y_values, color=color, linewidth=linewidth)
                else:
                    ax.plot(dates, y_values, color=color, marker=use_marker,
                            linewidth=linewidth, markersize=markersize, markevery=markevery)
                
                # Use auto date formatting
                locator = AutoDateLocator()
                formatter = ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.grid(True, alpha=0.3)
        except Exception as e:
            LOGGER.debug(f"Could not parse dates, using default formatting: {e}")
            pass
    
    # Rotate labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    filename = save_chart(fig, chart_id)
    
    return {
        "chart_id": chart_id,
        "filename": filename,
        "chart_type": "line",
        "title": title
    }

@mcp.tool()
def create_multi_line_chart(
    x_values: List[Any],
    y_series: Dict[str, List[float]],
    title: str = "Multi-Line Chart",
    xlabel: str = "",
    ylabel: str = "Value",
    colors: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create a multi-line chart with multiple data series.
    
    Args:
        x_values: List of x-axis values (shared by all series)
        y_series: Dictionary mapping series names to y-values
                  Example: {"Series1": [1,2,3], "Series2": [4,5,6]}
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Optional list of colors for each series
    
    Returns:
        Dictionary with chart_id and filename
    
    Example:
        create_multi_line_chart(
            x_values=["2020", "2021", "2022", "2023"],
            y_series={
                "Stocks": [40000, 45000, 50000, 60000],
                "Bonds": [20000, 22000, 25000, 28000]
            },
            title="Investment Growth",
            ylabel="Value ($)"
        )
    
    IMPORTANT: All y_series values will be truncated to match the length of x_values.
    """
    LOGGER.info(f"Creating multi-line chart: {title}")
    
    # Validate all series against x_values
    validated_series = {}
    for series_name, y_vals in y_series.items():
        x_validated, y_validated = validate_data_lengths(x_values, y_vals)
        validated_series[series_name] = y_validated
        x_values = x_validated  # Use the validated x_values
    
    if len(x_values) == 0:
        LOGGER.error("Cannot create chart with empty data")
        return {"error": "Cannot create chart with empty data", "chart_type": "multi_line"}
    
    # Adaptive rendering based on number of data points
    num_points = len(x_values)
    
    if num_points > 200:
        use_marker = None
        markersize = 0
        markevery = None
        linewidth = 1.5
        max_ticks = 8
        LOGGER.info(f"Very dense data ({num_points} points) - line only, no markers")
    elif num_points > 100:
        use_marker = 'o'
        markersize = 3
        markevery = 10
        linewidth = 2
        max_ticks = 10
        LOGGER.info(f"Dense data ({num_points} points) - showing every 10th marker")
    elif num_points > 50:
        use_marker = 'o'
        markersize = 4
        markevery = 3
        linewidth = 2
        max_ticks = 12
        LOGGER.info(f"Moderate data ({num_points} points) - showing every 3rd marker")
    else:
        use_marker = 'o'
        markersize = 6
        markevery = 1
        linewidth = 2
        max_ticks = 15
        LOGGER.info(f"Sparse data ({num_points} points) - showing all markers")
    
    chart_id = generate_chart_id("multiline", {"x": x_values, "y": validated_series, "title": title})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    default_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    colors = colors or default_colors
    
    # Plot each series with adaptive parameters
    for idx, (series_name, y_vals) in enumerate(validated_series.items()):
        color = colors[idx % len(colors)]
        
        if use_marker is None:
            ax.plot(x_values, y_vals, label=series_name, color=color, linewidth=linewidth)
        else:
            ax.plot(x_values, y_vals, label=series_name, color=color, 
                    marker=use_marker, linewidth=linewidth, markersize=markersize, markevery=markevery)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Smart x-axis decluttering
    if num_points > 20:
        # Limit the number of x-axis ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        
        # If x_values look like dates, try to parse and format them
        try:
            from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
            import pandas as pd
            
            # Try to detect if these are date strings
            if isinstance(x_values[0], str) and len(x_values[0]) >= 8:
                # Convert to datetime if they're date strings
                dates = pd.to_datetime(x_values)
                ax.clear()  # Clear and replot with proper dates
                
                # Replot all series with dates
                for idx, (series_name, y_vals) in enumerate(validated_series.items()):
                    color = colors[idx % len(colors)]
                    
                    if use_marker is None:
                        ax.plot(dates, y_vals, label=series_name, color=color, linewidth=linewidth)
                    else:
                        ax.plot(dates, y_vals, label=series_name, color=color,
                                marker=use_marker, linewidth=linewidth, markersize=markersize, 
                                markevery=markevery)
                
                # Use auto date formatting
                locator = AutoDateLocator()
                formatter = ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
        except Exception as e:
            LOGGER.debug(f"Could not parse dates, using default formatting: {e}")
            pass
    
    # Rotate labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    filename = save_chart(fig, chart_id)
    
    return {
        "chart_id": chart_id,
        "filename": filename,
        "chart_type": "multi_line",
        "title": title  
    }
    
    


@mcp.tool()
def create_goal_projection_chart(
    current_value: float,
    goal_value: float,
    years: int,
    monthly_contribution: float,
    annual_return_rate: float = 0.07,
    title: str = "Goal Projection"
) -> Dict[str, str]:
    """
    Create a financial goal projection chart showing growth over time.
    
    Args:
        current_value: Starting portfolio value
        goal_value: Target goal amount
        years: Number of years to project
        monthly_contribution: Monthly contribution amount
        annual_return_rate: Expected annual return (default 7% = 0.07)
        title: Chart title
    
    Returns:
        Dictionary with chart_id and filename
    
    Example:
        create_goal_projection_chart(
            current_value=50000,
            goal_value=1000000,
            years=30,
            monthly_contribution=1000,
            title="Retirement Goal Projection"
        )
    """
    LOGGER.info(f"Creating goal projection chart: {title}")
    
    chart_id = generate_chart_id("goal", {
        "current": current_value,
        "goal": goal_value,
        "years": years,
        "contribution": monthly_contribution
    })
    
    # Calculate month-by-month projection
    monthly_rate = annual_return_rate / 12
    months = years * 12
    
    timeline = []
    projected_values = []
    
    value = current_value
    
    for month in range(months + 1):
        timeline.append(month / 12)  # Convert to years
        projected_values.append(value)
        
        if month < months:
            # Add monthly contribution and apply returns
            value = (value + monthly_contribution) * (1 + monthly_rate)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot projection
    ax.plot(timeline, projected_values, color='#2ecc71', linewidth=3, label='Projected Value')
    
    # Plot goal line
    ax.axhline(y=goal_value, color='#e74c3c', linestyle='--', linewidth=2, label=f'Goal: ${goal_value:,.0f}')
    
    # Fill area under curve
    ax.fill_between(timeline, projected_values, alpha=0.2, color='#2ecc71')
    
    # Add final value annotation
    final_value = projected_values[-1]
    ax.annotate(
        f'Final: ${final_value:,.0f}',
        xy=(timeline[-1], final_value),
        xytext=(-80, 20),
        textcoords='offset points',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    filename = save_chart(fig, chart_id)
    
    return {
        "chart_id": chart_id,
        "filename": filename,
        "chart_type": "goal_projection",
        "title": title,
        "final_value": f"${final_value:,.2f}",
        "goal_reached": "true" if final_value >= goal_value else "false"
    }


@mcp.tool()
def list_generated_charts() -> Dict[str, Any]:
    """
    List all generated charts in the chart directory.
    
    Returns:
        Dictionary with list of chart files and count
    """
    charts = list(CHART_DIR.glob("*.png"))
    
    return {
        "chart_directory": str(CHART_DIR.absolute()),
        "chart_count": len(charts),
        "charts": [chart.name for chart in sorted(charts, key=lambda x: x.stat().st_mtime, reverse=True)]
    }


@mcp.tool()
def delete_chart(chart_id: str) -> Dict[str, str]:
    """
    Delete a specific chart by ID.
    
    Args:
        chart_id: The chart ID or filename to delete
    
    Returns:
        Status message
    """
    filename = chart_id if chart_id.endswith('.png') else f"{chart_id}.png"
    filepath = CHART_DIR / filename
    
    if filepath.exists():
        filepath.unlink()
        LOGGER.info(f"Deleted chart: {filename}")
        return {"status": "success", "message": f"Deleted {filename}"}
    else:
        LOGGER.warning(f"Chart not found: {filename}")
        return {"status": "error", "message": f"Chart {filename} not found"}


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    LOGGER.info("Starting Chart Generation MCP Server on port 8003")
    mcp.run(transport="sse", port=8003)