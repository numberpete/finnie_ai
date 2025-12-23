# src/servers/image_server.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.utils.tracing import setup_tracing, setup_logger_with_tracing
import logging

# Setup logging
# Setup tracing and logging
setup_tracing("image_server", enable_console_export=False)
LOGGER = setup_logger_with_tracing(__name__, service_name="image_server")


# Initialize FastAPI
app = FastAPI(title="Chart Image Server", version="1.0.0")

# Add CORS middleware to allow requests from Gradio/Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chart directory (same as MCP server)
CHART_DIR = Path("generated_charts")
CHART_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Chart Image Server",
        "status": "running",
        "chart_directory": str(CHART_DIR.absolute()),
        "endpoints": {
            "get_chart": "/chart/{filename}",
            "list_charts": "/charts"
        }
    }


@app.get("/chart/{filename}")
def get_chart(filename: str):
    """
    Serve a chart image by filename.
    
    Args:
        filename: The chart filename (e.g., 'abc123.png')
    
    Returns:
        The image file
    
    Example:
        GET http://localhost:8010/chart/abc123.png
    """
    # Security: Only allow .png files and prevent directory traversal
    if not filename.endswith('.png') or '/' in filename or '\\' in filename:
        LOGGER.warning(f"Invalid filename requested: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = CHART_DIR / filename
    
    if not filepath.exists():
        LOGGER.warning(f"Chart not found: {filename}")
        raise HTTPException(status_code=404, detail=f"Chart {filename} not found")
    
    LOGGER.info(f"Serving chart: {filename}")
    
    return FileResponse(
        filepath,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Disposition": f"inline; filename={filename}"
        }
    )


@app.get("/charts")
def list_charts():
    """
    List all available charts.
    
    Returns:
        JSON with list of chart filenames and count
    
    Example:
        GET http://localhost:8010/charts
    """
    charts = list(CHART_DIR.glob("*.png"))
    chart_list = [
        {
            "filename": chart.name,
            "size_bytes": chart.stat().st_size,
            "url": f"/chart/{chart.name}",
            "modified": chart.stat().st_mtime
        }
        for chart in sorted(charts, key=lambda x: x.stat().st_mtime, reverse=True)
    ]
    
    return {
        "chart_count": len(chart_list),
        "charts": chart_list
    }


@app.delete("/chart/{filename}")
def delete_chart(filename: str):
    """
    Delete a chart by filename.
    
    Args:
        filename: The chart filename to delete
    
    Returns:
        Success message
    """
    if not filename.endswith('.png') or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = CHART_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Chart {filename} not found")
    
    filepath.unlink()
    LOGGER.info(f"Deleted chart: {filename}")
    
    return {"status": "success", "message": f"Deleted {filename}"}


if __name__ == "__main__":
    import uvicorn
    
    LOGGER.info("Starting Chart Image Server on http://localhost:8010")
    LOGGER.info(f"Serving images from: {CHART_DIR.absolute()}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )