# Finnie AI - Single Container Deployment
# Runs all MCP servers + Streamlit UI in one container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for generated content
RUN mkdir -p /app/charts /app/src/data /var/log/supervisor

# Environment variables
ENV PYTHONPATH="/app"
ENV OPENAI_API_KEY=""
ENV CHART_URL="http://localhost:8010/chart/"
ENV CHART_PATH="/app/generated_charts"

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose only the Streamlit port (internal services stay internal)
EXPOSE 8501

# Health check on Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start all services via supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]