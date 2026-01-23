# Finnie AI - Single Container Deployment
# Runs all MCP servers + Streamlit UI in one container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    supervisor \
    nginx \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (including .git if present)
COPY . .

# IMPORTANT: pull real Git LFS objects (FAISS index, etc.)
# This MUST happen after COPY .
RUN git lfs pull

# Create directories for generated content
RUN mkdir -p /app/generated_charts /app/src/data /var/log/supervisor

# Environment variables
ENV PYTHONPATH="/app"
ENV OPENAI_API_KEY=""
ENV CHART_PATH="generated_charts"

# CHART_URL must be set to your public URL in production
ENV CHART_URL="http://localhost:8010/chart/"

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose single port (nginx proxies to internal services)
EXPOSE 8080

# Health check on nginx
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/__nginx_ok || exit 1

# Start all services via supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
