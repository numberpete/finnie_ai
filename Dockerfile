# Finnie AI - Single Container Deployment
# Runs all MCP servers + Streamlit UI in one container

FROM python:3.11-slim

WORKDIR /app

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    supervisor \
    nginx \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# --- build args (set these in Railway Variables) ---
# Public repo: set GIT_REPO_URL=https://github.com/<owner>/<repo>.git
ARG GIT_REPO_URL
# Optional: pin to a commit/branch/tag. If not set, defaults to main.
ARG GIT_REF=main

# Optional: for private repos, pass a token and use it to authenticate.
# Prefer a build secret mechanism if available; otherwise Railway Variable + build arg.
ARG GIT_TOKEN=""

# --- fetch source WITH .git so LFS works ---
# Clone into /src, then move into /app (keeps WORKDIR clean)
RUN set -eux; \
    if [ -z "$GIT_REPO_URL" ]; then echo "GIT_REPO_URL is required"; exit 1; fi; \
    if [ -n "$GIT_TOKEN" ]; then \
      REPO_URL="$(echo "$GIT_REPO_URL" | sed -e "s#https://#https://${GIT_TOKEN}@#")"; \
    else \
      REPO_URL="$GIT_REPO_URL"; \
    fi; \
    git clone "$REPO_URL" /src; \
    cd /src; \
    git fetch --all --tags; \
    git checkout "$GIT_REF"; \
    git lfs pull; \
    rm -rf /src/.git; \
    mv /src/* /src/.[!.]* /app/ 2>/dev/null || true

# --- python deps (now that requirements.txt exists from the cloned repo) ---
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for generated content
RUN mkdir -p /app/generated_charts /app/src/data /var/log/supervisor

# Environment variables
ENV PYTHONPATH="/app"
ENV OPENAI_API_KEY=""
ENV CHART_PATH="generated_charts"
ENV CHART_URL="http://localhost:8010/chart/"

# Supervisor / nginx configs (these should be in your repo)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/__nginx_ok || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
