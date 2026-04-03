# Email Triage OpenEnv — Dockerfile
# Builds a lightweight FastAPI server exposing the OpenEnv HTTP API.
# HF Spaces uses port 7860 by default.

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server source
COPY server/ ./server/

# Copy openenv spec
COPY openenv.yaml .

# Expose the port HF Spaces expects
EXPOSE 7860

# Health check so HF Spaces knows when the container is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]