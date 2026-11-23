# ==============================================================================
# Stage 1: Build React Frontend
# ==============================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Python Backend + Serving Frontend Static Bundle
# ==============================================================================
FROM python:3.10-slim

WORKDIR /app

# Install minimal system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python serving requirements
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Pre-download the embedding model so Render doesn't hang or OOM on the first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy backend, rag, common
COPY backend/ ./backend/
COPY common/ ./common/
COPY rag/ ./rag/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    LLM_BACKEND=fake \
    RAG_INDEX_DIR=rag/index

EXPOSE 8000 10000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

