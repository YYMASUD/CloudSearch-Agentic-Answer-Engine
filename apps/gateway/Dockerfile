FROM python:3.12-slim

WORKDIR /app

# Install system deps for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install packages
COPY packages/python-shared /app/packages/python-shared
COPY services/providers      /app/services/providers
COPY services/rag            /app/services/rag
COPY services/llm            /app/services/llm
COPY services/orchestrator   /app/services/orchestrator
COPY apps/gateway            /app/apps/gateway

# Install shared library + gateway deps
RUN pip install --no-cache-dir -e /app/packages/python-shared
RUN pip install --no-cache-dir -e /app/apps/gateway
RUN pip install --no-cache-dir \
    sentence-transformers>=3.3.1 \
    qdrant-client>=1.13.1 \
    meilisearch>=0.31.5 \
    rank-bm25>=0.2.2 \
    httpx>=0.28.1

EXPOSE 8000
ENV PYTHONPATH=/app

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
