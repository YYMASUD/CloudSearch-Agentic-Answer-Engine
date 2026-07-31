# CloudSearch — Agentic Answer Engine

> A production-grade, Perplexity-style multi-source agentic answer engine with citation-grounded streaming responses.

[![Architecture](docs/architecture.png)](docs/architecture.png)

---

## ✨ Features

| Feature | Details |
|---|---|
| **6-Layer Architecture** | Client → Gateway → Orchestrator → Providers → RAG → LLM |
| **5 Search Backends** | Meilisearch, Web (SearXNG/Brave/Serper), Code (zoekt), Private, Local |
| **Fusion Core** | RRF + Diversity + optional CrossEncoder re-ranking |
| **Streaming UI** | Token-by-token SSE, inline hoverable citations, source cards |
| **Citation Grounding** | Every answer claim mapped to source chunks |
| **Model Router** | Ollama → Groq → OpenAI → Mistral cost/latency/privacy ladder |
| **Full Observability** | OpenTelemetry → Prometheus → Grafana |
| **GraphQL API** | Strawberry schema alongside REST+SSE |

---

## 🚀 Quick Start

### 1. Prerequisites

- Docker Desktop (≥ 4.x) with 8 GB RAM allocated
- Node.js 20+ and pnpm 9+
- Python 3.12+
- (Optional) Ollama for local LLM inference

### 2. Clone & configure

```bash
git clone <repo-url> cloudsearch
cd cloudsearch
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start infrastructure

```bash
# Full stack (all services)
docker compose --profile full up -d

# Lite stack (Meilisearch + Qdrant + Postgres + Redis only)
docker compose --profile lite up -d
```

### 4. Initialize the search index

```bash
pip install -e packages/python-shared
python infra/meilisearch/init-index.py
```

### 5. Start the gateway

```bash
cd apps/gateway
pip install -e .
pip install -e ../../packages/python-shared
uvicorn gateway.main:app --reload --port 8000
```

### 6. Start the web app

```bash
cd apps/web
pnpm install
pnpm dev
```

Open **http://localhost:3000** 🎉

---

## 📁 Repository Structure

```
cloudsearch/
├── apps/
│   ├── gateway/         FastAPI REST+SSE+GraphQL gateway
│   └── web/             Next.js 15 App Router frontend
├── services/
│   ├── providers/       5 SearchProvider adapters
│   ├── rag/             RAG pipeline (parser, chunker, embedder, vector store, reranker, grounder)
│   ├── llm/             LLM layer (model router, 4 providers, synthesizer, SSE bridge)
│   └── orchestrator/    Agent layer (planner, router, fan-out, Fusion Core)
├── packages/
│   └── python-shared/   NormalizedDocument, Kafka events, OTel bootstrap
├── infra/               Docker, OTel, Prometheus, Grafana configs
└── tests/               Unit + integration test suite
```

---

## 🔌 API Reference

### REST

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/search` | Synchronous JSON search |
| `GET` | `/api/search/stream?q=...` | SSE streaming search |
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

### GraphQL

Navigate to **http://localhost:8000/graphql** for the GraphiQL explorer.

```graphql
query {
  search(query: "What is RAG?", mode: "web") {
    answer
    sources { title url snippet }
    citations { number url title }
  }
}
```

### SSE Events

Connect to `/api/search/stream?q=your+query` and listen for:

| Event | Payload | Timing |
|---|---|---|
| `source_card` | `{index, title, url, snippet, source_type, score, favicon_url}` | Before answer |
| `answer_chunk` | `{chunk, chunk_index}` | During generation |
| `answer_done` | `{answer, citations, source_count}` | End of generation |
| `citations` | `CitationItem[]` | Post-generation grounding |
| `error` | `{message, recoverable}` | On failure |

---

## 🧪 Running Tests

```bash
# Unit tests (no Docker required)
cd tests
pip install -e .
pytest unit/ -v

# With coverage
pytest unit/ --cov=services --cov-report=html
```

---

## 🐳 Docker Services

| Service | Port | Profile |
|---|---|---|
| Meilisearch | 7700 | full, lite |
| Qdrant | 6333 | full, lite |
| Postgres (pgvector) | 5432 | full, lite |
| Redis | 6379 | full, lite |
| MinIO | 9000/9001 | full |
| Kafka | 9092 | full |
| Elasticsearch | 9200 | full |
| OTel Collector | 4317/4318 | full |
| Prometheus | 9090 | full |
| Grafana | 3001 | full |
| SearXNG | 8888 | full |

---

## 🗺️ Roadmap

- [x] Phase 1 — Shared infra + Meilisearch provider
- [x] Phase 2 — RAG/Grounding layer
- [x] Phase 3 — LLM model router
- [x] Phase 4 — Orchestration + Fusion Core
- [x] Phase 5 — Gateway + streaming UI
- [x] Phase 6 — Web + Local providers
- [ ] Phase 7 — Code/GitHub mode + zoekt + GraphRAG + RAPTOR
- [ ] Phase 8 — Full test suite + CI + Grafana dashboards

---

## 📄 License

MIT
