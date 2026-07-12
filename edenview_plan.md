# Edenview — Project Planning Document

> A self-contained, open-source local AI stack launchable with a single command, exposing a localhost portal with modular AI "spaces." No cloud dependency by default.

---

## 1. Vision

Edenview is a pip-installable Python package or a github repo that spins up a full local AI stack via Docker Compose. Users run one command and get a localhost portal — the **Edenview Portal** — with modular AI workspaces called **Spaces**. The first and primary space is a full-featured RAG system. Future spaces (fine-tuning, evaluation, agents, knowledge graphs) slot in without architectural changes.

```bash
pip install edenview
edenview init     # detects system specs, pulls base models, generates config
edenview start    # docker compose up — portal live at localhost:3000
```

The system supports both **local models** (via Ollama) and **cloud API providers** (OpenAI, Anthropic, Gemini, Groq) with API keys stored in a locally mounted config file — never in a database.

---

## 2. Why Build This (Competitive Gap)

The "local RAG in Docker" problem is largely solved. What is **not** solved:

| Feature | Open WebUI | AnythingLLM | PrivateGPT | **Edenview** |
|---|---|---|---|---|
| Multiple chunking strategies (user-selectable) | ✗ | ✗ | ✗ | ✅ |
| Live chunking preview before full ingest | ✗ | ✗ | ✗ | ✅ |
| Agentic RAG (multi-agent retrieval loop) | Partial | Partial | ✗ | ✅ (ADK) |
| Global assistant for model + strategy advice | ✗ | ✗ | ✗ | ✅ |
| System spec detection → model recommendation | ✗ | ✗ | ✗ | ✅ |
| Pluggable multi-space architecture | ✗ | ✗ | ✗ | ✅ |
| Docling-based layout-aware extraction | ✗ | ✗ | ✗ | ✅ |
| Hybrid retrieval (dense + BM25 + RRF + reranker) | ✅ | ✗ | ✗ | ✅ |

The core bet: **depth of RAG quality + an intelligent setup layer + a space-extensible architecture.** Open WebUI is the closest competitor and moves fast, so the window for differentiation is on agentic retrieval and the guided setup experience.

---

## 3. Repository Structure

Monorepo with three installable packages:

```
edenview/
├── packages/
│   ├── edenview-core/          # Shared infra, CLI, config, model management
│   │   ├── cli.py              # edenview init / start / stop / status
│   │   ├── system_inspector.py # Detects CPU/GPU/RAM, platform, Ollama models
│   │   ├── model_manager.py    # Ollama pull, cloud API key validation
│   │   └── config/             # Config schema, .env template, docker-compose base
│   │
│   ├── edenview-rag/           # RAG Space — ingestion, retrieval, ADK agents
│   │   ├── ingestion/
│   │   │   ├── extractor.py    # Docling wrapper (PDF, DOCX, PPTX, XLSX, images)
│   │   │   ├── chunkers/
│   │   │   │   ├── overlap.py       # Fixed size + overlap
│   │   │   │   ├── parent_child.py  # Parent-child hierarchical
│   │   │   │   ├── semantic.py      # Sentence embedding boundary detection
│   │   │   │   └── contextual.py    # LLM-assisted contextual enrichment
│   │   │   ├── embedder.py     # Embedding model wrapper (Ollama / HuggingFace)
│   │   │   └── pipeline.py     # Orchestrates extract → chunk → embed → write
│   │   ├── retrieval/
│   │   │   ├── dense.py        # Qdrant dense vector search
│   │   │   ├── sparse.py       # BM25 sparse retrieval
│   │   │   ├── fusion.py       # Reciprocal Rank Fusion (RRF)
│   │   │   └── reranker.py     # Cross-encoder reranking
│   │   ├── agents/
│   │   │   ├── rag_agent/      # ADK Retriever → Relevance → Answer loop
│   │   │   └── assistant/      # Global assistant (ingestion + model advisor)
│   │   ├── db_manager.py       # Qdrant collection CRUD, metadata operations
│   │   └── api/                # FastAPI routes for RAG space
│   │
│   └── edenview-ui/            # Next.js frontend (built last)
│       ├── spaces/
│       │   └── rag/            # RAG space pages
│       ├── components/
│       └── global-assistant/   # Global assistant chatbot panel
│
├── docker/
│   ├── docker-compose.yml      # Full stack definition
│   └── services/               # Per-service Dockerfiles
│
└── pyproject.toml              # Monorepo root
```

---

## 4. Infrastructure Stack

All services run in Docker Compose. Edenview manages the lifecycle — users never touch Docker directly.

| Service | Role |
|---|---|
| **Qdrant** | Persistent vector store — one collection per ingestion job |
| **Ollama** | Local LLM and embedding model inference |
| **Docling worker** | Layout-aware document extraction (Celery task) |
| **Redis** | Celery task queue + response caching |
| **PostgreSQL** | Session persistence, ingestion job metadata, ADK sessions |
| **FastAPI** | Backend API layer + ADK agent host |
| **Next.js** | Frontend portal (the Edenview UI) |
| **Nginx** | Reverse proxy — routes `/api`, `/rag`, `/ws` |

---

## 5. Space 1 — RAG Space

### 5.1 Ingestion Module

Users select files and a chunking strategy. A fast synchronous **preview pipeline** runs on the first ~5 pages or ~2,000 tokens and returns chunk boundaries to the UI within 1–2 seconds. Full ingestion runs asynchronously through Celery.

**Supported file types (via Docling):**
PDF, DOCX, PPTX, XLSX, JPEG/PNG, TXT, Markdown

**Chunking strategies:**

| Mode | Description | Best for |
|---|---|---|
| Fixed + overlap | Sliding window with configurable size and overlap | General purpose, dense prose |
| Parent-child | Small retrieval chunks with large parent context preserved | Q&A over long documents |
| Semantic | Boundary detection via sentence embedding similarity | Thematically diverse docs |
| Contextual (LLM-assisted) | LLM prepends context summary to each chunk before embedding | Complex regulatory / technical docs |

**Qdrant collection naming convention:**
```
{user_slug}_{file_hash}_{chunk_mode}
```

This allows the same file to be re-ingested with different chunking strategies and stored as separate, addressable collections.

**Metadata in PostgreSQL (not in Qdrant payload):**
- Original filename, file type, file hash
- Chunking strategy + parameters used
- Embedding model used
- Ingestion timestamp, chunk count, job status

**Qdrant payload per chunk:**
- `chunk_text`, `page_number`, `source_path`, `chunk_id`

---

### 5.2 Chat / Query Module

Users select which Qdrant collections to include in the active context, then query across them in a persistent chat session.

**Retrieval pipeline:**

```
Query
  └─► Dense vector search  (Qdrant)  ─┐
  └─► BM25 sparse search   (Qdrant)  ─┤── RRF Fusion ──► Cross-encoder Reranker ──► Top-K chunks
```

**ADK Agentic RAG — sequential agent loop:**
1. **Retriever agent** — runs hybrid retrieval, assembles candidate chunks
2. **Relevance agent** — scores and filters candidates, decides if retrieval is sufficient or needs a follow-up query
3. **Answer agent** — generates response with inline citations from verified chunks

**Model selection:**
Users choose from a unified list populated at query time from:
- `Ollama /api/tags` — local models currently pulled
- Mounted config file — cloud providers with valid API keys

---

### 5.3 DB Manager

Displayed in the Chat view as a sidebar panel — users see all available Qdrant collections and toggle which ones are active for the current session.

**Operations:**
- List all collections (name, file, strategy, created at, chunk count)
- Toggle active / inactive per session
- Delete a collection (with confirmation)
- Trigger reindex (re-run ingestion with new settings)

---

### 5.4 Global Assistant

A persistent lightweight chatbot available across the entire portal. Powered by `qwen3:8b`, pulled automatically at `edenview init`. It has two advisory modes:

**Ingestion advisor:**
- User describes or uploads a sample of their documents
- Assistant analyzes document type, density, structure
- Recommends the most appropriate chunking strategy and why
- Suggests embedding model based on document language and complexity

**Model advisor:**
- `SystemInspector` class queries: `psutil` (RAM), `pynvml` / `nvidia-smi` (VRAM), `platform` (OS), `Ollama /api/tags` (currently pulled models)
- Assistant receives this as structured JSON via a tool call
- Recommends which local models are viable and what quality/speed trade-off to expect
- Warns if a selected model won't fit in available VRAM

**SystemInspector — abstracted for cross-platform support:**
```python
class SystemInspector:
    def get_specs(self) -> dict:
        # Returns: {ram_gb, vram_gb, gpu_name, cpu_cores, platform, ollama_models}
        # Platform-specific drivers for nvidia-smi / pynvml / Apple MPS
```

---

## 6. Key Architectural Decisions

**Space-aware from day one.** The Docker Compose stack, FastAPI router, and Next.js shell are all space-aware — not RAG-specific. Future spaces (`edenview-finetune`, `edenview-eval`, etc.) register themselves and are served by the same portal.

**Chunking preview is a separate pipeline from full ingestion.** The preview endpoint (`POST /rag/preview`) is fast, synchronous, and stateless — it runs on a small sample with no Celery involvement. Full ingestion (`POST /rag/ingest`) is async and goes through the Celery queue. These two paths never share code.

**API keys are never stored in the database.** Cloud provider API keys (OpenAI, Anthropic, Gemini, Groq) live in a `.env`-style config file that Docker mounts at runtime. The model selector reads from this file + Ollama at runtime, presenting a unified list.

**PostgreSQL owns metadata; Qdrant owns vectors only.** Qdrant payloads are minimal — just what's needed for citation display. All job metadata, session history, and user preferences live in PostgreSQL via ADK's `DatabaseSessionService`.

**bge-m3 is a future upgrade path, not default.** The default embedding model is `qwen3-embedding:0.6b` (fast, small). `qwen3-embedding:4b` is the production upgrade. `bge-m3` (dense + sparse in one forward pass) is the hybrid-search upgrade path — it requires Qdrant sparse vector support and is positioned as an advanced option.

---

## 7. Model Recommendations (Defaults)

| Role | Default | Upgrade |
|---|---|---|
| Global assistant | `qwen3:8b` | — |
| RAG LLM | `qwen3:14b` | `qwen3:30b-a3b` (MoE) |
| Agentic use (alt) | `qwen3:14b` | `gemma4:26b` (MoE, strong on multi-turn) |
| Embedding (default) | `qwen3-embedding:0.6b` | `qwen3-embedding:4b` |
| Embedding (hybrid) | — | `bge-m3` (requires Qdrant sparse support) |

**Gemma 3 1B and 4B are explicitly excluded** — broken/poor tool calling makes them unsuitable for agentic use.

**Evaluation metric for embedding models: nDCG@10** (not overall MTEB score) — this is the correct metric for retrieval quality in RAG contexts.

---

## 8. Build Order (RAG Space)

Build backend-first. UI is built last. Each step is testable independently before moving to the next.

```
Step 1 — Docker Compose stack
         All services healthy, volumes mounted, networking confirmed.

Step 2 — Ingestion pipeline (standalone script)
         Docling → fixed overlap chunking → embed → Qdrant write.
         Test end-to-end from CLI before any API wiring.

Step 3 — Additional chunk modes
         Parent-child → semantic → contextual (one at a time).

Step 4 — FastAPI ingestion endpoint + Celery wiring
         POST /rag/preview (sync, fast sample)
         POST /rag/ingest  (async, Celery job)
         GET  /rag/jobs/{id} (status polling)

Step 5 — Retrieval pipeline
         Dense search first → add BM25 → add RRF fusion → add reranker.

Step 6 — ADK RAG agent
         Retriever → Relevance → Answer loop.
         Session persistence via DatabaseSessionService → PostgreSQL.

Step 7 — Global Assistant
         SystemInspector class → ingestion advisor → model advisor.
         Lightweight always-on, isolated from RAG agent.

Step 8 — Next.js UI
         Document upload + job status → chunking config + preview →
         query/chat with citations + DB selector → report export.
```

---

## 9. Frontend Views (RAG Space)

| View | Key functionality |
|---|---|
| **Upload** | File drag-and-drop, chunking strategy selector, live chunk preview panel, ingest button, job status tracker |
| **Collections** | List of all Qdrant collections with metadata, active/inactive toggle, delete/reindex actions |
| **Chat** | Active collection selector sidebar, chat history, inline source citations, model selector, session persistence |
| **Report export** | Export chat session as PDF or JSON with full citation trace |

**Global Assistant** lives as a persistent floating panel / drawer, available on all views.

---

## 10. Future Spaces (Planned, Undefined)

The space architecture supports future additions without changes to the core portal:

- Fine-tuning space
- Model evaluation space  
- Knowledge graph space
- Autonomous agents workspace

Each future space ships as its own pip package (`edenview-finetune`, etc.) that registers with the portal on install.

---

## 11. Open Questions / Still To Decide

- [ ] Multi-user support in v1, or single-user only?
- [ ] Auth layer (simple token / local password) or fully open on localhost?
- [ ] Should the Global Assistant have memory across sessions, or be stateless?
- [ ] Embedding model auto-detection at ingest time, or user-selected per collection?
- [ ] GPU passthrough strategy in Docker on macOS (Ollama on host vs. containerized)?
- [ ] Packaging: pip only, or also a Homebrew formula / Docker-only path for non-Python users?
