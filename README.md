# Edenview

A local-first RAG (Retrieval-Augmented Generation) stack: ingest your own documents,
choose how they get chunked, and chat against them — either a single-pass retrieval
flow or a multi-step agentic pipeline that plans, searches, and checks its own work.
Everything runs on your own machine. No cloud account, no Docker, no data leaving
your disk.

## What's here

- **Multiple, user-selectable chunking strategies** — fixed-size overlap,
  structure-aware (headings/sections/tables), parent-child (precise match + wide
  context), and LLM-assisted contextual enrichment — so you can compare strategies on
  the same document instead of being locked into one.
- **Hybrid retrieval** — dense + sparse (BM25) search fused via Qdrant's native RRF,
  then cross-encoder reranked, across one collection or fanned out across many.
- **Agentic RAG** — an ADK-based pipeline that decomposes a question into
  sub-questions, researches each independently (search → evaluate → reword/deep-search
  as needed), and consolidates cited answers — with a live view of what it's doing at
  each step, not just a spinner.
- **Citation grounding** — click a citation to see the actual source PDF page,
  highlighted at the matched chunk's location.
- **A local system monitor and model-aware Settings UI** — see your real RAM/VRAM
  headroom, and pick models per role (embedding, chat, agent, vision) from what's
  actually pulled in Ollama, with capability checks (tool-calling, vision) enforced
  before you can save a bad choice.
- **Local models via [Ollama](https://ollama.com)**, embedded [Qdrant](https://qdrant.tech)
  (no server/Docker) for vectors, and [DuckDB](https://duckdb.org) for the
  Database/Collection/chat-session catalog — one long-lived process, one folder of
  local data.

## Quick start

Full prerequisites, per-OS install steps, and a guided first-session walkthrough live
in **[USER_GUIDE.md](USER_GUIDE.md)** — start there if this is your first time setting
this up. Short version:

```bash
# Prerequisites: Python 3.12, Node.js 20+, Ollama (https://ollama.com/download)
ollama pull bge-m3
ollama pull qwen3.5:2b     # or whatever chat model you prefer

python -m venv venv
venv\Scripts\activate                    # Windows
# source venv/bin/activate               # macOS / Linux
pip install -r requirements.txt
python scripts/install_torch.py          # optional, NVIDIA GPUs only

cd edenview-ui
npm install
npm run dev
```

Open **http://localhost:3000**, go to **Settings → Workspace** and confirm/set where
your data lives, then head to **Ingestion** to upload your first document.

## Documentation

| Doc | What it covers |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Full install guide, every Settings option explained, a chunking-strategy comparison, and a Simple RAG vs. Agentic RAG tutorial |
| [api/README.md](api/README.md) | Every backend endpoint, with a copy-pasteable curl walkthrough (upload → ingest → search) |

## Project status

Actively developed, local-first, single-user by design. Built and tuned against a
modest (~6GB VRAM) GPU, but every model choice and performance knob is
user-configurable from Settings for other hardware — see `USER_GUIDE.md`'s
prerequisites section for picking your own models.
