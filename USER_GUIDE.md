# Edenview — User Guide

A local-first RAG (Retrieval-Augmented Generation) stack: upload your own documents,
choose how they get chunked, and chat against them with either a single-pass retrieval
flow or a multi-step agentic pipeline — all running on your own machine, no cloud
dependency required. This guide walks a new user through installing prerequisites,
launching the app, and using every part of it (ingestion, collections, chat, settings).

> Companion docs: `edenview_plan.md` (product vision) and `edenview_progress.md`
> (build status / architecture decisions, for anyone modifying the code). This guide is
> for *using* the app, not building it.

---

## Table of contents

1. [What you're installing](#1-what-youre-installing)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Launching the app](#4-launching-the-app)
5. [First run: set your Workspace folder](#5-first-run-set-your-workspace-folder)
6. [Settings, in full](#6-settings-in-full)
7. [Ingestion tutorial](#7-ingestion-tutorial)
8. [Chunking strategies explained](#8-chunking-strategies-explained)
9. [Collections page](#9-collections-page)
10. [Chat tutorial](#10-chat-tutorial)
11. [Simple RAG vs. Agentic RAG](#11-simple-rag-vs-agentic-rag)
12. [End-to-end walkthrough](#12-end-to-end-walkthrough)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What you're installing

Edenview is two things running together as one local app:

- A **Python/FastAPI backend** that does document extraction, chunking, embedding,
  vector storage, and retrieval/chat — all on your machine.
- A **Next.js frontend** (the "portal") — Ingestion, Collections, Chat, and Settings
  pages.

Everything is stored locally under one folder you choose (your **Workspace**):

| Component | What it stores | Engine |
|---|---|---|
| Vector store | Chunk embeddings (dense + sparse), searched at query time | Qdrant, **embedded** (a local folder, no separate server/Docker) |
| Catalog | Databases, Collections, Documents, ingestion jobs, chat sessions | DuckDB (one local file) |
| Document store | Parsed document cache, extracted picture/table crops, preserved original PDFs (for citation grounding) | Plain files on disk |

There is no Docker requirement and no cloud account — the only external dependency is
**Ollama**, itself a local, offline model runner.

---

## 2. Prerequisites

Install these before touching the Edenview code.

### 2.1 Python

Python **3.12** (developed and tested against 3.12.5), 64-bit.

- **Windows / macOS:** download the installer from
  **https://www.python.org/downloads/** and run it. On Windows, tick **"Add
  python.exe to PATH"** on the first installer screen — easy to miss, and without it
  `python`/`pip` won't be found in a terminal afterward.
- **macOS (alternative):** `brew install python@3.12` (via [Homebrew](https://brew.sh)).
- **Linux:** install via your distro's package manager, e.g.
  `sudo apt install python3.12 python3.12-venv` (Debian/Ubuntu), or use
  [pyenv](https://github.com/pyenv/pyenv) if your distro only ships an older Python.

Confirm with:

```bash
python --version
```

(On macOS/Linux this may be `python3 --version` instead, depending on how it was
installed.)

### 2.2 Node.js

Node.js **20 or newer** (the frontend is Next.js 16 / React 19, which need a recent
Node).

- **All platforms (recommended):** download the **LTS** installer from
  **https://nodejs.org** and run it — this installs both `node` and `npm` together.
- **Alternative (any OS):** use a version manager like
  [nvm](https://github.com/nvm-sh/nvm) (macOS/Linux) or
  [nvm-windows](https://github.com/coreybutler/nvm-windows) if you want to switch Node
  versions per project: `nvm install 20 && nvm use 20`.
- **macOS (alternative):** `brew install node@20`.

Confirm with:

```bash
node --version
npm --version
```

### 2.3 Ollama

Ollama runs every local language/embedding model this app uses.

- **Windows / macOS:** download the installer from
  **https://ollama.com/download** and run it — it installs Ollama as a background
  service that starts automatically (including on subsequent reboots), so there's
  nothing further to launch manually.
- **Linux:** either grab the installer from the same page, or run:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
  then start the service (`sudo systemctl start ollama`, or `ollama serve` in a
  terminal you leave running if it isn't set up as a service).

Confirm it's installed and running:

```bash
ollama --version
ollama list
```

`ollama list` should return (an empty list is fine at this point) rather than a
connection error — a connection error means the Ollama service isn't running yet.

### 2.4 Models to pull

**The models below are just one example configuration, tuned for a modest ~6GB-VRAM
GPU** (the machine this app was developed against) — not a requirement. Ollama's model
library is large, and which models you actually pull is entirely your choice, based on
your own hardware and the answer quality you want. Everything below is changeable
later from **Settings**, model by model, as long as whatever you swap in is actually
pulled in Ollama first.

```bash
ollama pull bge-m3            # dense embedding model (also doubles as the tokenizer)
ollama pull qwen3.5:0.8b      # contextual-chunking LLM (used only by the "contextual" strategy)
ollama pull qwen3.5:2b        # Simple RAG chat/answer model
ollama pull qwen3.5:4b        # Agentic RAG model — must support tool-calling
ollama pull qwen3-vl:2b       # picture-description / vision model (optional, for image captioning)
```

You don't strictly need all five on day one — `bge-m3` is required for any ingestion at
all (every strategy embeds with it), and you need *a* Simple RAG chat model and/or *an*
Agentic RAG model depending on which chat mode(s) you plan to use. The contextual and
picture-description models are only needed if you use the `contextual` chunking
strategy or the "Generate image descriptions" ingestion option, respectively.

**If you have more VRAM (or none — CPU-only works too, just slower), pick whatever
models you actually want.** Bigger, more capable models generally answer better; browse
**https://ollama.com/library** for the full catalog. Check **Settings → "This
machine"** after you've launched the app once — it reports your actual RAM/VRAM so you
can judge headroom before committing to something larger.

There are exactly two hard constraints on model *choice*, regardless of size:

- **The Agentic RAG model must be tool-calling-capable.** Checked live via
  `ollama show <model>`'s reported capabilities — the Settings UI's Agent model
  dropdown only lists models that qualify, and won't let you save one that doesn't.
  Most modern instruction-tuned models qualify; a model's own listing page on
  `ollama.com/library` states whether it supports "Tools".
- **Any model you use for image captioning or image-grounded Q&A must be
  vision-capable.** This applies to two independent settings: the **picture
  description model** (Settings → LLMs, used by the "Generate image descriptions"
  ingestion checkbox) and, if you want Agentic RAG's deep-search to answer questions
  about specific images/pages, the **agent vision model** (Settings → Agentic RAG). A
  non-vision model plugged into either of these will fail every image-related call —
  a model's `ollama.com/library` page states whether it supports "Vision" (e.g.
  `qwen3-vl`, `llama3.2-vision`, `llava`, `gemma3` are vision-capable; plain
  `qwen3.5`/`llama3.1`/etc. text models are not). If you don't need image captioning
  or image Q&A at all, you can skip pulling a vision model entirely — it's optional,
  not a hard requirement of the app.

The reranker (`Xenova/ms-marco-MiniLM-L-6-v2`) and the sparse/BM25 model
(`Qdrant/bm25`) are **not** Ollama models — they run via FastEmbed (ONNX, CPU-only) and
download automatically from Hugging Face the first time they're used. No `ollama pull`
needed for either, regardless of which other models you choose.

### 2.5 GPU acceleration (optional, but recommended if you have an NVIDIA GPU)

Document *extraction* (layout analysis, OCR, table structure recognition — via Docling)
runs on `torch`, and is dramatically faster on a CUDA GPU than on CPU. This is separate
from Ollama, which manages its own GPU usage independently.

`pip install -r requirements.txt` (below) installs a CPU-only `torch` build by design —
which CUDA build (if any) actually works depends on your specific GPU/driver, so it's a
second, explicit step. Covered in [Installation](#3-installation) below.

---

## 3. Installation

From the project root:

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. (Optional, NVIDIA GPUs only) install a CUDA-enabled torch build
python scripts/install_torch.py

# 4. Install frontend dependencies
cd edenview-ui
npm install
cd ..
```

Notes:

- **`scripts/install_torch.py`** is safe to run even without an NVIDIA GPU — it
  detects hardware via `nvidia-smi` and leaves the CPU build in place if it finds
  nothing (or if no CUDA channel actually works on your driver). It reinstalls the
  *exact same* `torch` version already resolved by `requirements.txt`, just with CUDA
  support — verified by actually checking `torch.cuda.is_available()`, not just
  trusting the install exited cleanly. It's a one-time step you re-run only if you
  change GPUs or reinstall dependencies; **it needs a backend restart to take effect**
  (a running process doesn't pick up a torch reinstall on disk).
- If it doesn't find a working CUDA build, extraction still works — just on CPU, and
  Settings will show a banner reminding you a GPU was detected but isn't being used.

---

## 4. Launching the app

The simplest way — one command starts **both** the frontend and the backend together:

```bash
cd edenview-ui
npm run dev
```

This opens the portal at **http://localhost:3000**. The backend API comes up at
**http://localhost:8000** alongside it (Swagger UI / interactive API docs at
**http://localhost:8000/docs**, useful if you ever want to call an endpoint directly).

Always launch this way rather than starting the backend separately in another
terminal — two backend processes can end up bound to the same port from two different
Python interpreters, which corrupts the local vector/catalog storage's file locks.
`npm run dev` is wired to invoke the backend from the same virtual environment you set
up in step 3, every time.

To stop, `Ctrl+C` in that terminal. If a previous run didn't shut down cleanly and
ingestion jobs are stuck showing "queued"/"running" that clearly aren't actually
running anymore, use **Settings → Maintenance → "Clear stale ingestion jobs"**.

---

## 5. First run: set your Workspace folder

**Do this before ingesting anything.** Go to **Settings → Workspace**.

The Workspace is the one folder everything else lives under: the vector store, the
catalog (every Database/Collection/Document record), and every document you ingest
(parse cache, extracted images, preserved original PDFs used for citation grounding).
By default it points at `edenview_data/` inside the project folder.

Why this matters *before* you start ingesting:

- **Changing it later does not move your data.** It just points the app at a
  different (likely empty) folder — every collection, document, and chat session you
  already built will look like it vanished, because the app is now reading from
  somewhere else. It's still there under the old path; you'd need to move the folder
  manually and update the setting to see it again.
- **A restart is required** for a new Workspace path to take effect — it's read once
  at process startup, not per-request.
- **Keep it on a local, always-available disk.** A network drive or a removable drive
  that isn't always connected will make the embedded vector store and the DuckDB
  catalog unreliable — both rely on OS-level file locking that doesn't behave
  predictably over a network share, and you can end up with stalls, "database is
  locked" errors, or corruption on a dropped connection.

Use the **Browse…** button for a native folder picker, or type a path directly
(relative paths resolve against the project root; absolute paths are used as-is).
Click **Save workspace folder**, then restart `npm run dev`.

Once this is set the way you want it, you generally never touch it again.

---

## 6. Settings, in full

Everything here is read from and written back to `config.yaml` (with comments
preserved on save). Some changes apply immediately; others need a backend restart —
the Settings page tells you which via a toast message or a "restart required" badge
after you save.

### This machine

Read-only. Reports your CPU core count, RAM (available/total), detected GPU (name +
VRAM, NVIDIA and Apple Silicon supported; AMD/Intel aren't detected), and whether
Docling's extraction is actually using that GPU (**"Extraction acceleration"**: `CUDA
(<gpu>)`, `Apple Silicon (MPS)`, or `CPU only`). If a GPU is detected but extraction
isn't using it, a warning banner appears pointing you at
`python scripts/install_torch.py`. Use this card to judge how much headroom you have
before picking bigger models elsewhere in Settings.

### Workspace

Covered in [section 5](#5-first-run-set-your-workspace-folder) above.

### Performance

Tunes how Docling's document extraction uses your machine's resources. All three are
**auto-detected by default** (shown as a placeholder like `auto (14)`) — leave them
blank unless you have a specific reason to override:

| Field | What it controls | Default |
|---|---|---|
| Extraction threads | Threads Docling uses per document extraction | `CPU count − 2` (min 1) |
| Page batch size | Pages of one document Docling batches together internally | Docling's own default (4) |
| Max concurrent extractions | How many documents can run extraction (the actual resource-hungry step) at the same time | 4 |

**Max concurrent extractions matters more than it looks.** Ingesting many files at
once with no cap runs every extraction simultaneously — this has been confirmed to
starve RAM and oversubscribe a single shared GPU badly enough that the whole batch
finishes *slower* than a lower concurrency setting would. More CPU cores doesn't mean
raising this is safe; RAM and (if you have one) a single shared GPU are the actual
limiting resources. Extra files beyond the cap simply queue and show a "queued" badge
in the Ingestion page's job list until their turn comes.

Extraction threads / page batch size apply to the *next* extraction you run; max
concurrent extractions needs a backend restart (its underlying limiter isn't resizable
while running).

### Embedding & retrieval

The models used for chunking/embedding/retrieval at ingest and query time:

- **Dense embedding model** — the primary semantic embedding model (default `bge-m3`).
- **Dense embedding dimension** — read-only, auto-detected from whichever dense model
  you pick (by actually calling it once and measuring its output). Not independently
  editable, since a mismatch here would either fail loudly (Qdrant rejects the wrong
  vector size) or, worse, silently mix incompatible vector spaces if two models
  happened to share a dimension.
- **Sparse (BM25) model** — keyword-style sparse retrieval, fused with dense results
  via Qdrant's native RRF (Reciprocal Rank Fusion). Default `Qdrant/bm25` (FastEmbed,
  no Ollama pull needed).
- **Reranker model** — a cross-encoder that re-scores (query, chunk) pairs after
  fusion, which is what makes results comparable across multiple collections in one
  search. Default `Xenova/ms-marco-MiniLM-L-6-v2` (FastEmbed, CPU-only).
- **Tokenizer** — used to size chunks in tokens for the structure-aware chunking
  strategies. Kept aligned to the dense embedding model.

**Important:** changing the dense embedding model does **not** retroactively
re-embed anything already ingested. An existing collection stays tied to whatever
model it was built with; switching this setting only affects *new* ingestions. Mixing
models across collections is fine (each collection tracks its own), but you can't
change one already-built collection's embedding model without re-ingesting it.

### LLMs

- **Chat / answer model** — the model that answers your question in **Simple RAG**
  mode, from retrieved context. Overridable per-request from the Chat page too.
- **Contextual chunking model** — used only by the `contextual` chunking strategy, to
  generate one short context sentence per chunk at ingest time.
- **Picture description model** — a vision-capable model that captions retained
  images when you check "Generate image descriptions" during ingestion.

### Agentic RAG

Configures the model driving the multi-step **Agentic RAG** chat pipeline — separate
from the plain chat model above, since Simple RAG and Agentic RAG are independently
configurable and can even have different models resident in memory at once.

- **Agent model** — must support tool-calling. The dropdown only lists Ollama models
  that actually report this capability (checked live), and the backend refuses to
  save anything that doesn't qualify — there's no reasonable degraded mode for a
  tool-calling pipeline without tool-calling.
- **Agent vision model** — optional, used only for image-grounded sub-questions
  during Agentic RAG's "deep search" tools. Leave unset to reuse the agent model's own
  vision capability if it has one; set a dedicated smaller/larger vision model
  independently if you have VRAM headroom for two resident models.
- **Max refinement iterations** — the ceiling on how many reword→search→evaluate
  passes one sub-question's research loop can take before giving up. The loop
  normally exits early once its own evaluation step decides the findings are
  sufficient; this is just a safety cap for a weak model that never converges. Needs
  a restart to take effect.

Changing **agent model** or **max refinement iterations** always needs a backend
restart (the agentic pipeline builds one shared, cached model instance at process
startup) — the Chat page's scope panel shows these as read-only with a link back here
for exactly that reason.

### Connection

- **Ollama host** — where Ollama is running (default `http://localhost:11434`). Only
  change this if Ollama is running elsewhere (e.g. a different machine on your
  network, or a nonstandard port).
- **Model idle timeout (`keep_alive`)** — how long a model stays loaded in RAM/VRAM
  after its last use before Ollama evicts it. Default `30m`. Accepts Ollama's own
  duration syntax (`30m`, `1h`, `-1` for never, `0` for immediate unload). Applies to
  every model call this app makes and takes effect immediately, no restart needed.
  Raised from Ollama's own 5-minute default because that was evicting models between
  ordinary back-and-forth chat turns, forcing a disk reload mid-conversation.

### Maintenance

- **Clear stale ingestion jobs** — marks any job left "queued"/"running" by a crashed
  or restarted backend as "error", so it stops cluttering the Ingestion page's job
  list as if it were still active. Only touches job status rows — never real
  documents, collections, or chat data. Safe to click any time, including when
  nothing is actually stale.
- **Unload all Ollama models** — immediately frees RAM/VRAM held by every currently
  loaded model (embedding, chat, contextual chunking, picture description, agent) in
  one click, instead of waiting out each one's idle timeout. Doesn't interrupt a call
  that's actively generating right now.

The sidebar also has a live **system monitor** (RAM/VRAM usage over time, plus a list
of currently Ollama-loaded models with individual "Unload" buttons) — useful for
watching headroom in real time while you work, separate from the one-click "unload
all" in Settings.

---

## 7. Ingestion tutorial

Go to the **Ingestion** page. This is where you turn raw files into a searchable
collection.

### 7.1 The DB / Collection concept

Two nested naming concepts, worth understanding before your first upload:

- **Database ("DB")** — a purely organizational label, catalog-only. Qdrant itself has
  no concept of a "database" — think of it like a folder grouping related
  collections together (e.g. one DB per project, department, or document set).
- **Collection** — the actual underlying Qdrant collection. One collection = one
  chunking strategy + one embedding model, holding one or more documents. Collection
  names are **globally unique across every DB**, not just within one.

This split lets you re-ingest the *same* source document into two different
collections under different chunking strategies (e.g. compare `hybrid_docling` vs.
`contextual` on the same PDF) while keeping them organized under one DB.

### 7.2 Uploading

1. Drag and drop one or more files onto the upload area, or click it to browse.
   Supported formats (via Docling): PDF, DOCX, PPTX, XLSX, images (JPEG/PNG), plain
   text, and Markdown.
2. Type or pick a **Database** name (an existing one, or type a new one to create it).
3. Type or pick a **Collection name** (same — existing to add to it, or new to create
   it). Every file you selected in step 1 is ingested into this *same* collection, as
   separate documents.
4. Pick a **Chunking strategy** (see [section 8](#8-chunking-strategies-explained)).
5. Optionally check:
   - **Generate image descriptions** — runs your configured picture-description
     vision model once per retained image, folding a generated caption into a
     searchable chunk. Meaningfully slower (one extra model call per image); the
     vision model must already be pulled in Ollama or every call fails. Leave off
     unless you actually need images to be searchable by content.
   - **Scanned document (force full-page OCR)** — leave this **off** for ordinary
     digital documents. By default, OCR already runs automatically wherever Docling
     detects it's needed (it distinguishes real digital text from scanned bitmap
     regions per page). Only check this if you *know* a document is a scan (or a
     shaky one) and want every page forcibly OCR'd regardless of that detection — an
     opt-in safety net, not something to enable by default.
6. Click **Start ingestion**.

The request returns almost instantly with a queued job — the actual parsing,
chunking, and embedding runs in the background. A job list below the form tracks
progress live: a stage stepper (`extracting` → `chunking` → `embedding`), elapsed
time, and (during the embedding stage only, since it's the one phase with a real
known total) a live chunk count/percentage. A job genuinely waiting behind **Max
concurrent extractions** other jobs shows a distinct "queued" badge until it actually
starts.

Every file uploads and ingests **concurrently** with the others in that same batch —
useful for bulk-loading a folder of documents, bounded by the Performance setting
above so it doesn't overwhelm your machine.

If a job fails (bad file, model unreachable, etc.), its row gets a **Retry** button —
this re-runs the job from the originally uploaded file without needing to re-upload
(PDF sources that failed *after* extraction completed only; a failure mid-extraction
or a non-PDF source needs a manual re-upload). A **Cancel** button appears on any
still-running/queued job — cancellation is cooperative (it stops at the next safe
checkpoint, not necessarily instantly, since a Docling extraction call in progress
can't be interrupted mid-call).

### 7.3 SQL DB (not yet available)

The Ingestion page has a "Vector DB / SQL DB" toggle. **SQL DB is a placeholder for a
future feature** (DuckDB-native tabular ingestion, for querying structured data
directly rather than chunking it into a vector store) — not usable yet. Stick to
"Vector DB" for everything today.

---

## 8. Chunking strategies explained

Four strategies, selectable per ingestion. All four embed with the same configured
dense model and get hybrid dense+sparse search at query time — what differs is *how*
each document gets split into chunks in the first place.

| Strategy | How it works | Best for | Tunable in the UI |
|---|---|---|---|
| **`recursive_overlap`** | Fixed-size text splitting with overlap between consecutive chunks — falls back through paragraph → sentence → word → character boundaries. Simple, fast, no document-structure awareness. | Plain, dense prose with little internal structure. A reasonable default when you're not sure. | Chunk size (characters, default 512), chunk overlap (default 50) |
| **`hybrid_docling`** *(default)* | Token-aware chunking that respects the document's actual structure — headings, sections, tables — and merges undersized adjacent chunks. Boundaries derive from the tokenizer/embedding model, not a fixed character count. | Structured documents (reports, regulations, manuals) where headings and sections carry real meaning. | Nothing extra — token budget is inherited from the tokenizer/embedding model. |
| **`parent_child`** | Small "child" chunks are what's actually embedded and matched at search time (precise), but each carries a link back to a larger "parent" chunk. When a child matches, the LLM is fed the full parent for more context than the matched span alone. | Long documents / detailed Q&A, where you want a precise match but a wider context window for the answer. | Child chunk size (tokens, default 180), parent chunk size (tokens, default 2000) — child must stay smaller than parent. |
| **`contextual`** | Same structure-aware chunking as `hybrid_docling`, plus **one extra LLM call per chunk** that prepends a short sentence describing where that chunk sits in the overall document, before embedding. | Complex/technical/regulatory documents where a chunk read in isolation is ambiguous (e.g. "the exemption above applies" — *which* exemption?). | Nothing extra in the UI; uses the Settings → LLMs → "Contextual chunking model". Slower ingestion — one LLM call per chunk. |

A composable add-on, independent of which strategy above you pick: **"Generate image
descriptions"** (the checkbox from section 7.2) turns each retained picture into its
own additional, separately embedded, searchable chunk — it doesn't replace or change
your chosen strategy, it adds to it.

You can freely re-ingest the same document under a *different* collection with a
different strategy to compare results side-by-side — that's exactly what the
DB/Collection split in section 7.1 is designed to support. At query time, a
**strategy filter** (covered in [section 10](#10-chat-tutorial)) lets you scope a
search/chat to only collections built with a specific strategy, which matters mainly
when searching across a whole DB that mixes strategies.

---

## 9. Collections page

The **Collections** page is a browser over the catalog: every Database you've
created, and every Collection inside it, with:

- Metadata: chunking strategy, embedding model, chunk count, document count, created
  date.
- The list of source documents ingested into each collection.
- A **paginated chunk preview** — the actual chunk text/payload as stored in Qdrant
  (not the catalog — this reads straight off the vector store so you can see exactly
  what got embedded), including any linked images.
- **Delete** a collection (removes both the Qdrant collection and its catalog rows —
  confirmation required; doesn't delete a document's extracted images if another
  collection still references the same source file).
- **Delete** a DB — only succeeds once every collection under it is gone first.

Use this page to sanity-check that a chunking strategy actually produced the
boundaries you expected before relying on it in chat, and to clean up experimental
collections you no longer need.

---

## 10. Chat tutorial

Go to the **Chat** page. Layout: a collapsible **chat history** rail (left), the
transcript (center), and a collapsible **chat settings** panel (right) — plus, in
Agentic RAG mode, a live **agent pipeline** panel.

### 10.1 Chat settings panel

- **Chat mode** — toggle between **Simple RAG** and **Agentic RAG** (see
  [section 11](#11-simple-rag-vs-agentic-rag) for the difference).
- **Scope** — check a Database to search everything in it, or expand it to hand-pick
  specific Collections. You can mix collections from different DBs in one search.
  This scope is what actually gets sent to the backend as a flat list of collection
  names.
- **Top K** — how many chunks to retrieve and hand to the LLM (default 5).
- **Use reranker** — whether to re-score fused dense+sparse results with the
  cross-encoder reranker before truncating to Top K. Leave this on; it's what makes
  merging results across multiple collections meaningful (raw fusion scores are only
  comparable *within* one collection's own query).
- **Strategy filter** — restrict the search to collections built with one specific
  chunking strategy. Matters mainly when your scope spans a whole DB that mixes
  strategies — without it, near-duplicate chunks of the same underlying content from
  competing strategies can crowd out genuinely different results in the merged top-K.
- **Chat model** (Simple RAG only) — override the configured default chat model for
  this one conversation, picked from whatever's currently pulled in Ollama. In
  Agentic RAG mode this becomes a read-only display of the shared agent
  model/vision-model/max-iterations settings, with a link to Settings — the agentic
  model is a shared, restart-required config, not a per-message choice.

Your scope selection is saved in the browser and persists across page reloads/
navigation.

### 10.2 Sending a message and reading the answer

Type your question and send. The answer streams in with **inline numbered
citations** (`[1]`, `[2]`, …) — click one to open the **grounding panel**: it renders
the actual source PDF page with the matched chunk's location highlighted, so you can
verify the answer against the real document rather than trusting the citation blindly.
An expand button opens that page near-fullscreen for easier reading. (Grounding is
PDF-specific and depends on chunk-level bounding boxes — available for
`hybrid_docling`/`parent_child`/`contextual` hits; `recursive_overlap` hits show a
page number only, since fixed-size chunks don't have one well-defined source region.)

Answers render full Markdown (tables, formatting) via GitHub-flavored Markdown
support, not raw text.

### 10.3 Chat history

The left rail lists your past sessions (10 at a time, "Load more" to page further).
Click one to reload its full transcript. A session is created automatically on your
first message in a new chat — there's no separate "New chat" setup step.

If you reload the page or switch away mid-answer during an Agentic RAG turn, coming
back reattaches to the still-running turn and replays what already happened, then
continues live — the turn itself keeps running server-side regardless of whether
anyone's watching, so you never lose the answer by navigating away.

---

## 11. Simple RAG vs. Agentic RAG

### Simple RAG (default)

One retrieval pass, one LLM call: search → (optional rerank) → feed the top-K chunks
to the chat model → answer. Fast, predictable, and the right default for
straightforward, single-topic questions clearly covered by your documents.

### Agentic RAG

A multi-step pipeline (built on Google's ADK) that plans, searches, checks its own
work, and digs deeper before answering — slower, but noticeably better on compound or
ambiguous questions. Roughly, per turn:

1. **Decompose** — splits your question into one or more independent
   sub-questions (a two-part question about unrelated topics becomes two separate
   research threads; a single-topic question stays one).
2. For **each sub-question, independently**:
   - **Search** — runs hybrid retrieval (same dense+sparse+RRF+rerank pipeline as
     Simple RAG) and drafts a cited answer to just that sub-question from what it
     found.
   - **Evaluate** — a dedicated step judges whether the findings are actually
     sufficient to answer that sub-question.
   - If not sufficient: **Reword** (tries different search phrasing) and/or **Deep
     search** (narrower follow-up tools — reading specific pages in detail, pulling
     linked images, a literal text/regex scan of one document) kick in, then
     re-evaluates. This repeats up to the **Max refinement iterations** cap from
     Settings, exiting as soon as the evaluation step is satisfied (most turns exit
     well before hitting the cap).
3. **Answer formatter** — consolidates every sub-question's drafted answer into one
   cohesive final answer with clean, deduplicated, sequential citations.

While a turn runs, the **agent pipeline panel** (right side, Agentic RAG mode only)
shows a live flowchart lighting up the active step, and a collapsible **reasoning
trace** shows granular status down to individual tool calls — which sub-question
they belong to, what each step is doing, and how long it took. This is genuinely
useful for understanding *why* an answer came out the way it did, not just cosmetic.

**When to reach for Agentic RAG:** multi-part questions ("what's X, and separately how
does Y compare"), questions where a first search might not find the right terminology
on the first try, or anything where you want visible reasoning/citations you can
audit step by step. **When Simple RAG is enough:** a single, clearly-scoped factual
question — Agentic RAG's extra steps just add latency without changing the answer.

A turn can genuinely take anywhere from tens of seconds to several minutes depending
on how many sub-questions get decomposed and how many refinement passes each one
needs — this is expected, not a hang; watch the live pipeline panel for progress.

---

## 12. End-to-end walkthrough

A concrete first session, start to finish:

1. **Install prerequisites** ([section 2](#2-prerequisites)): Python, Node, Ollama,
   and pull at minimum `bge-m3` + `qwen3.5:2b` (Simple RAG) or `+ qwen3.5:4b` (Agentic
   RAG too).
2. **Install & launch** ([sections 3–4](#3-installation)): `pip install -r
   requirements.txt`, `npm install` in `edenview-ui/`, then `npm run dev`.
3. Open **http://localhost:3000**, go to **Settings → Workspace**, confirm/set your
   data folder, save, and restart `npm run dev` if you changed it.
4. Go to **Ingestion**. Create a new Database (e.g. `my-first-db`) and a new
   Collection (e.g. `my-first-collection`). Drop in a PDF. Leave the strategy at the
   default `hybrid_docling`. Click **Start ingestion**.
5. Watch the job list until it reaches `done` — for a typical document, extraction
   is the slowest stage.
6. Go to **Collections**, open your new collection, and check the chunk preview to
   confirm the content looks right.
7. Go to **Chat**. In the settings panel, expand your DB and check your new
   collection (or check the DB itself to include everything in it). Leave mode on
   **Simple RAG** for a first test.
8. Ask a question you know the document answers. Confirm the answer is grounded —
   click a citation to see the highlighted source page.
9. Try the same question in **Agentic RAG** mode and compare — watch the live
   pipeline panel to see the reword/search/evaluate loop in action.
10. If you want to compare chunking strategies: go back to **Ingestion**, re-upload
    the same file into a *different* collection name (e.g. `my-first-collection-pc`)
    with `parent_child` selected, then in Chat's scope panel switch which collection
    is checked and compare answers.

From here, everything else — Performance tuning, model swaps, workspace changes — is
optional and covered in [section 6](#6-settings-in-full) when you need it.

---

## 13. Troubleshooting

**"Ollama unreachable" banner in Settings.** Confirm `ollama list` works in a
terminal. If Ollama is running on a different host/port, update **Settings →
Connection → Ollama host**.

**Ingestion fails immediately on a specific model call.** Almost always a model
that's referenced in `config.yaml`/Settings but not actually pulled yet — check the
error message for the model name and `ollama pull` it.

**Image descriptions all fail.** The configured picture-description model (Settings →
LLMs) isn't pulled, or isn't vision-capable. Pull `qwen3-vl:2b` (the default) or
whatever you've configured there.

**Agentic RAG won't save a model in Settings, or fails to start.** The Agent model
must report tool-calling support — pick one from the filtered dropdown rather than
typing an arbitrary name; the dropdown only lists models that actually qualify.

**A GPU is detected but extraction still shows "CPU only".** Run
`python scripts/install_torch.py` from the project root, then fully restart
`npm run dev` (a running backend won't pick up a torch reinstall on disk).

**Ingested collections/chats seem to have vanished after a Settings change.** Check
**Settings → Workspace** — if the folder path changed, your data is still on disk
under the *old* path; either move it to the new path or point Workspace back at the
old one.

**A job is stuck showing "queued" or "running" and never finishes.** Usually left
over from a backend crash or restart mid-job. Use **Settings → Maintenance → "Clear
stale ingestion jobs"**, then re-upload if needed (or use the job's **Retry** button
if it's now marked `error` and was a PDF that got past extraction).

**Everything feels slow / VRAM keeps filling up.** Check the sidebar's system
monitor and **Settings → "This machine"** for real headroom. Lower **Max concurrent
extractions** if ingesting many files at once, use **Settings → Maintenance →
"Unload all Ollama models"** to reclaim memory between sessions, or pick smaller
models for chat/agent/picture-description if you're consistently VRAM-constrained.
