# Edenview — Progress & Next Steps

> Status snapshot of the whole Edenview stack — ingestion, chunking, the catalog and
> vector store, retrieval/RAG, every API router, and the Next.js portal (Ingestion,
> Collections, Chat, Settings). Renamed from `edenview_ingestion_progress.md` once it
> outgrew "ingestion" — if you're following an old code comment pointing at that name,
> this is the same doc. Companion to `edenview_plan.md` (overall product vision) and
> `rag_vector_db_ingestion_plan.md` (earlier reference design this work supersedes for
> chunking). Update this as work continues rather than creating a new snapshot each time.

---

## 1. Architecture decisions locked in

| Decision | Choice | Why |
|---|---|---|
| Build scope (current phase) | Minimal local-first: DuckDB + Qdrant only | Postgres/Redis/Celery/Next.js from `edenview_plan.md`'s full stack are deferred until the RAG core works end-to-end |
| Deployment model | **One long-lived Python process** handles both ingestion and chat querying (ingestion as a background task within that process, not a separate worker) | This is what makes embedded-mode storage (below) safe — its "one process at a time" restriction is a non-issue once there's only ever one process. Also the actual target: a non-technical user runs `pip install` + one command, opens `localhost` — no services to separately manage |
| Vector store | **Qdrant, embedded** (`QdrantClient(path=...)`, a local folder) — **not** Docker/server mode | Originally built as a Docker container (server mode), reasoning: embedded mode only allows one *process* to hold the store open, which breaks if ingestion and querying are separate processes. Reversed once the deployment model above was clarified — Docker is a real install barrier for a non-technical user, and the concurrency concern doesn't apply to a single-process app. Same engine either way, same hybrid dense+sparse query API |
| Catalog store | DuckDB, single local file | Backs the DB/Collection metadata catalog now *and* doubles as the engine for a future Databricks-Genie-style tabular SQL agent — one engine instead of two |
| "DB" vs "Collection" (UI concept) | **DB name** = a logical grouping label, catalog-only (like a BigQuery dataset). **Collection name** = the actual Qdrant collection (one chunking strategy + one embed model + one or more docs), globally unique across all DBs | Qdrant itself has no "database" layer, only *instance → collections → points* |
| Catalog vs. content browsing | DuckDB tables drive the DB/Collection picker UI (what do I have). Row-level chunk/payload browsing pages directly off Qdrant's `scroll()` API (what's inside one collection) | Avoids duplicating chunk content into DuckDB just to display it |
| DuckDB for vectors too? | No — Qdrant stays the vector store | DuckDB's `vss` extension has HNSW persistence marked experimental (crash/corruption risk on unclean shutdown) and no native dense+sparse hybrid fusion like Qdrant's `FusionQuery(RRF)` |
| Embedding model | **bge-m3** dense (via Ollama) + **FastEmbed BM25** sparse + Qdrant native RRF fusion — hybrid from the start | Diverges from `edenview_plan.md`'s stated default (`qwen3-embedding:0.6b`, dense-only). Already proven end-to-end (the old `ingest/shared.py`), fits the 6GB VRAM budget, doubles as the tokenizer `chunking` already used |
| Retrieval filtering granularity | **Collection-level**, **document-level** (`file_hash` payload filter), and **chunking-strategy-level** (catalog `chunking_strategy`, mainly for DB-wide fan-out) — all three compose | A DB can hold collections built with different strategies over the *same* documents (e.g. comparing `hybrid_docling` vs `contextual`); without the strategy filter, a DB-wide search's merged top-k can surface several near-duplicate chunks of the same content instead of genuinely different results |
| Reranker | Cross-encoder via **FastEmbed** (`Xenova/ms-marco-MiniLM-L-6-v2`), ONNX/CPU | No GPU cost, no new dependency. Also solves cross-collection comparability: raw RRF fusion scores are only meaningful *within* one collection's own query, so merging results across a DB-wide search needs the reranker's more absolute (query, chunk) scores |
| Multi-collection search | Fan out hybrid search to each collection independently, concatenate candidates, rerank the merged set globally, take top-k | Matches `edenview_plan.md`'s retrieval diagram extended to "search a DB" |
| Async ingestion inside one process | FastAPI `BackgroundTasks` (Starlette runs them in a worker thread), not Celery/a separate worker | Consistent with the single-process deployment model — `POST /ingest` does the fast synchronous part and returns a `job_id` immediately, `GET /jobs/{id}` polls |
| Job progress granularity | **Stage name always** (`extracting`/`chunking`/`embedding`) + **real `(current, total)` counts only during `embedding`** — no fabricated overall %, no ETA | Embedding is the only phase with a genuinely known total upfront. Extraction has no per-page callback from Docling and no known page count until parsing finishes — a percentage there would be a guess, not real progress |
| Picture description mechanism | **Our own module** (`docling_parsing/picture_description.py`), calling Ollama's *native* API directly on saved crop files — **not** Docling's `do_picture_description` pipeline option | Confirmed broken for reasoning-capable local VLMs (e.g. `qwen3-vl`): Ollama's OpenAI-compatible endpoint puts the model's real answer in a separate `reasoning` field, and Docling's parser only reads `content`, so it silently returns empty every time — a known issue pattern, not specific to this codebase. Ollama's native API correctly separates `content` from `thinking`; verified working end-to-end |
| Frontend stack | Next.js 16 (App Router, Turbopack) + shadcn/ui (on **Base UI**, not Radix) + Tailwind v4 + TanStack Query | Matches the user's explicit stack request; Base UI's `render` prop replaces Radix's `asChild` throughout |
| Chat persistence | Two new DuckDB tables (`chat_sessions`, `chat_messages`) alongside the existing catalog, not a separate store | One process, one catalog file — consistent with the rest of the catalog design; `POST /chat` lazily creates a session on first message and returns `session_id` |
| Visual grounding (click a citation → see the source page, highlighted) | **Python-driven, on-demand**: `pypdfium2` renders one page of a *preserved original PDF* to PNG + draws the chunk's bbox with PIL, per request — **not** pre-rendering every page as an image at ingest time | Explicit user preference ("why rebuild the wheel with bitmaps") over a bulk pre-render approach. Bbox is Docling's own normalized top-left-origin box (`prov.bbox.to_top_left_origin(...).normalized(...)`), stored on each chunk/hit; PDF-only, `recursive_overlap`-chunked collections stay page-number-only (no single well-defined source region per chunk) |
| Keeping a document's original file | `pipeline._preserve_original_pdf()` copies the source into `documents_dir/originals/{file_hash}.pdf` right after `register_document()`, PDF sources only | Needed for visual grounding (nothing Docling caches can reconstruct a renderable PDF) and, as of the retry feature below, for retrying a failed job without re-uploading |
| Ollama idle unload timing | Configurable `ollama.keep_alive` in `config.yaml` (default `30m`) + Settings UI field, threaded into every `client.chat()`/`client.embed()` call via `keep_alive=get_ollama_keep_alive()` | Ollama's own default (5 minutes, resets on every call rather than running on a fixed schedule) was evicting models from VRAM between ordinary back-and-forth chat turns |
| System monitor + model memory | Sidebar polls `GET /system/info` (RAM/VRAM history via `psutil`/`nvidia-smi`, plus `ollama.Client.ps()`'s loaded models) every ~4s; `POST /system/ollama/unload` calls `client.generate(model=X, prompt="", keep_alive=0)`, Ollama's documented immediate-unload pattern | Lets a user see and free VRAM without leaving the app |
| Multi-file ingestion concurrency | `POST /ingest` stays single-file; the frontend fires one call per selected file concurrently, relying on FastAPI `BackgroundTasks` already running each on its own worker thread | No new queue/worker infrastructure needed — but concurrency bugs found this way (see section 3) meant the shared DuckDB connection and shared embedded-Qdrant client both needed explicit thread-safety fixes, since neither is safe for unsynchronized concurrent use out of the box |
| Failed-job retry | `POST /jobs/{job_id}/retry` re-runs `ingest_document()` from the *preserved original file*, as a brand-new job row (the failed one stays as history) | Only possible for a job that failed after extraction completed (`doc_id` already set, so a preserved original exists) — a job that failed during extraction itself has nothing to retry from and needs a re-upload |
| Dev workflow | `npm run dev` (in `edenview-ui/`) runs the frontend *and* the backend together via `concurrently`, with the backend explicitly invoked as `..\venv\Scripts\python.exe -m uvicorn api.app:app --app-dir ..` | Prevents accidentally starting the backend against a different (non-venv) Python interpreter in a stray terminal — see the "duplicate backend process" bug in section 3, which this exists to prevent from recurring |
| Package audience & hardware defaults | Edenview is meant to be installed and run by any enterprise user on their own machine, not tuned to one dev machine — `num_threads`/`page_batch_size`/`max_concurrent_extractions` all auto-detect from *this* machine (CPU count, a flat conservative default) rather than a hardcoded number, `accelerator_device=AUTO` lets Docling pick CUDA/MPS/CPU per-machine, and GPU acceleration is an opt-in install step (`scripts/install_torch.py`), never baked into `requirements.txt` | See `torch_installation.md` for the full reasoning; pip itself has no way to conditionally install a CUDA vs. CPU wheel from a static requirements file |
| Extraction concurrency limiting | A `threading.Semaphore` (`pipeline._EXTRACTION_SEMAPHORE`, sized from `settings.get_max_concurrent_extractions()`, default 4) gates only the actual Docling extraction call — chunking/embedding proceed unthrottled once extraction finishes | Confirmed necessary by reproduction, not theory: ingesting 22 files at once with no limit ran every extraction simultaneously, RAM fell to ~4.6GB free and the one shared GPU was oversubscribed 22 ways, making the whole batch far slower than a few at a time. More CPU cores doesn't mean more concurrent extractions are safe — RAM and a single shared GPU are the actual limiting resources and neither scales with core count |
| Ingestion cancellation | Cooperative, not preemptive: an in-memory `job_id → threading.Event` registry (`pipeline._cancel_events`) is checked at safe checkpoints (before/after the extraction semaphore, before chunking, before each embedding batch) | Docling's own extraction call can't be interrupted mid-call (nothing inside it checks for cancellation) — a job already extracting stops as soon as that one call returns, not instantly. In-memory only: a cancel button only works against the same backend process actually running the job |
| Embedding batch-size mismatch | Every `embed_dense()` call now passes `options={"num_batch": 4096}` to Ollama, matching bge-m3's actual context window | Ollama's `llama-server` defaults its internal "physical batch size" to 2048 — a compute-batching knob, *not* the model's real context window (4096) — so a chunk between those two numbers (real chunks seen at 2133–3877 tokens, likely an oversized table `HybridChunker` couldn't fully split, a known upstream Docling limitation) was being rejected by an artificially low limit the model could actually handle fine |
| Agentic RAG framework | Google's **ADK** (`google-adk==2.5.0`) + **LiteLLM** (`litellm==1.77.7`, pinned specifically for wheel availability — see below) wrapping a local Ollama model via the `ollama_chat/` provider prefix | Matches `edenview_plan.md`'s original ADK bet; `ollama_chat/` (not bare `ollama/`) per ADK's own docs, which warn the latter causes infinite tool-call loops |
| Agentic RAG effort tiers | **Resolved in plain Python, not by the LLM** — three pre-built agent trees (`low`/`medium`/`high`), selected by a request field, not a prompt the model interprets | A small local model can't reliably self-regulate loop iteration counts or tool-call budgets; Python-side selection keeps that deterministic |
| Agentic RAG session persistence | ADK's `DatabaseSessionService` on a **dedicated local SQLite file** (`adk_sessions.db`, via `sqlite+aiosqlite`), same `session_id` as the existing DuckDB `chat_sessions` row | ADK's session persistence needs an async SQLAlchemy driver; no supported async DuckDB one exists. DuckDB stays the durable, UI-facing store (titles/message text/citations); the SQLite file is ADK's own internal conversation memory so multi-turn context survives a backend restart too |
| Agentic RAG tier structure | `low` = one flat agent (retrieve → answer, no subagent split). `medium`/`high` = thin root (holds session history) wrapping a `reframe → deterministic dispatch → critic/refiner loop → answer` subagent tree via `AgentTool`, differing only in `max_iterations` (2 vs 4) and `high`'s extra `get_page_context`/`inspect_image` tools | Reproduced directly why `low` is NOT split like the others: even with an explicit "relay verbatim" instruction, a root-agent relay hop unreliably narrated/second-guessed the subagent's already-correct answer on this model; removing the hop fixed it. `medium`/`high` keep the split anyway since their retrieved-text volume is large enough that root-context bloat matters more than the relay-reliability cost, and `AgentTool(skip_summarization=True)` (which removes the relay hop's extra generation entirely) turned out to work fine there once a real, unrelated bug — see below — was fixed |
| Agentic RAG retrieval fan-out | `RetrievalDispatchAgent`, a **custom non-LLM `BaseAgent`**, deterministically loops over reframe's split sub-questions calling `search()`/`search_db()` directly — no tool-calling involved in this hop at all | Removes the highest-risk tool-call shape (transcribing a JSON query list into a function argument) from the LLM's responsibility entirely. Sequential, not threaded — every real Qdrant call already goes through one process-wide `client_lock`, so threading would mostly re-serialize there anyway |
| Agentic RAG tool-response format | `retrieve()`/`get_page_context()` return a **plain numbered-text block** (mirroring `generate.py`'s own proven `_format_context()` style), not structured JSON per hit | Reproduced directly: returning JSON-shaped dicts made the local model treat tool responses as "data to process/describe" instead of "context to answer from" — a plain-text block reads the same whether it arrives via a tool response or directly in a prompt |
| Agentic RAG image-into-context | `tools.inspect_image` saves the chunk's image as an ADK **Artifact** (`InMemoryArtifactService` — ephemeral by design, only needs to live one turn); `callbacks.inject_pending_images` (a `before_model_callback`) loads it back and appends it to the model's next turn. Gated by `model_supports_vision(config.yaml's agent.model)`, checked live via `ollama show`'s `capabilities` field | A tool cannot return raw image bytes directly (ADK errors) — must go through an Artifact. Confirmed via a real Google codelab pattern, then a live spike test (a synthetic image with unmemorizable text) proved the whole chain works once a real litellm bug (below) was patched |
| Real bugs found in ADK/LiteLLM/Ollama integration (all reproduced, not guessed, all patched from `edenview_RAG/agentic_rag/config.py` — never by editing files under `venv/`, so every fix ships with the repo and applies automatically on any fresh install) | (1) litellm doesn't recognize any model this project uses as tool-calling-capable (its bundled registry predates them) — silently falls back to a legacy JSON-in-prompt hack that crashes on real `tool_calls` responses; fixed via `litellm.register_model(...)`. (2) `litellm.llms.ollama.chat.transformation._extract_reasoning_content` raises `KeyError: 'content'` on a tool-call-only assistant turn (`reasoning_content` set, no `content` key) — a normal turn for any tool-calling agent; patched. (3) Ollama's runtime `num_ctx` silently defaults to ~2048 regardless of the model's real max context, silently truncating large retrieved-context prompts *and their trailing question*; fixed via `config.yaml`'s `agent.num_ctx: 16384` (raising it further to "fix" a deeper-pipeline issue instead caused VRAM/CPU-offload timeouts on this 6GB GPU — the actual fix there was `include_contents="none"` on steps that don't need full history, not a bigger window). (4) `litellm`'s `GLOBAL_LOGGING_WORKER` is a process-wide singleton whose queue binds to whichever event loop touches it first — breaks every request after the first `asyncio.run()`-driven call in a long-running server; fixed by resetting its queue reference every turn. (5) litellm's Ollama image handling forwards a `data:image/png;base64,...` URI to Ollama's `images` field *verbatim, prefix included* — Ollama's decoder expects pure base64 and rejects it outright; patched to strip the prefix | Every fix has a full docstring at its call site in `config.py` explaining the reproduction, not just the patch — see that file directly for exact tracebacks/line numbers |

---

## 2. What's built

### `edenview_ingestion/docling_parsing/`

- `DoclingExtractor.extract(source) → ExtractionBundle` — parses any Docling-supported
  format into a `DoclingDocument` + `DocumentMetadata` + `TableRecord`s + `PictureRecord`s.
- **`picture_description.py`** — `generate_picture_descriptions(pictures)`, a
  post-extraction step (not a Docling pipeline option, see the architecture-decisions
  table above for why) that calls Ollama's native chat API directly on each retained
  picture's saved crop file, mutating `PictureRecord.description` in place. Only
  operates on pictures `images.build_picture_records()` already kept — logos/icons/etc.
  (`picture_exclude_labels`) never reach this step, since they were never saved as
  `PictureRecord`s in the first place.

### `edenview_ingestion/chunking/`

Four base strategies (`recursive_overlap`, `hybrid_docling`, `parent_child`, `contextual`)
sharing one `Chunk`/`ChunkImage` output shape, dispatchable by name via the `CHUNKERS`
registry. Plus `image_descriptions.py`, a composable addition (not a 5th strategy) that
turns each described picture (`PictureRecord.description`, from the module above) into
its own standalone embeddable chunk (`kind="image_description"`, `strategy="image_description"`
on the chunk itself — distinct from whichever base strategy the collection was built
with). Shared `_linking.py` resolves `Chunk.doc_item_refs` against extracted
pictures/tables to populate `Chunk.images` — how a retrieved chunk pulls its images back
in by file path, with no multimodal embedding involved anywhere.

### `edenview_ingestion/catalog/`

DuckDB-backed metadata store (`dbs`, `collections`, `documents`, `collection_documents`,
`ingestion_jobs`, `parent_chunks` — see `schema.py` for the full column list and
rationale per table). `crud.py` provides full CRUD including **delete**
(`delete_db`, `delete_collection_catalog_rows` — catalog rows only, see
`pipeline.delete_collection()` for the combined Qdrant+catalog operation). `delete_db`
refuses if collections still reference it.

`ingestion_jobs` carries `stage`/`stage_current`/`stage_total` (a computed `stage_pct`
lives on `IngestionJobRecord`, not stored) — see the job-progress row in section 1.
Schema changes ship with idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
migrations, safe to run against an already-existing `catalog.duckdb`.

Real bug caught and fixed: collection `chunk_count`/`doc_count` were originally tracked
by incrementing on every ingest call, which double-counts on re-ingestion (idempotent
upsert, not append). Replaced with `recompute_collection_counts()`, deriving both from
`SUM`/`COUNT` over `collection_documents` — the actual source of truth.

### `edenview_ingestion/vectorstore/`

Embedded Qdrant client, `embed_dense`/`embed_sparse`/`embed_texts` (model names from
`config.yaml`), collection lifecycle (`create_collection` idempotent; `delete_collection`
deregisters *and* removes the on-disk folder, since Qdrant's local-mode delete doesn't
reclaim disk space on its own), and `Chunk` → `PointStruct` conversion +
`upsert_chunks()` (skips `parent`-kind chunks — catalog-only, never embedded).

Real bug caught and fixed: same-process folder removal right after `delete_collection()`
raced a Windows file-lock release (a fresh process could always remove it immediately;
same-process removal right after sometimes lost that race) — fixed with a short bounded
retry.

### `edenview_ingestion/pipeline.py`

The orchestrator: `ingest_document(source, db_name, qdrant_collection_name, strategy, ...)`
runs extract → chunk → embed → write to both Qdrant and the catalog in one call.
`include_image_descriptions=True` runs `generate_picture_descriptions()` right after
extraction, then folds the resulting chunks into the strategy's own output.
`prepare_ingest()` is the fast synchronous half of the two-phase API flow (resolves/
creates the collection, creates a `"queued"` job) so a `job_id` can return immediately;
`ingest_document(..., job_id=...)` then does the slow work against that same job.
`delete_collection(qdrant_collection_name)` does the combined Qdrant+catalog delete.
Idempotent by design throughout.

Real bug caught and fixed: picture/table crop images were being written to
`docling_parsing`'s default *temp* workspace — a chunk's `image_path` in Qdrant would
point at a file no longer there by the time that chunk is retrieved. Fixed by pointing
`StorageConfig.base_dir` at a **permanent** location (`config.yaml`'s
`storage.documents_dir`, keyed by `doc_stem`) and never calling `extractor.cleanup()` in
the real ingestion path.

### `edenview_ingestion/system_inspector.py`

`get_system_specs() → SystemSpecs` — OS/platform, CPU core counts, total/available RAM
(via `psutil`), GPU detection (NVIDIA via `nvidia-smi` subprocess; Apple Silicon reported
as unified memory, no separate VRAM figure; AMD/Intel not detected — empty list, not an
error), and Ollama reachability + every pulled model with its size in GB. Matches
`edenview_plan.md`'s `SystemInspector` concept — feeds the not-yet-built model-selection
advisor. Verified against real hardware (this machine's actual GTX 1660 SUPER, 6GB VRAM,
and 5 pulled Ollama models all reported correctly) both standalone and through the live
API.

`torch_acceleration: TorchAccelerationInfo` (`installed`, `device: "cuda"|"mps"|"cpu"|None`,
`gpu_name`) — a *different* fact from `gpus` above: whether `torch` itself is actually
using a detected GPU, not just whether one exists. A machine can have an NVIDIA GPU
while still running `torch` CPU-only if the installed build has no CUDA support — the
Settings page surfaces this gap directly (see below) rather than silently running slow.

### Performance settings (`edenview_ingestion/settings.py` + `scripts/install_torch.py`)

- `get_num_threads()` — `cpu_count - 2` (min 1) unless overridden; `get_page_batch_size()`
  — Docling's own default (4) unless overridden, applied by setting the process-wide
  `docling.datamodel.settings.settings.perf.page_batch_size` right before each `.convert()`
  call (not a per-call `PdfPipelineOptions` field). `get_max_concurrent_extractions()` —
  see the architecture-decisions table. All three: `null` in `config.yaml`'s new
  `performance:` section means auto-detect; editable from **Settings → Performance**
  (`GET`/`PUT /system/performance`). `max_concurrent_extractions` needs a backend restart
  to take effect (a `threading.Semaphore`'s capacity isn't resizable live); the other two
  are re-read fresh per extraction.
- **`scripts/install_torch.py`** — one-time setup step (documented in `torch_installation.md`),
  run after `pip install -r requirements.txt`: detects an NVIDIA GPU via `nvidia-smi`,
  and if found, reinstalls the *exact same* torch version already resolved from PyTorch's
  CUDA wheel channels (newest first: `cu130`/`cu128`/`cu126`/`cu121`/`cu118`), verifying
  each attempt by actually checking `torch.cuda.is_available()` rather than trusting
  `pip install` alone. Deliberately not built on `light-the-torch` (monkey-patches pip
  internals, has broken before on a pip upgrade) — only plain, documented
  `pip install pkg==version --index-url ...` calls. **Confirmed working end-to-end on
  this machine**: `torch.__version__` → `2.12.1+cu130`, `torch.cuda.is_available()` →
  `True`, and Docling's own `AcceleratorDevice.AUTO` picked it up automatically once the
  backend was restarted.
- **Ingestion-time "Scanned document (force full-page OCR)" checkbox** — defaults
  **unchecked** (today's already-safe behavior: `force_full_page_ocr=False`, Docling's
  own default, already only OCRs detected bitmap regions per page rather than treating
  born-digital text as needing OCR at all). Checking it sets
  `OcrAutoOptions(force_full_page_ocr=True)` for that document — an opt-in for a document
  *already known* to be a scan, not an opt-out that risks silently losing text from a
  scanned page a user forgot to flag.

### `edenview_RAG/retrieval/` (new top-level package, sibling to `edenview_ingestion`)

`edenview_ingestion` owns the *write* path (extract/chunk/embed/write); `edenview_RAG`
owns the *query-time/serving* path (this module and `agentic_rag/` below) — both
depend on `edenview_ingestion`'s `catalog`/`vectorstore` as the shared storage layer.

- `search(collection_names, query, config, file_hashes=None, strategy=None)` and
  `search_db(db_name, query, config, file_hashes=None, strategy=None)` — dense + sparse
  (BM25) + Qdrant native RRF fusion per collection, fanned out across one or more
  collections, merged, and (if `use_reranker`, the default) reranked before truncating
  to `top_k`.
- **`strategy` filter**: restricts to collections whose catalog `chunking_strategy`
  matches — operates at the *collection* level, so an `image_description` chunk living
  inside a `hybrid_docling` collection still passes the `hybrid_docling` filter (it's
  the collection that's filtered, not each chunk's own `strategy` tag).
- **Parent swap**: `RetrievalHit.text` is always the precise matched chunk (citations);
  `context_text` is what to feed an LLM — swapped for the full parent chunk (from
  `catalog.parent_chunks`) on a `parent_child` strategy's `child`-kind hits.
- **Document-level filtering**: optional `file_hashes` payload filter, composes with
  collection-level and strategy-level scoping.

### `edenview_RAG/agentic_rag/` (ADK-based agentic RAG loop)

Powers the Chat UI's "Agentic RAG" mode (`POST /chat`/`POST /chat/stream` with
`agentic: true`) — three pre-built agent trees (`low`/`medium`/`high`, picked in
plain Python by the request's `effort` field, see the architecture-decisions table
above for why not left to the LLM):

- **`config.py`** — `get_agent_model_name()`/`get_agent_num_ctx()`/
  `get_default_effort()`/`get_max_iterations()` read `config.yaml`'s new `agent:`
  section; `model_supports_vision(model)` checks Ollama's live `capabilities` field
  (not assumed) to gate image-inspection; `get_shared_llm()` builds the **one**
  `LiteLlm(model="ollama_chat/...")` instance every agent across every tier shares
  (never a second/smaller model for the critic, never a separate vision model —
  qwen3.5:4b is itself multimodal). Also where every litellm/Ollama compatibility
  patch lives (see the architecture-decisions table's "real bugs found" row) and
  where `OLLAMA_API_BASE` gets set at import time.
- **`tools.py`** — `retrieve` (wraps `search`/`search_db`, reads scope from
  `tool_context.state["scope"]`, never as a function argument since the three
  trees are built once and reused across every request), `get_page_context`
  (reconstructs a page's full text from every chunk on it via a Qdrant `scroll()`
  filtered on `file_hash`+`page_no`, `include_adjacent` also pulls page±1),
  `inspect_image` (saves a chunk's image as an ADK Artifact for
  `callbacks.inject_pending_images` to attach to the model's next turn),
  `exit_loop` (the documented `tool_context.actions.escalate = True` pattern for
  ending the critic/refiner loop early).
- **`callbacks.py`** — `cap_tool_calls` (a `before_tool_callback` hard cap per tool
  per turn, a safety floor independent of the loop's own `max_iterations`),
  `harvest_citations`/`merge_hits_into_state` (accumulate every retrieval round's
  hits into `state["citations"]` keyed by chunk_id *and* a running plain-text
  `state["findings"]` block with continuous numbering — shared by every retrieval
  site so a refiner's mid-loop search adds to, not overwrites, what dispatch
  already found), `inject_pending_images` (the `before_model_callback` that
  actually attaches a saved image Artifact to the next model call).
- **`prompts.py`** — every tier/agent's instruction string, plus `STATUS_LABELS`/
  `AGENT_STATUS_LABELS` (tool/agent name → human-readable line) for the streaming
  endpoint.
- **`subagent.py`** — `build_reframe_agent` (rewrites + *conditionally* splits a
  compound question into up to 4 sub-questions — `output_schema=ReframeOutput`,
  no tools, since ADK's own docs warn `output_schema`+`tools` together is
  unreliable on non-Gemini models), `RetrievalDispatchAgent` (custom non-LLM
  `BaseAgent`, see the architecture-decisions table), `build_research_agent(effort,
  extra_refiner_tools=None)` — the shared `medium`/`high` tree (`reframe → dispatch
  → critic/refiner LoopAgent → answer`), with `critic`/`answer` using
  `include_contents="none"` (they get everything via explicit `{findings}`/
  `{critique?}`/`{original_question}` templating, so they don't need the full
  conversation transcript re-sent every call — `refiner` keeps the default since it
  needs to reason about what it already tried across loop iterations).
- **`agent.py`** — `build_low_agent` (single flat agent, no subagent split — see
  architecture-decisions table for why), `build_medium_agent`/`build_high_agent`
  (thin root wrapping the research tree via `AgentTool(..., skip_summarization=True)`).
- **`runtime.py`** — `Runner`/`DatabaseSessionService`/`InMemoryArtifactService`
  wiring, `run_turn()` (used by `POST /chat`, retries the whole turn once if the
  final answer comes back empty — a real, recurring local-model characteristic:
  occasionally an agent step's entire response lands in native "thinking" content
  with nothing in regular content), `run_turn_stream()` (used by `POST
  /chat/stream`, yields `{"type": "status", ...}` from ADK's own event stream, no
  retry-on-empty since a live stream can't silently restart mid-flight).

### `api/` (project-root package — one FastAPI app every domain mounts routers onto)

- `app.py` — `FastAPI()` + CORS (wide open; local single-user app) + router mounting.
  Run with `npm run dev` from `edenview-ui/` (starts both frontend and this backend
  from the venv together), or standalone with
  `venv/Scripts/python.exe -m uvicorn api.app:app --port 8000`; `/docs` for Swagger UI.
- `routers/catalog.py` — DBs/collections CRUD, `/collections/{name}/documents`,
  `/collections/{name}/preview` (paginated Qdrant `scroll()`), `/chunking/strategies`.
- `routers/ingest.py` — two-phase `POST /ingest` (`prepare_ingest()` synchronous, then a
  `BackgroundTasks` task; form fields include `include_image_descriptions` and
  `force_full_page_ocr`) + `GET /jobs/{job_id}` + `GET /jobs?limit=N&filename=...&status=...`
  (most-recent-first, backs the Ingestion page's job list server-side instead of
  browser `localStorage`; `filename` is a substring search, `status` is one of
  `active`/`done`/`error`/`cancelled`) + `POST /jobs/{job_id}/retry` (requeue a failed
  job from its preserved original file) + `POST /jobs/{job_id}/cancel` (signals a
  queued/running job to stop at its next checkpoint — `404` if it doesn't exist, `409`
  if it's already finished or isn't actually being run by this process).
- `routers/search.py` — `POST /search`, thin wrapper over `edenview_RAG.retrieval`.
- `routers/chat.py` — session-aware `POST /chat` (lazily creates/continues a
  `chat_sessions` row, runs either the Simple RAG search+generate flow or, with
  `agentic: true`, `edenview_RAG.agentic_rag`'s ADK loop, persists both the user
  and assistant messages either way), `POST /chat/stream` (SSE variant, agentic
  only, forwards live progress from the agent loop), `GET /chat/sessions`,
  `GET /chat/sessions/{id}`, `DELETE /chat/sessions/{id}`.
- `routers/documents.py` — `GET /documents/{file_hash}/pages/{page_no}?bbox=l,t,r,b`,
  the visual-grounding page renderer (pypdfium2 + PIL, see section 1).
- `routers/files.py` — `GET /files?path=...` serves a chunk's linked image, path-traversal
  guarded (`403` outside `settings.get_documents_dir()`).
- `routers/system.py` — `GET /system/info` (now also reports currently-loaded Ollama
  models and `torch_acceleration` status) + `POST /system/ollama/unload`.
- `routers/config.py` — `GET`/`PUT /system/config` (model names), `GET`/`PUT
  /system/performance` (`num_threads`/`page_batch_size`/`max_concurrent_extractions`,
  each with an `_is_auto` flag), `GET`/`PUT /system/workspace` + `POST
  /system/workspace/browse` (native OS folder-picker dialog, returns the chosen path).

Full reference + a copy-pasteable upload→ingest→search walkthrough: **`api/README.md`**
— covers every endpoint above with example requests/responses; keep it in sync with
this list when adding a new router or endpoint.

### `edenview-ui/` (Next.js 16 portal — App Router, Turbopack, shadcn/ui on Base UI, Tailwind v4)

- **Ingestion** (`app/ingestion/page.tsx`) — multi-file drag/drop upload (one file =
  one concurrent `POST /ingest` call, same db/collection/strategy for all), plus an
  ingestion-time **"Generate image descriptions"** checkbox and a **"Scanned document
  (force full-page OCR)"** checkbox (see the performance-settings section above for why
  the latter defaults unchecked). A server-backed job list (`GET /jobs`, polls while
  anything is `queued`/`running`) that is:
  - **Collapsible and contained** — a `max-h` scrollable box, not an unboundedly tall
    page section, with a chevron toggle showing a live count.
  - **Limited by default (10) with a filename search box** — typing searches server-side
    (`GET /jobs?filename=...`, bumps the fetch limit to 100) instead of a full
    offset/page UI; the job list is meant to stay a lightweight recent-activity feed.
  - Each row shows a live **stage stepper**, an **elapsed/total duration**
    (`started_at`/`finished_at`), a **Retry** button on any `error` row, and a
    **Cancel** button on any `queued`/`running` row (see the cancellation
    architecture-decision row — cooperative, not always instant).
  - A job genuinely queued behind `max_concurrent_extractions` other extractions shows
    a distinct **"queued"** badge (not "running") until it actually starts — fixed after
    being caught showing every queued job as "running"/"extracting" immediately, before
    it had even acquired its turn.
- **Collections** — DB/collection browser over the catalog + a paginated Qdrant
  `scroll()`-backed chunk preview.
- **Chat** (`app/chat/page.tsx`) — sessions list (left) + transcript (center) +
  collapsible scope panel (right, db/collection/strategy/top-k/reranker/model
  selection, persisted to `localStorage` across navigation/reload) + a grounding panel
  (opens on citation click, renders the source PDF page with the chunk's bbox
  highlighted, with an expand-to-near-fullscreen option for actually reading the page).
  Assistant messages render through `react-markdown` with `remark-gfm` + `rehype-raw`
  (tables, `<br>`, and other GFM/HTML constructs render properly instead of as raw text).
  **Agentic RAG mode is now live** (no longer a "coming soon" placeholder) — the scope
  panel's mode toggle plus an effort selector (low/medium/high) drive `POST
  /chat/stream` (`lib/api.ts`'s `runChatStream`, hand-parsed SSE over `fetch` since
  `EventSource` can't send a POST body), rendering each live status line above the
  pending message. Type-checked/linted clean; not yet visually verified in an actual
  browser (no browser-automation tool available when this was built) — worth a real
  click-through before relying on it.
- **Settings** — every model name in `config.yaml` (tokenizer, dense/sparse embedding,
  contextual/picture-description/chat LLM, reranker), Ollama host, `ollama_keep_alive`,
  the workspace root folder (with a native-dialog **Browse…** button via
  `POST /system/workspace/browse`), and a **Performance** card (extraction threads,
  page batch size, max concurrent extractions — each showing `auto (N)` as a
  placeholder unless overridden) — all editable and persisted back to `config.yaml` via
  `ruamel.yaml` round-trip (preserves comments/formatting). A **"This machine"** card
  reports CPU/RAM/GPU plus an **"Extraction acceleration"** line (`CUDA (<gpu>)` /
  `Apple Silicon (MPS)` / `CPU only`) with a warning banner specifically when a GPU is
  detected but `torch` isn't using it, pointing at `scripts/install_torch.py`.
- **Sidebar system monitor** (`components/layout/system-monitor.tsx`) — collapsible,
  Task-Manager-style RAM/VRAM area charts (polls `GET /system/info` every ~4s, ~60s
  rolling window) plus a loaded-Ollama-models list with per-model **Unload** buttons.
- **Copyright/tagline footer** under the sidebar's performance monitor.

### `config.yaml` + `edenview_ingestion/settings.py`

Single source of truth for every model name *and* local storage path — nothing
hardcoded in Python:

```yaml
models:
  tokenizer: BAAI/bge-m3
  dense_embedding: bge-m3
  dense_embedding_dim: 1024
  sparse_embedding: Qdrant/bm25
  contextual_llm: qwen3:4b
  picture_description_llm: qwen3-vl:2b
  reranker: Xenova/ms-marco-MiniLM-L-6-v2
ollama:
  host: http://localhost:11434
qdrant:
  path: edenview_data/qdrant_db        # embedded, no server
catalog:
  path: edenview_data/catalog.duckdb
storage:
  documents_dir: edenview_data/documents
```

`edenview_data/` (the whole local data root) is gitignored — real user data, not source.

A newer `agent:` section (not shown in the excerpt above, which predates it) backs
`edenview_RAG/agentic_rag`: `model` (the one shared agent/tool-calling model, default
`qwen3.5:4b` — a genuinely multimodal Ollama tag, unlike the plain `qwen3:*` family),
`num_ctx` (Ollama's real runtime context window — see the architecture-decisions
table's "real bugs found" row for why this has to be set explicitly), `default_effort`,
`max_iterations: {medium: 2, high: 4}`.

---

## 3. What's verified

- **`test/chunking/verify_chunking.py`** — all 4 chunking strategies + image-description
  addition, structural invariants, live Ollama calls included. **26/26 passed** (after
  the picture-description fix — previously 0 descriptions were ever generated, silently).
- **`test/pipeline/verify_pipeline.py`** — ingests a real sample PDF into two collections
  (`hybrid_docling` and `parent_child`) under one DB: catalog rows, Qdrant point counts,
  a real hybrid search round-trip, parent-chunk lookup, **strategy-filtered DB-wide
  search** (confirms `strategy=X` returns hits only from the matching collection),
  idempotent re-ingestion, and combined Qdrant+catalog deletion. **22/22 passed.**
- **`test/retrieval/verify_retrieval.py`** — single-collection search, reranker on/off,
  multi-collection fan-out+merge, `parent_child` context-text swap, document-level
  `file_hashes` filtering. **13/13 passed.**
- **Picture description, live**: verified qualitatively — a flow-chart image's generated
  description correctly summarized specific participant counts and exclusion criteria
  visible only in the image, not restated anywhere in the surrounding text.
- **`GET /system/info`**, live through the running API: correctly reported this
  machine's actual GPU (GTX 1660 SUPER, 6144MB total / free VRAM), RAM, CPU cores, and
  all 5 currently-pulled Ollama models with sizes.
- Manually verified end-to-end via `curl`/Swagger UI against a running server: DB
  create/list/duplicate-rejection, collection get/list/documents/preview, `POST /ingest`
  → background job → polled through `queued` → `running` (`stage` moving
  `extracting` → `chunking` → `embedding`, with real `stage_current`/`stage_total`/
  `stage_pct` only during embedding) → `done`, search by collection names / `db_name`
  fan-out / strategy filter, `/files` for a valid image / a traversal attempt (403) / a
  missing file (404), and delete-collection → delete-db cleanup.

All scripts run via `PYTHONPATH=. python test/<name>/verify_*.py` (not `-m` — `test`
collides with Python's own stdlib `test` package).

- **Chat persistence, live**: a `/chat` call with no `session_id` creates a session and
  returns one; a follow-up with that `session_id` appends to the same session;
  `GET /chat/sessions/{id}` returns full history; delete removes it. Verified in-browser
  across a page reload.
- **Visual grounding, live**: ingested a real PDF, confirmed `bbox` populated for
  `hybrid_docling`/`parent_child`/`contextual` hits and null for `recursive_overlap`;
  `GET /documents/{file_hash}/pages/{page_no}?bbox=...` returns a real page image with
  the highlight box in the right place; verified in-browser via the chat grounding panel.
- **Ollama `keep_alive`, live**: confirmed via `ollama ps`'s `expires_at` timestamp that
  a configured `30m` value is honored across chat/embed calls, instead of the
  5-minute default silently evicting models between ordinary turns.
- **System monitor + model unload, live**: sidebar RAM/VRAM charts confirmed updating
  over time against real `Get-CimInstance`/`nvidia-smi` numbers; unloading a model via
  the UI button confirmed to drop it from both `ollama ps` and the next poll.
- **CUDA torch install, live, end-to-end**: `scripts/install_torch.py` run against this
  machine's real GTX 1660 SUPER — before: `torch.cuda.is_available()` → `False`; after:
  `torch.__version__` → `2.12.1+cu130`, `torch.cuda.is_available()` → `True`, and
  `/system/info`'s `torch_acceleration` correctly flipped from `{"device": "cpu"}` to
  `{"device": "cuda", "gpu_name": "NVIDIA GeForce GTX 1660 SUPER"}` once the backend
  was restarted (a running process doesn't pick up a torch reinstall on disk until
  restarted — confirmed directly, not assumed).
- **Markdown rendering + grounding panel expand, live in-browser**: a chat answer with
  a real GFM table (`| col | col |` syntax) rendered as an actual bordered `<table>`
  (was showing raw pipe characters and literal `<br>` text before `remark-gfm`/
  `rehype-raw`); clicking a citation's "view source page" then the expand button opened
  a large, readable overlay of the highlighted PDF page.
- **Server-backed ingestion job tracker, live**: uploaded a real file, confirmed the
  job list shows its real filename (not `null`, since `filename` is now stored at
  job-creation time rather than waiting on `doc_id`), live-updates through
  `extracting` → `chunking` → `embedding` via polling, and reaches `done` — all driven
  by `GET /jobs`, not browser `localStorage`.
- **Multi-file concurrent ingestion — four real concurrency bugs found and fixed**,
  each reproduced directly with concurrent `curl` requests (not just inferred from code
  review) before and after its fix:
  1. `DuplicateNameError` races when two concurrent uploads targeting a brand-new
     db/collection/document all lose a check-then-create race — fixed by catching the
     duplicate-key error and falling back to a fetch.
  2. DuckDB's single process-wide connection isn't thread-safe for concurrent queries —
     fixed with one `cursor()` per thread (DuckDB's own documented pattern) behind a
     double-checked-locking init.
  3. The embedded Qdrant client had the identical lazy-init race — fixed the same way.
  4. DuckDB raises `ConstraintException` immediately on one connection but
     `TransactionException` (a generic "duplicate key" message) at commit time across
     concurrent cursors — fixed by catching both and checking the message.
- **Two more concurrency/environment bugs found from real usage (not synthetic
  testing) and fixed** — see the retry-tested confirmation below:
  5. **Duplicate backend processes**: a second `uvicorn` had been started against a
     different (non-venv) Python interpreter while the venv-backed one was still
     running, both bound to port 8000. Only one can ever hold the OS-level file locks
     Qdrant/DuckDB's embedded/local storage rely on — this alone was enough to produce
     the `/search` 500 seen in chat. Fixed by always launching the backend the same
     way (`npm run dev`, wired to the venv explicitly — see section 1).
  6. **Qdrant embedded client had no locking around actual use, only around its own lazy
     construction.** Its local-mode persistence (`qdrant_client/local/persistence.py`)
     is backed by sqlite3, which cannot tolerate genuinely concurrent, unsynchronized
     calls from multiple threads on one connection — this surfaced as sqlite's own
     `"bad parameter or other API misuse"` error, specifically on a `finance_act`
     multi-file ingestion where one large document's chunking/embedding phase ran long
     enough to overlap with a sibling file's Qdrant calls. Fixed by adding a shared
     `client_lock` (`vectorstore/client.py`) and wrapping every real Qdrant call —
     `create_collection`/`delete_collection` (`collections.py`), `upsert`
     (`points.py`), `query_points` (`search.py`), `scroll`
     (`api/routers/catalog.py`) — in `with client_lock:`. **Confirmed fixed**: retried
     the exact 65MB document that had failed twice before (same job, same
     `finance_audit` collection) through the new `/jobs/{id}/retry` endpoint — it ran
     extraction → chunking → embedding to completion this time, all 435 chunks
     embedded, `status: "done"`.
- **Six more real bugs found from live usage and fixed**, each confirmed by
  reproduction (not just code review) before and after:
  7. **`NameError: name 'docling_settings' is not defined`** — the `page_batch_size`
     feature (bug/feature added this session) referenced a name that was never
     imported in `docling_parsing/extractor.py`. Crashed every ingestion until fixed
     (`from docling.datamodel.settings import settings as docling_settings`); left 3
     jobs permanently stuck "running" (the crash happened before extraction returns,
     before anything marks a job "error") — cleaned up by hand since there's nothing
     to retry from at that stage (see the retry-coverage deferred item below).
  8. **DuckDB connection race on a fast dev-server restart** — stopping `npm run dev`
     and immediately restarting it can start the new backend process before the old
     one has actually released its file lock on `catalog.duckdb` (confirmed Windows-
     specific: the old process's shutdown isn't instant) — `duckdb.connect()` raised
     `IOException` on whichever request happened to be first, though the very next
     attempt a moment later always succeeded once the old process finished exiting.
     Fixed with a short bounded retry (5 attempts, 0.3s apart) in
     `catalog/connection.py`'s `get_connection()`.
  9. **Unbounded extraction concurrency** — ingesting 22 files at once (no limit
     existed) ran every extraction simultaneously: RAM fell to ~4.6GB free, and — once
     CUDA was enabled (see the GPU section above) — 22 processes fought over one 6GB
     GPU at once, making the whole batch far slower than a few at a time. Fixed with
     `pipeline._EXTRACTION_SEMAPHORE` (see architecture-decisions table). Stopping this
     for good in the moment required abruptly killing the backend process itself —
     the fix wasn't loaded yet in the already-running process, so cancellation
     couldn't help retroactively; the 24 resulting stuck "running" rows (2 from bug 7,
     22 from this) were deleted directly from the catalog.
  10. **Job status showing "running"/"extracting" while still queued behind the new
      concurrency semaphore** — `start_job()` (which flips status to "running") was
      being called *before* the semaphore acquire, not after, so a job waiting its
      turn behind `max_concurrent_extractions` others showed as actively running when
      it hadn't started at all. Fixed by moving `start_job()`/`update_job_stage(...,
      "extracting")` to inside the `with _EXTRACTION_SEMAPHORE:` block.
  11. **Embedding failures on oversized chunks** — see the embedding-batch-size-mismatch
      architecture-decision row above. **Confirmed fixed**: reproduced the exact
      failure (a ~2400-word/oversized text via `embed_dense()`) and confirmed it now
      embeds successfully (1024-dim vector back) with `options={"num_batch": 4096}` set.
  12. **Chat-scope `localStorage` read causing a hydration mismatch** — reading
      `window.localStorage` inside a `useState` initializer runs differently on the
      server (no `window`, falls back to the empty default → "Pick a database..."
      text) than on the client's first render (real saved scope → different text) —
      a textbook Next.js hydration error. Fixed by always starting from the SSR-safe
      default and loading the real value only inside a post-mount `useEffect`.
- **Agentic RAG (ADK) — built end-to-end, real bugs found and fixed at every layer**,
  each confirmed by direct reproduction against real Ollama/qdrant, not code review:
  - **`test/agentic_rag/verify_agentic_rag_low.py`** — two phases run as two
    *separate OS processes* (proving `DatabaseSessionService` persistence survives an
    actual restart, not just staying alive within one process): phase 1 ingests a
    real 30-page excerpt, asks a question, confirms a grounded cited answer; phase 2
    (fresh process, same `session_id`) asks a follow-up and confirms the agent
    recognized it as a follow-up (not just a fresh question) — the multi-turn memory
    genuinely round-tripped through the SQLite file. **4/4 then 3/3 checks passed**
    after the fixes below.
  - **`test/agentic_rag/verify_agentic_rag_medium.py`** — simple + genuinely compound
    (2-part) questions; confirms the compound one splits into >1 reframed queries and
    the simple one stays at exactly 1 (the "conditional split" prompt behavior, both
    directions). **8/9 checks passed** — the one residual failure is inherent model
    nondeterminism (occasionally an answer step's whole response lands in "thinking"
    with empty regular content), not a wiring bug; a bounded single-retry in
    `runtime.run_turn` mitigates but can't eliminate this.
  - **`test/agentic_rag/verify_agentic_rag_high.py`** — same tree plus
    `get_page_context`, confirmed to actually reconstruct a page's full text
    containing the original citation's own text. **8/8 checks passed.**
  - **`test/agentic_rag/spike_image_injection.py`** — a synthetic image with
    unmemorizable text (so a correct answer can only come from the model actually
    reading the injected image, not pretrained knowledge); confirmed the model
    correctly transcribed the image's text after the litellm base64 fix below.
  - Real bugs found and fixed along the way (see the architecture-decisions table's
    "real bugs found" row for the full list: litellm's tool-calling-capability
    detection, a `reasoning_content` `KeyError` crash, Ollama's silent `num_ctx`
    truncation, a cross-event-loop `GLOBAL_LOGGING_WORKER` crash, and litellm's
    Ollama image handler forwarding an unstripped `data:...;base64,` prefix) — plus
    two agentic-loop-specific ones not upstream bugs: (a) `include_contents="none"`
    on a sub-agent strips its OWN prior turns within the same invocation, not just
    the root's history, which broke a tool-calling agent's memory of what it was
    even asked (fixed by removing it where it wasn't actually needed for isolation —
    `AgentTool` already gives each call its own fresh session); (b)
    `AgentTool(skip_summarization=True)`'s final event shape (plain text part vs. raw
    `function_response`) isn't consistent run-to-run, silently producing an empty
    answer on the state that doesn't include a text part — fixed with a
    dual-path extractor in `runtime._extract_final_text`.
  - Live HTTP smoke tests (not just the standalone scripts): `POST /chat` with
    `agentic: true` at `low`/`medium`/`high` effort, and `POST /chat/stream` (`curl
    -N`) confirmed streaming real `status` events followed by one `result` event with
    a correct grounded answer + citations.

---

## 4. Deferred / open items

- **Document image cleanup on collection delete** — `pipeline.delete_collection()`
  deliberately leaves a document's permanent images alone (another collection might
  still reference the same doc). No "delete this doc's images if truly orphaned" sweep.
- **Qdrant client is now fully serialized, not just made safe** — `client_lock` forces
  every Qdrant call (search included) onto one thread at a time. Fine for a
  single-user local app at today's scale; if collections/traffic grow, this is the
  first place to revisit (e.g. read/write separation) rather than the DuckDB
  cursor-per-thread model, which does allow real read concurrency.
- **Retry only covers PDF sources that failed after extraction completed** — a job that
  fails mid-extraction, or one from a non-PDF source, has no preserved file to retry
  from and needs a manual re-upload. Not seen as a real gap yet (every failure so far
  has been post-extraction), but worth widening if that changes.
- **AMD/Intel GPU detection** not implemented in `system_inspector.py` — rare for this
  audience's local ML workloads, explicitly out of scope for now.
- **Agentic RAG's residual answer-empty rate** — a real, recurring characteristic of
  running a 4B local model through a multi-step agent loop (occasionally a step's
  whole response lands in "thinking" with nothing in regular content); mitigated with
  a bounded single retry in `runtime.run_turn`, not eliminated. Would need a
  meaningfully larger/more instruction-reliable model to improve further, per
  `config.yaml`'s `agent.model` being user-overridable if VRAM allows.
- **Agentic RAG's Chat UI wiring is type-checked/linted clean but not yet
  browser-verified** — no browser-automation tool was available when it was built;
  worth a real click-through (toggle Agentic RAG, watch live status lines render,
  confirm citations/grounding still work) before relying on it.
- **`get_page_context`/`inspect_image` promoted to "high" tier only, not "medium"** —
  a deliberate scope choice (the highest-argument-count tool built last, after loop
  mechanics were already proven), not a technical limitation; revisit if "medium"
  users would benefit from page/image tools too.
- **`google-adk`/`litellm` pinned to specific versions** (`2.5.0`/`1.77.7`) —
  `litellm`'s newer releases (1.9x+) have no prebuilt wheel on this platform and need
  a Rust toolchain to build from source; several of the compatibility patches in
  `edenview_RAG/agentic_rag/config.py` target exact internal module paths/behavior of
  the pinned versions and would need re-verifying (not blindly assumed still needed)
  before bumping either dependency.
- **Cancellation is in-memory and cooperative only** — a job orphaned by a crash/restart
  (nothing in the current process is actually running it) can't be cancelled; the
  endpoint reports this clearly (`409`) rather than silently no-op'ing, but there's no
  "force-clean a stuck row" button in the UI yet (done by hand against the catalog when
  it's come up so far).
- **`max_concurrent_extractions` needs a backend restart** to take effect, unlike
  `num_threads`/`page_batch_size` (re-read fresh per extraction) — a
  `threading.Semaphore`'s capacity isn't resizable live. Not surfaced as a distinct
  "restart required" badge in the Settings UI the way model-config keys are, just
  explained in the field's help text and the save-toast message.
- **No UI button to actually run `scripts/install_torch.py`** — deliberately kept as a
  documented one-time setup step (`torch_installation.md`), not a "click to install"
  action in the app. Running it live-replaces an already-imported `torch` module on
  disk, so it wouldn't take effect until a restart anyway, and it's a multi-GB
  download — a real environment change better done from a terminal the user controls.
- **`HybridChunker`'s `max_tokens` isn't strictly enforced for atomic oversized doc
  items** (e.g. a large/complex table) — a known upstream Docling limitation, not a bug
  in this codebase's chunking wrapper. Currently absorbed on the embedding side
  (`num_batch=4096`, matching the model's real context window) rather than by capping
  chunk size defensively, since the model can genuinely handle chunks in the size range
  actually seen (2000–4000 tokens) — only Ollama's lower-than-context-window default
  batch limit was the problem. Would need revisiting if a chunk ever exceeded the
  embedding model's actual context window, not just Ollama's batch-size default.

---

## 5. Next steps

1. Real browser click-through of the Agentic RAG Chat UI (toggle, effort selector,
   live status lines, citations/grounding) — the backend and the frontend wiring are
   both verified independently, but not yet together in an actual browser.
2. Wire `system_inspector` into an actual model-selection advisor (recommend/warn based
   on VRAM vs. model size, per `edenview_plan.md`'s Global Assistant concept).
3. Widen retry coverage (extraction-stage failures, non-PDF sources) if real usage
   shows it's needed.
4. SQL DB ingestion path — the Ingestion page's "Vector DB / SQL DB" toggle already has
   a "coming soon" placeholder for a DuckDB-native tabular ingestion flow.
5. Promote `get_page_context`/image inspection to "medium" tier if real usage shows
   it's worth the extra tool-calling surface there too.
