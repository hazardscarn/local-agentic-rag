# Edenview API — Reference & End-to-End Testing Guide

Every endpoint in `api/`, with what it does, how to call it, and what comes back — plus
a walkthrough that chains them together, upload → ingest → search, so you can see the
whole pipeline work for yourself.

---

## 1. Starting the server

From the project root:

```bash
PYTHONPATH=. venv/Scripts/python.exe -m uvicorn api.app:app --reload --port 8000
```

Leave that running in its own terminal. Everything below assumes it's up at
`http://localhost:8000`.

## 2. Two ways to test

**Swagger UI (recommended for exploring by hand)** — open
**http://localhost:8000/docs** in a browser. Every endpoint below is listed there with a
"Try it out" button — fill in the fields, click Execute, see the real request/response.
No curl syntax to get right; this is the easiest way to click through the whole flow
yourself.

**curl (scriptable, copy-pasteable)** — every endpoint below has a working curl example.
Run them from the project root so relative file paths resolve.

> **If you're in PowerShell, not git-bash**: PowerShell aliases `curl` to
> `Invoke-WebRequest`, which does **not** understand the `-X`/`-F`/`-d` flags used below.
> Either run these from a git-bash/WSL terminal, or explicitly call `curl.exe` instead of
> `curl` in PowerShell (`curl.exe -X POST ...`) to bypass the alias.

---

## 3. End-to-end walkthrough

Run these in order. Each step's output feeds the next.

### Step 1 — create a DB

```bash
curl -s -X POST http://localhost:8000/dbs \
  -H "Content-Type: application/json" \
  -d '{"name":"my-first-db"}'
```
```json
{"db_id":"6a3a8c61-...","name":"my-first-db","created_at":"2026-07-15T10:00:00"}
```

### Step 2 — see what chunking strategies exist

```bash
curl -s http://localhost:8000/chunking/strategies
```
```json
["recursive_overlap","hybrid_docling","parent_child","contextual"]
```

### Step 3 — upload and ingest a file

Use any PDF you have on hand:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/your.pdf" \
  -F "db_name=my-first-db" \
  -F "collection_name=my-first-collection" \
  -F "strategy=hybrid_docling"
```
```json
{"job_id":"596d700c-...","status":"queued","qdrant_collection_name":"my-first-collection"}
```

This returns **immediately** — the actual parsing/chunking/embedding runs in the
background. Save the `job_id`.

`strategy` is one of the four from Step 2. Optional extra form fields:
`include_image_descriptions=true` (runs the configured vision model over each retained
image — that model must already be pulled in Ollama, e.g. `ollama pull qwen3-vl:2b`, or
every image description call fails) and `force_full_page_ocr=true` (opt-in for a
document you already know is a scan — see the Ingest section below for why this
defaults off). Both are also checkboxes on the Ingestion page's upload form.

### Step 4 — poll until it's done

```bash
curl -s http://localhost:8000/jobs/596d700c-...
```
```json
{"job_id":"596d700c-...","collection_id":"...","doc_id":null,"status":"running","filename":"covid-19-risk-factors-Japan.pdf","stage":"extracting","stage_current":null,"stage_total":null,"started_at":"...","finished_at":null,"error_message":null,"qdrant_collection_name":"my-first-collection","db_name":"my-first-db","stage_pct":null}
```

`status` moves `queued` → `running` → `done` (or `error`, or `cancelled` if
`POST /jobs/{job_id}/cancel` was called — see the Ingest section below). A full
document (accurate table mode, OCR) can take a minute or two for extraction alone —
re-run the same curl command every few seconds until `status` is `done`. A job that's
merely queued behind `max_concurrent_extractions` other concurrent extractions stays
`status: "queued"` (not `"running"`) until it actually starts.

`stage` tells you *what's* running: `extracting` → `chunking` → `embedding`.
`stage_current`/`stage_total`/`stage_pct` are only ever non-null during `embedding` —
that's the one phase with a real, known chunk count (embedding runs in batches against a
total we already know), so the percentage there is an actual measurement, not an
estimate. Extraction has no equivalent: Docling doesn't expose per-page progress, and
page count itself isn't known until parsing finishes, so `extracting`/`chunking` report
their name only, no fake percentage. Confirmed live in testing:
```json
{"status":"running","stage":"embedding","stage_current":7,"stage_total":13,"stage_pct":53.8, ...}
```
→
```json
{"status":"done","stage":"embedding","stage_current":13,"stage_total":13,"stage_pct":100.0, ...}
```

### Step 5 — check the collection landed correctly

```bash
curl -s http://localhost:8000/collections/my-first-collection
```
```json
{"collection_id":"...","db_id":"...","qdrant_collection_name":"my-first-collection","chunking_strategy":"hybrid_docling","embedding_model":"bge-m3","dense_dim":1024,"sparse_model":"Qdrant/bm25","status":"ready","chunk_count":13,"doc_count":1,"created_at":"..."}
```

```bash
curl -s http://localhost:8000/collections/my-first-collection/documents
```
```json
[{"doc_id":"...","file_hash":"...","filename":"covid-19-risk-factors-Japan.pdf","source_path":"...","input_format":"pdf","num_pages":10,"first_ingested_at":"..."}]
```

### Step 6 — preview the actual chunks (the "table view")

```bash
curl -s "http://localhost:8000/collections/my-first-collection/preview?limit=5"
```
```json
{"chunks":[{"chunk_id":"...","text":"...","page_no":1,"kind":"text","strategy":"hybrid_docling","images":[]}, ...],"next_offset":"..."}
```

Pass `next_offset` as the `offset` query param to page through the rest. Any chunk whose
`images` array is non-empty has a picture linked to it — see Step 8.

### Step 7 — search it

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"risk factors for post-COVID condition","collection_names":["my-first-collection"],"top_k":3}'
```
```json
[{"chunk_id":"...","score":5.49,"text":"...","context_text":"...","collection_name":"my-first-collection","strategy":"hybrid_docling","kind":"text","page_no":1,"headings":[],"doc_stem":"covid-19-risk-factors-Japan","file_hash":"...","images":[]}, ...]
```

Or search every collection under the whole DB instead of naming one:

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"risk factors for post-COVID condition","db_name":"my-first-db","top_k":3}'
```

### Step 8 — view an image from a hit

If a search hit or preview chunk has a non-empty `images` array, grab its
`image_path` and fetch it directly (this returns actual image bytes — open the URL
in a browser, or save with curl):

```bash
curl -s --get --data-urlencode "path=<image_path from the hit>" \
  http://localhost:8000/files -o picture.png
```

Or just paste `http://localhost:8000/files?path=<image_path>` (URL-encoded) into a
browser to view it directly.

### Step 9 — clean up (optional)

```bash
curl -s -X DELETE http://localhost:8000/collections/my-first-collection
curl -s -X DELETE http://localhost:8000/dbs/<db_id from Step 1>
```
Deleting a DB only succeeds once every collection under it is gone.

---

## 4. How an endpoint gets built here (read before adding a new one)

Every router follows the same shape — copy an existing one rather than inventing a new
pattern:

1. **One file per domain** under `api/routers/` (`catalog.py`, `ingest.py`, `search.py`,
   `chat.py`, `documents.py`, `files.py`, `system.py`, `config.py`) — group by the
   resource/concern the endpoints operate on, not by HTTP verb. A new domain gets a new
   file; a new operation on an existing domain (e.g. another `/jobs/{id}/...` action)
   goes in that domain's existing file.
2. **`APIRouter(tags=[...])` per file**, mounted once in `app.py` via
   `app.include_router(...)` — the router itself never runs standalone.
3. **Request/response shapes live in `api/schemas.py`** as Pydantic models (or, for
   catalog-backed resources, the same Pydantic model already defined in
   `edenview_ingestion/catalog/models.py` / `edenview_RAG/retrieval/models.py` — reused
   directly as a route's `response_model`, not redeclared). Never return a raw
   dict/`JSONResponse` for structured data.
4. **The route function itself stays thin** — it validates/translates HTTP-shaped input
   (query params, form fields, request body) into a call against `edenview_ingestion`
   or `edenview_RAG`'s own functions (`pipeline.py`, `catalog.crud`, `vectorstore`,
   `edenview_RAG.retrieval`), and translates that layer's typed exceptions
   (`catalog.NotFoundError`, `catalog.DuplicateNameError`, `ValueError`, ...) into the
   right `HTTPException` status code. Business logic belongs in the underlying
   package, not in the route handler — this is what let every core function above also
   be verified standalone via the `test/*/verify_*.py` scripts, independent of FastAPI.
5. **Long-running work** (anything extraction/chunking/embedding-shaped) goes through
   the two-phase pattern `POST /ingest` established: a fast synchronous half that
   returns immediately with an id, plus a `BackgroundTasks` task (or, for a
   Settings-style action, a plain synchronous call if it's genuinely fast) — never
   block the event loop on Docling/Ollama work. See `pipeline.prepare_ingest()` +
   `ingest_document(..., job_id=...)` as the reference implementation.
6. **Update this doc** (add the new endpoint under the relevant `###` section below)
   and `edenview_progress.md`'s `api/` bullet list in the same change — both are meant
   to stay a complete, current reference, not just cover what existed when they were
   first written.

---

## 5. Full endpoint reference

### Health

**`GET /health`** — liveness check.
```bash
curl -s http://localhost:8000/health
```
`{"status":"ok"}`

### Catalog — DBs

**`GET /dbs`** — list every DB.
```bash
curl -s http://localhost:8000/dbs
```
→ `list[DBRecord]`: `{db_id, name, created_at}`

**`POST /dbs`** — create a DB.
```bash
curl -s -X POST http://localhost:8000/dbs -H "Content-Type: application/json" -d '{"name":"..."}'
```
Body: `{"name": str}`. → `201` + `DBRecord`. **`409`** if the name already exists.

**`DELETE /dbs/{db_id}`** — delete a DB.
```bash
curl -s -X DELETE http://localhost:8000/dbs/<db_id>
```
→ `204`. **`409`** if any collection still references this DB — delete those first.

### Catalog — Collections

**`GET /collections?db_name=...`** — list collections, optionally filtered to one DB.
```bash
curl -s "http://localhost:8000/collections?db_name=my-first-db"
```
→ `list[CollectionRecord]`: `{collection_id, db_id, qdrant_collection_name, chunking_strategy, embedding_model, dense_dim, sparse_model, status, chunk_count, doc_count, created_at}`

**`GET /collections/{name}`** — one collection's details (`name` = the Qdrant collection name you chose at ingest time).
```bash
curl -s http://localhost:8000/collections/my-first-collection
```
→ `CollectionRecord`. **`404`** if it doesn't exist.

**`DELETE /collections/{name}`** — deletes both the Qdrant collection and its catalog rows.
```bash
curl -s -X DELETE http://localhost:8000/collections/my-first-collection
```
→ `204`. **`404`** if it doesn't exist. Does **not** delete the source document's images
(another collection might still reference them).

**`GET /collections/{name}/documents`** — every source file ingested into this collection.
```bash
curl -s http://localhost:8000/collections/my-first-collection/documents
```
→ `list[DocumentRecord]`: `{doc_id, file_hash, filename, source_path, input_format, num_pages, first_ingested_at}`

**`GET /collections/{name}/preview?limit=20&offset=...`** — paginated raw chunk/payload
browsing, straight off Qdrant (not the catalog — see `edenview_progress.md`'s
"Catalog vs. content browsing" decision for why).
```bash
curl -s "http://localhost:8000/collections/my-first-collection/preview?limit=10"
```
→ `{"chunks": [{"chunk_id", "text", "page_no", "kind", "strategy", "images"}], "next_offset"}`.
Pass `next_offset` back in as `offset` to get the next page; `null` means you've reached the end.

### Catalog — chunking strategies

**`GET /chunking/strategies`** — the four available strategy names, for a dropdown.
```bash
curl -s http://localhost:8000/chunking/strategies
```
→ `["recursive_overlap", "hybrid_docling", "parent_child", "contextual"]`

### Ingest

**`POST /ingest`** — multipart upload, starts a background ingestion job.
```bash
curl -s -X POST http://localhost:8000/ingest \
  -F "file=@path/to/your.pdf" \
  -F "db_name=my-first-db" \
  -F "collection_name=my-first-collection" \
  -F "strategy=hybrid_docling" \
  -F "include_image_descriptions=false"
```
Form fields: `file` (required), `db_name` (required — created if it doesn't exist),
`collection_name` (required — becomes the literal Qdrant collection name; created if it
doesn't exist, reused if it does), `strategy` (required, one of the four from
`/chunking/strategies`), `include_image_descriptions` (optional, default `false`),
`force_full_page_ocr` (optional, default `false` — opt-in for a document *already known*
to be a scan; leave off for ordinary born-digital/mixed documents, since Docling's own
default already OCRs only the bitmap regions it detects per page).

→ `202` + `{"job_id", "status": "queued", "qdrant_collection_name"}` **immediately** —
the actual work happens after the response is sent. **`400`** if `strategy` is unknown.

**`GET /jobs/{job_id}`** — poll a job's status.
```bash
curl -s http://localhost:8000/jobs/<job_id>
```
→ `IngestionJobRecord`: `{job_id, collection_id, doc_id, status, filename, stage, stage_current, stage_total, stage_pct, started_at, finished_at, error_message, qdrant_collection_name, db_name}`.
`status`: `queued` → `running` → `done` | `error` | `cancelled`. `stage`: `extracting` →
`chunking` → `embedding` (see the walkthrough's Step 4 for why only `embedding` ever has
real `stage_current`/`stage_total`/`stage_pct` values — a job still queued behind
`max_concurrent_extractions` other extractions stays `status: "queued"`, `stage: null`
until it actually starts, not `"running"` prematurely). **`404`** if the job_id doesn't
exist.

**`GET /jobs?limit=N&filename=...&status=...`** — most-recently-started-first job list,
backs the Ingestion page's job list server-side (every job regardless of which
browser/device started it, unlike a browser-`localStorage`-backed list).
```bash
curl -s "http://localhost:8000/jobs?limit=10"
curl -s "http://localhost:8000/jobs?filename=finance&limit=100"
curl -s "http://localhost:8000/jobs?status=error"
```
→ `list[IngestionJobRecord]`. `limit` defaults to 50. `filename` is a case-insensitive
substring match (SQL `ILIKE`), letting you find an older job the default `limit` would
otherwise cut off, without a full offset/page UI. `status` is one of `active` (queued or
running), `done`, `error`, or `cancelled` — omit for every status. Both filters compose.

**`POST /jobs/{job_id}/retry`** — requeues a failed job from its preserved original file,
as a brand-new job row (the failed one stays as history).
```bash
curl -s -X POST http://localhost:8000/jobs/<job_id>/retry
```
→ `202` + `IngestAccepted` (same shape as `POST /ingest`). **`404`** if the job doesn't
exist. **`400`** if it's not in an `error` state, or if it failed *during* extraction
(no `doc_id` yet, so no preserved file to retry from — non-PDF sources have the same
gap) — re-upload the file instead in either case.

**`POST /jobs/{job_id}/cancel`** — signals a `queued`/`running` job to stop at its next
checkpoint.
```bash
curl -s -X POST http://localhost:8000/jobs/<job_id>/cancel
```
→ `204` on success. Cooperative, not instant: a job already inside Docling's own
extraction call stops as soon as that call returns (nothing inside it checks for
cancellation), a job still queued behind `max_concurrent_extractions` others or in
chunking/embedding stops within moments. **`404`** if the job doesn't exist. **`409`**
if it already finished, or if its row says active but this backend process isn't
actually the one running it (e.g. orphaned by an earlier restart — cancelling it here
can't help a job like that).

### Search

**`POST /search`** — hybrid (dense + BM25 + RRF) search, optionally reranked, optionally
across multiple collections at once.
```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question here",
    "collection_names": ["my-first-collection"],
    "top_k": 5,
    "use_reranker": true,
    "file_hashes": null
  }'
```
Body: `query` (required), exactly one of `collection_names` (list) or `db_name` (str —
searches every collection under that DB), `top_k` (default 5), `use_reranker` (default
`true`), `file_hashes` (optional list — restricts results to specific source documents
by their `file_hash`, from `/collections/{name}/documents`), `strategy` (optional —
restricts to collections whose catalog `chunking_strategy` matches; mainly relevant with
`db_name`, since a DB can hold collections built with different strategies over the same
underlying documents, and without this filter the merged top-k could mix near-duplicate
chunks of the same content from competing strategies). **`400`** if neither
`collection_names` nor `db_name` is given.

→ `list[RetrievalHit]`: `{chunk_id, score, text, context_text, collection_name, strategy, kind, page_no, headings, doc_stem, file_hash, images}`.

`text` is the precise matched chunk (good for citations); `context_text` is what to feed
an LLM — identical to `text` except for a `parent_child` strategy's `kind: "child"` hits,
where it's swapped for the full parent chunk.

Note `strategy` on a *request* filters by collection; `strategy` on a returned *hit* is
the tag on that specific chunk, which can differ from the collection's own
`chunking_strategy` for composable additions (e.g. an `image_description` chunk living
inside a `hybrid_docling` collection still carries `strategy: "image_description"`).

### System

**`GET /system/info`** — local machine specs, for a future model-selection UI to know
what's actually viable to run.
```bash
curl -s http://localhost:8000/system/info
```
```json
{
  "platform": "Windows", "platform_release": "11", "architecture": "AMD64",
  "cpu_cores_physical": 8, "cpu_cores_logical": 16,
  "ram_total_gb": 15.6, "ram_available_gb": 2.4,
  "gpus": [{"name": "NVIDIA GeForce GTX 1660 SUPER", "vendor": "nvidia", "vram_total_mb": 6144, "vram_free_mb": 1845, "unified_memory": false}],
  "ollama": {"available": true, "host": "http://localhost:11434", "models": [{"name": "bge-m3:latest", "size_gb": 1.08}, ...], "error": null},
  "loaded_models": [{"name": "bge-m3:latest", "size_gb": 0.62, "size_vram_gb": 0.62, "expires_at": "2026-07-17T00:10:27-04:00"}],
  "torch_acceleration": {"installed": true, "device": "cuda", "gpu_name": "NVIDIA GeForce GTX 1660 SUPER"}
}
```
GPU detection covers NVIDIA (`nvidia-smi`) and Apple Silicon (unified memory — no
separate `vram_total_mb`, just `unified_memory: true`); AMD/Intel GPUs aren't detected
(empty `gpus` list, not an error). `ollama.available: false` with an `error` message
means Ollama isn't reachable at the configured host — not a failed request, this
endpoint always returns `200`. `torch_acceleration.device` (`"cuda"`/`"mps"`/`"cpu"`/
`null` if torch isn't installed) is a *different* fact from `gpus` above — whether
`torch` itself is actually using a detected GPU, not just whether one exists; see
`torch_installation.md` if this shows `"cpu"` despite `gpus` being non-empty.

**`POST /system/ollama/unload`** — immediately evicts a model from Ollama's memory
instead of waiting for its `keep_alive` timeout.
```bash
curl -s -X POST http://localhost:8000/system/ollama/unload \
  -H "Content-Type: application/json" -d '{"model": "bge-m3:latest"}'
```
→ `204`. Uses Ollama's own documented immediate-unload pattern
(`generate(model=X, prompt="", keep_alive=0)`).

**`GET /system/ollama/models`** — every pulled Ollama model's name/size/capabilities
(e.g. `"tools"`, `"vision"`), for filtering a model-selection dropdown to only
options that will actually work. Deliberately separate from `GET /system/info`
(polled every 4s/30s by the sidebar) since this costs one extra `ollama show` call
per model — fetched once per Settings page load instead.
```bash
curl -s http://localhost:8000/system/ollama/models
```
```json
[{"name": "qwen3.5:2b", "size_gb": 2.47, "capabilities": ["tools", "completion"]}, ...]
```

### System — Maintenance

**`POST /system/jobs/clear-stale`** — marks every ingestion job left
`"queued"`/`"running"` by a backend that crashed or was restarted mid-job as
`"error"`. Only touches job status rows, never real documents/collections/chat
data — safe to call any time, including with nothing actually stale.
```bash
curl -s -X POST http://localhost:8000/system/jobs/clear-stale
```
```json
{"cleared_count": 2, "cleared_filenames": ["report.pdf", "resume_042.pdf"]}
```
Shares its backend logic (`catalog.crud.clear_stale_jobs()`) with
`scripts/fresh_start.py`, a standalone dev-environment reset script that also
kills stray processes on this project's dev ports — the API route only does the
job-clearing half, not the process-killing half, since that's a dev-only concern.

### System — Performance

**`GET`/`PUT /system/performance`** — Docling extraction pipeline tuning:
`num_threads`, `page_batch_size`, `max_concurrent_extractions` — each auto-detected
from this machine unless overridden (see `edenview_progress.md`'s architecture-decisions
table for what each one does and why the defaults are what they are).
```bash
curl -s http://localhost:8000/system/performance
curl -s -X PUT http://localhost:8000/system/performance \
  -H "Content-Type: application/json" -d '{"max_concurrent_extractions": 2}'
```
```json
{"num_threads": 14, "page_batch_size": 4, "max_concurrent_extractions": 4, "num_threads_is_auto": true, "page_batch_size_is_auto": true, "max_concurrent_extractions_is_auto": true}
```
`PUT` body: any subset of the three fields; `null` for a field reverts it to
auto-detecting. `num_threads`/`page_batch_size` are re-read fresh on the next
extraction; `max_concurrent_extractions` needs an API server restart (a
`threading.Semaphore`'s capacity isn't resizable live).

### System — Workspace

**`GET`/`PUT /system/workspace`** — where every DB/collection/document/chat session is
stored on disk (`config.yaml`'s `workspace.root`).
```bash
curl -s http://localhost:8000/system/workspace
curl -s -X PUT http://localhost:8000/system/workspace \
  -H "Content-Type: application/json" -d '{"root": "D:/edenview_data"}'
```
```json
{"root": "D:/edenview_data", "resolved_path": "D:\\edenview_data"}
```
`PUT` only updates the config value — it does **not** move existing data, and needs an
API server restart to actually point a fresh process at the new folder.

**`POST /system/workspace/browse`** — opens a native OS folder-picker dialog
server-side, returns the chosen path (or `null` if the user closed/canceled it).
```bash
curl -s -X POST http://localhost:8000/system/workspace/browse
```
```json
{"path": "D:/edenview_data"}
```
endpoint always returns `200`.

### Chat

**`POST /chat`** — hybrid search (same as `/search`) followed by either one Ollama
chat call over the retrieved context (**Simple RAG**, the default), or a full
ADK-based agentic pipeline (**Agentic RAG**, `edenview_RAG.agentic_rag` — one flat
`root_agent` → `query_pipeline` (`question_capture` → `decompose`, once per turn →
`subquestion_orchestrator`, running a full `search_executor`/`eval`/`reworder`/
`deep_search` research loop independently per decomposed sub-question, each
producing its own drafted, cited answer → `answer_formatter`, consolidating those
drafts into one final answer); see that package's own module docstrings for the
full design). Both modes return the same `ChatResponse` shape and persist to the
same `chat_sessions`/`chat_messages` tables.
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the fund balance as of the latest report?",
    "collection_names": ["my-first-collection"],
    "top_k": 5,
    "use_reranker": true
  }'
```
```json
{"answer":"The fund balance was $X as of... [1]","citations":[{"chunk_id":"...","score":5.49,"text":"...", "...": "..."}],"model_used":"qwen3:4b"}
```
Body: same scoping fields as `/search` (`query` required, exactly one of
`collection_names`/`db_name`, `top_k`, `use_reranker`, `file_hashes`, `strategy`)
plus optional `chat_model` (overrides config.yaml's `models.chat_llm` for this one
call, Simple RAG only) and `agentic` (bool, default `false`). No effort/tier
selector — the agentic pipeline is one flat design now, not a choice of tiers.
**`400`** if neither `collection_names` nor `db_name` is given.

→ `{"answer": str, "citations": list[RetrievalHit], "model_used": str, "session_id": str, "thinking": str | null}`.
`citations` is the same `RetrievalHit` list `/search` returns — index into it to
match the answer's `[1]`/`[2]` markers, which refer to citation position (1-based)
in that list. Simple RAG: if nothing relevant is found, `answer` is a canned "no
relevant information" message, `citations` is `[]`, and no LLM call is made.
Agentic RAG: `model_used` is `config.yaml`'s `agent.model`, ignoring `chat_model`
(the agentic pipeline always uses the one shared agent model, never a per-request
override); `thinking` carries the agent's own reasoning/planning narration for the
turn, kept separate from `answer` so a UI can show it as an expandable section
rather than as part of the displayed response.

**`POST /chat/stream`** — SSE variant of `POST /chat`, **agentic requests only**
(**`400`** if `agentic` isn't `true`). Same request body as `/chat`. Streams live
per-node and per-tool-call status as the pipeline runs — a real turn can genuinely
take several minutes (reworder + search + up to `agent.max_iterations` eval/
deep-search rounds + answer formatting, native "thinking" kept on for every LLM
step to reduce hallucination), and a bare spinner reads as broken for that long.
```bash
curl -s -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "collection_names": ["my-first-collection"], "agentic": true}'
```
```
data: {"type": "status", "node": "reworder", "phase": "start", "message": "Rewording your question..."}

data: {"type": "status", "node": "reworder", "phase": "end", "duration_s": 4.2}

data: {"type": "status", "node": "vector_search", "phase": "start", "message": "Searching your documents..."}

data: {"type": "thinking", "message": "...agent's reasoning chunk..."}

data: {"type": "result", "answer": "...", "citations": [...], "model_used": "qwen3.5:2b", "session_id": "...", "thinking": "..."}
```
Every frame is `data: {json}\n\n`. Zero or more `{"type": "status", ...}` frames,
interleaved with zero or more `{"type": "thinking", "message": str}` frames, then
exactly one `{"type": "result", ...}` frame — the same shape `/chat` returns,
persisted to `chat_messages` the same way once it fires. A `status` frame's `node`
is either an agent name (`reworder`, `eval`, `deep_search`, ...) or a tool name
(`vector_search`, `get_images`, ...) — granular down to individual tool calls, not
just top-level agent phases; `phase: "start"` always carries a human-readable
`message`, `phase: "end"` carries `duration_s` instead. This is the same event
shape a future flowchart-style status UI (lighting up the active node) would
consume, not just a simple status line — see `edenview-ui/src/lib/types.ts`'s
`AgenticStatusEvent`. The Chat UI's "Agentic RAG" mode uses this endpoint (via
manual SSE frame-parsing over `fetch`, not `EventSource`, since that API can't send
a POST body — see `edenview-ui/src/lib/api.ts`'s `runChatStream`); `POST /chat`'s
own `agentic: true` path stays available for scripts/tests where live progress
doesn't matter.

This turn actually runs via an in-memory per-session broadcast registry
(`api/routers/chat.py`'s `_active_turns`) — `POST /chat/stream`'s own connection is
just that turn's first subscriber, not the only possible one.

**`GET /chat/stream/{session_id}`** — reattaches to a still-running turn on this
session (page reload, switching chats and back, or any client that wasn't the one
that started it): replays every event already emitted, then continues live exactly
like the original `POST /chat/stream` connection.
```bash
curl -s -N http://localhost:8000/chat/stream/<session_id>
```
If nothing is actually in flight for that session (already finished, never
started, or its ~30s post-completion grace period already elapsed), yields exactly
one `{"type": "not_running"}` frame and closes — cheap and safe to call
unconditionally on every session load, no need to check first. A turn keeps
running and persists its answer regardless of whether any client is subscribed —
dropping every connection doesn't stop it.

### System — model config

**`GET /system/config`** — every model name from `config.yaml`'s `models:` section,
plus the configured Ollama host and the agentic pipeline's own model settings
(`agent_model`, `agent_vision_model`, `agent_max_iterations`, from `config.yaml`'s
`agent:` section), for a Settings UI to render as a form.
```bash
curl -s http://localhost:8000/system/config
```
```json
{"tokenizer":"BAAI/bge-m3","dense_embedding":"bge-m3","dense_embedding_dim":1024,"sparse_embedding":"Qdrant/bm25","contextual_llm":"qwen3:4b","picture_description_llm":"qwen3-vl:2b","chat_llm":"qwen3:4b","reranker":"Xenova/ms-marco-MiniLM-L-6-v2","ollama_host":"http://localhost:11434","agent_model":"qwen3.5:2b","agent_vision_model":null,"agent_max_iterations":3}
```

**`PUT /system/config`** — updates one or more of those keys, **persisted to
config.yaml on disk** (via `ruamel.yaml`'s round-trip mode, so the file's existing
comments survive), not just a session-scoped override.
```bash
curl -s -X PUT http://localhost:8000/system/config \
  -H "Content-Type: application/json" \
  -d '{"chat_llm": "qwen3:8b"}'
```
```json
{"updated": {"...": "...", "chat_llm": "qwen3:8b"}, "restart_required": []}
```
Body: any subset of the fields `GET /system/config` returns. **`400`** if an unknown
key is sent, if the body is empty, or if `agent_model` is set to a model that
doesn't report tool-calling support (checked live via `ollama show` at request
time — the agentic pipeline hard-requires this, see `edenview_progress.md`'s
architecture table). `agent_vision_model` has no equivalent hard check — an
unavailable vision model degrades gracefully instead of erroring.

`restart_required` lists which of the keys you just changed need the API server
restarted to actually take effect. Traced through the real call sites, not guessed:
`dense_embedding` and `ollama_host` are read fresh on every embed call
(`vectorstore/embedding.py`), and `chat_llm` is read fresh on every `/chat` call
(`api/routers/chat.py`), so those three apply immediately. `tokenizer`,
`sparse_embedding`, `contextual_llm`, `picture_description_llm`, `reranker`, and
all three `agent_*` keys are each baked into a Pydantic config class's default (or,
for `sparse_embedding`/the agentic pipeline's shared LLM, a module-global cached
instance) at process import time — changing them in config.yaml only takes effect
on the next `uvicorn` restart.

### Files

**`GET /files?path=...`** — serves an image referenced in a hit's or preview chunk's
`images[].image_path`.
```bash
curl -s --get --data-urlencode "path=<image_path>" http://localhost:8000/files -o out.png
```
`path` must resolve to somewhere under the configured documents directory
(`config.yaml`'s `storage.documents_dir`) — anything else gets **`403`**. A path that's
allowed but doesn't exist gets **`404`**. This is not a general file server.

---

## 6. Error responses

All errors come back as `{"detail": "human-readable message"}` with the HTTP status
indicating what went wrong: `400` bad input (unknown strategy, missing search target),
`403` file path outside the allowed directory, `404` not found (db/collection/job/file),
`409` conflict (duplicate DB name, deleting a DB that still has collections).
