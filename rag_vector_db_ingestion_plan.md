# RAG Vector DB Ingestion Plan
### Stack: Docling · bge-m3 (FlagEmbedding) · Qdrant local path · Ollama (qwen3:4b) · 5 chunking strategies

---

## Hardware budget

- GPU: NVIDIA GeForce GTX 1660 SUPER — **6GB VRAM** (not 16GB — confirmed via `nvidia-smi`)
- System RAM: 16GB
- Intel UHD iGPU is present but not used for inference (no CUDA; Ollama gets no meaningful acceleration from it)
- This rules out `gemma4:12b` (doesn't exist — no Gemma4 at original plan time) and any LLM whose Q4 quant exceeds ~5GB, since the embedder (bge-m3, ~1.2GB) needs to coexist in VRAM during S3/S4 batch runs.

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | Qdrant `path="./qdrant_db"` | Persistent local folder, no Docker, no server process. Verified: data persists across process restarts; concurrent opens raise a clean `RuntimeError` rather than corrupting data — ingest and query must run sequentially, not concurrently. |
| PDF parser | Docling | Structure-aware: preserves headings, tables, reading order |
| Embed model | bge-m3 via Ollama (dense) + FastEmbed `Qdrant/bm25` (sparse) | Keeps everything on Ollama for the dense vectors (no separate CUDA/torch install needed). Ollama's bge-m3 only returns dense vectors, so sparse/keyword search is covered by FastEmbed's BM25 sparse model (ONNX-based, CPU, no GPU needed) — raw term frequencies, with Qdrant's `Modifier.IDF` on the sparse field completing real BM25 scoring as points are added to the collection. |
| LLM | qwen3:4b via Ollama | ~2.5GB Q4_K_M, fits 6GB VRAM alongside the embedder. Used only in S3, S4, S5 at index time. Considered qwen3.5:4b (newer, multimodal) but Ollama has an open bug (qwen3.5 tool-calling/structured-output template mismatch — ollama/ollama#14493, #14745) so stuck with the proven qwen3 generation. |
| Hybrid search | Qdrant RRF fusion | Dense (semantic) + sparse (keyword) merged natively |
| Multi-doc scope | "Space" namespacing | Each ingest script takes `--pdf` and `--space` args. Collections are named `{space}_{strategy}` (e.g. `kerala_finance_s1_overlap`), so any PDF can be passed in and appended to the same logical knowledge base over time. Point IDs are deterministic (`uuid5` of space+strategy+doc+chunk-index) so re-running an ingest on the same PDF overwrites rather than duplicates. |

---

## Repo structure

```
rag-project/
├── data/
│   └── input.pdf                   # source document
├── cache/
│   ├── doc.json                    # Docling parse output (cached, committed to git)
│   ├── doc.md                      # Markdown export of PDF (cached, committed to git)
│   └── s3_contexts.json            # pre-generated LLM context strings for S3 (cache)
├── qdrant_db/                      # LOCAL Qdrant storage folder — in .gitignore
├── s2_docstore/
│   └── parents.json                # parent chunk store for S2 — in .gitignore
├── ingest/
│   ├── shared.py                   # Docling parsing, embed function, Qdrant client
│   ├── s1_overlap.py
│   ├── s2_parent_child.py
│   ├── s3_contextual.py
│   ├── s4_agentic.py
│   └── s5_summary.py
├── query/
│   ├── retriever.py                # shared hybrid search function
│   └── query.py                    # CLI entrypoint
├── requirements.txt
├── .gitignore
└── README.md
```

**.gitignore must include:**
```
qdrant_db/
s2_docstore/
__pycache__/
.env
```

**What IS committed to git** (so collaborators skip expensive steps):
- `cache/doc.json` — Docling parse (~1–3 min, needs GPU)
- `cache/doc.md` — Markdown export (derived from above)
- `cache/s3_contexts.json` — LLM-generated contexts (~2–15 min depending on async)

**What is NOT committed** (must be rebuilt locally):
- `qdrant_db/` — vector embeddings (model-specific, large)
- `s2_docstore/parents.json` — parent chunks (derived, fast to rebuild)

---

## Phase 1 — Environment setup

### Step 1.1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` (current, S1/S2 scope — llama-index deps for S5 added when we build that phase):
```
docling
qdrant-client
fastembed
ollama
langchain-text-splitters
tqdm
```

> **No torch/CUDA install needed.** Dense embeddings go through the Ollama server
> (which manages its own GPU offload); sparse BM25 embeddings via FastEmbed run on
> ONNX runtime, CPU-only. This was a deliberate simplification from an earlier
> FlagEmbedding-based design — see the embed model row in Key design decisions above.

### Step 1.2 — Pull Ollama models

```bash
# Embed model (dense vectors)
ollama pull bge-m3

# LLM — used for S3 context generation, S4 proposition extraction, S5 summaries
ollama pull qwen3:4b
```

### Step 1.3 — Verify Qdrant local mode works

```python
from qdrant_client import QdrantClient

# This creates ./qdrant_db/ on first run and loads it on subsequent runs
# No server, no Docker, no ports — just a folder on disk
client = QdrantClient(path="./qdrant_db")
print(client.get_collections())  # should print empty list on first run
```

---

## Phase 2 — Parse PDF with Docling (run once, cache result)

### Step 2.1 / 2.2 — Shared Docling parsing + embedding + Qdrant helpers

Implemented in [`ingest/shared.py`](ingest/shared.py) — see that file for the actual code.
Key differences from the original single-document sketch:

- **Multi-document, not single `PDF_PATH` constant.** `get_docling_doc(pdf_path)` /
  `get_docling_markdown(pdf_path)` take a path and derive a `doc_stem` from the filename.
  Cache lives at `cache/<doc_stem>/doc.json` and `cache/<doc_stem>/doc.md` — one subfolder
  per source document, so any PDF can be parsed and cached independently.
- **Embeddings are dense (Ollama) + sparse (FastEmbed BM25)**, not FlagEmbedding —
  see the embed model row in Key design decisions above.
- **Collections are namespaced by space**: `create_collection(client, space, strategy)`
  creates `{space}_{strategy}` (e.g. `kerala_finance_s1_overlap`) if it doesn't exist.
- **Point IDs are deterministic** (`make_point_id(space, strategy, stem, index)`, a `uuid5`
  hash) so re-running an ingest script on the same PDF overwrites existing points instead
  of duplicating them — ingestion is idempotent and incremental.

**What Docling preserves that plain PDF text extraction misses:**
- Heading hierarchy (`H1 > H2 > H3`) — used by HybridChunker and MarkdownHeaderTextSplitter
- Table structure (TableFormer model) — each table as a structured element, not garbled text
- Reading order across multi-column layouts
- Page number metadata per element
- Figure / caption associations

> **One parse per document, all strategies.** Every ingestion script imports
> `get_docling_doc()` / `get_docling_markdown()` from `shared.py`. Each document's
> Docling parse runs once and is cached; subsequent runs (even for a different
> strategy) load from `cache/<doc_stem>/doc.json` in under a second.

---

## Phase 3 — Strategy ingestion scripts

Run these independently in any order. Each script is idempotent — it checks if
the collection already exists and skips creation if so.

---

### S1 — Overlap chunking (baseline)

**File:** `ingest/s1_overlap.py`

**No LLM calls. Fastest to run. Run this first.**

```
Input:  cache/doc.md  (Markdown export)
Output: Qdrant collection "s1_overlap"
Time:   ~2–5 min for 300 pages (embed only)
```

**Steps:**

1. Load `cache/doc.md`
2. Split with `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`
   - Tries paragraph → sentence → word → character boundaries in order
   - `chunk_overlap=50` gives ~10% overlap for context continuity at boundaries
3. Collect all chunk texts into a list
4. Batch embed with `embed_texts()` → dense + sparse vectors per chunk
5. Build `PointStruct` per chunk with payload:
   ```python
   {
       "text":         chunk_text,
       "chunk_index":  i,
       "strategy":     "s1",
       "char_start":   chunk.metadata.get("start_index", 0)
   }
   ```
6. Upsert all points to `"s1_overlap"` collection in batches of 100

**Qdrant collection schema:**

| Field | Type | Notes |
|---|---|---|
| vector["dense"] | float32[1024] | bge-m3 dense |
| vector["sparse"] | SparseVector | bge-m3 lexical weights |
| payload.text | string | raw chunk text |
| payload.chunk_index | int | position in document |
| payload.strategy | string | "s1" |

---

### S2 — Parent document retriever (small-to-big)

**File:** `ingest/s2_parent_child.py`

**No LLM calls. Two-level split: only child chunks go to Qdrant.**

```
Input:  cache/doc.md
Output: Qdrant collection "s2_parent_child"
        s2_docstore/parents.json  (parent text lookup)
Time:   ~3–6 min for 300 pages
```

**Steps:**

1. Load `cache/doc.md`
2. Split into **parent chunks** (large context units):
   - `RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)`
   - Assign each parent a UUID: `parent_id`
3. Split each parent into **child chunks** (small retrieval units):
   - `RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)`
   - Each child stores `parent_id` in its metadata
4. Save parent docstore to `s2_docstore/parents.json`:
   ```python
   # {parent_id: parent_text}
   json.dump({p.metadata["parent_id"]: p.page_content for p in parents},
             open("s2_docstore/parents.json", "w"))
   ```
5. Embed only the **child** chunk texts → dense + sparse
6. Upsert child points to `"s2_parent_child"` with payload:
   ```python
   {
       "text":       child_text,
       "parent_id":  parent_id,       # key for docstore lookup
       "strategy":   "s2"
   }
   ```

**At query time** (not index time):
- Search Qdrant → get top-k child points
- Extract `payload["parent_id"]` from each hit
- Load parent text from `s2_docstore/parents.json`
- Inject **parent** texts (not child texts) into LLM prompt

**Why this works:** Child chunks are small and precise → high vector similarity.
Parent chunks are large and rich → LLM gets full context. Best of both.

---

### S3 — Contextual chunking (Anthropic-style)

**File:** `ingest/s3_contextual.py`

**LLM call per chunk. Most impactful for retrieval quality. Run as overnight job.**

```
Input:  DoclingDocument (from cache/doc.json)
        cache/s3_contexts.json  (if pre-generated, skip LLM step)
Output: Qdrant collection "s3_contextual"
Time:   ~2–15 min (async Ollama calls) + ~5 min embedding
```

**Steps:**

1. Load `DoclingDocument` from `cache/doc.json`
2. Chunk with Docling's `HybridChunker` (operates on native DoclingDocument):
   ```python
   from docling.chunking import HybridChunker
   chunks = list(HybridChunker().chunk(doc))
   # Each chunk has: chunk.text, chunk.meta.headings, chunk.meta.page_no
   ```
   > Use `HybridChunker` (not Markdown + splitter) here because it preserves
   > heading path metadata per chunk — this is used in the context generation prompt
   > without having to re-parse the document structure.

3. **Generate context strings** (check cache first):
   ```python
   # cache hit: load from s3_contexts.json and skip LLM calls
   # cache miss: run async Ollama calls
   ```
   Prompt template per chunk:
   ```
   You are helping build a retrieval system for a technical document.

   Document section: {chunk.meta.headings}

   Chunk text:
   {chunk.text}

   Write a single short sentence (max 30 words) that situates this chunk
   within the document and would help a search system find it. Do not
   restate the content — explain where it fits. Answer with the sentence only.
   ```
   Run with `asyncio` + semaphore (8 concurrent Ollama calls):
   ```python
   semaphore = asyncio.Semaphore(8)
   tasks = [generate_context(chunk, semaphore) for chunk in chunks]
   contexts = await asyncio.gather(*tasks)
   ```
   Save to `cache/s3_contexts.json` immediately after generation.

4. **Build enriched strings:**
   ```python
   enriched = f"{context}\n\n{chunk.text}"
   ```
   The enriched string is what gets embedded. The raw `chunk.text` is stored
   in payload for display at query time — never show the enriched version to users.

5. Embed enriched strings (not raw chunks) → dense + sparse
6. Upsert to `"s3_contextual"` with payload:
   ```python
   {
       "text":         chunk.text,        # display text (raw)
       "context":      context,           # generated context sentence
       "enriched":     enriched,          # what was embedded (for debugging)
       "headings":     chunk.meta.headings,
       "page_no":      chunk.meta.page_no,
       "strategy":     "s3"
   }
   ```

> **Performance note:** A 300-page PDF at 512-token chunks ≈ 400–500 chunks.
> With 8 async workers at ~1–2s per call → approximately 2–4 minutes total.
> Always save to cache before embedding — if embedding fails you don't
> want to re-run 500 LLM calls.

---

### S4 — Agentic / proposition chunker

**File:** `ingest/s4_agentic.py`

**LLM call per section. Highest quality, highest cost. Run on subset first.**

```
Input:  cache/doc.md
Output: Qdrant collection "s4_agentic"
Time:   20–60 min for 300 pages (many LLM calls + structured output parsing)
```

**Steps:**

1. Load `cache/doc.md`
2. Split into heading-level sections using `MarkdownHeaderTextSplitter`:
   ```python
   from langchain_text_splitters import MarkdownHeaderTextSplitter
   splitter = MarkdownHeaderTextSplitter(
       headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
   )
   sections = splitter.split_text(md_text)
   # Each section: page_content = body text, metadata = {h1, h2, h3}
   ```
3. **For each section, call Ollama for proposition extraction:**
   ```
   Extract every atomic factual proposition from the text below.
   Rules:
   - Each proposition must be a single, self-contained, verifiable sentence
   - Include all key facts, numbers, names, and relationships
   - Do not include meta-commentary or section headers
   - Return ONLY a valid JSON array of strings, no preamble or explanation

   Text:
   {section.page_content}
   ```
   Use `gemma4:12b` with `format="json"` (Ollama structured output mode).
   Validate JSON — retry up to 2 times on parse failure with a simpler prompt.

4. Each proposition → one chunk. Attach section metadata to each:
   ```python
   payload = {
       "text":     proposition,
       "section":  section.metadata.get("h2", section.metadata.get("h1", "")),
       "strategy": "s4"
   }
   ```
5. Embed all propositions → dense + sparse
6. Upsert to `"s4_agentic"`

> **Tip:** Run on one chapter or section first (`sections[:10]`) to assess
> proposition quality before committing to the full document. Expect
> 5–20 propositions per section → 1,000–3,000 total points for 300 pages.

> **Retry logic is critical** here. gemma4:12b reliably outputs valid JSON
> with `format="json"` but always wrap `json.loads()` in try/except and
> log failures for review.

---

### S5 — DocumentSummaryIndex (LlamaIndex native)

**File:** `ingest/s5_summary.py`

**LLM call per document/section. Fully managed by LlamaIndex.**

```
Input:  data/input.pdf  (DoclingReader handles parsing internally)
Output: Qdrant collection "s5_summary"
        cache/s5_index/  (LlamaIndex index state)
Time:   ~5–15 min (LLM summary generation + embedding)
```

**Steps:**

1. Configure LlamaIndex global settings:
   ```python
   from llama_index.core import Settings
   from llama_index.llms.ollama import Ollama
   from llama_index.embeddings.ollama import OllamaEmbedding

   Settings.llm = Ollama(model="gemma4:12b", request_timeout=120.0)
   Settings.embed_model = OllamaEmbedding(model_name="bge-m3")
   # Note: LlamaIndex's Ollama embed gives dense only
   # For hybrid, set enable_hybrid=True on QdrantVectorStore (uses FastEmbed BM25 internally)
   ```

2. Load PDF via DoclingReader:
   ```python
   from llama_index.readers.docling import DoclingReader
   docs = DoclingReader().load_data("data/input.pdf")
   # Returns list of LlamaIndex Document objects with Docling metadata
   ```

3. Connect Qdrant local path as vector store:
   ```python
   from qdrant_client import QdrantClient
   from llama_index.vector_stores.qdrant import QdrantVectorStore
   from llama_index.core import StorageContext

   qclient = QdrantClient(path="./qdrant_db")
   vector_store = QdrantVectorStore(
       collection_name="s5_summary",
       client=qclient,
       enable_hybrid=True   # LlamaIndex manages sparse via FastEmbed BM25
   )
   storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
   ```

4. Build the DocumentSummaryIndex:
   ```python
   from llama_index.core import DocumentSummaryIndex

   index = DocumentSummaryIndex.from_documents(
       docs,
       storage_context=storage_ctx,
       show_progress=True,
       # LLM generates a summary for each Document object
       # Summary nodes drive document-level routing at query time
   )
   ```

5. Persist index state to disk:
   ```python
   index.storage_context.persist("cache/s5_index")
   # Saves summary nodes, doc metadata, index state
   # Load later with: load_index_from_storage(StorageContext.from_defaults(...))
   ```

**At query time:**
```python
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("your question here")
print(response)  # LlamaIndex handles retrieval + LLM in one call
```

---

## Phase 4 — Qdrant collection reference

All five collections share the same vector schema. One collection per strategy
makes it easy to run side-by-side retrieval experiments.

| Collection | Strategy | Points (est. 300 pages) | LLM at index |
|---|---|---|---|
| `s1_overlap` | Overlap | ~800–1,200 | None |
| `s2_parent_child` | Parent-child | ~3,000–5,000 (children) | None |
| `s3_contextual` | Contextual | ~400–600 | 1 per chunk |
| `s4_agentic` | Agentic | ~1,000–3,000 | 1 per section |
| `s5_summary` | Summary index | varies | 1 per doc/section |

**Vector schema (S1–S4):**
```python
vectors_config={
    "dense": VectorParams(size=1024, distance=Distance.COSINE)
},
sparse_vectors_config={
    "sparse": SparseVectorParams()
}
```

**Payload fields across all collections:**

| Field | Present in | Type | Notes |
|---|---|---|---|
| `text` | S1–S4 | str | Display text shown to user |
| `strategy` | S1–S4 | str | "s1"/"s2"/"s3"/"s4" |
| `chunk_index` | S1 | int | Position in doc |
| `parent_id` | S2 | str | UUID for docstore lookup |
| `context` | S3 | str | LLM-generated context sentence |
| `headings` | S3 | list[str] | Docling heading path |
| `page_no` | S3 | int | Source page |
| `section` | S4 | str | Source heading |

---

## Phase 5 — Query pipeline

### Hybrid search function (shared by S1, S2, S3, S4)

```python
# query/retriever.py

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from ingest.shared import EMBED_MODEL

client = QdrantClient(path="./qdrant_db")

def hybrid_search(collection: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid search: dense (semantic) + sparse (keyword) fused with RRF.
    Returns list of {text, score, payload} dicts.
    """
    # Embed query with same model used at index time
    q_out = EMBED_MODEL.encode(
        [query],
        return_dense=True,
        return_sparse=True
    )
    dense_vec = q_out["dense_vecs"][0].tolist()
    sparse_weights = q_out["lexical_weights"][0]

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 4    # over-fetch for RRF to work well
            ),
            Prefetch(
                query=SparseVector(
                    indices=[int(k) for k in sparse_weights.keys()],
                    values=[float(v) for v in sparse_weights.values()]
                ),
                using="sparse",
                limit=top_k * 4
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
    )
    return [
        {"text": r.payload["text"], "score": r.score, "payload": r.payload}
        for r in results.points
    ]
```

### S2 query — parent swap

```python
# After hybrid_search on "s2_parent_child":
import json

parents = json.load(open("s2_docstore/parents.json"))

hits = hybrid_search("s2_parent_child", query, top_k=5)
context_texts = [
    parents[h["payload"]["parent_id"]]   # swap child for parent
    for h in hits
    if h["payload"]["parent_id"] in parents
]
```

### RAG prompt construction

```python
def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question using only the
provided context. If the context does not contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""
```

### Ollama LLM call

```python
import ollama

response = ollama.chat(
    model="gemma4:12b",
    messages=[{"role": "user", "content": prompt}]
)
print(response["message"]["content"])
```

---

## Phase 6 — Recommended run order

```bash
# 1. Parse PDF once (cached after first run)
python -c "from ingest.shared import get_docling_doc; get_docling_doc()"

# 2. Fast strategies first (no LLM calls)
python ingest/s1_overlap.py
python ingest/s2_parent_child.py

# 3. LLM-light strategy
python ingest/s5_summary.py

# 4. LLM-per-chunk (run async, results cached)
python ingest/s3_contextual.py

# 5. Most expensive — run on a subset first
python ingest/s4_agentic.py --sections 10  # test first 10 sections
python ingest/s4_agentic.py               # full run if quality looks good

# 6. Query any strategy
python query/query.py --strategy s1 --question "What is the capital requirement?"
python query/query.py --strategy s3 --question "What is the capital requirement?"
# Compare outputs to see which strategy retrieves better context
```

---

## Phase 7 — Evaluation

Before choosing a single strategy for production, evaluate all five with the
same test set.

**Build a test set of 20–30 questions** covering:
- Factual lookups (page-specific facts, numbers, dates)
- Concept questions (span multiple sections)
- Keyword-specific queries (good for sparse/hybrid advantage)
- Ambiguous phrasings (tests context quality)

**Metrics to track per strategy:**

| Metric | What it measures | How to compute |
|---|---|---|
| Recall@5 | Is the relevant chunk in top 5? | Manual annotation vs retrieval |
| MRR | Mean reciprocal rank of first hit | `1 / rank_of_first_hit` averaged |
| Faithfulness | Does LLM answer match context? | RAGAS library |
| Answer relevance | Is the answer actually on-topic? | RAGAS library |

```bash
pip install ragas
```

**Expected outcome pattern:**
- S4 (agentic) typically wins on precision for specific factual questions
- S3 (contextual) typically wins on cross-section reasoning questions
- S2 (parent-child) typically has the best precision/cost tradeoff
- S1 (overlap) is the baseline — if others don't beat it, simplify

---

## For collaborators cloning this repo

```bash
git clone https://github.com/your-repo/rag-project
cd rag-project
pip install -r requirements.txt

# Pull Ollama models
ollama pull bge-m3
ollama pull gemma4:12b

# Docling parse is cached — no re-parsing needed
# Just run ingestion (re-embeds from cached Markdown/JSON)
python ingest/s1_overlap.py    # ~5 min, no LLM needed
python ingest/s2_parent_child.py

# S3 context strings are cached too — just re-embeds them
python ingest/s3_contextual.py

# S4 and S5 require LLM calls — run separately
python ingest/s4_agentic.py
python ingest/s5_summary.py

# Query
python query/query.py --strategy s1 --question "your question here"
```

> **The expensive steps (Docling parse, LLM context generation) are cached and
> committed to the repo. Collaborators only need to re-embed, which is fast and
> requires no LLM — just bge-m3 via FlagEmbedding.**