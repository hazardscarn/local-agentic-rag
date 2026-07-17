"""Request/response models specific to the API layer. Deliberately thin -- wherever
catalog/retrieval already has a pydantic model that fits (DBRecord, CollectionRecord,
DocumentRecord, IngestionJobRecord, RetrievalHit), routers return that directly as the
response_model instead of a duplicate schema to keep in sync."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from edenview_ingestion.catalog import ChatMessageRecord
from edenview_RAG.retrieval import RetrievalHit


class CreateDBRequest(BaseModel):
    name: str


class IngestAccepted(BaseModel):
    job_id: str
    status: str
    qdrant_collection_name: str


class SearchRequest(BaseModel):
    query: str
    db_name: Optional[str] = None
    collection_names: Optional[list[str]] = None
    top_k: int = 5
    use_reranker: bool = True
    file_hashes: Optional[list[str]] = None
    # Restricts search to collections built with this chunking strategy -- mainly
    # relevant with db_name (fanning out across every collection in a DB), since those
    # collections can span multiple strategies over the same underlying documents.
    strategy: Optional[str] = None

    # Swagger UI's "Try it out" pre-fills a request body from each field's *type*, not
    # its actual Python default -- Optional[str] = None renders as "" and
    # Optional[list[str]] = None renders as [""], not null. Left as-is, pasting that
    # placeholder unedited silently turns "no filter" into "filter for the literal
    # empty string," which matches nothing and returns [] with no error (confirmed:
    # this happened in practice, twice). This explicit example replaces Swagger's
    # per-field guess with a real, working request.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the fund balance as of the latest report?",
                    "collection_names": ["fiscal_health"],
                    "top_k": 5,
                    "use_reranker": True,
                }
            ]
        }
    }

    @field_validator("db_name", "strategy", mode="before")
    @classmethod
    def _blank_string_to_none(cls, v):
        """Belt-and-suspenders alongside the example above: even if a caller submits
        Swagger's raw "" placeholder without editing it, treat it as "not provided"
        rather than a literal filter value that can never match anything."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("collection_names", "file_hashes", mode="before")
    @classmethod
    def _blank_list_to_none(cls, v):
        """Same as above for list[str] fields, where Swagger's placeholder is [""]."""
        if v is None:
            return None
        cleaned = [item for item in v if isinstance(item, str) and item.strip() != ""]
        return cleaned or None


class PreviewChunk(BaseModel):
    chunk_id: str
    text: str
    page_no: Optional[int] = None
    # Normalized (0..1), top-left-origin -- see RetrievalHit.bbox for the same field.
    bbox: Optional[tuple[float, float, float, float]] = None
    kind: str
    strategy: str
    file_hash: str = ""
    images: list[dict] = Field(default_factory=list)
    # Only set for a parent_child strategy's "child" chunks -- the full parent
    # context this fragment was split from, resolved from the catalog's
    # parent_chunks table (see api/routers/catalog.py's preview_collection()).
    parent_text: Optional[str] = None


class PreviewResponse(BaseModel):
    chunks: list[PreviewChunk]
    next_offset: Optional[str] = None


class ChatRequest(SearchRequest):
    """Same scoping fields as SearchRequest (query, db_name/collection_names, top_k,
    use_reranker, file_hashes, strategy) plus a chat-model override -- inherits
    SearchRequest's blank-string/blank-list validators too."""

    chat_model: Optional[str] = None
    # Omitted -> a new session is created lazily (title seeded from this query) and
    # its id comes back on the response. Provided -> the turn is appended to that
    # existing session instead.
    session_id: Optional[str] = None

    # False (default) -> today's single retrieval pass + one LLM call (generate_answer).
    # True -> edenview_RAG.agentic_rag's ADK-based loop (reframe/split, retrieval,
    # a critic/refiner refinement loop, page/image tools depending on `effort`) --
    # see edenview_RAG/agentic_rag/ for what each tier actually does. Matches the
    # Chat UI's "Simple RAG" / "Agentic RAG" toggle in the same scope panel.
    agentic: bool = False
    effort: Literal["low", "medium", "high"] = "high"

    @field_validator("chat_model", "session_id", mode="before")
    @classmethod
    def _blank_chat_model_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class ChatResponse(BaseModel):
    answer: str
    citations: list[RetrievalHit]
    model_used: str
    session_id: str
    # Agentic mode only -- the agent's own reasoning/planning narration
    # (qwen3.5's native "thinking" content) for this turn, kept separate from
    # `answer` so the UI can show it as an expandable section rather than mixing it
    # into the real response. None for non-agentic /chat calls.
    thinking: Optional[str] = None


class ChatSessionDetail(BaseModel):
    session_id: str
    title: str
    messages: list[ChatMessageRecord]


class ModelSettings(BaseModel):
    """Mirrors edenview_ingestion.settings.MODEL_KEYS -- keep in sync if that list
    changes."""

    tokenizer: str
    dense_embedding: str
    dense_embedding_dim: int
    sparse_embedding: str
    contextual_llm: str
    picture_description_llm: str
    chat_llm: str
    reranker: str
    ollama_host: Optional[str] = None
    # How long a model stays loaded after its last call before Ollama evicts it --
    # e.g. "30m", "1h", "-1" (never), "0" (immediately). Applies live, no restart --
    # see edenview_ingestion.settings.get_ollama_keep_alive().
    ollama_keep_alive: Optional[str] = None


class UpdateModelSettingsRequest(BaseModel):
    """Every field optional -- only the ones actually set are written back to
    config.yaml (see api/routers/config.py's use of model_dump(exclude_unset=True)).
    `extra="forbid"` so a typo'd key (e.g. "chat_model" instead of "chat_llm") is a
    loud 422 instead of being silently dropped and read back as "no fields changed"."""

    model_config = {"extra": "forbid"}

    tokenizer: Optional[str] = None
    dense_embedding: Optional[str] = None
    dense_embedding_dim: Optional[int] = None
    sparse_embedding: Optional[str] = None
    contextual_llm: Optional[str] = None
    picture_description_llm: Optional[str] = None
    chat_llm: Optional[str] = None
    reranker: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_keep_alive: Optional[str] = None


class UpdateModelSettingsResponse(BaseModel):
    updated: ModelSettings
    restart_required: list[str]


class WorkspaceSettings(BaseModel):
    # As stored in config.yaml -- relative ("edenview_data") or absolute.
    root: str
    # Absolute, resolved form (edenview_ingestion.settings.get_workspace_root()) --
    # what's actually in use, for display next to the editable `root` field.
    resolved_path: str


class UpdateWorkspaceRequest(BaseModel):
    root: str


class WorkspaceBrowseResponse(BaseModel):
    # None if the user closed/canceled the native folder-picker dialog.
    path: Optional[str] = None


class PerformanceSettings(BaseModel):
    # Threads Docling's own pipeline uses per extraction -- auto-detected from this
    # machine (cpu_count - 2, min 1) unless overridden.
    num_threads: int
    # Pages of a single document Docling batches together internally (its own
    # default is 4) -- a process-wide Docling setting, not per-document.
    page_batch_size: int
    # How many documents can run extraction at the same time -- default 4. Requires a
    # backend restart to take effect (see pipeline.py's _EXTRACTION_SEMAPHORE).
    max_concurrent_extractions: int
    # Whether each value above is the auto-detected default (True) or an explicit
    # override the user set (False) -- lets the Settings UI show "auto (N)" instead
    # of a bare number when nothing's been overridden.
    num_threads_is_auto: bool
    page_batch_size_is_auto: bool
    max_concurrent_extractions_is_auto: bool


class UpdatePerformanceRequest(BaseModel):
    # None reverts that field to auto-detecting/Docling's own default.
    num_threads: Optional[int] = None
    page_batch_size: Optional[int] = None
    max_concurrent_extractions: Optional[int] = None


class UnloadModelRequest(BaseModel):
    model: str
