"""Typed row models for the catalog tables -- what crud.py's read functions return,
instead of raw DuckDB tuples."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, computed_field


class DBRecord(BaseModel):
    db_id: str
    name: str
    created_at: datetime


class CollectionRecord(BaseModel):
    collection_id: str
    db_id: str
    qdrant_collection_name: str
    chunking_strategy: str
    embedding_model: str
    dense_dim: int
    sparse_model: Optional[str]
    status: str
    chunk_count: int
    doc_count: int
    created_at: datetime


class DocumentRecord(BaseModel):
    doc_id: str
    file_hash: str
    filename: str
    source_path: Optional[str]
    input_format: Optional[str]
    num_pages: Optional[int]
    first_ingested_at: datetime


class IngestionJobRecord(BaseModel):
    job_id: str
    collection_id: str
    doc_id: Optional[str]
    status: str
    # Stored at job-creation time (see schema.py's comment) -- known immediately,
    # unlike doc_id/documents.filename which only resolve once extraction finishes.
    filename: Optional[str] = None
    # stage: "extracting" | "chunking" | "embedding" | None (not yet started / done).
    # stage_current/stage_total: real counts, only populated during "embedding" -- see
    # catalog/schema.py's ingestion_jobs comment for why extraction has none.
    stage: Optional[str] = None
    stage_current: Optional[int] = None
    stage_total: Optional[int] = None
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]
    # Joined in from collections/dbs at read time (get_job()/list_jobs()) -- not
    # stored on this table itself, just resolved for display so the frontend doesn't
    # need a second round-trip per job to show where it's going.
    qdrant_collection_name: Optional[str] = None
    db_name: Optional[str] = None

    @computed_field
    @property
    def stage_pct(self) -> Optional[float]:
        """A real percentage (stage_current / stage_total), not an estimate -- only
        ever non-None during "embedding", the one phase with genuine progress counts.
        None for every other stage rather than a fabricated number (see
        catalog/schema.py's ingestion_jobs comment for why extraction has no
        equivalent)."""
        if self.stage_current is None or not self.stage_total:
            return None
        return round(100 * self.stage_current / self.stage_total, 1)


class ChatSessionRecord(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageRecord(BaseModel):
    message_id: str
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    citations: Optional[list[dict]] = None  # decoded from the citations JSON column
    model_used: Optional[str] = None
    created_at: datetime
