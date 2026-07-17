"""Output model for a retrieval hit -- what search.py's functions return."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float

    text: str
    # What to actually feed the LLM: the parent chunk's text if this hit is a "child"
    # (parent_child strategy), otherwise identical to `text`. Keeping both means a
    # citation can still point at the precise matched span while the LLM gets full
    # surrounding context -- see search.py's resolve_context().
    context_text: str

    collection_name: str
    strategy: str
    kind: str
    page_no: Optional[int] = None
    # Normalized (0..1), top-left-origin (left, top, right, bottom) -- only set for
    # HybridChunker-based strategies (hybrid_docling, parent_child, contextual), never
    # for recursive_overlap. Powers the /documents/{file_hash}/pages/{page_no}
    # grounding endpoint -- None means this hit can't be visually highlighted.
    bbox: Optional[tuple[float, float, float, float]] = None
    headings: list[str] = Field(default_factory=list)
    doc_stem: str
    file_hash: str
    images: list[dict] = Field(default_factory=list)
