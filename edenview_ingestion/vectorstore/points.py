"""Chunk -> Qdrant PointStruct conversion and the write path. "parent" kind Chunks are
skipped here entirely -- they never get embedded or land in Qdrant (see
chunking/parent_child.py and catalog/crud.py's parent_chunks table for where they do go)."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from edenview_ingestion.chunking import Chunk
from edenview_ingestion.errors import IngestionCancelledError

from .client import client_lock
from .embedding import embed_texts


def chunk_to_point(chunk: Chunk, vectors: dict, collection_id: str, db_id: str) -> PointStruct:
    return PointStruct(
        id=chunk.chunk_id,
        vector={
            "dense": vectors["dense"],
            "sparse": SparseVector(indices=vectors["sparse"]["indices"], values=vectors["sparse"]["values"]),
        },
        payload={
            "text": chunk.text,
            "strategy": chunk.strategy,
            "kind": chunk.kind,
            "doc_stem": chunk.doc_stem,
            "file_hash": chunk.file_hash,
            "page_no": chunk.page_no,
            "bbox": chunk.bbox,
            "headings": chunk.headings,
            "doc_item_refs": chunk.doc_item_refs,
            "parent_id": chunk.parent_id,
            "context": chunk.context,
            "images": [image.model_dump() for image in chunk.images],
            "collection_id": collection_id,
            "db_id": db_id,
        },
    )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    collection_id: str,
    db_id: str,
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    """Embeds and upserts every non-"parent" chunk. Returns how many points were
    written (not len(chunks) -- "parent" chunks are excluded, see module docstring).

    `progress_callback(done, total)`, if given, is called once per batch with real
    counts (not estimates) -- this is the one phase of ingestion where genuine progress
    is knowable upfront, see catalog/schema.py's ingestion_jobs comment.

    `cancel_event`, if given, is checked before each batch -- raises
    IngestionCancelledError instead of embedding/upserting further, so a cancelled job
    stops within one batch (batch_size=32 chunks) rather than running to completion."""
    embeddable = [c for c in chunks if c.kind != "parent"]
    total = len(embeddable)
    if not embeddable:
        return 0

    done = 0
    for i in range(0, total, batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise IngestionCancelledError()
        batch = embeddable[i : i + batch_size]
        vectors = embed_texts([c.embed_text for c in batch])
        points = [
            chunk_to_point(chunk, vec, collection_id, db_id) for chunk, vec in zip(batch, vectors)
        ]
        with client_lock:
            client.upsert(collection_name=collection_name, points=points)
        done += len(batch)
        if progress_callback is not None:
            progress_callback(done, total)

    return len(embeddable)
