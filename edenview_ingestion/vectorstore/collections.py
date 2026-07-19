"""Qdrant collection lifecycle -- create, delete, check existence. Every collection
uses the same two-vector schema (named "dense" + "sparse"), matching the proven
ingest/shared.py design, so every chunking strategy's output writes through one shared
point-building path (see points.py)."""

from __future__ import annotations

import shutil
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from edenview_ingestion.settings import get_dense_embedding_dim, get_qdrant_path

from .client import client_lock


def collection_exists(client: QdrantClient, name: str) -> bool:
    with client_lock:
        existing = {c.name for c in client.get_collections().collections}
        return name in existing


def create_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """No-op if the collection already exists -- ingestion is meant to be safely
    re-run (see chunking's deterministic chunk_id / this module's point-ID reuse).

    The exists-check and the create call are held under one client_lock acquisition
    (client_lock is reentrant) so two threads racing to create the same brand-new
    collection -- e.g. two files from a multi-file ingestion targeting the same
    collection name -- can't both pass the check before either has created it."""
    with client_lock:
        if collection_exists(client, name):
            return
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        )


def delete_collection(client: QdrantClient, name: str) -> None:
    """No-op if the collection doesn't exist -- deleting something already gone isn't
    an error for this store's purposes.

    Qdrant's local-mode delete_collection() deregisters the collection (meta.json,
    collection_exists() -> False, name immediately reusable) but doesn't reclaim its
    on-disk folder synchronously -- confirmed by inspection, not documented behavior to
    rely on blindly. The underlying storage file releases its lock within a beat of
    delete_collection() returning (confirmed: a fresh process could always remove it
    immediately; same-process removal right after delete_collection() sometimes lost
    that race on Windows), hence the short bounded retry rather than a single attempt."""
    with client_lock:
        if collection_exists(client, name):
            client.delete_collection(collection_name=name)

    collection_dir = get_qdrant_path() / "collection" / name
    for attempt in range(5):
        shutil.rmtree(collection_dir, ignore_errors=True)
        if not collection_dir.exists():
            return
        time.sleep(0.1 * (attempt + 1))


def default_dense_dim() -> int:
    return get_dense_embedding_dim()
