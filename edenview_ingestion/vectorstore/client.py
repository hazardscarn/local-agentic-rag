"""Embedded Qdrant client -- QdrantClient(path=...), no server/Docker (see
edenview_progress.md for why: a single long-lived Edenview process holds this
open, so the "one process at a time" restriction of embedded mode is a non-issue).

One process-wide client, opened lazily and reused -- opening a second one against the
same path from a different process is what triggers Qdrant's RuntimeError, so callers
should go through get_client() rather than constructing their own QdrantClient(path=...).

Local mode's on-disk persistence (qdrant_client/local/persistence.py) is backed by
sqlite3, which -- unlike DuckDB -- has no equivalent of "one cursor per thread" for a
single Python object to hand out safely. Concurrent, unsynchronized calls into the one
shared QdrantClient from multiple threads (e.g. several files of a multi-file ingestion,
each running its background extract/chunk/embed/write in its own thread) intermittently
raised sqlite's own "bad parameter or other API misuse" -- confirmed by reproducing it
with concurrent ingests, then eliminated by serializing every real operation on this
client through client_lock, not just its lazy construction below. Every call site that
touches the client returned by get_client() (collections.py, points.py, search.py,
the preview-scroll route) must wrap the actual client.<method>(...) call in
`with client_lock:`."""

from __future__ import annotations

import os
import threading

from qdrant_client import QdrantClient

from edenview_ingestion.settings import get_qdrant_path

_client: QdrantClient | None = None
_client_lock = threading.RLock()

# Exposed to every module that calls a method on the client returned by get_client() --
# reentrant so a function that already holds it (e.g. create_collection()) can safely
# call another locked helper (collection_exists()) without deadlocking itself.
client_lock = _client_lock


def get_client() -> QdrantClient:
    """Lock-guarded lazy init -- concurrent first callers (e.g. multiple files from a
    multi-file ingestion, each running its background extract/chunk/embed/write in its
    own thread) racing to construct the client would otherwise risk two
    QdrantClient(path=...) calls against the same embedded path at once; see
    catalog/connection.py's get_connection() for the identical pattern, confirmed
    necessary there by direct reproduction."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # re-check: another thread may have won the race
                path = get_qdrant_path()
                os.makedirs(path.parent, exist_ok=True)
                _client = QdrantClient(path=str(path))
    return _client


def reset_client_for_tests() -> None:
    """Closes and drops the cached client so the next get_client() call opens a fresh
    one -- only needed by verification scripts that want an isolated store, or that need
    to release the path lock before a second script run."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
