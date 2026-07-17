"""DuckDB connection management. One process-wide connection, opened lazily and reused
-- the catalog is low-write-volume metadata (see edenview_progress.md's
DuckDB-vs-SQLite reasoning), not a high-concurrency store, so a single underlying
connection is the right default rather than a pool.

That said, `get_connection()` is called concurrently: catalog reads/writes happen
synchronously in the request-handling thread (FastAPI's `run_in_threadpool`), so N
concurrent requests (e.g. multi-file ingestion, one POST /ingest per file) hit this
module from N different threads at once. A single `DuckDBPyConnection` is documented
by DuckDB as not safe for concurrent queries from multiple threads -- confirmed by
reproduction, not just the docs: firing 4 concurrent /ingest calls at a brand-new
collection threw a "Catalog write-write conflict" during schema init on one request
and corrupted a fetched row into an empty tuple on another. DuckDB's own fix is
exactly what's below: each thread gets its own `cursor()` off one shared base
connection -- same underlying database file/WAL, independent per-thread query/
transaction state. Cursors off the same base connection still can't execute queries
at the literal same instant (they serialize), which is fine here -- catalog metadata
operations are tiny; the slow extract/chunk/embed work never touches this connection.
"""

from __future__ import annotations

import os
import threading
import time

import duckdb

from edenview_ingestion.settings import get_catalog_path

from .schema import initialize_schema

_connection: duckdb.DuckDBPyConnection | None = None
_connection_lock = threading.Lock()
_thread_local = threading.local()

# A dev-server restart (Ctrl+C, then immediately `npm run dev` again) can start the new
# backend process before the old one has actually released its file lock on
# catalog.duckdb -- confirmed by reproduction (Windows specifically: the old uvicorn
# process's shutdown isn't instant), not just theoretical. duckdb.connect() raises
# IOException in that window; the very next attempt a moment later succeeds once the
# old process finishes exiting. Short bounded retry absorbs that instead of surfacing
# a 500 on whichever request happens to be first after a fast restart.
_CONNECT_RETRY_ATTEMPTS = 5
_CONNECT_RETRY_DELAY_SECONDS = 0.3


def get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        with _connection_lock:
            if _connection is None:  # re-check: another thread may have won the race
                path = get_catalog_path()
                os.makedirs(path.parent, exist_ok=True)
                for attempt in range(_CONNECT_RETRY_ATTEMPTS):
                    try:
                        _connection = duckdb.connect(str(path))
                        break
                    except duckdb.IOException:
                        if attempt == _CONNECT_RETRY_ATTEMPTS - 1:
                            raise
                        time.sleep(_CONNECT_RETRY_DELAY_SECONDS)
                initialize_schema(_connection)

    if not hasattr(_thread_local, "cursor"):
        _thread_local.cursor = _connection.cursor()
    return _thread_local.cursor


def reset_connection_for_tests() -> None:
    """Closes and drops the cached connection so the next get_connection() call opens a
    fresh one -- only needed by verification scripts that want an isolated catalog file.
    Only resets the calling thread's cursor state; fine since this is a single-threaded
    test helper, not used from concurrent request handling."""
    global _connection
    with _connection_lock:
        if _connection is not None:
            _connection.close()
            _connection = None
    if hasattr(_thread_local, "cursor"):
        del _thread_local.cursor
