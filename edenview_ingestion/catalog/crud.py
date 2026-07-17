"""Read/write operations on the catalog tables. Every write function takes an explicit
`connection` argument (default `get_connection()`) so verification scripts can pass an
isolated in-memory connection instead of touching the real catalog file."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

import duckdb

from .connection import get_connection
from .errors import CatalogError, DuplicateNameError, NotFoundError
from .models import CollectionRecord, DBRecord, DocumentRecord, IngestionJobRecord


def _new_id() -> str:
    return str(uuid.uuid4())


# Which DuckDB exception a uniqueness conflict surfaces as depends on how it's
# discovered, not just what kind of conflict it is: a single sequential connection
# raises it immediately as ConstraintException, but under concurrent per-thread
# cursors (see connection.py) the same conflict can instead surface at commit time as
# a TransactionException wrapping "constraint violation"/"duplicate key" in its
# message -- confirmed by direct reproduction (concurrent multi-file ingestion into a
# brand-new db/collection), not assumed. Catching both exception types AND checking
# the message keeps this from also swallowing a genuinely different transaction
# failure (e.g. real corruption) under the same except clause.
_DUPLICATE_KEY_EXCEPTIONS = (duckdb.ConstraintException, duckdb.TransactionException)


def _is_duplicate_key_error(e: Exception) -> bool:
    message = str(e).lower()
    return "constraint violation" in message or "duplicate key" in message


# --- dbs ---------------------------------------------------------------------------


def create_db(name: str, connection: duckdb.DuckDBPyConnection = None) -> DBRecord:
    connection = connection or get_connection()
    db_id = _new_id()
    created_at = datetime.now()
    try:
        connection.execute(
            "INSERT INTO dbs (db_id, name, created_at) VALUES (?, ?, ?)", [db_id, name, created_at]
        )
    except _DUPLICATE_KEY_EXCEPTIONS as e:
        if not _is_duplicate_key_error(e):
            raise
        raise DuplicateNameError(f"A DB named {name!r} already exists") from None
    return DBRecord(db_id=db_id, name=name, created_at=created_at)


def get_db(name: str, connection: duckdb.DuckDBPyConnection = None) -> DBRecord:
    connection = connection or get_connection()
    row = connection.execute("SELECT db_id, name, created_at FROM dbs WHERE name = ?", [name]).fetchone()
    if row is None:
        raise NotFoundError(f"No DB named {name!r}")
    return DBRecord(db_id=row[0], name=row[1], created_at=row[2])


def get_or_create_db(name: str, connection: duckdb.DuckDBPyConnection = None) -> DBRecord:
    """Check-then-create, with a fallback for the race it can't fully close: two
    concurrent callers can both see NotFoundError for a brand-new name (e.g.
    multi-file ingestion into a database that doesn't exist yet, one request per
    file, fired concurrently) and both attempt create_db() -- the loser hits
    DuplicateNameError against the UNIQUE constraint. Since that only means "someone
    else just created the exact row we wanted", fetch and return it instead of
    propagating the error."""
    connection = connection or get_connection()
    try:
        return get_db(name, connection)
    except NotFoundError:
        try:
            return create_db(name, connection)
        except DuplicateNameError:
            return get_db(name, connection)


def list_dbs(connection: duckdb.DuckDBPyConnection = None) -> list[DBRecord]:
    connection = connection or get_connection()
    rows = connection.execute("SELECT db_id, name, created_at FROM dbs ORDER BY created_at").fetchall()
    return [DBRecord(db_id=r[0], name=r[1], created_at=r[2]) for r in rows]


def delete_db(db_id: str, connection: duckdb.DuckDBPyConnection = None) -> None:
    """Refuses if any collection still references this DB -- delete those collections
    first (via vectorstore's delete_collection, which cleans up the Qdrant side too;
    this function only ever touches catalog rows, never Qdrant)."""
    connection = connection or get_connection()
    remaining = connection.execute("SELECT COUNT(*) FROM collections WHERE db_id = ?", [db_id]).fetchone()[0]
    if remaining:
        raise CatalogError(
            f"DB {db_id!r} still has {remaining} collection(s) -- delete those first"
        )
    connection.execute("DELETE FROM dbs WHERE db_id = ?", [db_id])


# --- collections -------------------------------------------------------------------

_COLLECTION_COLUMNS = (
    "collection_id, db_id, qdrant_collection_name, chunking_strategy, embedding_model, "
    "dense_dim, sparse_model, status, chunk_count, doc_count, created_at"
)


def _collection_from_row(row) -> CollectionRecord:
    return CollectionRecord(
        collection_id=row[0],
        db_id=row[1],
        qdrant_collection_name=row[2],
        chunking_strategy=row[3],
        embedding_model=row[4],
        dense_dim=row[5],
        sparse_model=row[6],
        status=row[7],
        chunk_count=row[8],
        doc_count=row[9],
        created_at=row[10],
    )


def create_collection(
    db_name: str,
    qdrant_collection_name: str,
    chunking_strategy: str,
    embedding_model: str,
    dense_dim: int,
    sparse_model: Optional[str] = None,
    status: str = "ready",
    connection: duckdb.DuckDBPyConnection = None,
) -> CollectionRecord:
    connection = connection or get_connection()
    db = get_or_create_db(db_name, connection)
    collection_id = _new_id()
    created_at = datetime.now()
    try:
        connection.execute(
            f"INSERT INTO collections ({_COLLECTION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                collection_id,
                db.db_id,
                qdrant_collection_name,
                chunking_strategy,
                embedding_model,
                dense_dim,
                sparse_model,
                status,
                0,
                0,
                created_at,
            ],
        )
    except _DUPLICATE_KEY_EXCEPTIONS as e:
        if not _is_duplicate_key_error(e):
            raise
        raise DuplicateNameError(
            f"A collection named {qdrant_collection_name!r} already exists "
            "(Qdrant collection names must be globally unique across all DBs)"
        ) from None
    return CollectionRecord(
        collection_id=collection_id,
        db_id=db.db_id,
        qdrant_collection_name=qdrant_collection_name,
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        dense_dim=dense_dim,
        sparse_model=sparse_model,
        status=status,
        chunk_count=0,
        doc_count=0,
        created_at=created_at,
    )


def update_collection_status(
    collection_id: str, status: str, connection: duckdb.DuckDBPyConnection = None
) -> None:
    """Flips a collection's status after ingestion finishes (or fails) -- see
    pipeline.py's ingest_document(). create_collection() sets the initial status
    ("ingesting" for a brand-new collection, via pipeline._get_or_create_collection());
    this is the only other writer of this column."""
    connection = connection or get_connection()
    connection.execute("UPDATE collections SET status = ? WHERE collection_id = ?", [status, collection_id])


def get_collection(qdrant_collection_name: str, connection: duckdb.DuckDBPyConnection = None) -> CollectionRecord:
    connection = connection or get_connection()
    row = connection.execute(
        f"SELECT {_COLLECTION_COLUMNS} FROM collections WHERE qdrant_collection_name = ?",
        [qdrant_collection_name],
    ).fetchone()
    if row is None:
        raise NotFoundError(f"No collection named {qdrant_collection_name!r}")
    return _collection_from_row(row)


def list_collections(db_name: Optional[str] = None, connection: duckdb.DuckDBPyConnection = None) -> list[CollectionRecord]:
    connection = connection or get_connection()
    if db_name is None:
        rows = connection.execute(f"SELECT {_COLLECTION_COLUMNS} FROM collections ORDER BY created_at").fetchall()
    else:
        db = get_db(db_name, connection)
        rows = connection.execute(
            f"SELECT {_COLLECTION_COLUMNS} FROM collections WHERE db_id = ? ORDER BY created_at", [db.db_id]
        ).fetchall()
    return [_collection_from_row(r) for r in rows]


def recompute_collection_counts(collection_id: str, connection: duckdb.DuckDBPyConnection = None) -> None:
    """Sets chunk_count/doc_count from the actual collection_documents rows, rather
    than incrementing -- ingestion is idempotent (re-ingesting the same doc upserts its
    collection_documents row, it doesn't add a second one), so counts have to be
    recomputed from that source of truth each time, not accumulated. Call this *after*
    link_document_to_collection()."""
    connection = connection or get_connection()
    connection.execute(
        "UPDATE collections SET "
        "chunk_count = (SELECT COALESCE(SUM(chunk_count), 0) FROM collection_documents WHERE collection_id = ?), "
        "doc_count = (SELECT COUNT(*) FROM collection_documents WHERE collection_id = ?) "
        "WHERE collection_id = ?",
        [collection_id, collection_id, collection_id],
    )


def delete_collection_catalog_rows(collection_id: str, connection: duckdb.DuckDBPyConnection = None) -> None:
    """Removes this collection's catalog rows only -- never touches Qdrant. Callers that
    also need the underlying Qdrant collection gone should call
    vectorstore.collections.delete_collection() themselves (see pipeline.py's
    delete_collection() for the combined operation)."""
    connection = connection or get_connection()
    connection.execute("DELETE FROM parent_chunks WHERE collection_id = ?", [collection_id])
    connection.execute("DELETE FROM ingestion_jobs WHERE collection_id = ?", [collection_id])
    connection.execute("DELETE FROM collection_documents WHERE collection_id = ?", [collection_id])
    connection.execute("DELETE FROM collections WHERE collection_id = ?", [collection_id])


# --- documents -----------------------------------------------------------------------


def _document_from_row(row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row[0], file_hash=row[1], filename=row[2], source_path=row[3],
        input_format=row[4], num_pages=row[5], first_ingested_at=row[6],
    )


def get_document(doc_id: str, connection: duckdb.DuckDBPyConnection = None) -> DocumentRecord:
    connection = connection or get_connection()
    row = connection.execute(
        "SELECT doc_id, file_hash, filename, source_path, input_format, num_pages, first_ingested_at "
        "FROM documents WHERE doc_id = ?",
        [doc_id],
    ).fetchone()
    if row is None:
        raise NotFoundError(f"Document {doc_id!r} not found")
    return _document_from_row(row)


def register_document(
    file_hash: str,
    filename: str,
    source_path: Optional[str],
    input_format: Optional[str],
    num_pages: Optional[int],
    connection: duckdb.DuckDBPyConnection = None,
) -> DocumentRecord:
    """Idempotent -- the same file_hash is returned as-is on a second call rather than
    erroring, since the same source file legitimately gets ingested into multiple
    collections over time. Also covers the concurrent case: multi-file ingestion can
    fire two requests for byte-identical content (same file_hash) at the same time --
    the loser's INSERT hits the file_hash UNIQUE constraint, so that's treated the
    same as "already registered" rather than an error."""
    connection = connection or get_connection()
    row = connection.execute(
        "SELECT doc_id, file_hash, filename, source_path, input_format, num_pages, first_ingested_at "
        "FROM documents WHERE file_hash = ?",
        [file_hash],
    ).fetchone()
    if row is not None:
        return _document_from_row(row)

    doc_id = _new_id()
    first_ingested_at = datetime.now()
    try:
        connection.execute(
            "INSERT INTO documents (doc_id, file_hash, filename, source_path, input_format, num_pages, first_ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [doc_id, file_hash, filename, source_path, input_format, num_pages, first_ingested_at],
        )
    except _DUPLICATE_KEY_EXCEPTIONS as e:
        if not _is_duplicate_key_error(e):
            raise
        row = connection.execute(
            "SELECT doc_id, file_hash, filename, source_path, input_format, num_pages, first_ingested_at "
            "FROM documents WHERE file_hash = ?",
            [file_hash],
        ).fetchone()
        return _document_from_row(row)
    return DocumentRecord(
        doc_id=doc_id, file_hash=file_hash, filename=filename, source_path=source_path,
        input_format=input_format, num_pages=num_pages, first_ingested_at=first_ingested_at,
    )


def link_document_to_collection(
    collection_id: str, doc_id: str, chunk_count: int, connection: duckdb.DuckDBPyConnection = None
) -> None:
    """Upsert -- re-ingesting the same doc into the same collection updates chunk_count
    rather than erroring or duplicating."""
    connection = connection or get_connection()
    existing = connection.execute(
        "SELECT 1 FROM collection_documents WHERE collection_id = ? AND doc_id = ?", [collection_id, doc_id]
    ).fetchone()
    if existing:
        connection.execute(
            "UPDATE collection_documents SET chunk_count = ?, ingested_at = ?, status = 'ready' "
            "WHERE collection_id = ? AND doc_id = ?",
            [chunk_count, datetime.now(), collection_id, doc_id],
        )
    else:
        connection.execute(
            "INSERT INTO collection_documents (collection_id, doc_id, ingested_at, chunk_count, status) "
            "VALUES (?, ?, ?, ?, 'ready')",
            [collection_id, doc_id, datetime.now(), chunk_count],
        )


def list_documents_in_collection(collection_id: str, connection: duckdb.DuckDBPyConnection = None) -> list[DocumentRecord]:
    connection = connection or get_connection()
    rows = connection.execute(
        "SELECT d.doc_id, d.file_hash, d.filename, d.source_path, d.input_format, d.num_pages, d.first_ingested_at "
        "FROM documents d JOIN collection_documents cd ON cd.doc_id = d.doc_id "
        "WHERE cd.collection_id = ? ORDER BY cd.ingested_at",
        [collection_id],
    ).fetchall()
    return [_document_from_row(r) for r in rows]


# --- ingestion_jobs ------------------------------------------------------------------


_JOB_SELECT = """
    SELECT j.job_id, j.collection_id, j.doc_id, j.status, j.filename, j.stage,
           j.stage_current, j.stage_total, j.started_at, j.finished_at, j.error_message,
           c.qdrant_collection_name, d.name
    FROM ingestion_jobs j
    LEFT JOIN collections c ON j.collection_id = c.collection_id
    LEFT JOIN dbs d ON c.db_id = d.db_id
"""


def _job_from_row(row) -> IngestionJobRecord:
    return IngestionJobRecord(
        job_id=row[0], collection_id=row[1], doc_id=row[2], status=row[3], filename=row[4],
        stage=row[5], stage_current=row[6], stage_total=row[7],
        started_at=row[8], finished_at=row[9], error_message=row[10],
        qdrant_collection_name=row[11], db_name=row[12],
    )


def create_job(
    collection_id: str,
    doc_id: Optional[str] = None,
    status: str = "running",
    filename: Optional[str] = None,
    connection: duckdb.DuckDBPyConnection = None,
) -> IngestionJobRecord:
    """`status="queued"` is for the FastAPI layer's two-phase ingest: a fast synchronous
    call creates the job (and, if needed, the collection) so a job_id can be returned to
    the client immediately, then a background task does the slow extract/chunk/embed
    work and calls start_job()/complete_job() on that same job_id -- see
    pipeline.prepare_ingest() / pipeline.ingest_document()'s job_id parameter.

    `filename` is stored directly rather than waiting on doc_id (which only resolves
    once extraction finishes and the file_hash is known) -- see schema.py's comment;
    this is what lets GET /jobs show which file a still-running job belongs to."""
    connection = connection or get_connection()
    job_id = _new_id()
    started_at = datetime.now()
    connection.execute(
        "INSERT INTO ingestion_jobs (job_id, collection_id, doc_id, status, filename, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [job_id, collection_id, doc_id, status, filename, started_at],
    )
    return get_job(job_id, connection)


def start_job(job_id: str, connection: duckdb.DuckDBPyConnection = None) -> None:
    """Transitions a "queued" job to "running" -- called when a background task
    actually picks up the work, as opposed to when the job record was first created."""
    connection = connection or get_connection()
    connection.execute("UPDATE ingestion_jobs SET status = 'running' WHERE job_id = ?", [job_id])


def get_job(job_id: str, connection: duckdb.DuckDBPyConnection = None) -> IngestionJobRecord:
    connection = connection or get_connection()
    row = connection.execute(f"{_JOB_SELECT} WHERE j.job_id = ?", [job_id]).fetchone()
    if row is None:
        raise NotFoundError(f"No ingestion job {job_id!r}")
    return _job_from_row(row)


def list_jobs(
    limit: int = 50,
    filename: Optional[str] = None,
    statuses: Optional[list[str]] = None,
    connection: duckdb.DuckDBPyConnection = None,
) -> list[IngestionJobRecord]:
    """Most-recently-created first -- backs GET /jobs, the Ingestion page's status
    tracker. Server-side by design: browser localStorage only shows jobs kicked off
    from that one browser, and disappears on a cache clear or a different device.

    `filename`, if given, is a case-insensitive substring match (SQL ILIKE) letting a
    user find an older job the default `limit` would otherwise cut off, without needing
    real pagination -- the job list is meant to stay a lightweight recent-activity feed,
    not a full audit log with an offset/page UI.

    `statuses`, if given, restricts to jobs whose status is one of these exact values
    (e.g. ["queued", "running"] for an "active" filter) -- composes with `filename`."""
    connection = connection or get_connection()
    where_clauses = []
    params: list = []
    if filename:
        where_clauses.append("j.filename ILIKE ?")
        params.append(f"%{filename}%")
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        where_clauses.append(f"j.status IN ({placeholders})")
        params.extend(statuses)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)
    rows = connection.execute(f"{_JOB_SELECT} {where_sql} ORDER BY j.started_at DESC LIMIT ?", params).fetchall()
    return [_job_from_row(r) for r in rows]


def update_job_stage(
    job_id: str,
    stage: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    connection: duckdb.DuckDBPyConnection = None,
) -> None:
    """`current`/`total` are real counts, not estimates -- leave them None for stages
    where there's nothing genuine to report (see catalog/schema.py's ingestion_jobs
    comment). Called repeatedly during "embedding" (once per batch) -- see
    vectorstore/points.py's upsert_chunks() progress_callback."""
    connection = connection or get_connection()
    connection.execute(
        "UPDATE ingestion_jobs SET stage = ?, stage_current = ?, stage_total = ? WHERE job_id = ?",
        [stage, current, total, job_id],
    )


def complete_job(
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
    doc_id: Optional[str] = None,
    connection: duckdb.DuckDBPyConnection = None,
) -> None:
    """`doc_id`, if given, backfills the job's doc_id -- needed for the two-phase flow
    above, where the job is created before the document (and its doc_id) is known."""
    connection = connection or get_connection()
    if doc_id is not None:
        connection.execute(
            "UPDATE ingestion_jobs SET status = ?, finished_at = ?, error_message = ?, doc_id = ? WHERE job_id = ?",
            [status, datetime.now(), error_message, doc_id, job_id],
        )
    else:
        connection.execute(
            "UPDATE ingestion_jobs SET status = ?, finished_at = ?, error_message = ? WHERE job_id = ?",
            [status, datetime.now(), error_message, job_id],
        )


# --- parent_chunks -------------------------------------------------------------------
# "parent" kind Chunks from the parent_child strategy -- never embedded, looked up here
# at query time when a matched "child" chunk's parent_id needs swapping in.


def save_parent_chunk(
    collection_id: str,
    chunk_id: str,
    text: str,
    page_no: Optional[int],
    headings: list[str],
    connection: duckdb.DuckDBPyConnection = None,
) -> None:
    connection = connection or get_connection()
    connection.execute(
        "INSERT OR REPLACE INTO parent_chunks (collection_id, chunk_id, text, page_no, headings) "
        "VALUES (?, ?, ?, ?, ?)",
        [collection_id, chunk_id, text, page_no, json.dumps(headings)],
    )


def get_parent_chunk_text(
    collection_id: str, chunk_id: str, connection: duckdb.DuckDBPyConnection = None
) -> Optional[str]:
    connection = connection or get_connection()
    row = connection.execute(
        "SELECT text FROM parent_chunks WHERE collection_id = ? AND chunk_id = ?", [collection_id, chunk_id]
    ).fetchone()
    return row[0] if row else None
