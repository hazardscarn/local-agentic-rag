"""DuckDB table DDL for the Edenview catalog -- see edenview_progress.md
section 2 for the design rationale. This catalog never stores vectors or chunk text,
only what's needed to list/filter what exists in Qdrant:

  dbs                 -- user-facing grouping label (catalog-only; Qdrant has no
                         concept of this, see edenview_progress.md's
                         "DB vs Collection" decision)
  collections         -- one row per actual Qdrant collection
  documents           -- one row per unique source file (keyed by content hash, so
                         re-uploading the same file doesn't duplicate)
  collection_documents -- join table: a doc can be in multiple collections (different
                         chunking strategies), a collection can hold multiple docs
  ingestion_jobs      -- async job status, for the POST /rag/ingest + GET /rag/jobs/{id}
                         polling pattern in edenview_plan.md
  parent_chunks       -- "parent" kind Chunks from the parent_child strategy (see
                         chunking/parent_child.py). These never get embedded/written to
                         Qdrant -- they're a lookup table swapped in at query time for a
                         matched "child" chunk, same role the old ingest/s2_parent_child.py
                         played with a JSON docstore, kept in DuckDB instead so there's
                         no third storage mechanism to manage
  chat_sessions,
  chat_messages       -- persisted /chat conversations (see catalog/chat_crud.py) --
                         one process-wide catalog, so chat history survives a page
                         reload/restart the same way ingestion metadata already does

Every primary key is a Python-generated UUID (see catalog/crud.py), not a DB-side
autoincrement -- consistent with how chunk_id/point IDs are already generated
elsewhere in this codebase (chunking/models.py's make_chunk_id).
"""

from __future__ import annotations

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS dbs (
        db_id TEXT PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS collections (
        collection_id TEXT PRIMARY KEY,
        db_id TEXT NOT NULL REFERENCES dbs(db_id),
        qdrant_collection_name TEXT UNIQUE NOT NULL,
        chunking_strategy TEXT NOT NULL,
        embedding_model TEXT NOT NULL,
        dense_dim INTEGER NOT NULL,
        sparse_model TEXT,
        status TEXT NOT NULL DEFAULT 'ready',
        chunk_count INTEGER NOT NULL DEFAULT 0,
        doc_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        file_hash TEXT UNIQUE NOT NULL,
        filename TEXT NOT NULL,
        source_path TEXT,
        input_format TEXT,
        num_pages INTEGER,
        first_ingested_at TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS collection_documents (
        collection_id TEXT NOT NULL REFERENCES collections(collection_id),
        doc_id TEXT NOT NULL REFERENCES documents(doc_id),
        ingested_at TIMESTAMP NOT NULL,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ready',
        PRIMARY KEY (collection_id, doc_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        job_id TEXT PRIMARY KEY,
        collection_id TEXT NOT NULL REFERENCES collections(collection_id),
        doc_id TEXT REFERENCES documents(doc_id),
        status TEXT NOT NULL DEFAULT 'queued',
        -- The uploaded filename, stored at job-creation time (before extraction even
        -- starts) -- doc_id (and the documents.filename it points at) isn't known
        -- until extraction finishes and the file_hash can be computed, so this is the
        -- only way a job list can show which file a still-running job belongs to.
        filename TEXT,
        -- stage/stage_current/stage_total: which phase of extract->chunk->embed the job
        -- is in, and real (not estimated) progress counts where they exist -- only
        -- populated for "embedding" (batched, so the count is exact); "extracting" has
        -- no sub-progress since Docling exposes no per-page callback and page count
        -- isn't known until parsing finishes. See pipeline.py's ingest_document().
        stage TEXT,
        stage_current INTEGER,
        stage_total INTEGER,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS parent_chunks (
        collection_id TEXT NOT NULL REFERENCES collections(collection_id),
        chunk_id TEXT NOT NULL,
        text TEXT NOT NULL,
        page_no INTEGER,
        headings TEXT,
        PRIMARY KEY (collection_id, chunk_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        -- JSON-encoded list[RetrievalHit] -- NULL for user messages, which have no
        -- citations of their own. Kept as a JSON blob rather than a normalized table
        -- since citations are never queried/filtered on their own, only displayed
        -- alongside the message that produced them.
        citations TEXT,
        model_used TEXT,
        created_at TIMESTAMP NOT NULL
    )
    """,
]


# Columns added after ingestion_jobs already shipped -- CREATE TABLE IF NOT EXISTS is a
# no-op against an existing table, so these run every startup too (each individually
# idempotent via IF NOT EXISTS) to upgrade a catalog.duckdb created before this change,
# like the one already running against this session's server.
MIGRATION_STATEMENTS = [
    "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS stage TEXT",
    "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS stage_current INTEGER",
    "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS stage_total INTEGER",
    "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS filename TEXT",
]


def initialize_schema(connection) -> None:
    for statement in DDL_STATEMENTS:
        connection.execute(statement)
    for statement in MIGRATION_STATEMENTS:
        connection.execute(statement)
