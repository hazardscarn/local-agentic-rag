"""Shared exception types used across edenview_ingestion's submodules -- kept in its
own leaf module (no imports of its own) specifically so both pipeline.py and
vectorstore/points.py can raise/catch the same IngestionCancelledError without a
circular import (pipeline.py imports the vectorstore package; vectorstore/points.py
needs this same exception type during upsert_chunks()'s per-batch cancellation check)."""

from __future__ import annotations


class IngestionCancelledError(Exception):
    """Raised at a cancellation checkpoint once a job's cancel event has been set via
    pipeline.request_cancel() -- caught in pipeline.ingest_document()'s own except
    blocks and turned into a "cancelled" job status, not "error"."""
