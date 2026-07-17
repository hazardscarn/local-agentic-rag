"""Ingestion pipeline orchestrator: extract -> chunk -> embed -> write to Qdrant +
catalog. One call = one document ingested into one collection under one strategy.

Safe to re-run (idempotent): chunk_id is deterministic (chunking/models.py's
make_chunk_id), document registration is idempotent by file_hash, and Qdrant
upserts overwrite existing points rather than duplicating.

Picture/table crop images are written to a *permanent* location
(settings.get_documents_dir(), keyed by doc_stem) rather than docling_parsing's default
temp workspace, and the extractor's temp-workspace cleanup is deliberately never called
here -- a chunk's payload in Qdrant references these image files by path indefinitely,
so they have to actually still be there when that chunk is retrieved later.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from edenview_ingestion import catalog, vectorstore
from edenview_ingestion.chunking import CHUNKERS, generate_image_description_chunks
from edenview_ingestion.errors import IngestionCancelledError
from docling.datamodel.pipeline_options import OcrAutoOptions

from edenview_ingestion.docling_parsing import (
    DoclingExtractor,
    ExtractionConfig,
    StorageConfig,
    generate_picture_descriptions,
)
from edenview_ingestion.settings import (
    get_dense_embedding_dim,
    get_documents_dir,
    get_max_concurrent_extractions,
    get_model,
)

# Caps how many documents run Docling extraction (model loading + inference -- the
# actual resource-hungry step, as opposed to chunking/embedding) at the same time.
# Confirmed necessary by reproduction: firing many files at once with no limit ran
# every extraction simultaneously, oversubscribing CPU threads, RAM, and (if present)
# the one shared GPU badly enough that the whole batch was far slower than a few at a
# time would have been. Sized once at process start from
# settings.get_max_concurrent_extractions() -- changing that setting needs a backend
# restart to take effect, since a threading.Semaphore's capacity isn't resizable live.
_EXTRACTION_SEMAPHORE = threading.Semaphore(get_max_concurrent_extractions())


# In-memory registry, not persisted: a job's cancel button only works against the
# same backend process that's actually running it (true for this app's single-process
# deployment model -- see edenview_progress.md). Cleared in
# ingest_document()'s finally block regardless of how the job ends, so this never
# grows unbounded.
_cancel_events: dict[str, threading.Event] = {}
_cancel_events_lock = threading.Lock()


def _register_cancellable(job_id: str) -> threading.Event:
    event = threading.Event()
    with _cancel_events_lock:
        _cancel_events[job_id] = event
    return event


def _unregister_cancellable(job_id: str) -> None:
    with _cancel_events_lock:
        _cancel_events.pop(job_id, None)


def request_cancel(job_id: str) -> bool:
    """Signals a running job to stop at its next checkpoint. Returns True if a job
    with this id was actually found running in this process, False otherwise (already
    finished, or never existed) -- the caller (the API route) uses this to tell a
    real cancellation apart from a no-op.

    Cooperative, not preemptive: a job already inside Docling's own extraction call
    can't be interrupted mid-call (nothing in that call checks for cancellation), so
    "extracting"-stage jobs stop as soon as that call returns, not instantly. A job
    still waiting on _EXTRACTION_SEMAPHORE (queued behind other concurrent
    extractions) or in "chunking"/"embedding" stops within moments -- those stages
    check at fine-grained boundaries (each embedding batch, in particular)."""
    with _cancel_events_lock:
        event = _cancel_events.get(job_id)
    if event is None:
        return False
    event.set()
    return True


def _check_cancelled(event: Optional[threading.Event]) -> None:
    if event is not None and event.is_set():
        raise IngestionCancelledError()


class IngestResult(BaseModel):
    db_name: str
    qdrant_collection_name: str
    strategy: str
    doc_filename: str
    chunk_count: int
    embedded_count: int
    parent_count: int
    job_id: str
    job_status: str


def _default_extraction_config(force_full_page_ocr: bool = False) -> ExtractionConfig:
    """`force_full_page_ocr=True` is an explicit opt-in for a document the caller
    already knows is a scan (or otherwise unreliable for Docling's own per-page bitmap
    detection) -- Docling's default (force_full_page_ocr=False, its own bitmap-area
    heuristic deciding per page whether OCR is even needed) already handles ordinary
    born-digital and mixed documents well, so this stays off unless asked for.
    `OcrAutoOptions` (not a specific engine) so Docling still auto-picks the best OCR
    engine available on this machine (e.g. EasyOCR if a GPU is present, Tesseract
    otherwise) -- only the full-page-vs-bitmap-detection behavior is being overridden."""
    ocr_options = OcrAutoOptions(force_full_page_ocr=True) if force_full_page_ocr else None
    return ExtractionConfig.full(storage=StorageConfig(base_dir=str(get_documents_dir())), ocr_options=ocr_options)


def _preserve_original_pdf(source: Path, file_hash: str, input_format: Optional[str]) -> None:
    """Keeps a permanent copy of the uploaded PDF under get_documents_dir()/originals/,
    keyed by file_hash (skipped if a copy already exists -- the catalog already
    dedupes documents by file_hash, so a re-uploaded duplicate doesn't copy twice).

    Without this, the source file is deleted right after ingestion (see
    api/routers/ingest.py's _run_ingest_background()) and nothing Docling caches
    (doc.json, picture/table crops) can reconstruct a renderable PDF afterward --
    needed for the /documents/{file_hash}/pages/{page_no} visual-grounding endpoint,
    which renders a page with pypdfium2 on demand. PDF-only: that's the only format
    the renderer handles, and the only one grounding is offered for."""
    if input_format != "pdf":
        return
    originals_dir = get_documents_dir() / "originals"
    dest = originals_dir / f"{file_hash}{source.suffix}"
    if dest.exists():
        return
    originals_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _get_or_create_collection(db_name: str, qdrant_collection_name: str, strategy: str) -> catalog.CollectionRecord:
    """Check-then-create, with a fallback for the race it can't fully close: multi-
    file ingestion (the frontend's ingestion form) fires one POST /ingest per file
    concurrently, all targeting the same collection_name -- if that collection is
    brand-new, more than one request can see NotFoundError here before any of them
    finish creating it, and every loser hits DuplicateNameError against the
    collections table's UNIQUE constraint. That only means "a sibling request in
    this same batch just created the exact row we wanted", so fetch and return it
    instead of failing that file's ingest outright."""
    try:
        return catalog.crud.get_collection(qdrant_collection_name)
    except catalog.NotFoundError:
        try:
            return catalog.crud.create_collection(
                db_name=db_name,
                qdrant_collection_name=qdrant_collection_name,
                chunking_strategy=strategy,
                embedding_model=get_model("dense_embedding"),
                dense_dim=get_dense_embedding_dim(),
                sparse_model=get_model("sparse_embedding"),
                # Brand-new collection: nothing's actually ingested yet. Flipped to
                # "ready" (or "error") by ingest_document() below once the job finishes --
                # an already-existing collection (the get_collection() branch above)
                # keeps whatever status it already has, so adding a second document to an
                # already-"ready" collection doesn't regress it back to "ingesting".
                status="ingesting",
            )
        except catalog.DuplicateNameError:
            return catalog.crud.get_collection(qdrant_collection_name)


def prepare_ingest(
    db_name: str, qdrant_collection_name: str, strategy: str, filename: Optional[str] = None
) -> catalog.IngestionJobRecord:
    """Fast, synchronous half of a two-phase ingest: resolves/creates the collection and
    creates a "queued" job, so a job_id can be returned to a caller (e.g. a FastAPI
    handler) immediately. Pass the returned job's job_id into ingest_document() -- run
    that part as a background task -- to do the slow extract/chunk/embed work against
    this same job record.

    `filename`, if given, is stored on the job immediately (see catalog/schema.py's
    ingestion_jobs.filename comment for why this can't wait on doc_id)."""
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {list(CHUNKERS)}")
    collection = _get_or_create_collection(db_name, qdrant_collection_name, strategy)
    return catalog.crud.create_job(collection.collection_id, doc_id=None, status="queued", filename=filename)


def ingest_document(
    source: str | Path,
    db_name: str,
    qdrant_collection_name: str,
    strategy: str,
    extraction_config: Optional[ExtractionConfig] = None,
    chunk_config=None,
    include_image_descriptions: bool = False,
    force_full_page_ocr: bool = False,
    job_id: Optional[str] = None,
) -> IngestResult:
    """`include_image_descriptions=True` runs docling_parsing.generate_picture_descriptions()
    against the extracted bundle's pictures right after extraction, then folds the
    resulting standalone image-description chunks into the strategy's own chunks -- see
    that module for why picture description is a separate step here rather than a
    Docling pipeline option.

    `force_full_page_ocr=True` is an opt-in for a document already known to be a scan --
    see _default_extraction_config()'s docstring. Ignored if `extraction_config` is
    passed explicitly (that caller already made its own OCR decision).

    `job_id`, if given, must come from a prior prepare_ingest() call -- this function
    transitions that job to "running" and completes it, instead of creating its own.
    Without it (the direct-call/script usage from before this parameter existed), a job
    is created and completed within this one call, same as always."""
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {list(CHUNKERS)}")

    source = Path(source)
    extraction_config = extraction_config or _default_extraction_config(force_full_page_ocr)
    cancel_event = _register_cancellable(job_id) if job_id is not None else None

    try:
        try:
            _check_cancelled(cancel_event)
            # Holds this document's spot in line for the resource-heavy part --
            # everything after (chunking, embedding via Ollama) proceeds unthrottled
            # once extraction is done, same as before. The job stays "queued" (not
            # "running") until the semaphore is actually acquired -- otherwise a job
            # still waiting its turn behind max_concurrent_extractions others would
            # show "running"/"extracting" in the UI despite doing nothing yet, which
            # is exactly what was happening before this was moved inside the `with`.
            with _EXTRACTION_SEMAPHORE:
                _check_cancelled(cancel_event)  # may have been cancelled while queued here
                if job_id is not None:
                    catalog.crud.start_job(job_id)
                    catalog.crud.update_job_stage(job_id, "extracting")
                extractor = DoclingExtractor(extraction_config)
                bundle = extractor.extract(source, persist=True)
            _check_cancelled(cancel_event)
        except IngestionCancelledError:
            # No doc/job-with-doc_id exists yet (register_document() hasn't run) --
            # nothing to mark beyond the job row prepare_ingest() already created.
            if job_id is not None:
                catalog.crud.complete_job(job_id, "cancelled", "Cancelled by user")
            raise

        if include_image_descriptions:
            generate_picture_descriptions(bundle.pictures)

        collection = _get_or_create_collection(db_name, qdrant_collection_name, strategy)

        doc = catalog.crud.register_document(
            file_hash=bundle.metadata.file_hash,
            filename=source.name,
            source_path=str(source),
            input_format=bundle.metadata.input_format,
            num_pages=bundle.metadata.num_pages,
        )
        _preserve_original_pdf(source, bundle.metadata.file_hash, bundle.metadata.input_format)

        job = catalog.crud.get_job(job_id) if job_id is not None else catalog.crud.create_job(collection.collection_id, doc.doc_id)

        try:
            catalog.crud.update_job_stage(job.job_id, "chunking")
            _check_cancelled(cancel_event)
            chunk_kwargs = {"config": chunk_config} if chunk_config is not None else {}
            chunks = CHUNKERS[strategy](bundle, **chunk_kwargs)
            if include_image_descriptions:
                chunks = chunks + generate_image_description_chunks(bundle)

            parents = [c for c in chunks if c.kind == "parent"]
            for parent in parents:
                catalog.crud.save_parent_chunk(
                    collection.collection_id, parent.chunk_id, parent.text, parent.page_no, parent.headings
                )

            client = vectorstore.get_client()
            vectorstore.collections.create_collection(client, qdrant_collection_name, collection.dense_dim)

            def _report_embedding_progress(done: int, total: int) -> None:
                catalog.crud.update_job_stage(job.job_id, "embedding", current=done, total=total)

            embedded_count = vectorstore.upsert_chunks(
                client,
                qdrant_collection_name,
                chunks,
                collection.collection_id,
                collection.db_id,
                progress_callback=_report_embedding_progress,
                cancel_event=cancel_event,
            )

            catalog.crud.link_document_to_collection(collection.collection_id, doc.doc_id, len(chunks))
            catalog.crud.recompute_collection_counts(collection.collection_id)
            catalog.crud.complete_job(job.job_id, "done", doc_id=doc.doc_id)
            catalog.crud.update_collection_status(collection.collection_id, "ready")
        except IngestionCancelledError:
            catalog.crud.complete_job(job.job_id, "cancelled", "Cancelled by user", doc_id=doc.doc_id)
            raise
        except Exception as e:
            catalog.crud.complete_job(job.job_id, "error", str(e), doc_id=doc.doc_id)
            catalog.crud.update_collection_status(collection.collection_id, "error")
            raise
    finally:
        if job_id is not None:
            _unregister_cancellable(job_id)

    return IngestResult(
        db_name=db_name,
        qdrant_collection_name=qdrant_collection_name,
        strategy=strategy,
        doc_filename=source.name,
        chunk_count=len(chunks),
        embedded_count=embedded_count,
        parent_count=len(parents),
        job_id=job.job_id,
        job_status="done",
    )


class RetryPlan(BaseModel):
    job: catalog.IngestionJobRecord
    source_path: Path
    db_name: str
    qdrant_collection_name: str
    strategy: str


def prepare_retry(job_id: str) -> RetryPlan:
    """Fast, synchronous half of a job retry -- validates the failed job can be
    retried and creates its replacement "queued" job, mirroring prepare_ingest()'s
    split so the caller (a FastAPI handler) can return immediately and run the actual
    re-ingest via ingest_document() as a background task, same as a first attempt.

    The uploaded file itself is deleted right after every attempt (see
    api/routers/ingest.py's _run_ingest_background()'s finally block), so a retry can
    only work from _preserve_original_pdf()'s permanent copy under
    get_documents_dir()/originals/. That copy only exists once register_document() has
    run (it's written right after), so this only supports retrying a job that failed
    *after* extraction -- i.e. one that already has a doc_id, which every
    "chunking"/"embedding"-stage failure does. A job that failed during extraction
    itself (doc_id still null) has nothing to retry from; the only path forward there
    is re-uploading the file, same as any first attempt.

    Creates a brand-new job row rather than mutating the failed one, so the failed
    attempt stays visible in the job list as history -- exactly the pattern already
    happening organically when a user re-ingests the same file by hand after a failure."""
    job = catalog.crud.get_job(job_id)
    if job.status != "error":
        raise ValueError(f"Job {job_id!r} is not in an error state (status={job.status!r})")
    if job.doc_id is None:
        raise ValueError("This job failed before extraction completed, so there's no preserved file to retry from -- re-upload it instead.")

    doc = catalog.crud.get_document(job.doc_id)
    if doc.input_format != "pdf":
        raise ValueError("Only PDF sources keep a preserved copy for retry -- re-upload this file instead.")

    original_path = get_documents_dir() / "originals" / f"{doc.file_hash}.pdf"
    if not original_path.exists():
        raise ValueError("Preserved original file is missing -- re-upload this file instead.")

    collection = catalog.crud.get_collection(job.qdrant_collection_name)
    new_job = catalog.crud.create_job(job.collection_id, doc_id=job.doc_id, status="queued", filename=job.filename)
    return RetryPlan(
        job=new_job,
        source_path=original_path,
        db_name=job.db_name,
        qdrant_collection_name=job.qdrant_collection_name,
        strategy=collection.chunking_strategy,
    )


def delete_collection(qdrant_collection_name: str) -> None:
    """Deletes both the Qdrant collection and its catalog rows -- the combined
    operation catalog.crud.delete_collection_catalog_rows() deliberately doesn't do on
    its own, since that function has no knowledge of Qdrant. Does not delete the
    permanent document images under settings.get_documents_dir() -- a document may still
    be referenced by other collections."""
    collection = catalog.crud.get_collection(qdrant_collection_name)
    client = vectorstore.get_client()
    vectorstore.collections.delete_collection(client, qdrant_collection_name)
    catalog.crud.delete_collection_catalog_rows(collection.collection_id)
