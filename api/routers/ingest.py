"""Two-phase ingest: POST /ingest resolves/creates the collection and a "queued" job
synchronously (fast), returns job_id immediately, then hands the slow extract/chunk/
embed/write work to a FastAPI BackgroundTask (which Starlette runs in a worker thread,
not blocking the event loop -- see pipeline.py's prepare_ingest()/job_id parameter for
why this needed a pipeline.py change first). Client polls GET /jobs/{job_id}."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from edenview_ingestion import catalog, pipeline
from edenview_ingestion.chunking import CHUNKERS

from ..schemas import IngestAccepted

router = APIRouter(tags=["ingest"])


def _run_ingest_background(
    tmp_path: Path,
    db_name: str,
    collection_name: str,
    strategy: str,
    include_image_descriptions: bool,
    force_full_page_ocr: bool,
    job_id: str,
) -> None:
    try:
        pipeline.ingest_document(
            tmp_path,
            db_name,
            collection_name,
            strategy,
            include_image_descriptions=include_image_descriptions,
            force_full_page_ocr=force_full_page_ocr,
            job_id=job_id,
        )
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


@router.post("/ingest", response_model=IngestAccepted, status_code=202)
def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db_name: str = Form(...),
    collection_name: str = Form(...),
    strategy: str = Form(...),
    include_image_descriptions: bool = Form(False),
    force_full_page_ocr: bool = Form(False),
):
    if strategy not in CHUNKERS:
        raise HTTPException(400, f"Unknown strategy {strategy!r}. Available: {list(CHUNKERS)}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="edenview_upload_"))
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job = pipeline.prepare_ingest(db_name, collection_name, strategy, filename=file.filename)
    background_tasks.add_task(
        _run_ingest_background,
        tmp_path,
        db_name,
        collection_name,
        strategy,
        include_image_descriptions,
        force_full_page_ocr,
        job.job_id,
    )
    return IngestAccepted(job_id=job.job_id, status=job.status, qdrant_collection_name=collection_name)


@router.get("/jobs/{job_id}", response_model=catalog.IngestionJobRecord)
def get_job(job_id: str):
    try:
        return catalog.crud.get_job(job_id)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/jobs/{job_id}/cancel", status_code=204)
def cancel_job(job_id: str):
    """Signals a queued/running job to stop at its next checkpoint -- see
    pipeline.request_cancel()'s docstring for why this is cooperative (a job already
    inside Docling's own extraction call stops as soon as that call returns, not
    instantly) rather than instant. 404s if the job doesn't exist; 409s if it's
    already finished, or if its DB row still says queued/running but this process
    has no record of actually running it (e.g. orphaned by an earlier backend
    restart) -- that job is stuck and cancelling it here can't help; see the
    Ingestion page's job list for how those get cleaned up."""
    try:
        job = catalog.crud.get_job(job_id)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e
    if job.status not in ("queued", "running"):
        raise HTTPException(409, f"Job is already {job.status!r} -- nothing to cancel")
    if not pipeline.request_cancel(job_id):
        raise HTTPException(409, "This job isn't being actively run by this backend process (possibly orphaned by an earlier restart) -- cancelling it here has no effect")


@router.post("/jobs/{job_id}/retry", response_model=IngestAccepted, status_code=202)
def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Requeues a failed job using its preserved original file (see
    pipeline.prepare_retry() for why only a job that failed after extraction --
    doc_id already set -- can be retried this way)."""
    try:
        plan = pipeline.prepare_retry(job_id)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    background_tasks.add_task(
        pipeline.ingest_document,
        plan.source_path,
        plan.db_name,
        plan.qdrant_collection_name,
        plan.strategy,
        job_id=plan.job.job_id,
    )
    return IngestAccepted(job_id=plan.job.job_id, status=plan.job.status, qdrant_collection_name=plan.qdrant_collection_name)


_STATUS_FILTER_MAP = {
    "active": ["queued", "running"],
    "done": ["done"],
    "error": ["error"],
    "cancelled": ["cancelled"],
}


@router.get("/jobs", response_model=list[catalog.IngestionJobRecord])
def list_jobs(limit: int = 50, filename: Optional[str] = None, status: Optional[str] = None):
    """Most-recently-created first -- backs the Ingestion page's status tracker
    server-side instead of the browser's own localStorage, so it shows every job
    regardless of which browser/device kicked it off. `filename`, if given, searches
    by that substring instead of just returning the most recent `limit` jobs.
    `status` is one of "active" (queued or running), "done", or "error" -- omit for
    every status."""
    statuses = _STATUS_FILTER_MAP.get(status) if status else None
    return catalog.crud.list_jobs(limit=limit, filename=filename, statuses=statuses)
