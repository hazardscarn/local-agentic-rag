"""Serves picture/table crop image files referenced in a chunk's payload. Not a general
file server -- `path` must resolve to somewhere under settings.get_documents_dir(),
rejected otherwise, so this endpoint can't be used to read arbitrary files off disk."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from edenview_ingestion.settings import get_documents_dir

router = APIRouter(tags=["files"])


@router.get("/files")
def get_file(path: str = Query(..., description="Absolute path from a chunk's images[].image_path")):
    documents_dir = get_documents_dir().resolve()
    requested = Path(path).resolve()

    try:
        requested.relative_to(documents_dir)
    except ValueError:
        raise HTTPException(403, f"Path must be under {documents_dir}") from None

    if not requested.is_file():
        raise HTTPException(404, "File not found")

    return FileResponse(requested)
