"""DB/Collection catalog endpoints -- thin wrappers over edenview_ingestion.catalog.crud
and pipeline.delete_collection() (the combined Qdrant+catalog delete)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from edenview_ingestion import catalog, pipeline, vectorstore
from edenview_ingestion.chunking import CHUNKERS
from edenview_ingestion.vectorstore.client import client_lock

from ..schemas import CreateDBRequest, PreviewChunk, PreviewResponse

router = APIRouter(tags=["catalog"])


@router.get("/dbs", response_model=list[catalog.DBRecord])
def list_dbs():
    return catalog.crud.list_dbs()


@router.post("/dbs", response_model=catalog.DBRecord, status_code=201)
def create_db(body: CreateDBRequest):
    try:
        return catalog.crud.create_db(body.name)
    except catalog.DuplicateNameError as e:
        raise HTTPException(409, str(e)) from e


@router.delete("/dbs/{db_id}", status_code=204)
def delete_db(db_id: str):
    try:
        catalog.crud.delete_db(db_id)
    except catalog.CatalogError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/collections", response_model=list[catalog.CollectionRecord])
def list_collections(db_name: Optional[str] = None):
    return catalog.crud.list_collections(db_name)


@router.get("/collections/{name}", response_model=catalog.CollectionRecord)
def get_collection(name: str):
    try:
        return catalog.crud.get_collection(name)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.delete("/collections/{name}", status_code=204)
def delete_collection(name: str):
    try:
        pipeline.delete_collection(name)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/collections/{name}/documents", response_model=list[catalog.DocumentRecord])
def list_collection_documents(name: str):
    try:
        collection = catalog.crud.get_collection(name)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return catalog.crud.list_documents_in_collection(collection.collection_id)


@router.get("/collections/{name}/preview", response_model=PreviewResponse)
def preview_collection(name: str, limit: int = 20, offset: Optional[str] = None):
    """Row-level chunk/payload browsing straight off Qdrant's scroll() -- never held in
    the catalog, see edenview_progress.md's "Catalog vs. content browsing"
    decision. Resolves parent_text for parent_child "child" chunks the same way
    edenview_RAG/retrieval/search.py's _resolve_context_text() does for search hits."""
    try:
        collection = catalog.crud.get_collection(name)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e

    client = vectorstore.get_client()
    try:
        with client_lock:
            points, next_page = client.scroll(collection_name=name, limit=limit, offset=offset, with_payload=True)
    except Exception as e:
        raise HTTPException(404, f"Collection {name!r} not found or scroll failed: {e}") from e

    chunks = []
    for p in points:
        payload = p.payload or {}
        parent_text = None
        if payload.get("kind") == "child" and payload.get("parent_id"):
            parent_text = catalog.crud.get_parent_chunk_text(collection.collection_id, payload["parent_id"])
        chunks.append(
            PreviewChunk(
                chunk_id=str(p.id),
                text=payload.get("text", ""),
                page_no=payload.get("page_no"),
                bbox=payload.get("bbox"),
                kind=payload.get("kind", "text"),
                strategy=payload.get("strategy", ""),
                file_hash=payload.get("file_hash", ""),
                images=payload.get("images") or [],
                parent_text=parent_text,
            )
        )
    return PreviewResponse(chunks=chunks, next_offset=str(next_page) if next_page is not None else None)


@router.get("/chunking/strategies", response_model=list[str])
def list_strategies():
    return list(CHUNKERS)
