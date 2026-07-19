"""Retrieval endpoint -- thin wrapper over edenview_RAG.retrieval.search()/search_db()."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from edenview_RAG.retrieval import RetrievalConfig, RetrievalHit, search, search_db

from ..schemas import SearchRequest

router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[RetrievalHit])
def run_search(body: SearchRequest):
    if not body.db_name and not body.collection_names:
        raise HTTPException(400, "Provide either db_name or collection_names")

    config = RetrievalConfig(top_k=body.top_k, use_reranker=body.use_reranker)
    if body.collection_names:
        return search(body.collection_names, body.query, config, body.file_hashes, body.strategy)
    return search_db(body.db_name, body.query, config, body.file_hashes, body.strategy)
