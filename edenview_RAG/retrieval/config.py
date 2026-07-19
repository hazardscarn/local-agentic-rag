"""Retrieval configuration. Model name (reranker) is never hardcoded -- read from
config.yaml via edenview_ingestion.settings, same pattern as chunking/config.py."""

from __future__ import annotations

from pydantic import BaseModel

from edenview_ingestion.settings import get_model

DEFAULT_RERANKER_MODEL = get_model("reranker")


class RetrievalConfig(BaseModel):
    top_k: int = 5
    # Each collection's dense/sparse prefetch fetches top_k * prefetch_multiplier
    # candidates before RRF fusion narrows to top_k -- standard hybrid-search
    # over-fetch so RRF has enough candidates to actually fuse over.
    prefetch_multiplier: int = 4
    use_reranker: bool = True
    reranker_model: str = DEFAULT_RERANKER_MODEL
    # When searching multiple collections at once (a whole DB), each collection
    # contributes this many candidates *before* the merge+rerank step narrows to
    # top_k -- higher than top_k so a collection with only mediocre matches doesn't
    # crowd out a different collection's stronger ones during the merge.
    per_collection_candidates: int = 15
