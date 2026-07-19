"""Hybrid search: dense + sparse (BM25) + Qdrant native RRF fusion per collection, then
(optionally) a cross-encoder rerank pass. The rerank pass isn't just a quality nicety --
it's also what makes searching *multiple* collections at once sound: RRF fusion scores
are only meaningfully comparable within one collection's own query, not across separate
ones, while a cross-encoder scores every (query, chunk) pair on the same absolute-ish
scale regardless of which collection it came from.

`text` on a returned hit is always the precise matched chunk; `context_text` is what to
actually feed an LLM -- identical to `text` except for a "child" (parent_child strategy)
hit, where it's swapped for the full parent chunk from the catalog's parent_chunks table.

Lives under edenview_RAG (not edenview_ingestion) because this is the query-time/serving
side of the stack -- edenview_ingestion owns the write path (extract/chunk/embed/write),
edenview_RAG owns the read path (this module, and later the ADK agent loop + API). Both
depend on edenview_ingestion's catalog/vectorstore as the shared storage layer.
"""

from __future__ import annotations

from typing import Optional

from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, MatchAny, Prefetch, SparseVector

from edenview_ingestion import catalog, vectorstore
from edenview_ingestion.vectorstore.client import client_lock
from edenview_ingestion.vectorstore.embedding import embed_dense, embed_sparse

from .config import RetrievalConfig
from .models import RetrievalHit
from .reranker import rerank


def _build_filter(file_hashes: Optional[list[str]]) -> Optional[Filter]:
    if not file_hashes:
        return None
    return Filter(must=[FieldCondition(key="file_hash", match=MatchAny(any=file_hashes))])


def _query_one_collection(collection_name: str, dense_vec, sparse_vec, limit: int, query_filter: Optional[Filter]):
    client = vectorstore.get_client()
    with client_lock:
        return client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=limit * 4, filter=query_filter),
                Prefetch(
                    query=SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"]),
                    using="sparse",
                    limit=limit * 4,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        ).points


def _resolve_context_text(payload: dict, collection_id: Optional[str]) -> str:
    text = payload.get("text", "")
    if payload.get("kind") != "child" or not payload.get("parent_id") or collection_id is None:
        return text
    parent_text = catalog.crud.get_parent_chunk_text(collection_id, payload["parent_id"])
    return parent_text if parent_text else text


def search(
    collection_names: list[str],
    query: str,
    config: RetrievalConfig = RetrievalConfig(),
    file_hashes: Optional[list[str]] = None,
    strategy: Optional[str] = None,
) -> list[RetrievalHit]:
    """Searches one or more Qdrant collections and returns the merged, ranked top_k.
    `file_hashes`, if given, restricts results to those specific source documents
    (payload filter on `file_hash`) -- collection-level and document-level scoping
    compose, per edenview_progress.md's retrieval filtering decision.

    `strategy`, if given, restricts the search to collections whose catalog record has
    that `chunking_strategy` (see catalog.CollectionRecord). Matters most when
    `collection_names` spans collections built from different strategies over
    overlapping documents (typically via search_db(), searching a whole DB) -- without
    this, the merged top-k can surface several near-duplicate chunks of the same
    underlying content from competing strategies instead of genuinely different
    results. A collection_name with no catalog record is excluded when filtering by
    strategy (nothing to match against), but still searched normally when strategy is
    None, same as before this parameter existed.

    A `collection_names` entry with no catalog record at all (never created, or
    deleted since whatever selected it) is dropped unconditionally, not just when a
    `strategy` filter happens to be active -- every real collection is registered in
    the catalog at creation time, so a name absent from it can never be a real,
    queryable Qdrant collection. Without this, a single stale name (e.g. a leftover
    browser-side selection referencing a collection that no longer exists, or was
    created under a different workspace) reached Qdrant directly and crashed the
    *entire* request with an unhandled ValueError -- even though every other,
    genuinely-selected collection in the same request would have searched fine.
    Confirmed by direct reproduction: a request scoped to one real collection plus one
    stale name 500'd outright instead of just skipping the stale one."""
    if not collection_names:
        return []

    collection_records: dict[str, Optional[catalog.CollectionRecord]] = {}
    for name in collection_names:
        try:
            collection_records[name] = catalog.crud.get_collection(name)
        except catalog.NotFoundError:
            collection_records[name] = None

    collection_names = [name for name in collection_names if collection_records[name] is not None]
    if not collection_names:
        return []

    if strategy is not None:
        collection_names = [
            name for name in collection_names if collection_records[name].chunking_strategy == strategy
        ]
        if not collection_names:
            return []

    sparse_vec = embed_sparse([query])[0]
    query_filter = _build_filter(file_hashes)
    per_collection_limit = config.top_k if len(collection_names) == 1 else config.per_collection_candidates

    hits: list[RetrievalHit] = []
    # chunk_id is deterministic on (file_hash, chunking_strategy, chunk_index) alone
    # (edenview_ingestion/chunking/models.py's make_chunk_id()) and is used directly
    # as the Qdrant point ID -- Qdrant only guarantees that ID is unique WITHIN one
    # collection, not globally. If the same source document is ingested into two
    # different collections under the same strategy (nothing prevents this --
    # catalog/schema.py's collection_documents join table explicitly allows one doc
    # in many collections), both collections produce chunks with the IDENTICAL
    # chunk_id, and a query spanning both would otherwise return the same chunk_id
    # twice in one hits list -- confirmed directly causing a real symptom: a React
    # "duplicate key" warning where the frontend renders the same citation twice.
    # Deduped here (first-seen wins, before reranking so a genuine duplicate never
    # even costs a rerank call) rather than downstream, since every caller of
    # search()/search_db() (Simple RAG's /chat, Agentic RAG's vector_search tool)
    # should get a clean, duplicate-free hit list, not have to work around this
    # collection-spanning ID collision individually.
    seen_chunk_ids: set[str] = set()
    for name in collection_names:
        record = collection_records[name]
        # Each collection's own stored embedding_model, not config.yaml's current
        # dense_embedding -- otherwise switching the configured model silently breaks
        # (or, if dimensions happen to coincide, silently corrupts) search over every
        # collection built with the old one. Falls back to the current config default
        # only when there's no catalog record at all to know better from.
        dense_vec = embed_dense([query], model=record.embedding_model if record is not None else None)[0]
        points = _query_one_collection(name, dense_vec, sparse_vec, per_collection_limit, query_filter)
        collection_id = record.collection_id if record is not None else None
        for point in points:
            chunk_id = str(point.id)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            payload = point.payload or {}
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    score=point.score,
                    text=payload.get("text", ""),
                    context_text=_resolve_context_text(payload, collection_id),
                    collection_name=name,
                    strategy=payload.get("strategy", ""),
                    kind=payload.get("kind", "text"),
                    page_no=payload.get("page_no"),
                    bbox=payload.get("bbox"),
                    headings=payload.get("headings") or [],
                    doc_stem=payload.get("doc_stem", ""),
                    file_hash=payload.get("file_hash", ""),
                    images=payload.get("images") or [],
                )
            )

    if config.use_reranker and hits:
        scores = rerank(query, [h.text for h in hits], config.reranker_model)
        for hit, score in zip(hits, scores):
            hit.score = float(score)

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: config.top_k]


def search_db(
    db_name: str,
    query: str,
    config: RetrievalConfig = RetrievalConfig(),
    file_hashes: Optional[list[str]] = None,
    strategy: Optional[str] = None,
) -> list[RetrievalHit]:
    """Convenience wrapper: search every collection currently registered under a DB.
    `strategy`, if given, narrows the fan-out to just that chunking strategy -- see
    search()'s docstring for why that matters for a DB spanning multiple strategies."""
    collections = catalog.crud.list_collections(db_name)
    return search([c.qdrant_collection_name for c in collections], query, config, file_hashes, strategy)
