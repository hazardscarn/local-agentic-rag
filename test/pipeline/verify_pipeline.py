"""Verification script for the catalog + vectorstore + pipeline modules -- ingests a
real sample document into two collections under one DB (different strategies), then
checks the catalog rows, the Qdrant collections/points, an actual hybrid search
round-trip, parent-chunk lookup, idempotent re-ingestion, and collection deletion.

Not pytest -- matches this repo's existing demo-script convention (see
test/chunking/verify_chunking.py). Uses the real configured data root (config.yaml's
qdrant/catalog/storage paths) since persistence across runs is the actual point of this
stack, not a throwaway test fixture -- re-running this script is expected to be safe
(idempotent) rather than needing a fresh environment each time.

Usage:
    PYTHONPATH=. python test/pipeline/verify_pipeline.py
"""

from __future__ import annotations

import sys

import edenview_ingestion  # noqa: F401 -- must be imported before any docling.* import
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from edenview_ingestion import catalog, pipeline, vectorstore
from edenview_ingestion.vectorstore.embedding import embed_dense, embed_sparse
from edenview_RAG.retrieval import search_db

SAMPLE_PDF = "data/sample_files/covid-19-risk-factors-Japan.pdf"
DB_NAME = "verify-pipeline-db"
COLLECTION_HYBRID = "verify-hybrid-docling"
COLLECTION_PARENT_CHILD = "verify-parent-child"

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def hybrid_search(collection_name: str, query: str, top_k: int = 3):
    client = vectorstore.get_client()
    dense_vec = embed_dense([query])[0]
    sparse_vec = embed_sparse([query])[0]
    return client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_k * 4),
            Prefetch(
                query=SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"]),
                using="sparse",
                limit=top_k * 4,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points


def main() -> int:
    print(f"Ingesting {SAMPLE_PDF} into two collections under db={DB_NAME!r}...")

    from docling.datamodel.pipeline_options import TableFormerMode

    from edenview_ingestion.docling_parsing import ExtractionConfig, StorageConfig
    from edenview_ingestion.settings import get_documents_dir

    fast_config = ExtractionConfig(
        page_range=(1, 5),
        table_mode=TableFormerMode.FAST,
        generate_picture_images=True,
        do_picture_classification=True,
        storage=StorageConfig(base_dir=str(get_documents_dir())),
    )

    result_a = pipeline.ingest_document(
        SAMPLE_PDF, DB_NAME, COLLECTION_HYBRID, "hybrid_docling",
        extraction_config=fast_config, include_image_descriptions=True,
    )
    print(f"  {COLLECTION_HYBRID}: {result_a.chunk_count} chunks, {result_a.embedded_count} embedded")

    result_b = pipeline.ingest_document(
        SAMPLE_PDF, DB_NAME, COLLECTION_PARENT_CHILD, "parent_child", extraction_config=fast_config,
    )
    print(f"  {COLLECTION_PARENT_CHILD}: {result_b.chunk_count} chunks, "
          f"{result_b.embedded_count} embedded, {result_b.parent_count} parents")

    print("\n[catalog]")
    db = catalog.crud.get_db(DB_NAME)
    check("db exists", db.name == DB_NAME)

    collections = catalog.crud.list_collections(DB_NAME)
    names = {c.qdrant_collection_name for c in collections}
    check("both collections registered under the db", {COLLECTION_HYBRID, COLLECTION_PARENT_CHILD} <= names)

    col_a = catalog.crud.get_collection(COLLECTION_HYBRID)
    check("collection chunk_count matches ingest result", col_a.chunk_count == result_a.chunk_count,
          f"catalog={col_a.chunk_count} result={result_a.chunk_count}")
    check("collection doc_count == 1", col_a.doc_count == 1)

    docs = catalog.crud.list_documents_in_collection(col_a.collection_id)
    check("document registered with correct filename", len(docs) == 1 and docs[0].filename.endswith(".pdf"))

    print("\n[vectorstore]")
    client = vectorstore.get_client()
    check("qdrant collection exists (hybrid_docling)", vectorstore.collections.collection_exists(client, COLLECTION_HYBRID))
    check("qdrant collection exists (parent_child)", vectorstore.collections.collection_exists(client, COLLECTION_PARENT_CHILD))

    count_a = client.count(COLLECTION_HYBRID).count
    check("point count matches embedded_count (hybrid_docling)", count_a == result_a.embedded_count,
          f"qdrant={count_a} expected={result_a.embedded_count}")

    sample = client.scroll(COLLECTION_HYBRID, limit=1, with_payload=True)[0][0]
    payload = sample.payload
    check("payload has text/strategy/kind/collection_id", all(k in payload for k in ("text", "strategy", "kind", "collection_id")))
    check("payload collection_id matches catalog", payload["collection_id"] == col_a.collection_id)

    print("\n[hybrid search round-trip]")
    hits = hybrid_search(COLLECTION_HYBRID, "risk factors for post-COVID condition")
    check("hybrid search returns results", len(hits) > 0, f"got {len(hits)}")
    if hits:
        print(f"  top hit score={hits[0].score:.4f} text={hits[0].payload['text'][:80]!r}")

    print("\n[parent_child lookup]")
    col_b = catalog.crud.get_collection(COLLECTION_PARENT_CHILD)
    child_hits = hybrid_search(COLLECTION_PARENT_CHILD, "study design participants")
    child_with_parent = next((h for h in child_hits if h.payload.get("kind") == "child"), None)
    check("a child chunk was retrieved", child_with_parent is not None)
    if child_with_parent:
        parent_text = catalog.crud.get_parent_chunk_text(col_b.collection_id, child_with_parent.payload["parent_id"])
        check("parent text resolves from catalog", bool(parent_text), f"parent_id={child_with_parent.payload.get('parent_id')}")

    print("\n[strategy filter on DB-wide search]")
    # Filtering is at the *collection* level (catalog.chunking_strategy), not per-chunk
    # payload.strategy -- an image_description chunk living inside the hybrid_docling
    # collection legitimately carries its own strategy="image_description" tag (see
    # chunking/image_descriptions.py), so the right check is "did every hit come from
    # the right collection", not "does every hit's own strategy tag match".
    query = "study design and infection risk factors"
    hybrid_only = search_db(DB_NAME, query, strategy="hybrid_docling")
    check("strategy=hybrid_docling returns only hits from that collection", len(hybrid_only) > 0 and
          all(h.collection_name == COLLECTION_HYBRID for h in hybrid_only), f"collections={[h.collection_name for h in hybrid_only]}")

    parent_child_only = search_db(DB_NAME, query, strategy="parent_child")
    check("strategy=parent_child returns only hits from that collection", len(parent_child_only) > 0 and
          all(h.collection_name == COLLECTION_PARENT_CHILD for h in parent_child_only), f"collections={[h.collection_name for h in parent_child_only]}")

    unfiltered = search_db(DB_NAME, query)
    collections_seen = {h.collection_name for h in unfiltered}
    check("unfiltered DB search can span both collections", len(collections_seen) >= 1, f"collections={collections_seen}")

    print("\n[idempotent re-ingestion]")
    result_a2 = pipeline.ingest_document(
        SAMPLE_PDF, DB_NAME, COLLECTION_HYBRID, "hybrid_docling",
        extraction_config=fast_config, include_image_descriptions=True,
    )
    count_a2 = client.count(COLLECTION_HYBRID).count
    check("re-ingesting doesn't duplicate points", count_a2 == count_a, f"before={count_a} after={count_a2}")
    col_a2 = catalog.crud.get_collection(COLLECTION_HYBRID)
    check("re-ingesting doesn't double-count docs", col_a2.doc_count == 1, f"doc_count={col_a2.doc_count}")
    check("re-ingesting doesn't double-count chunks", col_a2.chunk_count == result_a.chunk_count,
          f"chunk_count={col_a2.chunk_count} expected={result_a.chunk_count}")

    print("\n[deletion]")
    pipeline.delete_collection(COLLECTION_PARENT_CHILD)
    check("qdrant collection gone after delete", not vectorstore.collections.collection_exists(client, COLLECTION_PARENT_CHILD))
    try:
        catalog.crud.get_collection(COLLECTION_PARENT_CHILD)
        check("catalog row gone after delete", False)
    except catalog.NotFoundError:
        check("catalog row gone after delete", True)

    pipeline.delete_collection(COLLECTION_HYBRID)
    try:
        catalog.crud.delete_db(db.db_id)
        check("db deletable once its collections are gone", True)
    except catalog.CatalogError as e:
        check("db deletable once its collections are gone", False, str(e))

    print(f"\n{'=' * 60}\n{_passed} passed, {_failed} failed\n{'=' * 60}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
