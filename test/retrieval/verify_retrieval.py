"""Verification script for the retrieval module -- ingests a real sample document into
two collections under one DB, then checks single-collection search, multi-collection
fan-out+merge, cross-encoder reranking, parent-chunk swap, and document-level filtering.

Not pytest -- matches this repo's existing demo-script convention (see
test/pipeline/verify_pipeline.py). Uses the real configured data root; cleans up (deletes
its collections/db) at the end so repeated runs stay idempotent.

Usage:
    PYTHONPATH=. python test/retrieval/verify_retrieval.py
"""

from __future__ import annotations

import sys

import edenview_ingestion  # noqa: F401 -- must be imported before any docling.* import

from edenview_ingestion import catalog, pipeline
from edenview_RAG.retrieval import RetrievalConfig, search, search_db

SAMPLE_PDF = "data/sample_files/covid-19-risk-factors-Japan.pdf"
DB_NAME = "verify-retrieval-db"
COLLECTION_HYBRID = "verify-retrieval-hybrid"
COLLECTION_PARENT_CHILD = "verify-retrieval-parent-child"
QUERY = "risk factors for post-COVID condition"

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


def main() -> int:
    from docling.datamodel.pipeline_options import TableFormerMode

    from edenview_ingestion.docling_parsing import ExtractionConfig, StorageConfig
    from edenview_ingestion.settings import get_documents_dir

    fast_config = ExtractionConfig(
        page_range=(1, 5),
        table_mode=TableFormerMode.FAST,
        generate_picture_images=True,
        storage=StorageConfig(base_dir=str(get_documents_dir())),
    )

    print(f"Ingesting {SAMPLE_PDF} into two collections under db={DB_NAME!r}...")
    result_a = pipeline.ingest_document(SAMPLE_PDF, DB_NAME, COLLECTION_HYBRID, "hybrid_docling", extraction_config=fast_config)
    result_b = pipeline.ingest_document(SAMPLE_PDF, DB_NAME, COLLECTION_PARENT_CHILD, "parent_child", extraction_config=fast_config)
    print(f"  {COLLECTION_HYBRID}: {result_a.embedded_count} embedded")
    print(f"  {COLLECTION_PARENT_CHILD}: {result_b.embedded_count} embedded")
    file_hash = catalog.crud.list_documents_in_collection(catalog.crud.get_collection(COLLECTION_HYBRID).collection_id)[0].file_hash

    try:
        print("\n[single-collection search, reranker on]")
        hits = search([COLLECTION_HYBRID], QUERY, RetrievalConfig(top_k=3, use_reranker=True))
        check("returns results", len(hits) > 0, f"got {len(hits)}")
        check("results capped at top_k", len(hits) <= 3)
        check("all hits from the requested collection", all(h.collection_name == COLLECTION_HYBRID for h in hits))
        check("scores are descending", hits == sorted(hits, key=lambda h: h.score, reverse=True))
        if hits:
            print(f"  top hit (score={hits[0].score:.4f}): {hits[0].text[:80]!r}")

        print("\n[single-collection search, reranker off]")
        hits_no_rerank = search([COLLECTION_HYBRID], QUERY, RetrievalConfig(top_k=3, use_reranker=False))
        check("returns results without reranker too", len(hits_no_rerank) > 0)

        print("\n[multi-collection search (fan-out + merge)]")
        multi_hits = search_db(DB_NAME, QUERY, RetrievalConfig(top_k=6, use_reranker=True))
        check("returns results", len(multi_hits) > 0, f"got {len(multi_hits)}")
        sources = {h.collection_name for h in multi_hits}
        check("pulled from more than one collection", len(sources) > 1, f"sources={sources}")

        print("\n[parent_child swap]")
        pc_hits = search([COLLECTION_PARENT_CHILD], "study design participants", RetrievalConfig(top_k=5, use_reranker=True))
        child_hit = next((h for h in pc_hits if h.kind == "child"), None)
        check("a child-kind hit was retrieved", child_hit is not None)
        if child_hit:
            check("context_text differs from text (parent swapped in)", child_hit.context_text != child_hit.text)
            check("context_text is longer (parent is bigger than child)", len(child_hit.context_text) > len(child_hit.text))
            print(f"  child text: {child_hit.text[:60]!r}")
            print(f"  parent context_text: {child_hit.context_text[:60]!r}...")

        print("\n[document-level filtering]")
        filtered_hits = search([COLLECTION_HYBRID], QUERY, RetrievalConfig(top_k=3), file_hashes=[file_hash])
        check("filtering by the doc's own file_hash still returns results", len(filtered_hits) > 0)
        empty_hits = search([COLLECTION_HYBRID], QUERY, RetrievalConfig(top_k=3), file_hashes=["nonexistent-hash"])
        check("filtering by a bogus file_hash returns nothing", len(empty_hits) == 0, f"got {len(empty_hits)}")

    finally:
        print("\n[cleanup]")
        pipeline.delete_collection(COLLECTION_HYBRID)
        pipeline.delete_collection(COLLECTION_PARENT_CHILD)
        try:
            db = catalog.crud.get_db(DB_NAME)
            catalog.crud.delete_db(db.db_id)
            check("cleanup: db deleted", True)
        except catalog.NotFoundError:
            pass

    print(f"\n{'=' * 60}\n{_passed} passed, {_failed} failed\n{'=' * 60}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
