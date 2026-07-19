"""Verification script for the agentic RAG "high" tier -- the same tree as "medium"
(reframe -> dispatch -> critic/refiner loop -> answer) plus a higher max_iterations
and the refiner's extra get_page_context tool. Since medium's mechanics are already
proven (verify_agentic_rag_medium.py), this focuses on what's new here:
get_page_context actually reconstructing a page's full text.

Usage:
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_agentic_rag_high.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

import edenview_ingestion  # noqa: F401 -- must be imported before any docling.* import

from edenview_ingestion import catalog, pipeline

SAMPLE_PDF = "data/Finance_act_mini/ITA2025_definitions_excerpt.pdf"
DB_NAME = "verify-agentic-rag-high-db"
COLLECTION = "verify-agentic-rag-high-v2"

QUERY = "According to the Income Tax Act 2025, what does the term 'tax year' mean?"

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
    from edenview_RAG.agentic_rag import RetrievalScope
    from edenview_RAG.agentic_rag.runtime import run_turn
    from edenview_RAG.agentic_rag.tools import get_page_context

    fast_config = ExtractionConfig(table_mode=TableFormerMode.FAST, storage=StorageConfig(base_dir=str(get_documents_dir())))

    print(f"Ingesting {SAMPLE_PDF} into db={DB_NAME!r}...")
    result = pipeline.ingest_document(SAMPLE_PDF, DB_NAME, COLLECTION, "hybrid_docling", extraction_config=fast_config)
    print(f"  {COLLECTION}: {result.embedded_count} embedded")

    try:
        scope = RetrievalScope(collection_names=[COLLECTION], top_k=5, use_reranker=True)
        session_id = str(uuid.uuid4())
        print(f"\n[high tier end-to-end]\n  query: {QUERY!r}")
        answer, thinking, citations = asyncio.run(run_turn(QUERY, scope, "high", session_id))
        print(f"  answer: {answer[:300]!r}")
        print(f"  thinking: {thinking[:200]!r}")
        print(f"  citations: {len(citations)}")
        check("non-empty answer", bool(answer.strip()))
        check("at least one citation", len(citations) > 0, f"got {len(citations)}")
        check(
            "answer mentions the tax year definition (twelve months/April)",
            any(t in answer.lower() for t in ["twelve month", "1st april", "financial year"]),
            f"answer was: {answer[:200]!r}",
        )
        check(
            "answer doesn't contain leaked internal-reasoning narration",
            not any(
                marker in answer.lower()
                for marker in ["i need to use the", "let me search for", "i'll call the", "as my request parameter"]
            ),
            f"answer was: {answer[:300]!r}",
        )

        print("\n[get_page_context tool mechanics]")
        tax_year_citation = next((c for c in citations if c.page_no), None)
        check("a citation with a page_no exists to test against", tax_year_citation is not None)
        if tax_year_citation:
            page_result = get_page_context(
                file_hash=tax_year_citation.file_hash,
                collection_name=COLLECTION,
                page_no=tax_year_citation.page_no,
                include_adjacent=False,
                tool_context=None,  # unused by get_page_context -- it never reads scope/state
            )
            check("get_page_context returns ok status", page_result["status"] == "ok", f"got {page_result}")
            check(
                "get_page_context's reconstructed page text contains the citation's own text",
                tax_year_citation.text.strip()[:40] in page_result.get("text", ""),
                f"page text: {page_result.get('text', '')[:200]!r}",
            )

    finally:
        print("\n[cleanup]")
        try:
            pipeline.delete_collection(COLLECTION)
            check("cleanup: collection deleted", True)
        except Exception as e:  # noqa: BLE001
            check("cleanup: collection deleted", False, str(e))
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
