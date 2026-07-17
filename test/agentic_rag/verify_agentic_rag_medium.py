"""Verification script for the agentic RAG "medium" tier -- reframe (rewrite +
conditional split) -> deterministic retrieval dispatch -> critic/refiner refinement
loop -> final cited answer, wrapped by a thin root via AgentTool(skip_summarization=True).

Single process (unlike the "low" tier script, this doesn't re-test cross-restart
persistence -- that mechanism is already proven by verify_agentic_rag_low.py; this
script focuses on what's new here: reframe's conditional split behavior and the
refinement loop actually running end-to-end).

Usage:
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_agentic_rag_medium.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

import edenview_ingestion  # noqa: F401 -- must be imported before any docling.* import

from edenview_ingestion import catalog, pipeline

SAMPLE_PDF = "data/Finance_act_mini/ITA2025_definitions_excerpt.pdf"
DB_NAME = "verify-agentic-rag-medium-db"
COLLECTION = "verify-agentic-rag-medium-v2"

SIMPLE_QUERY = "According to the Income Tax Act 2025, what does the term 'tax year' mean?"
COMPOUND_QUERY = (
    "What does the Income Tax Act 2025 say a 'tax year' is, and separately, how does "
    "it define 'virtual digital asset'?"
)

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


async def _run_and_inspect(query: str, session_id: str):
    from edenview_RAG.agentic_rag import RetrievalScope
    from edenview_RAG.agentic_rag.runtime import _APP_NAME, _USER_ID, _session_service, run_turn

    scope = RetrievalScope(collection_names=[COLLECTION], top_k=5, use_reranker=True)
    answer, thinking, citations = await run_turn(query, scope, "medium", session_id)
    session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
    reframed = session.state.get("reframed_queries") if session else None
    return answer, citations, reframed


def main() -> int:
    from docling.datamodel.pipeline_options import TableFormerMode

    from edenview_ingestion.docling_parsing import ExtractionConfig, StorageConfig
    from edenview_ingestion.settings import get_documents_dir

    fast_config = ExtractionConfig(table_mode=TableFormerMode.FAST, storage=StorageConfig(base_dir=str(get_documents_dir())))

    print(f"Ingesting {SAMPLE_PDF} into db={DB_NAME!r}...")
    result = pipeline.ingest_document(SAMPLE_PDF, DB_NAME, COLLECTION, "hybrid_docling", extraction_config=fast_config)
    print(f"  {COLLECTION}: {result.embedded_count} embedded")

    try:
        print(f"\n[simple query]\n  query: {SIMPLE_QUERY!r}")
        answer, citations, reframed = asyncio.run(_run_and_inspect(SIMPLE_QUERY, str(uuid.uuid4())))
        print(f"  answer: {answer[:250]!r}")
        print(f"  reframed_queries: {reframed}")
        check("non-empty answer", bool(answer.strip()))
        check("at least one citation", len(citations) > 0, f"got {len(citations)}")
        check(
            "simple single-topic question stays as exactly 1 reframed query",
            isinstance(reframed, dict) and len(reframed.get("queries", [])) == 1,
            f"got {reframed}",
        )
        check(
            "answer mentions the tax year definition (twelve months/April)",
            any(t in answer.lower() for t in ["twelve month", "1st april", "financial year"]),
            f"answer was: {answer[:200]!r}",
        )

        print(f"\n[compound query]\n  query: {COMPOUND_QUERY!r}")
        answer2, citations2, reframed2 = asyncio.run(_run_and_inspect(COMPOUND_QUERY, str(uuid.uuid4())))
        print(f"  answer: {answer2[:250]!r}")
        print(f"  reframed_queries: {reframed2}")
        check("non-empty answer", bool(answer2.strip()))
        check(
            "genuinely compound question splits into >1 reframed queries",
            isinstance(reframed2, dict) and len(reframed2.get("queries", [])) > 1,
            f"got {reframed2}",
        )
        check(
            "compound answer covers both tax year and VDA",
            all(t in answer2.lower() for t in ["tax year"]) and any(t in answer2.lower() for t in ["virtual digital", "vda"]),
            f"answer was: {answer2[:300]!r}",
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
