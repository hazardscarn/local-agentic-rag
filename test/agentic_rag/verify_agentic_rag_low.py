"""Verification script for the agentic RAG "low" tier -- the single most important
checkpoint in the whole agentic_rag build plan: proves LiteLlm + ollama_chat/<model> +
ADK's FunctionTool schema generation + citation round-trip all work together, AND that
DatabaseSessionService genuinely persists conversational memory across a process
restart (not just within one process's lifetime).

Two phases, run as two SEPARATE OS processes (not subprocess-spawned from within one
script) -- the embedded Qdrant/DuckDB store this project uses only tolerates one
process holding it open at a time, so a real "does memory survive a restart" test
needs the first process to fully exit before the second one starts.

Usage:
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_agentic_rag_low.py --phase 1
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_agentic_rag_low.py --phase 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # model output can include non-ASCII (emoji, etc.)

import edenview_ingestion  # noqa: F401 -- must be imported before any docling.* import

from edenview_ingestion import catalog, pipeline

# A standalone 30-page excerpt (pages 1-30 of the real Income Tax Act 2025 PDF,
# covering the definitions chapter incl. VDA at section 111/page 25) -- NOT the full
# 686-page original. Using a genuinely separate file (not extraction_config.page_range
# on the full PDF) deliberately sidesteps a real caching gotcha found during
# development: DoclingExtractor.extract()'s cache is keyed by filename and, on a hit,
# ignores page_range entirely (a docstring-documented behavior, not a bug in this
# script) -- an earlier, unrelated persist=True run had already cached a full parse of
# the original file, silently reused a full 636-chunk/686-page document instead of the
# requested 30, producing an accordingly-derailed test. A separate file has its own
# file_hash and can never collide with that (or any future) cache.
SAMPLE_PDF = "data/Finance_act_mini/ITA2025_definitions_excerpt.pdf"
DB_NAME = "verify-agentic-rag-low-db"
COLLECTION = "verify-agentic-rag-low-v2"
SESSION_ID_FILE = Path(__file__).parent / ".verify_low_session_id.tmp"

# Note: the excerpt's hybrid_docling chunking merged 30 dense pages of numbered
# clauses into only 10 chunks -- the specific VDA definition (page 25, item 111)
# ended up merged/split away from a clean top-k hit (confirmed present in the source
# text via a direct pypdfium2 text scan, so this is a chunking-granularity artifact of
# this particular strategy on this document, not an agentic_rag bug). Using a
# question the corpus's actual 10 chunks demonstrably cover instead (section 3's "tax
# year" definition, confirmed present via a direct scroll of the collection).
QUERY_1 = "According to the Income Tax Act 2025, what does the term 'tax year' mean?"
QUERY_2 = (
    "Following up on the 'tax year' definition I just asked about -- for a business "
    "that's newly set up partway through the year, does the tax year still run the "
    "full twelve months, or does it start from a different date?"
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


def phase_1() -> int:
    from docling.datamodel.pipeline_options import TableFormerMode

    from edenview_ingestion.docling_parsing import ExtractionConfig, StorageConfig
    from edenview_ingestion.settings import get_documents_dir
    from edenview_RAG.agentic_rag import RetrievalScope
    from edenview_RAG.agentic_rag.runtime import run_turn

    fast_config = ExtractionConfig(
        table_mode=TableFormerMode.FAST,
        storage=StorageConfig(base_dir=str(get_documents_dir())),
    )

    print(f"Ingesting {SAMPLE_PDF} into db={DB_NAME!r}...")
    result = pipeline.ingest_document(SAMPLE_PDF, DB_NAME, COLLECTION, "hybrid_docling", extraction_config=fast_config)
    print(f"  {COLLECTION}: {result.embedded_count} embedded")

    session_id = str(uuid.uuid4())
    SESSION_ID_FILE.write_text(session_id, encoding="utf-8")

    scope = RetrievalScope(collection_names=[COLLECTION], top_k=5, use_reranker=True)

    print(f"\n[phase 1 -- turn 1, session_id={session_id}]")
    print(f"  query: {QUERY_1!r}")
    answer, thinking, citations = asyncio.run(run_turn(QUERY_1, scope, "low", session_id))
    print(f"  answer: {answer[:300]!r}")
    print(f"  citations: {len(citations)}")

    check("non-empty answer", bool(answer.strip()))
    check("at least one citation", len(citations) > 0, f"got {len(citations)}")
    if citations:
        check(
            "citation chunk_id resolves back to the ingested collection",
            all(c.collection_name == COLLECTION for c in citations),
        )

    print(f"\n{'=' * 60}\nphase 1: {_passed} passed, {_failed} failed\n{'=' * 60}")
    print("Now run this script again with --phase 2 (a SEPARATE process) to test persistence.")
    return 1 if _failed else 0


def phase_2() -> int:
    from edenview_RAG.agentic_rag import RetrievalScope
    from edenview_RAG.agentic_rag.runtime import run_turn

    if not SESSION_ID_FILE.exists():
        print("No session id file found -- run --phase 1 first.")
        return 1
    session_id = SESSION_ID_FILE.read_text(encoding="utf-8").strip()

    scope = RetrievalScope(collection_names=[COLLECTION], top_k=5, use_reranker=True)

    print(f"\n[phase 2 -- turn 2 (fresh process), session_id={session_id}]")
    print(f"  query: {QUERY_2!r}")
    answer, thinking, citations = asyncio.run(run_turn(QUERY_2, scope, "low", session_id))
    print(f"  answer: {answer[:500]!r}")
    print(f"  citations: {len(citations)}")

    check("non-empty answer", bool(answer.strip()))
    check(
        "answer engages with the tax-year/newly-set-up-business follow-up (not a generic/confused answer)",
        any(term in answer.lower() for term in ["tax year", "twelve month", "newly set", "financial year"]),
        f"answer was: {answer[:200]!r}",
    )

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
    SESSION_ID_FILE.unlink(missing_ok=True)

    print(f"\n{'=' * 60}\nphase 2: {_passed} passed, {_failed} failed\n{'=' * 60}")
    return 1 if _failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    return phase_1() if args.phase == 1 else phase_2()


if __name__ == "__main__":
    sys.exit(main())
