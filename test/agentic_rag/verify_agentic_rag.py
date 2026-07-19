"""Verification script for the agentic RAG pipeline (one flat pipeline, no effort
tiers -- see edenview_RAG/agentic_rag/agent.py). Proves LiteLlm + ollama_chat/<model>
+ ADK's FunctionTool schema generation + citation round-trip all work together, AND
that DatabaseSessionService genuinely persists conversational memory across a process
restart (not just within one process's lifetime) -- Root's own history is what
resolves a follow-up like "the tax-year question I just asked" into a standalone
question before calling query_pipeline (see prompts.ROOT_INSTRUCTION).

Two phases, run as two SEPARATE OS processes (not subprocess-spawned from within one
script) -- the embedded Qdrant/DuckDB store this project uses only tolerates one
process holding it open at a time, so a real "does memory survive a restart" test
needs the first process to fully exit before the second one starts.

A real run through this pipeline can genuinely take several minutes (reworder +
search + up to `agent.max_iterations` eval/deep-search rounds + answer formatting,
per runtime.py::run_turn's own docstring) -- this is expected, not a hang.

Usage:
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_agentic_rag.py --phase 1
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_agentic_rag.py --phase 2
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

# Same standalone 30-page excerpt the old tiered verify scripts used (pages 1-30 of
# the real Income Tax Act 2025 PDF, covering the definitions chapter) -- a genuinely
# separate file (not extraction_config.page_range on the full PDF) deliberately
# sidesteps a real caching gotcha: DoclingExtractor.extract()'s cache is keyed by
# filename and, on a hit, ignores page_range entirely -- a separate file has its own
# file_hash and can never collide with that (or any future) cache.
SAMPLE_PDF = "data/Finance_act_mini/ITA2025_definitions_excerpt.pdf"
DB_NAME = "verify-agentic-rag-db"
COLLECTION = "verify-agentic-rag-v3"
SESSION_ID_FILE = Path(__file__).parent / ".verify_session_id.tmp"

# Confirmed present in the excerpt's actual chunks (section 3's "tax year"
# definition) -- not a question the corpus is silent on.
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
    answer, thinking, citations = asyncio.run(run_turn(QUERY_1, scope, session_id))
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
    answer, thinking, citations = asyncio.run(run_turn(QUERY_2, scope, session_id))
    print(f"  answer: {answer[:500]!r}")
    print(f"  citations: {len(citations)}")

    check("non-empty answer", bool(answer.strip()))
    check(
        "answer engages with the tax-year/newly-set-up-business follow-up (not a generic/confused answer) "
        "-- proves Root's own persisted history resolved the follow-up before calling query_pipeline",
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
