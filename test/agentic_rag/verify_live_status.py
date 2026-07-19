"""Verification script for run_turn_stream's live per-node/per-tool status events --
confirms callbacks.track_agent_start/end + track_tool_start/end (via the
contextvars-based queue in runtime.py) actually surface every agent/tool boundary
inside the agentic pipeline in real time, not just the final answer.

Drives runtime.run_turn_stream directly (the exact function POST /chat/stream calls)
rather than going through FastAPI/SSE, since the goal here is to verify the event
stream itself -- that individual nodes (reworder, search_executor, eval,
escalation_checker, deep_search) and their tool calls each report a "start" (with a
human-readable message) and an "end" (with duration_s), in real time as the pipeline
actually runs, not just that a final answer eventually comes back.

Uses its own separate db/collection (not verify_agentic_rag.py's) so it can run
independently of that script's lifecycle -- but still needs to run as its own
process with nothing else holding the embedded Qdrant/DuckDB store open at the same
time (same single-process constraint as every other script in this directory).

Usage:
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/verify_live_status.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8")  # model output can include non-ASCII (emoji, etc.)

import edenview_ingestion  # noqa: F401 -- must be imported before any docling.* import

from edenview_ingestion import catalog, pipeline

SAMPLE_PDF = "data/Finance_act_mini/ITA2025_definitions_excerpt.pdf"
DB_NAME = "verify-live-status-db"
COLLECTION = "verify-live-status-v1"

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


async def main() -> int:
    from docling.datamodel.pipeline_options import TableFormerMode

    from edenview_ingestion.docling_parsing import ExtractionConfig, StorageConfig
    from edenview_ingestion.settings import get_documents_dir
    from edenview_RAG.agentic_rag import RetrievalScope
    from edenview_RAG.agentic_rag.runtime import run_turn_stream

    fast_config = ExtractionConfig(
        table_mode=TableFormerMode.FAST,
        storage=StorageConfig(base_dir=str(get_documents_dir())),
    )
    print(f"Ingesting {SAMPLE_PDF} into db={DB_NAME!r}...")
    result = pipeline.ingest_document(SAMPLE_PDF, DB_NAME, COLLECTION, "hybrid_docling", extraction_config=fast_config)
    print(f"  {COLLECTION}: {result.embedded_count} embedded")

    session_id = str(uuid.uuid4())
    scope = RetrievalScope(collection_names=[COLLECTION], top_k=5, use_reranker=True)

    print(f"\n[live-status run, session_id={session_id}]")
    print(f"  query: {QUERY!r}")

    seen_nodes: set[str] = set()
    start_events = 0
    end_events = 0
    thinking_events = 0
    result_event = None
    t0 = time.time()

    async for event in run_turn_stream(QUERY, scope, session_id):
        elapsed = time.time() - t0
        etype = event.get("type")
        if etype == "status":
            node = event.get("node")
            phase = event.get("phase")
            if node:
                seen_nodes.add(node)
            if phase == "start":
                start_events += 1
                print(f"  [{elapsed:6.1f}s] START {node or '(root)'}: {event.get('message')}")
            elif phase == "end":
                end_events += 1
                dur = event.get("duration_s")
                dur_str = f"{dur:.2f}s" if dur is not None else "?"
                print(f"  [{elapsed:6.1f}s] END   {node or '(root)'}  ({dur_str})")
            else:
                print(f"  [{elapsed:6.1f}s] STATUS (root-relayed): {event.get('message')}")
        elif etype == "thinking":
            thinking_events += 1
        elif etype == "result":
            result_event = event
            print(f"  [{elapsed:6.1f}s] RESULT received")

    check("at least one status 'start' event with a node name", bool(seen_nodes), f"nodes seen: {seen_nodes}")
    check("decompose node tracked", "decompose" in seen_nodes, f"nodes seen: {seen_nodes}")
    check("subquestion_orchestrator node tracked", "subquestion_orchestrator" in seen_nodes, f"nodes seen: {seen_nodes}")
    # reworder is only invoked on a genuine retry (Eval said needs_requery) under
    # the decompose/subquestion_loop design -- NOT asserted here, since a run
    # where Eval succeeds on the first pass for every sub-question legitimately
    # never calls it at all. See agent.py's SubquestionLoop docstring.
    check(
        "search_executor or vector_search node tracked",
        "search_executor" in seen_nodes or "vector_search" in seen_nodes,
        f"nodes seen: {seen_nodes}",
    )
    check("eval node tracked", "eval" in seen_nodes, f"nodes seen: {seen_nodes}")
    check("start/end event counts roughly balanced", abs(start_events - end_events) <= 2, f"start={start_events} end={end_events}")
    check("at least one thinking event streamed", thinking_events > 0, f"count={thinking_events}")
    check("final result event received", result_event is not None)
    if result_event:
        check("result has non-empty answer", bool(result_event.get("answer", "").strip()))
        check("result has at least one citation", len(result_event.get("citations", [])) > 0)

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

    print(f"\n{'=' * 60}\nlive-status verification: {_passed} passed, {_failed} failed\n{'=' * 60}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
