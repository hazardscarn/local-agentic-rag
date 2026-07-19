"""Runner + DatabaseSessionService wiring + run_turn()/run_turn_stream() -- the async
orchestration layer kept separate from agent.py (pure tree construction).

Session persistence: a dedicated local SQLite file (via aiosqlite), NOT the project's
existing catalog.duckdb -- ADK's DatabaseSessionService needs an async SQLAlchemy
driver and there's no supported async DuckDB one. Uses the SAME session_id as the
existing DuckDB chat_sessions row (see api/routers/chat.py) so the two stores track
one conversation in lockstep: DuckDB stays the durable, UI-facing source of truth for
session titles/message text/citations; this SQLite file is ADK's own internal
conversation/state bookkeeping so the agent actually has multi-turn memory.
"""

from __future__ import annotations

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from edenview_ingestion.settings import get_workspace_root
from edenview_RAG.retrieval import RetrievalHit

from . import agent as agent_trees
from . import prompts
from .callbacks import _status_queue_var
from .config import RetrievalScope

_APP_NAME = "edenview_agentic_rag"
_USER_ID = "local"  # single-user local app, matching every other part of this codebase

_ADK_DB_PATH = get_workspace_root() / "adk_sessions.db"
_session_service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{_ADK_DB_PATH}")

# One Runner for the one pipeline -- no effort tiers anymore, so no per-tier cache.
_runner = Runner(agent=agent_trees.root_agent, app_name=_APP_NAME, session_service=_session_service)


def _extract_final_text(parts) -> str:
    """The root agent's `AgentTool(..., skip_summarization=True)` call produces a
    final event whose content.parts shape is NOT consistent run-to-run (confirmed by
    direct reproduction, running the identical query 3 times): sometimes a plain
    `text` part mirroring the sub-agent's answer is present alongside the raw
    `function_response` part, sometimes only the `function_response` part is -- in
    which case relying on `p.text` alone silently returns an empty string even though
    the sub-agent's real answer is sitting in `function_response.response["result"]`.
    Checks plain text parts first, falls back to the function_response payload.

    Real bug fixed here: `p.thought` parts (qwen3.5's native "thinking" content) were
    never excluded, so a run where the model's own reasoning/planning narration
    ("I need to use the high_research tool... let me search for...") landed in a
    `thought=True` text part got returned as if it were the actual answer -- this is
    exactly the leaked-reasoning bug reported against a real "high" tier question.
    ADK's own `AgentTool.run_async` excludes thought parts the same way
    (`if not p.thought`, confirmed directly in the installed google-adk source) --
    this mirrors that, since our own extraction has to redo the same job."""
    text_parts = [p.text for p in parts if p.text and not getattr(p, "thought", False)]
    if text_parts:
        return "".join(text_parts)
    for p in parts:
        fr = getattr(p, "function_response", None)
        if fr and isinstance(fr.response, dict):
            result = fr.response.get("result")
            if isinstance(result, str) and result:
                return result
    return ""


def _extract_thought_text(parts) -> str:
    """Companion to _extract_final_text: pulls out ONLY the `thought=True` text parts
    -- the model's own internal reasoning/planning narration for this step. Surfaced
    separately (not discarded) so the UI can show it as an expandable "thinking"
    section, distinct from the actual final answer."""
    return "".join(p.text for p in parts if p.text and getattr(p, "thought", False))


async def _ensure_session(session_id: str) -> None:
    session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
    if session is None:
        await _session_service.create_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id, state={})


def _reset_litellm_logging_worker() -> None:
    """Real, reproduced bug: litellm.litellm_core_utils.logging_worker.
    GLOBAL_LOGGING_WORKER is a process-wide singleton whose internal asyncio.Queue is
    lazily created tied to whichever event loop happens to be running the first time
    it's used. `run_turn`/`run_turn_stream` are driven via `asyncio.run(...)` from a
    FastAPI sync route -- each call gets a brand-new event loop (that's how
    `asyncio.run` works), but the singleton is shared process-wide across every
    request regardless of which thread/loop handles it. Confirmed directly: the
    SECOND `asyncio.run`-driven call after the first raised `RuntimeError: <Queue ...>
    is bound to a different event loop` on every subsequent Ollama call, repeatedly,
    inside litellm's background logging task -- which doesn't crash the actual
    completion request itself (it's a fire-and-forget logging/telemetry worker we
    don't use), but is exactly the kind of noisy per-request failure a long-running
    server must not hit on every request after the first. Resetting the queue
    reference before each turn forces it to lazily rebuild against the current
    (correct) event loop next time it's touched."""
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        GLOBAL_LOGGING_WORKER._queue = None
    except Exception:
        pass  # best-effort -- never let a logging-plumbing reset break a real turn


def _ordered_citations(session) -> list[RetrievalHit]:
    """Builds the final citations list in the same order the answer's own
    inline [N] markers use, via state["final_citation_order"] (the
    {global_number: chunk_id} map callbacks.prepare_consolidated_findings
    writes) -- so citations[N-1] in the API response really is the chunk
    marker [N] in the answer text refers to. Required for the frontend's
    clickable inline citations (chat-message.tsx) to jump to the right source.
    Falls back to today's dict-insertion order if that key is absent -- e.g.
    Simple RAG's non-agentic /chat path, which never runs this pipeline's
    callbacks at all. Dict keys may come back as strings (JSON round-trip
    through session persistence), hence `key=int` rather than assuming int."""
    if not session:
        return []
    citations_raw: dict = session.state.get("citations", {})
    final_citation_order: dict = session.state.get("final_citation_order") or {}
    if not final_citation_order:
        return [RetrievalHit(**d) for d in citations_raw.values()]
    ordered = []
    for number in sorted(final_citation_order, key=int):
        chunk_id = final_citation_order[number]
        hit = citations_raw.get(chunk_id)
        if hit:
            ordered.append(RetrievalHit(**hit))
    return ordered


async def _run_once(query: str, scope: RetrievalScope, session_id: str) -> tuple[str, str, list[RetrievalHit]]:
    await _ensure_session(session_id)

    final_text = ""
    thinking_chunks: list[str] = []
    async for event in _runner.run_async(
        user_id=_USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        state_delta={"scope": scope.model_dump(), "citations": {}, "sub_answers": [], "final_answer": ""},
    ):
        if event.content and event.content.parts:
            thought = _extract_thought_text(event.content.parts)
            if thought:
                thinking_chunks.append(thought)
        if event.is_final_response() and event.content and event.content.parts:
            text = _extract_final_text(event.content.parts)
            if text:
                final_text = text

    session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
    citations = _ordered_citations(session)
    return final_text, "\n\n".join(thinking_chunks), citations


async def run_turn(query: str, scope: RetrievalScope, session_id: str) -> tuple[str, str, list[RetrievalHit]]:
    """Runs one turn against a persistent ADK session (reused across turns/restarts
    for conversational memory), reading the final response + this turn's own
    reasoning narration + accumulated citations back out. `state_delta` on run_async
    both seeds/refreshes "scope" (per-request, can legitimately change turn to turn)
    and resets "citations" to empty at the start of every turn -- citations must
    reflect only THIS turn's retrieval, not the whole session's history, even though
    conversation memory itself does persist.

    Returns (answer, thinking, citations) -- `thinking` is every agent step's own
    `thought=True` narration concatenated in event order (reworder/eval/deep_search/
    root reasoning), kept separate from `answer` so a caller can show it as an
    expandable "thinking" section rather than mixing it into the real answer (see
    _extract_final_text's docstring for the bug this separation fixes).

    If the outer-event-extracted answer comes back empty, first checks
    state["final_answer"] (populated durably by callbacks.finalize_answer,
    registered on answer_formatter, which reads its own generation regardless of
    `thought` flag -- safe there specifically since answer_formatter's only job
    is ever to produce user-facing prose, unlike other pipeline steps whose
    `thought` content is deliberately excluded to stop leaked reasoning from
    being mistaken for a real answer). This salvage covers the common case
    directly: a real, recurring characteristic of this local model is that its
    entire response sometimes lands in native "thinking" content with nothing in
    regular content -- when that happens to answer_formatter specifically, the
    text is still sitting in state, just not where the outer extraction looks.
    Only if state["final_answer"] is ALSO empty (a genuine double failure, not
    just a routing mismatch) does this retry the WHOLE turn once, as a last
    resort -- not a fix for that underlying nondeterminism, but an empty
    response reaching the user is a worse outcome than the modest extra latency
    of one full retry."""
    _reset_litellm_logging_worker()
    text, thinking, citations = await _run_once(query, scope, session_id)
    if not text.strip():
        session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
        salvaged = (session.state.get("final_answer", "") if session else "").strip()
        if salvaged:
            text = salvaged
        else:
            _reset_litellm_logging_worker()
            text, thinking, citations = await _run_once(query, scope, session_id)
    return text, thinking, citations


async def _run_turn_and_push(query: str, scope: RetrievalScope, session_id: str, queue: "asyncio.Queue") -> tuple[str, str]:
    """Runs one turn, pushing every live-status/thinking update into `queue` as it
    happens, instead of yielding directly -- this is what actually makes
    run_turn_stream's status granular.

    `queue` is carried via callbacks._status_queue_var, a contextvars.ContextVar,
    NOT session state -- confirmed by direct reproduction that state can't do this
    job (see that variable's own docstring for the full story: ADK's
    InMemorySessionService silently drops `temp:`-prefixed keys when AgentTool
    seeds a nested sub-agent's brand-new session, and a plain key would make our
    real DatabaseSessionService try to persist a raw Queue object). Setting the
    contextvar here, before calling run_async(), means every nested node inside
    query_pipeline (reworder, search_executor, eval, deep_search, and each of
    their own tool calls) sees the exact same queue object via
    callbacks.track_agent_start/track_agent_end/track_tool_start/track_tool_end,
    confirmed working with ~0.01s latency by direct isolated testing -- this is
    the actual fix for a real, confirmed ADK limitation: AgentTool.run_async()
    consumes and discards its own inner events, so the OUTER event stream this
    function also reads below only ever sees root's own function-call to
    query_pipeline and the final answer -- never anything from inside it.

    Always pushes a final `None` sentinel (even on error, via `finally`) so
    run_turn_stream's drain loop knows when to stop waiting."""
    final_text = ""
    thinking_chunks: list[str] = []
    token = _status_queue_var.set(queue)
    try:
        async for event in _runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=query)]),
            state_delta={"scope": scope.model_dump(), "citations": {}, "sub_answers": [], "final_answer": ""},
        ):
            for call in event.get_function_calls() or []:
                label = prompts.STATUS_LABELS.get(call.name, f"Working ({call.name})...")
                await queue.put({"type": "status", "message": label})
            if event.author and event.author != "user":
                tier_label = prompts.AGENT_STATUS_LABELS.get(event.author)
                if tier_label:
                    await queue.put({"type": "status", "message": tier_label})
            if event.content and event.content.parts:
                thought = _extract_thought_text(event.content.parts)
                if thought:
                    thinking_chunks.append(thought)
                    await queue.put({"type": "thinking", "message": thought})
            if event.is_final_response() and event.content and event.content.parts:
                text = _extract_final_text(event.content.parts)
                if text:
                    final_text = text
    finally:
        _status_queue_var.reset(token)
        await queue.put(None)  # sentinel -- always fires, success or exception
    return final_text, "\n\n".join(thinking_chunks)


async def run_turn_stream(query: str, scope: RetrievalScope, session_id: str):
    """Same turn as run_turn(), but yields live progress as it happens, instead of
    only returning once everything is done -- a run can genuinely take several
    minutes (reworder + search + up to `max_iterations` eval/deep_search rounds +
    answer formatting), which looks broken in a UI with no feedback for that long.
    Yields plain dicts: {"type": "status", "node": ..., "phase": "start"|"end",
    "message": ..., "duration_s": ...} for every agent/tool boundary anywhere in
    the pipeline (not just root-level -- see callbacks.track_agent_start/
    track_agent_end/track_tool_start/track_tool_end, which push these into the
    shared queue this function drains), {"type": "thinking", "message": ...} for
    each step's own `thought=True` narration, and exactly one final {"type":
    "result", "answer": ..., "thinking": ..., "citations": [...]} once the run
    completes.

    If the outer-event-extracted answer comes back empty, first checks
    state["final_answer"] (see run_turn's own docstring for the full mechanism
    -- callbacks.finalize_answer, registered on answer_formatter, captures its
    generation regardless of `thought` flag, since that node has no legitimate
    reasoning to hide). Only if that's ALSO empty does this retry the WHOLE
    turn once, same real, recurring nondeterminism run_turn's own docstring
    documents. A "trying again" status line is yielded before that full retry
    so it isn't a silent stall -- the cheap state salvage doesn't need one,
    since it doesn't re-run anything."""
    final_text = ""
    thinking_chunks: list[str] = []
    for attempt in range(2):
        if attempt == 1:
            yield {"type": "status", "message": "Didn't get a clear answer -- trying again..."}
        _reset_litellm_logging_worker()
        await _ensure_session(session_id)
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_run_turn_and_push(query, scope, session_id, queue))
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
            if item["type"] == "thinking":
                thinking_chunks.append(item["message"])
        final_text, _ = await task
        if not final_text.strip():
            session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
            salvaged = (session.state.get("final_answer", "") if session else "").strip()
            if salvaged:
                final_text = salvaged
        if final_text.strip():
            break

    session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
    citations = _ordered_citations(session)
    yield {
        "type": "result",
        "answer": final_text,
        "thinking": "\n\n".join(thinking_chunks),
        "citations": [c.model_dump(mode="json") for c in citations],
    }
