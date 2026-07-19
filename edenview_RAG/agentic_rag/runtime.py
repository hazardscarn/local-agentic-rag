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

from functools import lru_cache

from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from edenview_ingestion.settings import get_workspace_root
from edenview_RAG.retrieval import RetrievalHit

from . import agent as agent_trees
from . import prompts
from .config import Effort, RetrievalScope

_APP_NAME = "edenview_agentic_rag"
_USER_ID = "local"  # single-user local app, matching every other part of this codebase

_ADK_DB_PATH = get_workspace_root() / "adk_sessions.db"
_session_service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{_ADK_DB_PATH}")
# In-memory (not disk-backed) on purpose -- retrieved-chunk images used by
# tools.inspect_image only need to live for the duration of one turn, not survive a
# restart, so there's no need for a custom disk-backed ArtifactService (core ADK only
# ships InMemoryArtifactService/GcsArtifactService, neither local-disk).
_artifact_service = InMemoryArtifactService()

_BUILDERS = {
    "low": agent_trees.build_low_agent,
    "medium": lambda: agent_trees.build_medium_agent(),
    "high": lambda: agent_trees.build_high_agent(),
}


@lru_cache(maxsize=3)
def _runner_for(effort: Effort) -> Runner:
    return Runner(
        agent=_BUILDERS[effort](),
        app_name=_APP_NAME,
        session_service=_session_service,
        artifact_service=_artifact_service,
    )


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


async def _run_once(query: str, scope: RetrievalScope, effort: Effort, session_id: str) -> tuple[str, str, list[RetrievalHit]]:
    runner = _runner_for(effort)
    await _ensure_session(session_id)

    final_text = ""
    thinking_chunks: list[str] = []
    async for event in runner.run_async(
        user_id=_USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        state_delta={"scope": scope.model_dump(), "citations": {}},
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
    citations_raw = session.state.get("citations", {}) if session else {}
    citations = [RetrievalHit(**d) for d in citations_raw.values()]
    return final_text, "\n\n".join(thinking_chunks), citations


async def run_turn(query: str, scope: RetrievalScope, effort: Effort, session_id: str) -> tuple[str, str, list[RetrievalHit]]:
    """Runs one turn against a persistent ADK session (reused across turns/restarts
    for conversational memory), reading the final response + this turn's own
    reasoning narration + accumulated citations back out. `state_delta` on run_async
    both seeds/refreshes "scope" (per-request, can legitimately change turn to turn)
    and resets "citations" to empty at the start of every turn -- citations must
    reflect only THIS turn's retrieval, not the whole session's history, even though
    conversation memory itself does persist.

    Returns (answer, thinking, citations) -- `thinking` is every agent step's own
    `thought=True` narration concatenated in event order (reframe/critic/refiner/
    root reasoning), kept separate from `answer` so a caller can show it as an
    expandable "thinking" section rather than mixing it into the real answer (see
    _extract_final_text's docstring for the bug this separation fixes).

    Retries the whole turn ONCE if the final answer comes back empty -- a real,
    recurring characteristic of this local model observed throughout development
    (occasionally an agent step's entire response lands in native "thinking" content
    with nothing in regular content, at every position in the pipeline this project
    tried: low tier's research step, the critic step, the final answer step). Not a
    fix for that underlying nondeterminism -- retrying the same prompt on the same
    small model doesn't guarantee a better roll -- but empirically a second attempt
    frequently succeeds where the first didn't, and an empty response reaching the
    user is a worse outcome than the modest extra latency of one retry."""
    _reset_litellm_logging_worker()
    text, thinking, citations = await _run_once(query, scope, effort, session_id)
    if not text.strip():
        _reset_litellm_logging_worker()
        text, thinking, citations = await _run_once(query, scope, effort, session_id)
    return text, thinking, citations


async def run_turn_stream(query: str, scope: RetrievalScope, effort: Effort, session_id: str):
    """Same turn as run_turn(), but yields live progress as ADK's own event stream
    arrives, instead of only returning once everything is done -- for the "high"
    tier especially, a run can genuinely take 30-90+ seconds (reframe + dispatch +
    up to `max_iterations` critic/refiner rounds + answer), which looks broken in a
    UI with no feedback for that long. Yields plain dicts:
    {"type": "status", "message": <human-readable line>} for each tool call/agent
    phase observed, {"type": "thinking", "message": <reasoning text>} for each
    step's own `thought=True` narration as it arrives (each agent step's reasoning
    is its own chunk -- Ollama/ADK doesn't stream token-by-token here, so this is
    "one thinking chunk per model call," not a live token feed, but it's still
    real-time relative to the whole run), and exactly one final
    {"type": "result", "answer": ..., "thinking": ..., "citations": [...]} once the
    run completes ("thinking" there is every chunk already streamed, concatenated,
    so a client that only listens for "result" still gets it). No retry-on-empty
    here (unlike run_turn) -- a caller consuming a live stream has already seen the
    status trail and would need special handling to silently restart a whole run
    mid-stream; retry logic stays in the non-streaming path."""
    _reset_litellm_logging_worker()
    runner = _runner_for(effort)
    await _ensure_session(session_id)

    final_text = ""
    thinking_chunks: list[str] = []
    async for event in runner.run_async(
        user_id=_USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        state_delta={"scope": scope.model_dump(), "citations": {}},
    ):
        for call in event.get_function_calls() or []:
            label = prompts.STATUS_LABELS.get(call.name, f"Working ({call.name})...")
            yield {"type": "status", "message": label}
        if event.author and event.author != "user":
            tier_label = prompts.AGENT_STATUS_LABELS.get(event.author)
            if tier_label:
                yield {"type": "status", "message": tier_label}
        if event.content and event.content.parts:
            thought = _extract_thought_text(event.content.parts)
            if thought:
                thinking_chunks.append(thought)
                yield {"type": "thinking", "message": thought}
        if event.is_final_response() and event.content and event.content.parts:
            text = _extract_final_text(event.content.parts)
            if text:
                final_text = text

    session = await _session_service.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id)
    citations_raw = session.state.get("citations", {}) if session else {}
    citations = [RetrievalHit(**d) for d in citations_raw.values()]
    yield {
        "type": "result",
        "answer": final_text,
        "thinking": "\n\n".join(thinking_chunks),
        "citations": [c.model_dump(mode="json") for c in citations],
    }
