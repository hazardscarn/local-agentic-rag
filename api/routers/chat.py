"""Minimal RAG chat endpoint -- reuses edenview_RAG.retrieval.search()/search_db()
exactly like api/routers/search.py, then one Ollama call over the retrieved context
via generate_answer(). Not the future ADK Retriever->Relevance->Answer agent loop
(still "not started" per edenview_progress.md) -- a real chat experience
now, without pretending it's more than one retrieval pass + one LLM call.

Every turn is persisted to the catalog's chat_sessions/chat_messages tables (see
edenview_ingestion.catalog.chat_crud) so a conversation survives a page reload --
POST /chat creates a session lazily when no session_id is given."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from edenview_ingestion import catalog
from edenview_ingestion.settings import get_model, get_ollama_host
from edenview_RAG import agentic_rag
from edenview_RAG.agentic_rag.runtime import run_turn, run_turn_stream
from edenview_RAG.retrieval import RetrievalConfig, generate_answer, reword_query_for_retrieval, search, search_db

from ..schemas import ChatRequest, ChatResponse, ChatSessionDetail

router = APIRouter(tags=["chat"])

_NO_CONTEXT_ANSWER = "No relevant information found in the selected collection(s)."
# Simple RAG's generate_answer() is a single, stateless Ollama call with no session
# management of its own (unlike the agentic tier's ADK session) -- capped here rather
# than sending unbounded history, since a long session's full transcript could push a
# small local model's default context window (no num_ctx override on this path)
# straight into truncation. 6 messages = last 3 user/assistant turns.
_MAX_HISTORY_MESSAGES = 6


def _get_or_create_session(body: ChatRequest) -> str:
    if body.session_id:
        try:
            catalog.chat_crud.get_session(body.session_id)
        except catalog.NotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return body.session_id
    return catalog.chat_crud.create_session(body.query).session_id


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if not body.db_name and not body.collection_names:
        raise HTTPException(400, "Provide either db_name or collection_names")

    session_id = _get_or_create_session(body)
    # Fetched BEFORE appending this turn's user message -- prior turns only, the
    # current question is passed to generate_answer()/run_turn() separately.
    prior_messages = catalog.chat_crud.list_messages(session_id)
    catalog.chat_crud.append_message(session_id, "user", body.query)

    if body.agentic:
        # edenview_RAG.agentic_rag's ADK-based pipeline -- see that package for the
        # full node breakdown (no effort tiers, one flat pipeline). asyncio.run() is
        # safe here even though this is a sync `def` route: FastAPI already runs sync
        # routes in its own threadpool, so this doesn't block the event loop, and it
        # avoids restructuring the non-agentic branch below (still fully synchronous)
        # into `async def` just for this mode.
        scope = agentic_rag.RetrievalScope(
            collection_names=body.collection_names,
            db_name=body.db_name,
            file_hashes=body.file_hashes,
            strategy=body.strategy,
            top_k=body.top_k,
            use_reranker=body.use_reranker,
        )
        model = agentic_rag.get_agent_model_name()
        answer, thinking, hits = asyncio.run(run_turn(body.query, scope, session_id))
        citations_json = [hit.model_dump(mode="json") for hit in hits]
    else:
        thinking = None
        model = body.chat_model or get_model("chat_llm")
        history = [{"role": m.role, "content": m.content} for m in prior_messages[-_MAX_HISTORY_MESSAGES:]]
        # Retrieval has no notion of conversation -- a raw follow-up like "what about
        # the second one" would otherwise be embedded/searched on literally and match
        # nothing. Only rewritten when there IS history (see
        # reword_query_for_retrieval's docstring); the ORIGINAL body.query still goes
        # to generate_answer() below so the final answer is grounded in what the user
        # actually asked.
        retrieval_query = reword_query_for_retrieval(body.query, history, model, get_ollama_host())
        config = RetrievalConfig(top_k=body.top_k, use_reranker=body.use_reranker)
        if body.collection_names:
            hits = search(body.collection_names, retrieval_query, config, body.file_hashes, body.strategy)
        else:
            hits = search_db(body.db_name, retrieval_query, config, body.file_hashes, body.strategy)

        if not hits:
            answer = _NO_CONTEXT_ANSWER
            citations_json = []
        else:
            answer, thinking = generate_answer(body.query, hits, model, get_ollama_host(), history=history)
            citations_json = [hit.model_dump(mode="json") for hit in hits]

    # Not persisted -- thinking is per-turn reasoning narration for the UI, not part
    # of the durable chat transcript (ChatMessageRecord has no column for it).
    catalog.chat_crud.append_message(session_id, "assistant", answer, citations=citations_json, model_used=model)
    catalog.chat_crud.touch_session(session_id)

    return ChatResponse(answer=answer, citations=hits, model_used=model, session_id=session_id, thinking=thinking)


@dataclass
class _ActiveTurn:
    """One in-flight agentic turn, keyed by session_id in `_active_turns` below.
    `events` is a full replay buffer (oldest first) -- what makes reattachment
    possible: a client that (re)subscribes mid-turn (page reload, switching chats
    and back, or just never having been the tab that started it) gets caught up
    from `events` before continuing to receive whatever's pushed live afterward.
    In-memory only, process-wide -- a backend restart loses in-flight tracking,
    the same accepted limitation this project's ingestion-job cancellation already
    has (see edenview_progress.md's deferred items)."""

    events: list[dict] = field(default_factory=list)
    subscribers: list["asyncio.Queue"] = field(default_factory=list)
    done: bool = False


# session_id -> its one in-flight turn, if any. A session can only ever have one
# turn running at a time (this pipeline has no concept of concurrent turns on the
# same session), so keying by session_id alone is unambiguous.
_active_turns: dict[str, _ActiveTurn] = {}


def _broadcast(session_id: str, event: dict) -> None:
    turn = _active_turns.get(session_id)
    if turn is None:
        return
    turn.events.append(event)
    for q in turn.subscribers:
        q.put_nowait(event)


async def _run_turn_and_broadcast(query: str, scope, session_id: str, model: str) -> None:
    """Runs the turn exactly once, regardless of how many clients end up
    subscribed to it (the original POST /chat/stream caller, plus anyone who
    later reattaches via GET /chat/stream/{session_id}) -- persists the assistant
    message on the result event exactly as the old single-subscriber
    event_generator did, then broadcasts every event instead of yielding it
    directly to one caller. Requires `_active_turns[session_id]` to already
    exist (chat_stream sets it synchronously before scheduling this as a task) --
    this function only looks it up, never creates it, to avoid a race between
    task scheduling and the first subscriber attaching."""
    try:
        async for event in run_turn_stream(query, scope, session_id):
            if event["type"] == "result":
                catalog.chat_crud.append_message(
                    session_id, "assistant", event["answer"], citations=event["citations"], model_used=model
                )
                catalog.chat_crud.touch_session(session_id)
                event = {**event, "model_used": model, "session_id": session_id}
            _broadcast(session_id, event)
    finally:
        turn = _active_turns.get(session_id)
        if turn is not None:
            turn.done = True
            for q in turn.subscribers:
                q.put_nowait(None)  # sentinel -- tells each subscriber's drain loop to stop
        asyncio.create_task(_cleanup_active_turn(session_id))


async def _cleanup_active_turn(session_id: str) -> None:
    # A short grace period rather than immediate removal -- a client reattaching
    # in the same instant a turn finishes should still get the full replay
    # (including the result event) instead of a bare "not_running" that would
    # undersell what just happened moments ago.
    await asyncio.sleep(30)
    _active_turns.pop(session_id, None)


async def _subscribe(session_id: str):
    """Shared by POST /chat/stream (the turn's first subscriber) and
    GET /chat/stream/{session_id} (a reattaching one) -- replays whatever's
    already been emitted, then keeps streaming live until the turn ends. Yields
    exactly one {"type": "not_running"} event and closes if no turn is in
    flight for this session_id at all (already finished, never started, or its
    grace period already expired) -- cheap and safe to call unconditionally on
    every session load."""
    turn = _active_turns.get(session_id)
    if turn is None:
        yield f"data: {json.dumps({'type': 'not_running'})}\n\n"
        return

    for event in list(turn.events):
        yield f"data: {json.dumps(event)}\n\n"
    if turn.done:
        return

    q: asyncio.Queue = asyncio.Queue()
    turn.subscribers.append(q)
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        turn.subscribers.remove(q)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """SSE variant of POST /chat, agentic mode only -- forwards every node/tool's
    live status (a real run can genuinely take several minutes: reword + search +
    up to max_iterations eval/deep-search rounds + answer formatting, per
    edenview_RAG.agentic_rag.runtime.run_turn_stream's own docstring) as
    {"type": "status", "node": ..., "phase": "start"|"end", "message": ...,
    "duration_s": ...} events -- granular down to individual tool calls, not just
    top-level agent phases, since run_turn_stream's queue-based mechanism reaches
    every level of the pipeline (see runtime.py/callbacks.py for why a plain event
    stream alone can't do this). Ends with one "result" event carrying the same
    {answer, citations} shape POST /chat returns. The frontend's Agentic RAG mode
    uses this; POST /chat's agentic branch stays available too for scripts/tests
    where latency doesn't warrant streaming.

    The actual turn runs via _run_turn_and_broadcast() as a background task, and
    this route becomes just its first subscriber (via _subscribe()) -- the same
    path GET /chat/stream/{session_id} uses to reattach later. This is what lets
    a page reload or a switch to a different chat and back pick the live view
    back up instead of showing nothing for a turn that's still genuinely running
    server-side (confirmed directly: dropping the client connection does not
    stop the turn -- it keeps running and persists its answer regardless)."""
    if not body.db_name and not body.collection_names:
        raise HTTPException(400, "Provide either db_name or collection_names")
    if not body.agentic:
        raise HTTPException(400, "POST /chat/stream is for agentic requests only -- use POST /chat otherwise")

    session_id = _get_or_create_session(body)
    catalog.chat_crud.append_message(session_id, "user", body.query)
    scope = agentic_rag.RetrievalScope(
        collection_names=body.collection_names,
        db_name=body.db_name,
        file_hashes=body.file_hashes,
        strategy=body.strategy,
        top_k=body.top_k,
        use_reranker=body.use_reranker,
    )
    model = agentic_rag.get_agent_model_name()

    # Registered synchronously, before scheduling the task -- avoids a race where
    # _subscribe() (called via StreamingResponse just below) could run before the
    # entry exists and wrongly report "not_running".
    _active_turns[session_id] = _ActiveTurn()
    asyncio.create_task(_run_turn_and_broadcast(body.query, scope, session_id, model))

    return StreamingResponse(_subscribe(session_id), media_type="text/event-stream")


@router.get("/chat/stream/{session_id}")
async def chat_stream_reattach(session_id: str):
    """Reattaches to a still-running agentic turn on this session, if one exists
    -- replays every event emitted so far, then continues live exactly like
    POST /chat/stream's own connection does, via the same _subscribe(). Yields a
    single {"type": "not_running"} event and closes immediately if nothing is
    actually in flight (session already finished, never had one, or its 30s
    grace period already elapsed) -- safe to call on every session load without
    checking first."""
    return StreamingResponse(_subscribe(session_id), media_type="text/event-stream")


@router.get("/chat/sessions", response_model=list[catalog.ChatSessionRecord])
def list_chat_sessions(limit: int = 10, offset: int = 0):
    """Most-recently-updated-first, paginated -- backs the sidebar's "last N + Load
    more" (default 10 per page). The frontend infers "there might be more" from
    getting back a full page (`len(result) == limit`), not a separate flag."""
    return catalog.chat_crud.list_sessions(limit=limit, offset=offset)


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionDetail)
def get_chat_session(session_id: str):
    try:
        session = catalog.chat_crud.get_session(session_id)
    except catalog.NotFoundError as e:
        raise HTTPException(404, str(e)) from e
    messages = catalog.chat_crud.list_messages(session_id)
    return ChatSessionDetail(session_id=session.session_id, title=session.title, messages=messages)


@router.delete("/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(session_id: str):
    catalog.chat_crud.delete_session(session_id)
