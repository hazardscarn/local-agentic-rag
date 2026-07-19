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

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from edenview_ingestion import catalog
from edenview_ingestion.settings import get_model, get_ollama_host
from edenview_RAG import agentic_rag
from edenview_RAG.agentic_rag.runtime import run_turn, run_turn_stream
from edenview_RAG.retrieval import RetrievalConfig, generate_answer, search, search_db

from ..schemas import ChatRequest, ChatResponse, ChatSessionDetail

router = APIRouter(tags=["chat"])

_NO_CONTEXT_ANSWER = "No relevant information found in the selected collection(s)."


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
    catalog.chat_crud.append_message(session_id, "user", body.query)

    if body.agentic:
        # edenview_RAG.agentic_rag's ADK-based loop -- see that package for what each
        # effort tier does. asyncio.run() is safe here even though this is a sync
        # `def` route: FastAPI already runs sync routes in its own threadpool, so this
        # doesn't block the event loop, and it avoids restructuring the non-agentic
        # branch above (still fully synchronous) into `async def` just for this mode.
        scope = agentic_rag.RetrievalScope(
            collection_names=body.collection_names,
            db_name=body.db_name,
            file_hashes=body.file_hashes,
            strategy=body.strategy,
            top_k=body.top_k,
            use_reranker=body.use_reranker,
        )
        model = agentic_rag.get_agent_model_name()
        answer, thinking, hits = asyncio.run(run_turn(body.query, scope, body.effort, session_id))
        citations_json = [hit.model_dump(mode="json") for hit in hits]
    else:
        thinking = None
        config = RetrievalConfig(top_k=body.top_k, use_reranker=body.use_reranker)
        if body.collection_names:
            hits = search(body.collection_names, body.query, config, body.file_hashes, body.strategy)
        else:
            hits = search_db(body.db_name, body.query, config, body.file_hashes, body.strategy)

        model = body.chat_model or get_model("chat_llm")
        if not hits:
            answer = _NO_CONTEXT_ANSWER
            citations_json = []
        else:
            answer = generate_answer(body.query, hits, model, get_ollama_host())
            citations_json = [hit.model_dump(mode="json") for hit in hits]

    # Not persisted -- thinking is per-turn reasoning narration for the UI, not part
    # of the durable chat transcript (ChatMessageRecord has no column for it).
    catalog.chat_crud.append_message(session_id, "assistant", answer, citations=citations_json, model_used=model)
    catalog.chat_crud.touch_session(session_id)

    return ChatResponse(answer=answer, citations=hits, model_used=model, session_id=session_id, thinking=thinking)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """SSE variant of POST /chat, agentic mode only -- forwards ADK's own event
    stream as short "status" lines (a "high" tier run can genuinely take 30-90+
    seconds: reframe + dispatch + up to max_iterations critic/refiner rounds +
    answer, per edenview_RAG.agentic_rag.runtime.run_turn_stream's own docstring),
    ending with one "result" event carrying the same {answer, citations} shape
    POST /chat returns. The frontend's Agentic RAG mode uses this; POST /chat's
    agentic branch stays available too for scripts/tests and for "low" tier where
    latency doesn't warrant streaming."""
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

    async def event_generator():
        async for event in run_turn_stream(body.query, scope, body.effort, session_id):
            if event["type"] == "result":
                catalog.chat_crud.append_message(
                    session_id, "assistant", event["answer"], citations=event["citations"], model_used=model
                )
                catalog.chat_crud.touch_session(session_id)
                event = {**event, "model_used": model, "session_id": session_id}
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/chat/sessions", response_model=list[catalog.ChatSessionRecord])
def list_chat_sessions():
    return catalog.chat_crud.list_sessions()


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
