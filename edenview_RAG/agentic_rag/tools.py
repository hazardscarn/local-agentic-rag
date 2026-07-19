"""FunctionTools for the agentic RAG loop. Tools stay simple -- they return retrieval
results, they don't manage shared state directly (citation harvesting and tool-call
capping happen in callbacks.py, registered separately on each agent)."""

from __future__ import annotations

from google.adk.tools.tool_context import ToolContext

from edenview_RAG.retrieval import RetrievalConfig, RetrievalHit, search, search_db

from .config import RetrievalScope


def _scope(tool_context: ToolContext) -> RetrievalScope:
    return RetrievalScope(**tool_context.state["scope"])


def _format_hits_for_llm(hits: list[RetrievalHit]) -> str:
    """Plain numbered-text block -- deliberately mirrors
    edenview_RAG/retrieval/generate.py's `_format_context()` exactly (the proven-
    working format today's non-agentic /chat already feeds a small local model), not
    a list of structured dicts. Reproduced directly why this matters: returning
    results as JSON-shaped dicts (index/chunk_id/page_no/score/... per hit) instead of
    plain numbered prose made qwen3.5:4b consistently treat the tool response as "data
    to process/describe" rather than "context to answer from" -- e.g. reasoning
    "the user is asking me to extract information from this text" instead of just
    answering the original question. A single plain-text block reads the same to the
    model whether it arrives via a tool response or directly in a prompt, matching the
    one style already confirmed to work well in this codebase."""
    return "\n\n".join(f"[{i}] {hit.context_text}" for i, hit in enumerate(hits, start=1))


def retrieve(query: str, tool_context: ToolContext) -> dict:
    """Search the user's selected document scope for chunks relevant to `query`.
    Call this whenever you need to look something up in the documents before
    answering. Returns numbered snippets ordered from MOST to LEAST relevant
    (snippet [1] is the best match) -- cite them in your answer using their number,
    like [1] or [2][3]."""
    scope = _scope(tool_context)
    config = RetrievalConfig(top_k=scope.top_k, use_reranker=scope.use_reranker)
    hits = (
        search(scope.collection_names, query, config, scope.file_hashes, scope.strategy)
        if scope.collection_names
        else search_db(scope.db_name, query, config, scope.file_hashes, scope.strategy)
    )
    # Full hit data (collection_name/strategy/headings/bbox/images -- everything the
    # frontend's citation display needs, but the LLM doesn't) stashed under a temp:
    # key for callbacks.harvest_citations to pick up right after this call returns --
    # kept out of the LLM-visible response below entirely.
    tool_context.state["temp:last_hits"] = {h.chunk_id: h.model_dump(mode="json") for h in hits}
    if not hits:
        return {"status": "no_results", "snippets": ""}
    return {"status": "ok", "snippets": _format_hits_for_llm(hits)}


def get_page_context(
    file_hash: str, collection_name: str, page_no: int, include_adjacent: bool, tool_context: ToolContext
) -> dict:
    """Reconstructs a page's full text from every chunk on it, for when a retrieved
    snippet needs more surrounding context than the chunk alone gives. Set
    include_adjacent=True to also pull the page before and after. file_hash and
    collection_name come from a citation's own metadata (a retrieve() result)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    from edenview_ingestion import vectorstore
    from edenview_ingestion.vectorstore.client import client_lock

    pages = [page_no - 1, page_no, page_no + 1] if include_adjacent else [page_no]
    query_filter = Filter(
        must=[
            FieldCondition(key="file_hash", match=MatchValue(value=file_hash)),
            FieldCondition(key="page_no", match=MatchAny(any=pages)),
        ]
    )
    # Same client_lock pattern every other real Qdrant call in this codebase uses
    # (embedded Qdrant's local sqlite-backed persistence isn't safe for concurrent
    # unsynchronized access -- see edenview_ingestion/vectorstore/client.py).
    with client_lock:
        points, _ = vectorstore.get_client().scroll(
            collection_name=collection_name, scroll_filter=query_filter, limit=200, with_payload=True
        )
    if not points:
        return {"status": "no_results", "text": ""}
    points.sort(key=lambda p: p.payload.get("page_no", 0))
    text = "\n\n".join(p.payload.get("text", "") for p in points)
    return {"status": "ok", "text": text}


async def inspect_image(chunk_id: str, tool_context: ToolContext) -> dict:
    """Loads a picture/table crop belonging to a previously-retrieved chunk so you can
    look at it directly on your next reasoning turn -- use this when a result's text
    alone might be missing information a chart, table, or figure shows (e.g. specific
    numbers, a diagram's structure). Only call this for a chunk_id whose result you
    already have from retrieve/get_page_context. Only usable when the configured
    model is actually multimodal -- this tool is only registered when it is."""
    citations: dict = tool_context.state.get("citations", {})
    hit = citations.get(chunk_id)
    if not hit or not hit.get("images"):
        return {"status": "no_image", "chunk_id": chunk_id}

    from pathlib import Path

    from google.genai import types

    image = hit["images"][0]
    # Already an absolute path -- confirmed via api/routers/files.py's own docstring
    # ("Absolute path from a chunk's images[].image_path"), no /files HTTP round-trip
    # needed for server-side tool code.
    data = Path(image["image_path"]).read_bytes()
    part = types.Part.from_bytes(data=data, mime_type="image/png")
    artifact_id = f"chunk_img_{chunk_id}.png"
    # A tool cannot return raw image bytes in its response dict directly -- ADK raises
    # an error (confirmed against a real Google codelab on this exact pattern). It
    # must be saved as an Artifact and only the artifact's id returned; the actual
    # Part gets attached to the model's next turn by
    # callbacks.inject_pending_images (a before_model_callback), not by this tool.
    await tool_context.save_artifact(filename=artifact_id, artifact=part)
    return {"status": "ok", "tool_response_artifact_id": artifact_id, "caption": image.get("caption")}


def exit_loop(tool_context: ToolContext) -> dict:
    """Call this ONLY when the current findings are sufficient to answer the
    question well -- this stops the refinement loop. Do not call any other tool
    after calling this one."""
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
    return {"status": "ok"}
