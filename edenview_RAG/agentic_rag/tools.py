"""FunctionTools for the agentic RAG pipeline. Tools stay simple -- they return
retrieval results, they don't manage shared state directly (citation harvesting and
tool-call capping happen in callbacks.py, registered separately on each agent).

Every docstring below is the ONLY description the LLM ever sees for that tool (ADK's
FunctionTool derives the tool description shown to the model directly from the
Python docstring/signature) -- written to state precisely what each tool does, what
each argument means, when to call it, and what its response looks like, not as an
afterthought once the function body worked."""

from __future__ import annotations

import re
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from edenview_RAG.retrieval import RetrievalConfig, RetrievalHit, search, search_db

from .config import RetrievalScope, get_agent_model_name, get_vision_model
from .errors import AgenticRAGError


def _scope(tool_context: ToolContext) -> RetrievalScope:
    """`state["scope"]` is always seeded by runtime.py's run_turn/run_turn_stream
    (via state_delta on every real request through the FastAPI app) -- there is no
    sensible default here, since which collections/db to search is inherently a
    required, request-specific input with no universal "right" answer. Only missing
    when driving this agent directly through `adk web` or its REST API without first
    seeding session state, which raises a bare, confusing `KeyError: 'scope'` by
    default -- this turns that into an actionable message instead."""
    try:
        return RetrievalScope(**tool_context.state["scope"])
    except KeyError:
        raise AgenticRAGError(
            "No 'scope' in session state -- this agent needs to know which "
            "collections/database to search before it can run a turn. When "
            "driving it directly (adk web or its REST API, bypassing this "
            "project's own FastAPI /chat route), seed it once when creating the "
            "session, e.g.: POST /apps/agentic_rag/users/<user>/sessions/<id> "
            'with body {"scope": {"collection_names": ["<a real collection '
            'name>"], "top_k": 5, "use_reranker": true}, "citations": {}}.'
        ) from None


def _format_hits_for_llm(hits: list[RetrievalHit]) -> str:
    """Plain numbered-text block -- deliberately mirrors
    edenview_RAG/retrieval/generate.py's `_format_context()` exactly (the proven-
    working format today's non-agentic /chat already feeds a small local model), not
    a list of structured dicts. Reproduced directly why this matters: returning
    results as JSON-shaped dicts (index/chunk_id/page_no/score/... per hit) instead of
    plain numbered prose made a small local model consistently treat the tool
    response as "data to process/describe" rather than "context to answer from". A
    single plain-text block reads the same to the model whether it arrives via a
    tool response or directly in a prompt, matching the one style already confirmed
    to work well in this codebase."""
    return "\n\n".join(f"[{i}] {hit.context_text}" for i, hit in enumerate(hits, start=1))


def _resolve_ref(ref: int, tool_context: ToolContext) -> Optional[dict]:
    """Resolves a `ref` (the [N] number the model already sees next to a finding in
    `{findings}`) back to the full RetrievalHit-json dict a prior vector_search call
    stashed in state["citations"], via state["ref_to_chunk_id"] (both written by
    callbacks.merge_hits_into_state). Every Deep Search tool below takes `ref`, never
    a raw chunk_id/file_hash -- the model was never shown a chunk_id, only the [N]
    number, so asking it to transcribe an ID it never saw would be a real
    transcription-fidelity risk on a small local model."""
    chunk_id = tool_context.state.get("ref_to_chunk_id", {}).get(ref)
    if chunk_id is None:
        return None
    return tool_context.state.get("citations", {}).get(chunk_id)


def _scroll_page_points(file_hash: str, collection_name: str, page_no: int, include_adjacent: bool) -> list:
    """Shared by get_pages_detailed/get_images/get_answer_from_images/
    get_answer_from_detailed_pages -- one Qdrant scroll filtered by file_hash +
    page_no (or page_no-1/page_no/page_no+1 if include_adjacent), sorted by page_no.
    Every chunk on the matched page(s) comes back, not just the one `ref` originally
    pointed to, since a page can have multiple chunks."""
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
    points.sort(key=lambda p: p.payload.get("page_no", 0))
    return points


def vector_search(query: str, tool_context: ToolContext) -> dict:
    """Searches the user's selected document scope for chunks relevant to `query`.
    `query` should be a single, focused question or topic -- if the user's original
    question covers multiple distinct topics, call this tool once per topic with a
    separate, focused query each time, never one call combining several topics.
    Always runs the full hybrid retrieval pipeline (dense + sparse + reranking), so
    there is no separate "quality" toggle to worry about.

    Returns numbered snippets ordered from MOST to LEAST relevant (snippet [1] is
    the best match) -- cite them in your answer using their number, like [1] or
    [2][3]. Returns {"status": "no_results", "snippets": ""} if nothing matched."""
    scope = _scope(tool_context)
    config = RetrievalConfig(top_k=scope.top_k, use_reranker=True)  # always reranked, not agent-gated
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


def get_pages_detailed(ref: int, include_adjacent: bool, tool_context: ToolContext) -> dict:
    """Reconstructs a finding's full page text from every chunk on that page, for
    when a finding's snippet looks cut off or you need more surrounding context than
    the snippet alone gives. `ref` is the [N] reference number of the finding whose
    page you want (e.g. pass 3 for finding [3]) -- not a file name or page number
    yourself, the tool looks those up from the finding automatically. Set
    include_adjacent=True to also pull the page immediately before and after.

    Returns {"status": "ok", "ref": ref, "text": "..."}, or {"status": "invalid_ref"}
    if `ref` doesn't match a real finding, or {"status": "no_page"} if that finding
    has no page number recorded."""
    hit = _resolve_ref(ref, tool_context)
    if hit is None:
        return {"status": "invalid_ref", "ref": ref}
    if hit.get("page_no") is None:
        return {"status": "no_page", "ref": ref}
    points = _scroll_page_points(hit["file_hash"], hit["collection_name"], hit["page_no"], include_adjacent)
    if not points:
        return {"status": "no_results", "ref": ref, "text": ""}
    text = "\n\n".join(p.payload.get("text", "") for p in points)
    return {"status": "ok", "ref": ref, "text": text}


def get_images(ref: int, tool_context: ToolContext) -> dict:
    """Looks for pictures, tables, or figures on a finding's page -- use this when a
    finding's page might contain a chart/table/figure with information the snippet's
    text doesn't cover. `ref` is the [N] reference number of the finding whose page
    to check (e.g. pass 3 for finding [3]). Every image found on that page is
    returned once each, even if multiple findings on the same page reference the
    same image.

    Returns {"status": "ok", "ref": ref, "images": [{"picture_id", "caption",
    "kind"}, ...]} if any were found, {"status": "no_images"} if the page has none,
    or {"status": "invalid_ref"} if `ref` doesn't match a real finding. After
    finding images here, call get_answer_from_images with the SAME `ref` and a
    specific question to actually get an answer from them."""
    hit = _resolve_ref(ref, tool_context)
    if hit is None:
        return {"status": "invalid_ref", "ref": ref}
    if hit.get("page_no") is None:
        return {"status": "no_page", "ref": ref}
    points = _scroll_page_points(hit["file_hash"], hit["collection_name"], hit["page_no"], include_adjacent=False)
    seen_ids: set = set()
    images = []
    for point in points:
        for image in point.payload.get("images", []):
            picture_id = image.get("picture_id")
            if picture_id and picture_id not in seen_ids:
                seen_ids.add(picture_id)
                images.append({"picture_id": picture_id, "caption": image.get("caption"), "kind": image.get("kind")})
    if not images:
        return {"status": "no_images", "ref": ref, "images": []}
    return {"status": "ok", "ref": ref, "images": images}


def grep(ref: int, pattern: str, regex: bool, tool_context: ToolContext) -> dict:
    """Searches for the EXACT, literal text of `pattern` within the one document a
    finding came from -- use this only when you need a specific term, section
    number, defined phrase, or figure verbatim, and a finding's snippet doesn't show
    it word-for-word (semantic search can miss or under-rank exact tokens even in a
    document it correctly identified as relevant). This is NOT a better version of
    search -- it only looks within the single document `ref` points to, never the
    whole document collection, and it only finds literal text, not related concepts.
    `ref` is the [N] reference number of the finding whose document to search (e.g.
    pass 3 for finding [3]). Set regex=True to treat `pattern` as a regular
    expression instead of literal text; case-insensitive either way.

    Returns {"status": "ok", "ref": ref, "matches": ["...surrounding text...", ...]}
    (each match includes roughly a sentence of context on either side, capped at 20
    matches), {"status": "no_matches"} if nothing matched, {"status":
    "invalid_pattern"} if `regex=True` and `pattern` isn't valid regex, or
    {"status": "invalid_ref"} if `ref` doesn't match a real finding."""
    hit = _resolve_ref(ref, tool_context)
    if hit is None:
        return {"status": "invalid_ref", "ref": ref}

    from edenview_ingestion.settings import get_documents_dir

    doc_path = get_documents_dir() / "cache" / hit["doc_stem"] / "doc.md"
    if not doc_path.exists():
        return {"status": "no_matches", "ref": ref, "matches": []}
    text = doc_path.read_text(encoding="utf-8")

    if regex:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"status": "invalid_pattern", "ref": ref, "error": str(e)}
    else:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)

    matches = []
    for m in compiled.finditer(text):
        if len(matches) >= 20:
            break
        start, end = max(0, m.start() - 100), min(len(text), m.end() + 100)
        matches.append(text[start:end].strip())
    if not matches:
        return {"status": "no_matches", "ref": ref, "matches": []}
    return {"status": "ok", "ref": ref, "matches": matches}


async def get_answer_from_images(ref: int, question: str, tool_context: ToolContext) -> dict:
    """Asks a focused question directly against the picture(s)/table(s) found by
    get_images for the SAME `ref` -- call get_images first to confirm images exist
    on that page. Makes its own separate call to a vision-capable model, so ask one
    specific question (e.g. "what values does this chart show for 2023?"), not a
    broad "describe this image" -- you'll get a direct text answer back, not the
    image itself.

    Returns {"status": "ok", "ref": ref, "answer": "..."}, {"status": "no_images"}
    if the page has none, {"status": "no_vision_model"} if no vision-capable model
    is configured, or {"status": "invalid_ref"}/{"status": "no_page"} same as
    get_images."""
    hit = _resolve_ref(ref, tool_context)
    if hit is None:
        return {"status": "invalid_ref", "ref": ref}
    if hit.get("page_no") is None:
        return {"status": "no_page", "ref": ref}
    vision_model = get_vision_model()
    if not vision_model:
        return {"status": "no_vision_model", "ref": ref}

    points = _scroll_page_points(hit["file_hash"], hit["collection_name"], hit["page_no"], include_adjacent=False)
    seen_ids: set = set()
    image_paths = []
    for point in points:
        for image in point.payload.get("images", []):
            picture_id, path = image.get("picture_id"), image.get("image_path")
            if picture_id and picture_id not in seen_ids and path:
                seen_ids.add(picture_id)
                image_paths.append(path)
    if not image_paths:
        return {"status": "no_images", "ref": ref}

    # Direct ollama.Client() call -- bypasses ADK/LiteLLM entirely. No image ever
    # enters ADK's own session/conversation state, so LiteLLM's ollama_chat image
    # forwarding path (the thing the old design needed monkeypatched) is never
    # exercised for this at all. Same proven direct-client pattern already used by
    # edenview_RAG/retrieval/generate.py's generate_answer().
    import ollama

    from edenview_ingestion.settings import get_ollama_host, get_ollama_keep_alive

    client = ollama.Client(host=get_ollama_host()) if get_ollama_host() else ollama.Client()
    response = client.chat(
        model=vision_model,
        messages=[{"role": "user", "content": question, "images": image_paths}],
        keep_alive=get_ollama_keep_alive(),
    )
    return {"status": "ok", "ref": ref, "answer": response["message"]["content"].strip()}


async def get_answer_from_detailed_pages(
    ref: int, question: str, include_adjacent: bool, tool_context: ToolContext
) -> dict:
    """Asks a focused question directly against a finding's full page text (same
    page get_pages_detailed would return for this `ref`) instead of you reading the
    whole page yourself -- use this to keep your own context short when you only
    need one specific answer out of a long page, e.g. "what is the exact deadline
    mentioned on this page?" rather than pulling the whole page into your own
    reasoning. Set include_adjacent=True to also consider the page before/after.

    Returns {"status": "ok", "ref": ref, "answer": "..."}, or {"status":
    "invalid_ref"}/{"status": "no_page"}/{"status": "no_results"} if the page
    couldn't be found."""
    hit = _resolve_ref(ref, tool_context)
    if hit is None:
        return {"status": "invalid_ref", "ref": ref}
    if hit.get("page_no") is None:
        return {"status": "no_page", "ref": ref}
    points = _scroll_page_points(hit["file_hash"], hit["collection_name"], hit["page_no"], include_adjacent)
    if not points:
        return {"status": "no_results", "ref": ref}
    page_text = "\n\n".join(p.payload.get("text", "") for p in points)

    # Direct ollama.Client() call, same reasoning as get_answer_from_images -- keeps
    # a full page's text out of Deep Search's own growing conversation transcript.
    import ollama

    from edenview_ingestion.settings import get_ollama_host, get_ollama_keep_alive

    prompt = f"Page content:\n{page_text}\n\nQuestion: {question}\n\nAnswer using only the page content above."
    client = ollama.Client(host=get_ollama_host()) if get_ollama_host() else ollama.Client()
    response = client.chat(
        model=get_agent_model_name(),
        messages=[{"role": "user", "content": prompt}],
        keep_alive=get_ollama_keep_alive(),
    )
    return {"status": "ok", "ref": ref, "answer": response["message"]["content"].strip()}
