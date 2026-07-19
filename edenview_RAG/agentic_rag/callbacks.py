"""Shared callbacks registered on the pipeline's tool-calling agents. Keeping this
logic in callbacks rather than inside each tool function means tools stay simple
(return results, don't manage shared state) and the citation-harvesting/tool-cap
behavior is applied uniformly no matter how deep in an AgentTool/loop hierarchy a
tool call happens -- AgentTool.run_async DOES forward state_delta changes back to
the parent context (confirmed directly in its source), which is what makes
merge_hits_into_state/harvest_citations work correctly from inside a sub-agent
wrapped as a tool.

Live status tracking (track_agent_start/end, track_tool_start/end below) does NOT
use session state for its queue, though -- see _push_status's docstring for why
that specific mechanism needs contextvars instead, a real, confirmed ADK limitation
this project hit directly (not assumed from docs).

Callback parameter names are load-bearing: ADK invokes these by keyword
(`tool=`, `args=`, `tool_context=`, `tool_response=` -- confirmed directly against the
installed google-adk package's flows/llm_flows/functions.py), so renaming a parameter
breaks the call with a TypeError."""

from __future__ import annotations

import contextvars
import re
import time
from typing import Any, Optional

from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from . import prompts

# Generous enough for legitimate multi-round use, tight enough to stop a weak local
# model stuck re-issuing the same tool call (the "infinite tool call loop" failure
# mode ADK's own Ollama docs warn about for small local models). Independent of --
# and a defensive floor underneath -- subquestion_loop's own max_iterations, which only
# bounds the whole loop, not a single agent step calling one tool repeatedly within
# one turn.
_TOOL_CALL_LIMITS = {
    # Real, confirmed failure mode (not hypothetical): root_agent's own
    # instruction says "call the research tool exactly once", but nothing
    # enforced that -- on a compound "X, and separately Y" question, the model
    # sometimes decided to call query_pipeline TWICE itself (once per topic),
    # duplicating the job `decompose` already exists to do in a single call.
    # ADK executes multiple same-turn tool calls concurrently (see this dict's
    # own module docstring), so both ran as fully independent, isolated
    # AgentTool sessions -- and root_agent, with skip_summarization=True,
    # simply concatenated both real answers with no transition, no merge, no
    # awareness they were both live. AgentTool.name resolves to the wrapped
    # agent's own name ("query_pipeline", confirmed against installed ADK
    # source), so this is capped by the exact same mechanism as any other
    # tool here -- no new pattern needed.
    "query_pipeline": 1,
    "vector_search": 6,
    "get_pages_detailed": 4,
    "get_images": 4,
    "grep": 4,
    "get_answer_from_images": 4,
    "get_answer_from_detailed_pages": 4,
}


def cap_tool_calls(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> Optional[dict]:
    limit = _TOOL_CALL_LIMITS.get(tool.name)
    if limit is None:
        return None
    counts: dict = tool_context.state.get("temp:tool_call_counts", {})
    counts[tool.name] = counts.get(tool.name, 0) + 1
    tool_context.state["temp:tool_call_counts"] = counts
    if counts[tool.name] > limit:
        return {
            "status": "limit_reached",
            "message": (
                f"You've already called {tool.name} {limit} times this turn -- "
                "stop calling it and answer with what you have."
            ),
        }
    return None  # under the cap, let the real tool run


def merge_hits_into_state(state, new_hits: dict[str, dict]) -> None:
    """ONE mechanism for turning a batch of {chunk_id: RetrievalHit-json} hits into
    state["citations"] (the full accumulated set, keyed by chunk_id, for the API
    route to read back after the run completes), state["findings"] (a plain
    numbered-text block the eval/deep_search/answer_formatter steps read via
    {findings} instruction templating -- deliberately mirroring edenview_RAG/
    retrieval/generate.py's proven _format_context() style, see tools.py's
    _format_hits_for_llm for why plain text reads far better to a small local model
    than structured JSON), and state["ref_to_chunk_id"] (the SAME [N] reference
    number visible in `findings` -> that hit's real chunk_id, so deep_search's tools
    accept a plain `ref: int` -- the number already on screen -- instead of asking
    the model to transcribe a UUID it was never even shown).

    Called every time vector_search finds new hits, across however many calls fire
    in one turn (search_executor calling it once per reworded query) or across
    multiple loop iterations -- numbering must stay continuous, never restart or
    diverge, which is why this one function owns both the findings text and the
    ref_to_chunk_id map together. `state` accepts either a ToolContext.state or a
    plain dict (both support the same __getitem__/get/__setitem__ interface ADK's
    State wrapper provides)."""
    citations: dict = state.get("citations", {})
    already_had = set(citations.keys())
    citations.update(new_hits)
    state["citations"] = citations

    new_ones = [(cid, h) for cid, h in new_hits.items() if cid not in already_had]
    if not new_ones:
        return
    count = state.get("_findings_count", 0)
    lines = [state["findings"]] if state.get("findings") else []
    ref_to_chunk_id: dict = state.get("ref_to_chunk_id", {})
    for chunk_id, hit in new_ones:
        count += 1
        lines.append(f"[{count}] {hit['context_text']}")
        ref_to_chunk_id[count] = chunk_id
    state["findings"] = "\n\n".join(lines)
    state["_findings_count"] = count
    state["ref_to_chunk_id"] = ref_to_chunk_id


def harvest_citations(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[dict]:
    """Registered on search_executor -- picks up vector_search's temp:last_hits
    (full RetrievalHit json, stashed by the tool itself since the compact
    tool_response the LLM sees deliberately omits those fields, see tools.py) and
    merges it into state via merge_hits_into_state. A no-op if vector_search found
    nothing new."""
    last_hits: dict = tool_context.state.get("temp:last_hits", {})
    if not last_hits:
        return None
    merge_hits_into_state(tool_context.state, last_hits)
    return None  # don't alter what the LLM sees, just observe


def harvest_page_references(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[dict]:
    """Registered on deep_search -- when get_images actually pulls a picture/table
    for a `ref`, the picture may belong to a DIFFERENT chunk on the same page than
    the one that `ref` originally pointed to (image linkage is per-chunk at
    ingestion, but get_images scrolls the whole page), so it isn't already sitting
    in that ref's citation entry. This merges any newly-surfaced images into
    state["citations"][chunk_id]["images"] (deduped by picture_id) so the UI's
    citation/page-reference panel reflects what Deep Search actually used to answer
    the question, not just what the initial vector_search hit happened to carry.
    get_pages_detailed needs no equivalent handling -- the ref's underlying chunk_id
    is already a citation entry, there's nothing new to merge. A no-op for any tool
    that isn't get_images or didn't find anything."""
    if tool.name != "get_images" or not isinstance(tool_response, dict):
        return None
    images = tool_response.get("images")
    ref = tool_response.get("ref")
    if not images or ref is None:
        return None
    chunk_id = tool_context.state.get("ref_to_chunk_id", {}).get(ref)
    if chunk_id is None:
        return None
    citations: dict = tool_context.state.get("citations", {})
    citation = citations.get(chunk_id)
    if citation is None:
        return None
    existing_ids = {img.get("picture_id") for img in citation.get("images", [])}
    for image in images:
        if image.get("picture_id") not in existing_ids:
            citation.setdefault("images", []).append(image)
            existing_ids.add(image.get("picture_id"))
    citations[chunk_id] = citation
    tool_context.state["citations"] = citations
    return None  # don't alter what the LLM sees, just observe


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def _renumber_sub_answers(sub_answers: list[dict]) -> tuple[str, dict[int, str]]:
    """Pure function behind prepare_consolidated_findings, split out for direct
    unit-testing without a fake ToolContext. Walks `sub_answers` (each a
    {"question", "answer", "ref_map": {local_marker: chunk_id}} dict, built by
    agent.py's SubquestionOrchestrator from a sub-question's finished
    sub_answer_draft) in order, assigning ONE global citation number per unique
    chunk_id in first-appearance order across the whole list -- a chunk_id cited
    in two different sub-answers under different local numbers gets the SAME
    global number both times. Rewrites each entry's local [N] markers to the
    assigned global numbers and joins the results under per-topic headers.

    Returns (consolidated_text, {global_number: chunk_id}) -- the second is
    stored as state["final_citation_order"] so runtime.py can build the API's
    final `citations` list in the same order these markers use, which is what
    makes an inline [N] in the answer and citations[N-1] in the response the
    same chunk (see runtime.py's citation-list construction)."""
    global_number_by_chunk_id: dict[str, int] = {}

    def assign(chunk_id: str) -> int:
        if chunk_id not in global_number_by_chunk_id:
            global_number_by_chunk_id[chunk_id] = len(global_number_by_chunk_id) + 1
        return global_number_by_chunk_id[chunk_id]

    sections = []
    for entry in sub_answers:
        ref_map: dict = entry.get("ref_map", {})
        # ref_map's keys arrive as either int or str depending on whether they
        # crossed a JSON boundary (session state persistence) -- normalize once.
        local_to_global = {int(local): assign(chunk_id) for local, chunk_id in ref_map.items()}

        def _replace(match: "re.Match[str]", _map=local_to_global) -> str:
            local = int(match.group(1))
            return f"[{_map[local]}]" if local in _map else match.group(0)

        rewritten = _CITATION_MARKER_RE.sub(_replace, entry.get("answer", ""))
        question = entry.get("question", "")
        sections.append(f"### {question}\n\n{rewritten}" if question else rewritten)

    consolidated_text = "\n\n".join(sections)
    final_citation_order = {number: chunk_id for chunk_id, number in global_number_by_chunk_id.items()}
    return consolidated_text, final_citation_order


async def prepare_consolidated_findings(callback_context) -> None:
    """before_agent_callback, registered on answer_formatter only. Reads
    state["sub_answers"] (one entry per finished sub-question, snapshotted by
    SubquestionOrchestrator) and writes state["consolidated_sub_answers"] (the
    {consolidated_sub_answers} instruction template's input) and
    state["final_citation_order"] via _renumber_sub_answers above. A no-op
    (writes an empty string) if there are no sub-answers at all -- shouldn't
    normally happen since SubquestionOrchestrator always runs at least one
    sub-question, but answer_formatter's instruction already handles empty
    findings gracefully ("if the notes don't contain the answer, say so")."""
    sub_answers = callback_context.state.get("sub_answers", [])
    consolidated_text, final_citation_order = _renumber_sub_answers(sub_answers)
    callback_context.state["consolidated_sub_answers"] = consolidated_text
    callback_context.state["final_citation_order"] = final_citation_order


def _normalize_citation_sequence(text: str) -> str:
    """Cosmetic safety net over answer_formatter's own generated text -- makes
    the citation sequence gapless and monotonic (1, 2, 3, ...) in order of first
    appearance, in case the model slipped slightly despite being told the
    numbers from prepare_consolidated_findings are already final. Cannot fix
    the model reusing one number for two different underlying chunks -- that's
    a rarer failure the instruction's "these numbers are final, reuse verbatim"
    wording is relied on to prevent, not this pass."""
    seen: dict[str, str] = {}

    def _replace(match: "re.Match[str]") -> str:
        original = match.group(1)
        if original not in seen:
            seen[original] = str(len(seen) + 1)
        return f"[{seen[original]}]"

    return _CITATION_MARKER_RE.sub(_replace, text)


async def finalize_answer(callback_context, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """after_model_callback, registered on answer_formatter only.

    PREFERS non-`thought` text parts, exactly mirroring _extract_final_text's
    (runtime.py) own priority -- real, reproduced bug fixed here: an earlier
    version of this callback joined ALL parts regardless of `thought` flag
    unconditionally, which defeated the entire "hide the model's own scratch
    reasoning from the user" mechanism the rest of this pipeline relies on --
    confirmed directly: a real run's returned answer started with a multi-
    paragraph "Thinking Process: 1. Analyze the Request..." preamble, verified
    NOT present in that same turn's separately-tracked `thinking` field, proving
    the model DID correctly flag it as `thought=True` and this callback was the
    one that stripped the flag and reglued it onto the real answer.

    Only falls back to thought-flagged content when there is NO non-thought
    text at all -- the genuine salvage case this callback exists for (a real,
    recurring characteristic of this local model: its entire response
    sometimes lands in native "thinking" content with nothing in regular
    content). answer_formatter never has a legitimate reason to produce
    thought-only output otherwise, since it has no tools and its only job is
    the user-facing answer -- so this fallback path is safe specifically here,
    unlike blindly reading thought content anywhere else in the pipeline.

    Writes the (citation-normalized) result into state["final_answer"] --
    durable regardless of whether ADK's own outer event-extraction later finds
    real text in this response or not -- and returns a replacement LlmResponse
    only when the text actually needs to change (citation-sequence fix, or the
    thought-fallback substitution), leaving the original multi-part response
    (with its correct thought flags intact) untouched otherwise."""
    if not llm_response.content or not llm_response.content.parts:
        return None
    parts = llm_response.content.parts
    non_thought_text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
    used_thought_fallback = not non_thought_text.strip()
    raw_text = "".join(p.text for p in parts if p.text) if used_thought_fallback else non_thought_text
    if not raw_text.strip():
        return None
    normalized = _normalize_citation_sequence(raw_text)
    callback_context.state["final_answer"] = normalized
    if not used_thought_fallback and normalized == raw_text:
        return None  # no rewrite needed, let the original (correctly thought-flagged) response stand
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=normalized)]))


# Carries the current turn's live-status queue across ADK's own internal session
# boundaries -- NOT session state. Confirmed by direct reproduction (an isolated
# 2-agent test, not assumed) that session state cannot do this job: ADK's
# InMemorySessionService.create_session() -- what AgentTool.run_async() calls
# internally to seed a nested sub-agent's brand-new session -- routes any state key
# through extract_state_delta(), which silently DROPS every `temp:`-prefixed key
# entirely when initializing a new session (by design: temp: state is defined as
# "this invocation only," so a fresh session correctly starts without it) -- and a
# plain, non-`temp:`-prefixed key isn't safe either, since it WOULD get treated as
# durable session state and our actual outer session uses DatabaseSessionService,
# which would try to persist a raw asyncio.Queue object as if it were JSON-
# serializable durable state. Neither option survives the trip into a nested
# AgentTool call. A plain Python contextvars.ContextVar sidesteps this entirely --
# it propagates through the real async call chain (any coroutine/task descended
# from a context where it's set sees the same value), independent of whatever
# session/state boundaries ADK draws internally, since AgentTool invokes its nested
# Runner via a plain `await`/`async for`, not a new thread or a reset execution
# context. Confirmed directly: an isolated test showed the nested agent's own
# before_agent_callback receiving the queue 0.01s after the parent's tool call
# fired, this way -- and never receiving it at all via the state-based approach,
# despite waiting through a full multi-minute run.
_status_queue_var: "contextvars.ContextVar[Optional[Any]]" = contextvars.ContextVar("status_queue", default=None)


async def _push_status(item: dict) -> None:
    """Pushes a status update into the current turn's queue (if one is set --
    a no-op for run_turn's non-streaming path, or when driving the agent directly
    without setting one, e.g. adk web). See _status_queue_var's own comment for
    why this is a contextvar and not session state."""
    queue = _status_queue_var.get()
    if queue is not None:
        await queue.put(item)


# Agent/tool names that belong to ONE sub-question's own research (everything
# inside subquestion_loop, see agent.py) -- as opposed to question_capture/decompose/
# answer_formatter, which run once per turn, not once per sub-question. Used to
# attach subquestion_index/total/text to only the events a live status UI actually
# needs to group by research thread; a naive flat node-name map would otherwise
# conflate e.g. `reworder`'s events across different sub-questions and loop
# iterations, since the same names repeat.
_SUBQUESTION_SCOPED_NODES = {
    "subquestion_loop",
    "reworder",
    "search_executor",
    "eval",
    "deep_search",
    "vector_search",
    "get_pages_detailed",
    "get_images",
    "grep",
    "get_answer_from_images",
    "get_answer_from_detailed_pages",
}


def _subquestion_context(name: str, state) -> dict:
    """Returns {} for anything not in _SUBQUESTION_SCOPED_NODES (e.g.
    answer_formatter, decompose) -- deliberately, so stale sub-question context from
    an earlier research thread never leaks onto a node that isn't actually part of
    one. For scoped nodes, reads back the three state keys
    SubquestionOrchestrator._run_async_impl (agent.py) sets before each
    sub-question's own subquestion_loop invocation."""
    if name not in _SUBQUESTION_SCOPED_NODES:
        return {}
    ctx = {
        "subquestion_index": state.get("current_subquestion_index"),
        "subquestion_total": state.get("current_subquestion_total"),
        "subquestion_text": state.get("current_subquestion"),
    }
    return {k: v for k, v in ctx.items() if v is not None}


async def track_agent_start(callback_context) -> None:
    """before_agent_callback, registered on every agent in the pipeline -- pushes
    a "this node started" status update and records the start time so the matching
    track_agent_end can report how long the node took. Start-time tracking itself
    stays in session state (unlike the queue) since both callbacks for one agent
    fire within that same agent's own single invocation -- no AgentTool boundary
    crossed between a node's own start and end. Keyed by plain agent name, not a
    per-call id: safe because this pipeline runs fully sequentially (no two
    invocations of the same agent are ever in flight at once), so a start always
    immediately precedes its own matching end."""
    name = callback_context.agent_name
    callback_context.state[f"temp:step_start:{name}"] = time.time()
    label = prompts.AGENT_STATUS_LABELS.get(name, f"Running {name}...")
    await _push_status(
        {"type": "status", "node": name, "phase": "start", "message": label}
        | _subquestion_context(name, callback_context.state)
    )


async def track_agent_end(callback_context) -> None:
    """after_agent_callback, registered on every agent in the pipeline -- pushes a
    "this node finished" status update including how long it took."""
    name = callback_context.agent_name
    started = callback_context.state.get(f"temp:step_start:{name}")
    duration_s = (time.time() - started) if started else None
    await _push_status(
        {"type": "status", "node": name, "phase": "end", "duration_s": duration_s}
        | _subquestion_context(name, callback_context.state)
    )


async def track_tool_start(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> Optional[dict]:
    """before_tool_callback, registered alongside cap_tool_calls on every
    tool-calling agent -- pushes a "this tool started" status update. Duration
    tracking here is keyed by plain tool name, same sequential-pipeline assumption
    as track_agent_start -- the one known edge case is two calls of the SAME tool
    name genuinely running concurrently within one model turn (ADK does support
    this via asyncio.gather when a model requests several tool calls at once),
    where the second call's start would overwrite the first's before it's read --
    a minor, accepted imprecision in reported duration, not a correctness bug in
    the status feed itself (both calls still get their own start/end events)."""
    tool_context.state[f"temp:tool_start:{tool.name}"] = time.time()
    label = prompts.STATUS_LABELS.get(tool.name, f"Running {tool.name}...")
    await _push_status(
        {"type": "status", "node": tool.name, "phase": "start", "message": label}
        | _subquestion_context(tool.name, tool_context.state)
    )
    return None


async def track_tool_end(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[dict]:
    """after_tool_callback, registered alongside harvest_citations/
    harvest_page_references -- pushes a "this tool finished" status update
    including how long it took."""
    started = tool_context.state.get(f"temp:tool_start:{tool.name}")
    duration_s = (time.time() - started) if started else None
    await _push_status(
        {"type": "status", "node": tool.name, "phase": "end", "duration_s": duration_s}
        | _subquestion_context(tool.name, tool_context.state)
    )
    return None
