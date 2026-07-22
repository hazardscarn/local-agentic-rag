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
    # No more "query_pipeline" cap -- the researcher calls vector_search directly, not
    # through a nested agent/tool. Cap is now on the tool itself.
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


def _next_call_letter(state) -> str:
    """Spreadsheet-style letter sequence (a, b, ..., z, aa, ab, ...) -- one assigned
    per distinct tool call that discovers new citable hits, so a call's own hits stay
    visibly grouped under one letter (e.g. every one of "b"'s findings came from the
    same vector_search call). Real, reproduced problem this fixes: with a flat global
    integer counter, every call's hits just got the next few consecutive numbers with
    nothing in the number itself indicating which call they came from -- confirmed
    directly that a small local model (qwen3.5:4b), asked to write a multi-topic
    answer citing from 15-20 accumulated findings across 5+ search calls, would
    frequently grab a stale number from an EARLIER topic's call while writing about a
    LATER topic, since nothing distinguished "number 5" as belonging to a specific
    call. The letter is the model's anchor for "which batch of results is this" --
    the per-call index after the dot is secondary. See RESEARCHER_INSTRUCTION's
    citation section for the matching instruction wording."""
    n = state.get("_call_letter_count", 0) + 1
    state["_call_letter_count"] = n
    letters = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("a") + rem) + letters
    return letters


def merge_hits_into_state(state, new_hits: dict[str, dict]) -> None:
    """ONE mechanism for turning a batch of {chunk_id: RetrievalHit-json} hits into
    state["citations"] (the full accumulated set, keyed by chunk_id, for the API
    route to read back after the run completes) and state["ref_to_chunk_id"] (the
    SAME call-scoped ref string a tool's own response shows, e.g. "b.2" -> that hit's
    real chunk_id, so a deep-dive tool's `ref` argument accepts that ref directly
    instead of asking the model to transcribe a chunk_id it was never shown -- see
    _next_call_letter for why refs are call-scoped rather than one flat counter).

    Called every time vector_search finds new hits, or a deep-dive tool like
    get_pages_detailed pulls in a genuinely new page (see tools.py) -- across however
    many calls fire in one turn, which is why this one function owns both `citations`
    and `ref_to_chunk_id` together, keeping them consistent. `state` accepts either a
    ToolContext.state or a plain dict (both support the same __getitem__/get/
    __setitem__ interface ADK's State wrapper provides). A no-op (no letter consumed)
    if every hit in `new_hits` is already known -- e.g. the same chunk turning up
    again from a different search."""
    citations: dict = state.get("citations", {})
    already_had = set(citations.keys())
    citations.update(new_hits)
    state["citations"] = citations

    new_chunk_ids = [cid for cid in new_hits if cid not in already_had]
    if not new_chunk_ids:
        return
    letter = _next_call_letter(state)
    ref_to_chunk_id: dict = state.get("ref_to_chunk_id", {})
    for i, chunk_id in enumerate(new_chunk_ids, start=1):
        ref_to_chunk_id[f"{letter}.{i}"] = chunk_id
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
    """Registered on the researcher agent's after_tool_callback (agent.py) -- when
    get_images actually pulls a picture/table for a `ref`, the picture may belong to
    a DIFFERENT chunk on the same page than the one that `ref` originally pointed to
    (image linkage is per-chunk at
    ingestion, but get_images scrolls the whole page), so it isn't already sitting
    in that ref's citation entry. This merges any newly-surfaced images into
    state["citations"][chunk_id]["images"] (deduped by picture_id) so the UI's
    citation/page-reference panel reflects what the researcher actually used to
    answer the question, not just what the initial vector_search hit happened to
    carry. get_pages_detailed needs no equivalent handling -- the ref's underlying chunk_id
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


# Matches anything bracketed that COULD be a raw call-scoped ref -- deliberately
# broad (not just the exact "[a.1]" shape), since a small local model does not
# reliably reproduce that exact syntax. Confirmed directly: qwen3.5:4b, shown
# "[a.1]"/"[a.2]" in tool responses, went on to write "[a.a1]" (doubling the call
# letter), "[c.C2]" (wrong case), and even "[b.b1-b.b4]" (a hyphen range spanning 4
# findings in one marker) in its actual answer -- a consistent, self-coherent but
# WRONG mental model of the format, not random noise. _expand_ref_candidates below
# tolerates exactly these near-miss shapes. Must start with a letter (excludes
# non-citation bracket content like a plain number range "[0,1]").
_RAW_CITATION_MARKER_RE = re.compile(r"\[([a-zA-Z][a-zA-Z0-9.\-]{0,20})\]")
# Matches a NORMALIZED, purely-sequential citation marker -- what
# _normalize_citation_sequence below rewrites every raw composite marker into, in
# first-appearance order (e.g. "[a.1]" then "[c.2]" then "[a.1]" again -> "[1]" "[2]"
# "[1]"). This is the format the final answer text and the frontend's citations[]
# array both use.
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
# One raw candidate ref, split out of a (possibly doubled-letter) bracket body by
# _expand_ref_candidates: an optional letter group, an optional '.', an optional
# SECOND letter group (the observed "a.a1" duplication), then the digits. A bare
# digit run with no letter at all (the tail of a hyphen range like "b1-4") is also
# accepted -- _expand_ref_candidates supplies the carried-forward letter for it.
_REF_PART_RE = re.compile(r"^([a-z]+)?\.?([a-z]*)(\d+)$")


def _expand_ref_candidates(raw: str, valid_refs: set[str]) -> list[str]:
    """Turns one raw bracket's inner text into zero or more real refs from
    `valid_refs` (state["ref_to_chunk_id"]'s keys), tolerating the near-miss formats
    _RAW_CITATION_MARKER_RE's own comment documents. Splits on '-' first to expand a
    range like "b.b1-b.b4" into every ref from b.1 through b.4 (inclusive) -- a range
    whose end doesn't repeat the letter (e.g. "b1-4") reuses the start's letter.
    Never invents a ref that isn't actually in valid_refs, even if its shape looks
    plausible -- an unrecognized bracket is left untouched by the caller instead."""
    parts = raw.lower().split("-")
    start_match = _REF_PART_RE.match(parts[0])
    if not start_match:
        return []
    start_letter = start_match.group(1) or start_match.group(2)
    start_num = int(start_match.group(3))
    if len(parts) == 1:
        candidate = f"{start_letter}.{start_num}" if start_letter else None
        return [candidate] if candidate in valid_refs else []

    # A hyphenated range -- only ever a single "start-end" pair in practice.
    end_match = _REF_PART_RE.match(parts[-1])
    if not end_match:
        return []
    end_letter = end_match.group(1) or end_match.group(2) or start_letter
    end_num = int(end_match.group(3))
    if end_letter != start_letter or end_num < start_num or end_num - start_num > 20:
        return []  # not a sane same-call range -- ignore rather than guess
    return [f"{start_letter}.{n}" for n in range(start_num, end_num + 1) if f"{start_letter}.{n}" in valid_refs]


def _normalize_citation_sequence(text: str, valid_refs: set[str]) -> tuple[str, list[str]]:
    """Rewrites the researcher's raw call-scoped ref markers (e.g. "[b.2]", or a
    tolerated near-miss like "[b.b2]" -- see _expand_ref_candidates) into a gapless,
    monotonic sequence (1, 2, 3, ...) in order of first appearance -- the format the
    end user actually sees. A single bracket can expand into SEVERAL normalized
    markers (a hyphen range like "[b.b1-b.b4]" becomes "[1][2][3][4]"). Any bracket
    that doesn't resolve to at least one real ref is left completely untouched, text
    and brackets as the model wrote them -- never silently dropped or replaced with
    something that looks plausible but isn't real. Cannot fix the model reusing one
    ref for two different underlying chunks -- that's a rarer failure the
    instruction's "these refs are final, reuse verbatim" wording is relied on to
    prevent, not this pass.

    Returns (normalized_text, refs_in_order) where refs_in_order[i] is the ORIGINAL
    raw ref (e.g. "b.2") that normalized marker "[i+1]" replaced -- callers that need
    to resolve a normalized marker back to a real chunk_id (see finalize_answer) use
    this instead of re-deriving it from the now-plain-integer normalized text."""
    seen: dict[str, str] = {}
    ordered_refs: list[str] = []

    def _replace(match: "re.Match[str]") -> str:
        candidates = _expand_ref_candidates(match.group(1), valid_refs)
        if not candidates:
            return match.group(0)  # not a recognizable ref -- leave untouched
        out = []
        for original in candidates:
            if original not in seen:
                seen[original] = str(len(seen) + 1)
                ordered_refs.append(original)
            out.append(f"[{seen[original]}]")
        return "".join(out)

    normalized = _RAW_CITATION_MARKER_RE.sub(_replace, text)
    return normalized, ordered_refs


async def finalize_answer(callback_context, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """after_model_callback, registered on the researcher agent (agent.py) --
    NOT a standalone answer_formatter agent, which no longer exists since the
    unified-researcher rewrite. researcher is a multi-turn tool-calling agent, so
    this fires on EVERY one of its model turns, not just the final one -- the
    `function_call` guard right below is what makes that safe: any turn that
    still contains a function_call is a mid-research turn (more tool calls
    coming), never the final answer, so it's left completely untouched. Only a
    turn with zero function_call parts is eligible to be treated as the answer.

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
    content). A tool-call-free turn never has a legitimate reason to produce
    thought-only output otherwise, since by construction it's the answer --
    so this fallback path is safe specifically here, unlike blindly reading
    thought content anywhere else in the pipeline.

    Writes the (citation-normalized) result into state["final_answer"] --
    durable regardless of whether ADK's own outer event-extraction later finds
    real text in this response or not -- and returns a replacement LlmResponse
    only when the text actually needs to change (citation-sequence fix, or the
    thought-fallback substitution), leaving the original multi-part response
    (with its correct thought flags intact) untouched otherwise."""
    if not llm_response.content or not llm_response.content.parts:
        return None
    parts = llm_response.content.parts
    if any(getattr(p, "function_call", None) for p in parts):
        return None  # mid-research tool-calling turn -- never the final answer
    non_thought_text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
    used_thought_fallback = not non_thought_text.strip()
    raw_text = "".join(p.text for p in parts if p.text) if used_thought_fallback else non_thought_text
    if not raw_text.strip():
        return None
    ref_to_chunk_id: dict = callback_context.state.get("ref_to_chunk_id", {})
    normalized, ordered_refs = _normalize_citation_sequence(raw_text, set(ref_to_chunk_id))
    callback_context.state["final_answer"] = normalized
    if not used_thought_fallback and normalized == raw_text:
        return None  # no rewrite needed, let the original (correctly thought-flagged) response stand
    # Write final_citation_order mapping each NORMALIZED position -> real chunk_id,
    # using ordered_refs (the raw "b.2"-style ref that ended up at each position) so
    # _ordered_citations returns chunks matching what the answer text displays. Since
    # every raw marker gets rewritten here (composite refs are never left as-is --
    # unlike the old plain-integer version, "already gapless" isn't a possible state
    # for a composite ref), this always has something to write whenever there were
    # any citations at all.
    if ref_to_chunk_id and not callback_context.state.get("final_citation_order"):
        callback_context.state["final_citation_order"] = {
            i + 1: ref_to_chunk_id[r] for i, r in enumerate(ordered_refs) if r in ref_to_chunk_id
        }

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


# Tool names that used to belong to one sub-question's own research thread under the
# old decompose -> subquestion_orchestrator -> subquestion_loop architecture (see
# agent.py's module docstring for why that was replaced by the single researcher
# agent below). The old agent-level names from that tree (subquestion_loop, reworder,
# search_executor, eval, deep_search) are gone from this set since nothing in the
# current pipeline ever emits them -- kept here only for the tool names, which are
# still real. In practice this whole mechanism is currently inert either way: the
# state keys _subquestion_context reads (current_subquestion_index/total/text) were
# only ever set by SubquestionOrchestrator, which no longer exists, so every call
# below returns {} regardless of `name` -- harmless (the frontend's legacy
# SubquestionThread rendering path is simply never populated), not actively wrong.
_SUBQUESTION_SCOPED_NODES = {
    "vector_search",
    "get_pages_detailed",
    "get_images",
    "grep",
    "get_answer_from_images",
    "get_answer_from_detailed_pages",
}


def _subquestion_context(name: str, state) -> dict:
    """Returns {} for anything not in _SUBQUESTION_SCOPED_NODES -- deliberately, so stale
    sub-question context from an earlier research thread never leaks onto a node that isn't
    actually part of one. For scoped nodes, reads back the three state keys the old
    SubquestionOrchestrator (removed, see _SUBQUESTION_SCOPED_NODES's own comment) used to
    set before each sub-question's own subquestion_loop invocation -- always {} now."""
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
    as track_agent_start. For vector_search, extracts the ACTUAL query text so the
    UI can display what's being searched (not just a generic "Searching...")."""
    tool_context.state[f"temp:tool_start:{tool.name}"] = time.time()

    # Extract actual query for vector_search so the UI shows real search terms
    if tool.name == "vector_search":
        query = args.get("query", "")
        label = f'Searching: "{query}"'
    else:
        label = prompts.STATUS_LABELS.get(tool.name, f"Running {tool.name}...")

    await _push_status({"type": "status", "node": tool.name, "phase": "start", "message": label})
    return None


async def track_tool_end(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[dict]:
    """after_tool_callback, registered alongside harvest_citations/
    harvest_page_references -- pushes a "this tool finished" status update
    including how long it took. Carries the actual query text for vector_search
    so the completed search line shows what was searched."""
    started = tool_context.state.get(f"temp:tool_start:{tool.name}")
    duration_s = (time.time() - started) if started else None

    event: dict[str, Any] = {
        "type": "status", "node": tool.name, "phase": "end", "duration_s": duration_s,
    }

    # Carry forward query text for vector_search end events
    if tool.name == "vector_search":
        query = args.get("query", "")
        event["message"] = f'Search complete: "{query}"'

    await _push_status(event)
    return None
