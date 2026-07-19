"""Shared callbacks registered on every agent with retrieval-shaped tools. Keeping
this logic in callbacks rather than inside each tool function means tools stay simple
(return results, don't manage shared state) and the behavior is applied uniformly no
matter how deep in an AgentTool/loop hierarchy a tool call happens -- AgentTool.run_async
forwards state changes back to the parent context, confirmed in ADK's own docs, so this
works correctly even from inside a sub-agent wrapped as a tool.

Callback parameter names are load-bearing: ADK invokes these by keyword
(`tool=`, `args=`, `tool_context=`, `tool_response=` -- confirmed directly against the
installed google-adk package's flows/llm_flows/functions.py), so renaming a parameter
breaks the call with a TypeError."""

from __future__ import annotations

from typing import Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

# Generous enough for legitimate multi-round use, tight enough to stop a weak local
# model stuck re-issuing the same tool call (the "infinite tool call loop" failure
# mode ADK's own Ollama docs warn about for small local models). Independent of --
# and a defensive floor underneath -- the critic/refiner LoopAgent's own max_iterations,
# which only bounds that specific loop, not a single agent step calling a tool
# repeatedly within one turn.
_TOOL_CALL_LIMITS = {"retrieve": 6, "get_page_context": 4, "inspect_image": 4}


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
    """Shared by harvest_citations (below) and subagent.RetrievalDispatchAgent --
    ONE mechanism for turning a batch of {chunk_id: RetrievalHit-json} hits into both
    state["citations"] (the full accumulated set, keyed by chunk_id, for the API
    route to read back after the run completes) and state["findings"] (a plain
    numbered-text block for the critic/refiner/answer steps to read via {findings}
    instruction templating -- deliberately mirroring edenview_RAG/retrieval/
    generate.py's proven _format_context() style, see tools.py's _format_hits_for_llm
    for why plain text reads far better to a small local model than structured JSON).

    Keeping this in one place (not duplicated between the dispatch step and every
    later retrieve() call in the refinement loop) matters because BOTH need to
    contribute to the SAME running findings text with continuous numbering -- a
    refiner's own retrieve() call mid-loop must add to, not overwrite or diverge
    from, whatever the initial dispatch step already found. `state` accepts either a
    ToolContext.state or a plain dict (RetrievalDispatchAgent doesn't have a
    ToolContext, only ctx.session.state -- both support the same __getitem__/get/
    __setitem__ interface ADK's State wrapper provides)."""
    citations: dict = state.get("citations", {})
    already_had = set(citations.keys())
    citations.update(new_hits)
    state["citations"] = citations

    new_ones = [h for cid, h in new_hits.items() if cid not in already_had]
    if not new_ones:
        return
    count = state.get("_findings_count", 0)
    lines = [state["findings"]] if state.get("findings") else []
    for hit in new_ones:
        count += 1
        lines.append(f"[{count}] {hit['context_text']}")
    state["findings"] = "\n\n".join(lines)
    state["_findings_count"] = count


async def inject_pending_images(callback_context, llm_request) -> None:
    """before_model_callback -- registered on `refiner` in the "high" tier only, and
    only when model_supports_vision() is true. Confirmed pattern (a real Google
    codelab, "ADK with Multimodal Tool Interaction Part 1", not improvised): a tool
    cannot return raw image bytes in its response, only an Artifact id
    (tools.inspect_image saves one); this callback is what actually attaches the
    image to the model's NEXT turn, by scanning the just-built request for an
    inspect_image function_response, loading the artifact it names, and appending
    the image Part into the conversation right after that function_response part --
    exactly where the model will see "here is the image you asked to inspect" as
    its next input.

    Callback parameter names (`callback_context`, `llm_request`) are load-bearing,
    confirmed directly against the installed google-adk package's
    flows/llm_flows/base_llm_flow.py, same as the tool callbacks above."""
    for content in llm_request.contents:
        if not content.parts:
            continue
        modified = []
        for part in content.parts:
            modified.append(part)
            fr = getattr(part, "function_response", None)
            if fr and fr.name == "inspect_image" and isinstance(fr.response, dict):
                artifact_id = fr.response.get("tool_response_artifact_id")
                if artifact_id:
                    image_part = await callback_context.load_artifact(filename=artifact_id)
                    if image_part:
                        modified.append(image_part)
        content.parts = modified
    return None


def harvest_citations(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[dict]:
    """Picks up tools.retrieve()'s temp:last_hits (full RetrievalHit json, stashed by
    the tool itself since the compact tool_response the LLM sees deliberately omits
    those fields -- see tools.py) and merges it into state via merge_hits_into_state,
    so the API route can read the full accumulated hit set back out after the run
    completes -- regardless of how many retrieval rounds fired, or how deep in an
    AgentTool/loop hierarchy they ran. A no-op for any other tool (get_page_context,
    inspect_image, exit_loop don't set temp:last_hits)."""
    last_hits: dict = tool_context.state.get("temp:last_hits", {})
    if not last_hits:
        return None
    merge_hits_into_state(tool_context.state, last_hits)
    return None  # don't alter what the LLM sees, just observe
