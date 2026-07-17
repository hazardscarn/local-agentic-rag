"""Research/retrieval sub-agents for the "medium"/"high" tiers -- where all the
actual work happens (reframing, retrieval, evaluation, answer writing). The
"medium"/"high" roots in agent.py never call `retrieve` themselves; they only ever
wrap one of these as a tool via `AgentTool`. This keeps the root's own persisted
conversation history (DatabaseSessionService, survives restarts) lean over a long
chat -- a sub-agent's intermediate tool calls/retrieved chunk text never enter the
root's context, only its finished, cited answer does. ("low" tier is a single flat
agent instead, no split -- see agent.py's docstring for why.)

Two lessons already learned building "low" that apply here too:

1. Do NOT set include_contents="none" on these sub-agents. AgentTool.run_async
   already gives every call its own brand-new InMemorySessionService (confirmed in
   ADK's own source, google/adk/tools/agent_tool.py) -- root-history isolation is
   already structurally guaranteed with no extra config needed.
   include_contents="none" does something different and much more aggressive: it
   strips ALL prior turns from context on EVERY generation call the agent makes,
   including the original request from earlier in the SAME tool-calling sequence --
   reproduced directly causing a sub-agent to lose track of what it was even asked,
   answering with a fabricated guess instead of the real retrieved content.

2. Do NOT set planner=PlanReActPlanner() when the configured agent.model already has
   native "thinking" (qwen3.5 does). Reproduced directly: layering ADK's
   framework-level ReAct scaffolding on top of a model that already reasons natively
   caused the model's real answer to end up trapped entirely inside its thinking
   block instead of regular content -- which AgentTool.run_async explicitly discards
   (filters out any part marked thought=True) -- producing a silently empty final
   answer despite the tool call itself succeeding.

Not yet built (Step 4 of the build plan): build_medium_research_agent /
build_high_research_agent (reframe -> dispatch -> critic/refiner LoopAgent)."""

from __future__ import annotations

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions

from edenview_RAG.retrieval import RetrievalConfig, search, search_db

from . import callbacks, prompts, tools
from .config import Effort, RetrievalScope, get_agent_model_name, get_max_iterations, get_shared_llm, model_supports_vision
from .models import ReframeOutput


def build_reframe_agent() -> LlmAgent:
    # No tools here on purpose -- ADK's own docs warn output_schema + tools together
    # is only reliably supported on specific models (e.g. Gemini 3.0), not something
    # to rely on for a local Ollama model via LiteLLM. This is a pure text
    # transformation (rewrite + conditional split), it doesn't need retrieval itself.
    return LlmAgent(
        name="reframe",
        description="Rewrites and, if needed, splits the user's question for retrieval.",
        model=get_shared_llm(),
        instruction=prompts.REFRAME_INSTRUCTION,
        output_schema=ReframeOutput,
        output_key="reframed_queries",
    )


class RetrievalDispatchAgent(BaseAgent):
    """Deterministic (non-LLM) fan-out over the reframed sub-questions -- no
    tool-calling involved, so a weak local model never has to correctly transcribe a
    JSON query list into a function call argument (the highest-risk tool-call shape
    identified in the build plan). Loops over state["reframed_queries"]["queries"]
    (a plain dict, confirmed directly -- output_schema stores a parsed dict, not a
    JSON string, in the installed google-adk version) calling search()/search_db()
    once per query, SEQUENTIALLY (not threaded -- every real Qdrant call in this
    codebase already goes through one process-wide client_lock, so threading would
    mostly re-serialize there anyway, and fastembed's sparse embedder's thread-safety
    under concurrent calls is unverified).

    Writes state["citations"]/state["findings"] via callbacks.merge_hits_into_state --
    the SAME mechanism harvest_citations uses for every later retrieve() call in the
    refinement loop, so a refiner's own follow-up search contributes to the running
    findings text with continuous numbering instead of overwriting or diverging from
    what this step already found.

    State changes are built into a plain dict here and committed via a yielded
    Event's EventActions.state_delta, not direct ctx.session.state mutation -- ADK's
    own guidance: direct mutation of a Session object bypasses append_event and isn't
    tracked/persisted correctly."""

    async def _run_async_impl(self, ctx: InvocationContext):
        original_question = (
            ctx.user_content.parts[0].text
            if ctx.user_content and ctx.user_content.parts and ctx.user_content.parts[0].text
            else ""
        )
        scope = RetrievalScope(**ctx.session.state["scope"])
        reframed = ctx.session.state.get("reframed_queries") or {}
        queries = reframed.get("queries") if isinstance(reframed, dict) else None
        if not queries:
            # Reframe produced nothing usable -- fall back to the original question
            # rather than searching for nothing.
            queries = [original_question] if original_question else []

        config = RetrievalConfig(top_k=scope.top_k, use_reranker=scope.use_reranker)
        # A plain dict standing in for state here (merge_hits_into_state only needs
        # get/__getitem__/__setitem__) -- seeded from the real session state so this
        # step composes correctly even if something upstream already populated
        # citations/findings (it won't today, dispatch always runs first, but this
        # keeps the function itself correct independent of tree position).
        # original_question is written to state explicitly (not left for the critic/
        # answer steps to recall from conversation history several turns back) --
        # reproduced directly why that matters: the "answer" step's own generation
        # came back either empty or reasoning "there is no specific user question...
        # explicitly asked" despite findings being correct, because by that point the
        # actual conversational history is dominated by reframe's structured JSON,
        # this step's own (contentless) event, critique text, and refiner tool calls
        # -- nothing that plainly restates what the original question even was.
        state = {
            "citations": dict(ctx.session.state.get("citations", {})),
            "findings": ctx.session.state.get("findings", ""),
            "_findings_count": ctx.session.state.get("_findings_count", 0),
            "original_question": original_question,
        }
        for query in queries:
            hits = (
                search(scope.collection_names, query, config, scope.file_hashes, scope.strategy)
                if scope.collection_names
                else search_db(scope.db_name, query, config, scope.file_hashes, scope.strategy)
            )
            callbacks.merge_hits_into_state(state, {h.chunk_id: h.model_dump(mode="json") for h in hits})

        if not state["findings"]:
            state["findings"] = "(no results found)"
        yield Event(author=self.name, actions=EventActions(state_delta=state))


def build_research_agent(effort: Effort, extra_refiner_tools: list | None = None) -> SequentialAgent:
    """Shared tree for "medium" and "high" (they're structurally identical --
    reframe -> deterministic dispatch -> critic/refiner refinement loop -- differing
    only in max_iterations and, for "high", extra tools on the refiner; see
    agent.py's build_medium_agent/build_high_agent). `effort` must be "medium" or
    "high" (not "low", which has no loop at all).

    Uses SequentialAgent/LoopAgent despite ADK 2.5.0 flagging both as deprecated in
    favor of a new "Workflow" system -- confirmed the replacement isn't a viable
    substitute yet (its own deprecation warning states "Workflow cannot yet be used
    as an LlmAgent sub-agent", which is exactly how this tree gets consumed, wrapped
    in an AgentTool on the root). Revisit once ADK's Workflow API supports that."""
    # include_contents="none" on critic/answer (but NOT refiner) -- both get
    # everything they need explicitly templated into their own instruction
    # ({findings}/{critique?}/{original_question}), so neither needs the full
    # accumulated conversation transcript re-sent on every call. Reproduced directly
    # why this matters beyond just efficiency: with the default (full history),
    # a genuinely large multi-step run (e.g. a compound/split question -> two
    # dispatch rounds -> loop) needed num_ctx raised to 32768 just to stop silently
    # truncating, and even then a heavier run (2 reframed sub-queries) timed out
    # after 600s -- reprocessing the ENTIRE growing transcript from scratch on every
    # single step is real, unnecessary compute on a 6GB consumer GPU. `refiner` keeps
    # the default (sees history) because it actually needs to reason about what it
    # already tried/tool-called across loop iterations, which its own instruction
    # doesn't restate for it.
    critic = LlmAgent(
        name="critic",
        description="Reviews current findings and decides if they sufficiently answer the question.",
        model=get_shared_llm(),
        instruction=prompts.MEDIUM_HIGH_RESEARCH_INSTRUCTION,
        include_contents="none",
        output_key="critique",
    )
    refiner_instruction = prompts.REFINER_INSTRUCTION
    refiner_tools = [tools.retrieve, tools.exit_loop, *(extra_refiner_tools or [])]
    refiner_before_model_callback = None
    if extra_refiner_tools:
        # "high" tier only -- get_page_context needs its own guidance appended.
        refiner_instruction += prompts.PAGE_CONTEXT_GUIDANCE
    if effort == "high" and model_supports_vision(get_agent_model_name()):
        # Image inspection is high-tier-only and gated on the configured model
        # actually being multimodal (checked live against Ollama, not assumed) --
        # if it isn't, inspect_image/its callback are simply never registered, no
        # error, that one capability is just silently unavailable.
        refiner_instruction += prompts.INSPECT_IMAGE_GUIDANCE
        refiner_tools.append(tools.inspect_image)
        refiner_before_model_callback = callbacks.inject_pending_images
    refiner = LlmAgent(
        name="refiner",
        description="Either fetches one more targeted result or ends the refinement loop.",
        model=get_shared_llm(),
        instruction=refiner_instruction,
        tools=refiner_tools,
        before_tool_callback=callbacks.cap_tool_calls,
        after_tool_callback=callbacks.harvest_citations,
        before_model_callback=refiner_before_model_callback,
    )
    refine_loop = LoopAgent(name="refine_loop", sub_agents=[critic, refiner], max_iterations=get_max_iterations(effort))
    answer = LlmAgent(
        name="answer",
        model=get_shared_llm(),
        instruction=prompts.RESEARCH_ANSWER_INSTRUCTION,
        include_contents="none",
    )
    return SequentialAgent(
        name=f"{effort}_research",
        description=(
            "Searches the user's selected documents to thoroughly answer one question -- "
            "reframes/splits it if needed, retrieves, reviews and refines the findings, "
            "then writes a fully-cited final answer. Call with the user's exact question."
        ),
        sub_agents=[build_reframe_agent(), RetrievalDispatchAgent(name="dispatch"), refine_loop, answer],
    )


def build_medium_research_agent() -> SequentialAgent:
    return build_research_agent("medium")


def build_high_research_agent() -> SequentialAgent:
    return build_research_agent("high", extra_refiner_tools=[tools.get_page_context])
