"""Agent tree construction -- one flat pipeline, no effort tiers. Every agent object
below is a plain module-level assignment (mirrors Google's own ADK sample style,
e.g. adk-samples/python/agents/deep-search/app/agent.py), not a builder function
wrapping a builder function -- read top to bottom.

Root -> AgentTool(query_pipeline) -> SequentialAgent[question_capture, decompose,
subquestion_orchestrator, answer_formatter]. See edenview_RAG/edenview-agentic-rag-spec.md
for the original design intent.

Decompose-once, research-per-sub-question (this shape replaces an earlier design
where a single `reworder` node ran inside one big loop over the WHOLE turn, deciding
on every iteration whether the question was multi-part): `decompose` runs exactly
ONCE per turn and is the only place topic-splitting happens; `subquestion_orchestrator`
then runs `subquestion_loop` once per decomposed sub-question, fully independently.
Real testing (test/agentic_rag/verify_live_status.py) showed the old design
re-deciding the topic-count split on every pass instead of just rephrasing the same
fixed sub-question, causing runaway extra search/eval/deep_search work and, in one
run, a full empty-answer failure after ~1000s. See prompts.py's own module docstring
for the full story.

`subquestion_loop` is a custom BaseAgent, not a declarative LoopAgent, for one
specific reason: a declarative LoopAgent always invokes every sub-agent in its list
on every iteration, regardless of what's actually needed. reworder should only ever
run when Eval said needs_requery -- a sub-question's FIRST search never needs
rephrasing (decompose already produced clean standalone phrasing), so it's seeded
directly from `current_subquestion` with zero LLM cost. Likewise deep_search only
runs when Eval said needs_deep_search. Both were previously always-invoked LLM steps
whose instructions just told them to no-op when not needed -- still a full round
trip paid for nothing. This cuts that cost entirely rather than instructing around
it.

`subquestion_orchestrator` (looping `subquestion_loop` once per sub-question) is
also a custom BaseAgent -- confirmed against google/adk-python's own maintainer
guidance (github.com/google/adk-python discussion #4346) that this is the
recommended way to run a dynamic/runtime-sized number of sub-tasks through the same
sub-pipeline: define the sub-agent ONCE, statically, as a real registered
`sub_agents` entry, and invoke its `run_async` repeatedly from a custom orchestrator
-- NOT construct ad-hoc agent instances at runtime (which breaks ADK's state
management/resumability/observability, per that discussion). This is exactly the
same idiom ADK's own LoopAgent/SequentialAgent use internally (confirmed by direct
source reading: both just call `sub_agent.run_async(ctx)` for each of their
sub_agents in a plain loop).

Two lessons already learned building the previous version, still true here:
1. Do NOT set `include_contents="none"` on whatever's wrapped by an AgentTool at the
   Root boundary -- AgentTool.run_async already gives every call its own brand-new
   InMemorySessionService (confirmed in ADK's own source), so history isolation from
   Root is already structural. `include_contents="none"` on the INTERNAL nodes below
   is a different, deliberate choice (see each node's own comment).
2. Do NOT set planner=PlanReActPlanner() when agent.model already has native
   "thinking" (reproduced directly: traps the real answer inside a thought=True part
   that AgentTool.run_async discards, producing a silently empty final answer)."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.tools.agent_tool import AgentTool

from . import callbacks, prompts, tools
from .config import get_max_iterations, get_reword_llm, get_shared_llm, get_vision_model, require_tool_calling_model
from .models import DecomposedQuestions, EvalOutput, RewordedQueries
from edenview_ingestion.settings import model_supports_capability, get_ollama_host

require_tool_calling_model()  # fail loudly at import time, before building anything


class QuestionCapture(BaseAgent):
    """Tiny deterministic step, not an LLM call -- writes state["original_question"]
    from query_pipeline's own initiating message so `decompose`/`answer_formatter`
    can reference {original_question} via instruction templating instead of relying
    on conversational recall several tool-calls deep, which proved unreliable in an
    earlier version of this package. Runs exactly ONCE at the top of query_pipeline
    (not per sub-question, not per loop iteration) -- decompose is what actually
    consumes this value, everything downstream works off decomposed sub-questions.

    Also derives state["scope_description"] from state["scope"] (already seeded by
    runtime.py before this agent tree even starts) -- a human-readable name for the
    actual collection(s)/database this turn is scoped to, e.g. "the 'resumes'
    collection". Real, confirmed failure mode this fixes: decompose, given a query
    about resume/candidate data with no concrete grounding in what it's actually
    searching, sometimes pattern-matched "private candidate data" against its own
    baked-in refusal training and wrote REFUSAL TEXT as if it were a sub-question
    (e.g. "I am unable to access private candidate resumes..."), which then burned
    a full, wasted research cycle searching the vector DB for that literal sentence.
    Naming a concrete, real collection is harder to rationalize a refusal around
    than an abstract "the documents" -- see DECOMPOSE_INSTRUCTION's own use of this."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        question = (
            ctx.user_content.parts[0].text
            if ctx.user_content and ctx.user_content.parts and ctx.user_content.parts[0].text
            else ""
        )
        scope = ctx.session.state.get("scope") or {}
        names = scope.get("collection_names") or []
        db_name = scope.get("db_name")
        if names:
            joined = ", ".join(f"'{n}'" for n in names)
            scope_description = f"the {joined} collection(s)"
        elif db_name:
            scope_description = f"every collection in the '{db_name}' database"
        else:
            scope_description = "the document collection(s) selected for this conversation"
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"original_question": question, "scope_description": scope_description}),
        )


class SubquestionLoop(BaseAgent):
    """Runs reword-if-needed -> search -> eval -> deep-search-if-needed for ONE
    sub-question (state["current_subquestion"], set by SubquestionOrchestrator
    before each invocation), up to max_iterations passes, exiting as soon as Eval
    says sufficient. sub_agents[0..3] = [reworder, search_executor, eval_agent,
    deep_search], in that fixed order -- referenced by position (matching how
    ADK's own LoopAgent/SequentialAgent reference their children), not named
    fields, to avoid adding Pydantic fields beyond what BaseAgent already declares.

    See this module's own docstring for why reworder/deep_search are only invoked
    when Eval's verdict actually calls for them, not on every pass regardless."""

    max_iterations: int

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        reworder, search_executor, eval_agent, deep_search = self.sub_agents

        for _ in range(self.max_iterations):
            evaluation = ctx.session.state.get("evaluation") or {}
            is_first_attempt = not evaluation

            if is_first_attempt:
                # decompose already produced a clean, standalone sub-question --
                # no rephrasing needed, search with it directly. Zero LLM cost.
                current_subquestion = ctx.session.state.get("current_subquestion", "")
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={"reworded_queries": {"queries": [current_subquestion]}}),
                )
                async for event in search_executor.run_async(ctx):
                    yield event
            elif evaluation.get("needs_requery"):
                async for event in reworder.run_async(ctx):
                    yield event
                async for event in search_executor.run_async(ctx):
                    yield event
            # else: prior pass only needed deeper reading (needs_deep_search), not
            # a new search -- skip straight to re-grading the now-enriched findings
            # below, rather than redundantly re-running the exact same search.

            async for event in eval_agent.run_async(ctx):
                yield event

            evaluation = ctx.session.state.get("evaluation") or {}
            if evaluation.get("sufficient"):
                return

            if evaluation.get("needs_deep_search"):
                async for event in deep_search.run_async(ctx):
                    yield event
            elif not evaluation.get("needs_requery"):
                # Eval found no specific, nameable problem to fix -- looping
                # further can't help, stop rather than burning remaining
                # iterations re-grading identical findings.
                return


class SubquestionOrchestrator(BaseAgent):
    """Runs subquestion_loop (sub_agents[0]) once per sub-question in
    state["sub_questions"] (written once by `decompose`), resetting `evaluation`
    to {} and swapping in `current_subquestion` before each one so a previous
    sub-question's verdict never leaks into the next one's reworder/eval
    templating. See this module's own docstring for why this dynamic-fan-out
    shape (not a declarative ParallelAgent/LoopAgent) is the ADK-recommended
    pattern here.

    Also resets state["findings"]/state["ref_to_chunk_id"]/state["_findings_count"]/
    state["sub_answer_draft"] per sub-question -- these previously only ever
    accumulated for the WHOLE turn (callbacks.merge_hits_into_state), a confirmed
    real bug: a later sub-question's eval/draft-writing saw an earlier
    sub-question's totally unrelated chunks pooled in, directly implicated in a
    real observed failure (a buyback-taxation question's answer came back
    polluted with an unrelated "Business Trust" sub-question's content).
    state["citations"] (the UI-facing dict keyed by chunk_id) is deliberately
    left global/never reset -- it naturally dedupes by chunk_id, only the
    *numbering* needs to stay sub-question-local. After each sub-question's loop
    finishes (before the next iteration's reset wipes ref_to_chunk_id), the
    finished draft is snapshotted into state["sub_answers"] -- what
    callbacks.prepare_consolidated_findings later reads to build
    answer_formatter's consolidated input."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        (subquestion_loop,) = self.sub_agents
        sub_questions = ctx.session.state.get("sub_questions", {}).get("questions") or []
        if not sub_questions:
            sub_questions = [ctx.session.state.get("original_question", "")]

        total = len(sub_questions)
        for i, question in enumerate(sub_questions, start=1):
            label = question if total == 1 else f"({i}/{total}) {question}"
            # subquestion_index/total/text (structured, not just baked into the
            # human-readable message above) let a UI group the same repeating node
            # names (reworder/search_executor/eval/deep_search fire once per loop
            # iteration, per sub-question) into the correct independent research
            # thread -- see callbacks._subquestion_context(), which reads these
            # same three state keys back out for every event inside this
            # sub-question's own loop.
            await callbacks._push_status({
                "type": "status",
                "node": "subquestion_orchestrator",
                "phase": "start",
                "message": f"Researching: {label}",
                "subquestion_index": i,
                "subquestion_total": total,
                "subquestion_text": question,
            })
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={
                        "current_subquestion": question,
                        "current_subquestion_index": i,
                        "current_subquestion_total": total,
                        "evaluation": {},
                        "findings": "",
                        "ref_to_chunk_id": {},
                        "_findings_count": 0,
                        "sub_answer_draft": "",
                    }
                ),
            )

            async for event in subquestion_loop.run_async(ctx):
                yield event

            draft = (ctx.session.state.get("sub_answer_draft") or "").strip()
            if draft:
                sub_answers = ctx.session.state.get("sub_answers", [])
                yield Event(
                    author=self.name,
                    actions=EventActions(
                        state_delta={
                            "sub_answers": sub_answers
                            + [
                                {
                                    "question": question,
                                    "answer": draft,
                                    "ref_map": dict(ctx.session.state.get("ref_to_chunk_id", {})),
                                }
                            ]
                        }
                    ),
                )


# include_contents="none" on every node below: each one gets everything it needs
# explicitly templated into its own instruction ({evaluation}/{reworded_queries}/
# {current_subquestion}/{original_question}/{findings}), so none of them need the
# full accumulated transcript re-sent on every call -- a real, reproduced problem in
# an earlier version of this package (reprocessing an ever-growing transcript on
# every single step was real unnecessary compute on a 6GB consumer GPU, and the
# actual transcript by that point is dominated by tool-call/response noise, not the
# plain question/findings a step actually needs). This does NOT affect a single
# node's ability to make multiple tool calls within its own turn -- that's ADK's
# native function-calling loop, unaffected by include_contents.

# before_agent_callback/after_agent_callback=callbacks.track_agent_start/track_agent_end
# on every node below (including the container SequentialAgent/custom BaseAgents
# themselves) push live "this node started/finished, took Ns" updates into the
# shared status queue (see callbacks.py::_push_status / runtime.py::run_turn_stream)
# -- this is the fix for a real, confirmed ADK limitation: events from inside an
# AgentTool-wrapped sub-agent tree never reach the outer event stream
# (AgentTool.run_async() consumes and discards its own inner events; also an open
# upstream issue, google/adk-python#3147). Tool-calling agents ALSO get
# track_tool_start/track_tool_end (as a list alongside the existing
# before/after_tool_callback) for per-tool-call granularity.

decompose = LlmAgent(
    name="decompose",
    description="Splits the user's question into standalone sub-questions, once per turn, only if it genuinely covers more than one distinct topic.",
    # get_shared_llm() (thinking ON), not get_reword_llm() -- this runs ONCE per
    # turn, not once per loop iteration, so real deliberation here is paid for
    # once, not multiplied across every pass the way the old combined
    # reword+split node was.
    model=get_shared_llm(),
    instruction=prompts.DECOMPOSE_INSTRUCTION,
    include_contents="none",
    output_schema=DecomposedQuestions,
    output_key="sub_questions",
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
)

reworder = LlmAgent(
    name="reworder",
    description="Rephrases the current sub-question differently after a search attempt came up short -- never splits or invents new sub-questions.",
    # Only ever invoked (by SubquestionLoop) when Eval already said needs_requery
    # -- a real retry-rephrase task, not the old dual-purpose (split-or-rephrase)
    # job, so keeping thinking off here is safe: there's no topic-count judgment
    # left for this node to get wrong.
    model=get_reword_llm(),
    instruction=prompts.REWORDER_INSTRUCTION,
    include_contents="none",
    output_schema=RewordedQueries,
    output_key="reworded_queries",
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
)

search_executor = LlmAgent(
    name="search_executor",
    description="Runs vector_search, then drafts a cited answer to the current sub-question from what it found.",
    model=get_shared_llm(),
    instruction=prompts.SEARCH_EXECUTOR_INSTRUCTION,
    include_contents="none",
    tools=[tools.vector_search],
    output_key="sub_answer_draft",
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
    before_tool_callback=[callbacks.cap_tool_calls, callbacks.track_tool_start],
    after_tool_callback=[callbacks.track_tool_end, callbacks.harvest_citations],
)

eval_agent = LlmAgent(
    name="eval",
    description="Grades whether the findings so far are enough to answer the current sub-question, and what to do next if not.",
    model=get_shared_llm(),
    instruction=prompts.EVAL_INSTRUCTION,
    include_contents="none",
    output_schema=EvalOutput,
    output_key="evaluation",
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
)

deep_search_tools = [tools.get_images, tools.get_pages_detailed, tools.grep, tools.get_answer_from_detailed_pages]
if model_supports_capability(get_vision_model() or "", "vision", get_ollama_host()):
    deep_search_tools.append(tools.get_answer_from_images)

deep_search = LlmAgent(
    name="deep_search",
    description="Looks deeper into already-found chunks (fuller page text, images, exact terms), then rewrites the sub-question's draft answer with what it found -- does not re-search the corpus.",
    model=get_shared_llm(),
    instruction=prompts.DEEP_SEARCH_INSTRUCTION,
    include_contents="none",
    tools=deep_search_tools,
    output_key="sub_answer_draft",  # same key search_executor writes -- whichever node ran last owns the current draft
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
    before_tool_callback=[callbacks.cap_tool_calls, callbacks.track_tool_start],
    after_tool_callback=[callbacks.track_tool_end, callbacks.harvest_page_references],
)

subquestion_loop = SubquestionLoop(
    name="subquestion_loop",
    max_iterations=get_max_iterations(),
    sub_agents=[reworder, search_executor, eval_agent, deep_search],
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
)

subquestion_orchestrator = SubquestionOrchestrator(
    name="subquestion_orchestrator",
    sub_agents=[subquestion_loop],
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
)

answer_formatter = LlmAgent(
    name="answer_formatter",
    description="Writes the final, cited, well-formatted answer from the already-synthesized per-sub-question drafts.",
    # get_shared_llm() (thinking ON), NOT get_reword_llm() -- tried thinking-off here
    # first (see git history/this comment's own prior version) to fix a real
    # truncation bug (this node's num_ctx budget exhausted mid-generation on a
    # multi-part question), but confirmed by direct reproduction that thinking-off
    # traded that for a worse problem: synthesizing several sub-questions' worth of
    # real research (13,879 chars of consolidated notes, in one observed case) into
    # ONE coherent answer is genuine integration/judgment work -- deciding what to
    # keep, merge, or compress -- not mechanical formatting, closer to eval's
    # sufficiency grading (thinking deliberately kept ON there too) than to
    # reworder's rewrite task. Without thinking, it flattened the richest, most
    # relevant sub-answer into one vague sentence while keeping others detailed --
    # exactly the "confident but thin" failure mode generate.py's own docstring
    # documents for thinking-off on this same local model. agent.num_ctx was
    # separately raised to 32768 (double the old 16384) specifically so this node
    # doesn't need to give up thinking to afford it -- that's the fix for the
    # truncation risk now, not disabling thinking.
    model=get_shared_llm(),
    instruction=prompts.ANSWER_FORMATTER_INSTRUCTION,
    include_contents="none",
    before_agent_callback=[callbacks.track_agent_start, callbacks.prepare_consolidated_findings],
    after_agent_callback=callbacks.track_agent_end,
    after_model_callback=callbacks.finalize_answer,
)

query_pipeline = SequentialAgent(
    name="query_pipeline",
    description=(
        "Searches the user's selected documents to thoroughly answer one question -- "
        "decomposes it into sub-questions if it genuinely covers more than one topic, "
        "researches each independently, then writes a fully-cited final answer. Call "
        "with the user's exact, standalone question as `request`."
    ),
    sub_agents=[
        QuestionCapture(
            name="question_capture",
            before_agent_callback=callbacks.track_agent_start,
            after_agent_callback=callbacks.track_agent_end,
        ),
        decompose,
        subquestion_orchestrator,
        answer_formatter,
    ],
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
)

root_agent = LlmAgent(
    name="root_agent",
    model=get_shared_llm(),
    instruction=prompts.ROOT_INSTRUCTION,
    # skip_summarization=True -- Root does not re-narrate query_pipeline's already-
    # finished, already-cited answer; it passes it through verbatim (see
    # ROOT_INSTRUCTION). AgentTool isolates query_pipeline's retrieval internals
    # from Root's own persisted conversation history (see module docstring).
    tools=[AgentTool(agent=query_pipeline, skip_summarization=True)],
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
    # Real, confirmed failure mode: ROOT_INSTRUCTION says "call the research
    # tool exactly once", but nothing enforced that -- on a compound question,
    # the model sometimes called query_pipeline TWICE itself, and ADK ran both
    # concurrently as fully isolated sessions, each producing its own complete
    # answer, which root_agent (skip_summarization=True) then just concatenated
    # with no merge. cap_tool_calls (via callbacks._TOOL_CALL_LIMITS) already
    # caps every other tool in this pipeline the same way -- capping
    # query_pipeline to 1 here closes this specific gap with the same
    # mechanism, no new pattern needed.
    before_tool_callback=callbacks.cap_tool_calls,
)
