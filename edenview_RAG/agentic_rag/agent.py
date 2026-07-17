"""Root agent tree construction -- build_low_agent / build_medium_agent /
build_high_agent.

"medium"/"high" roots are deliberately thin: they hold the persisted, cross-turn
conversation history (via runtime.py's DatabaseSessionService) and wrap ONE research
sub-agent (subagent.py) as a tool -- they never call `retrieve` themselves. All of the
actual work (reframing, retrieval, evaluation, answer writing) happens inside the
wrapped sub-agent, whose own intermediate tool calls/retrieved chunk text never enter
the root's own context/history, only its finished answer does -- this is what keeps a
long-running chat session's persisted history from ballooning with every past turn's
full retrieved text.

"low" is deliberately NOT split into a root+subagent pair, unlike the other two tiers
-- reproduced directly why a relay layer hurt more than it helped at this tier: even
with an explicit "relay word-for-word, add no commentary" instruction, the root
agent (running on the same small local model) unreliably second-guessed the
sub-agent's already-correct answer instead of passing it through ("this doesn't seem
to directly address the user's question..."), on top of an earlier-confirmed
event-shape nondeterminism with `AgentTool(skip_summarization=True)` (empty final
text on ~1 in 4 runs) that using a plain generation call instead of relying on the
tool-result shortcut avoided but didn't fully resolve the reliability problem. One
flat agent removes that whole unreliable hop for the cheapest/simplest tier -- at the
cost of "low" mode's retrieved chunk text entering the root's own persisted history
each turn, an acceptable trade-off since "medium"/"high" (the default effort) keep the
full split and do the actual heavy lifting."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from . import callbacks, prompts, subagent, tools
from .config import get_shared_llm


def build_low_agent() -> LlmAgent:
    return LlmAgent(
        name="edenview_low",
        model=get_shared_llm(),
        instruction=prompts.LOW_ROOT_INSTRUCTION,
        tools=[tools.retrieve],
        before_tool_callback=callbacks.cap_tool_calls,
        after_tool_callback=callbacks.harvest_citations,
    )


def build_medium_agent() -> LlmAgent:
    # skip_summarization=True -- eliminates the root's own extra generation pass
    # after the research tool returns, which is exactly what was reproduced adding
    # unwanted meta-commentary ("I need to relay this exact answer...") in front of
    # the subagent's already-correct, already-cited answer no matter how the root's
    # instruction was worded (three separate rewrites tried, all still narrated).
    # An earlier attempt at skip_summarization here crashed
    # (KeyError: 'Context variable not found: `critique`') -- traced to a different
    # root cause entirely (subagent.py's critic step occasionally producing no
    # output_key value at all, fixed via REFINER_INSTRUCTION's `{critique?}`), not to
    # skip_summarization itself; once that was fixed, skip_summarization worked
    # cleanly (4/4 runs, no narration, no crash). runtime.py's _extract_final_text
    # also handles both possible final-event shapes (plain text part or raw
    # function_response) regardless, same safety net "low" needed before it was
    # collapsed to a flat agent.
    research = subagent.build_medium_research_agent()
    return LlmAgent(
        name="edenview_medium",
        model=get_shared_llm(),
        instruction=prompts.MEDIUM_HIGH_ROOT_INSTRUCTION,
        tools=[AgentTool(agent=research, skip_summarization=True)],
    )


def build_high_agent() -> LlmAgent:
    # Same tree as "medium" (see subagent.build_research_agent) plus a higher
    # max_iterations (config.yaml's agent.max_iterations.high) and refiner's extra
    # get_page_context tool. Image inspection (inspect_image + its
    # before_model_callback, gated by model_supports_vision()) isn't wired in yet --
    # Step 5 of the build plan.
    research = subagent.build_high_research_agent()
    return LlmAgent(
        name="edenview_high",
        model=get_shared_llm(),
        instruction=prompts.MEDIUM_HIGH_ROOT_INSTRUCTION,
        tools=[AgentTool(agent=research, skip_summarization=True)],
    )
