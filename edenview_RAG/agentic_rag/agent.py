"""Unified researcher agent — one LlmAgent with all tools, decides dynamically
how to research, searches in parallel, digs deeper where needed, writes the answer.

Replaces the old multi-agent tree (decompose → reworder → search_executor → eval →
deep_search → answer_formatter) that made 15-25+ sequential LLM calls per turn and
produced worse answers than simple RAG due to signal degradation across intermediate
agent transformations (see agentic_rag_debug.md for the full analysis).

The new design mirrors how a human researcher works:
1. THINK internally — generate multiple query angles (zero LLM cost)
2. SEARCH in parallel — fire all queries simultaneously (~same latency as one)
3. DEEP DIVE selectively — only on promising incomplete findings
4. ANSWER directly from raw evidence — no synthesis layer to corrupt signal

Root → AgentTool(researcher, skip_summarization=True) — root passes through the
researcher's answer verbatim (ROOT_INSTRUCTION says "output the tool's returned
answer verbatim"). AgentTool isolates researcher's internal state from root's history.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from . import callbacks, prompts, tools
from .config import get_shared_llm, get_vision_model, model_supports_capability, require_tool_calling_model, get_ollama_host

require_tool_calling_model()  # fail loudly at import time

# Build the researcher's tool list — same tools the old deep_search + search_executor had.
researcher_tools = [tools.vector_search]  # primary retrieval

if model_supports_capability(get_vision_model() or "", "vision", get_ollama_host()):
    researcher_tools.append(tools.get_answer_from_images)

researcher_tools.extend([
    tools.get_pages_detailed,
    tools.grep,
    tools.get_images,
    tools.get_answer_from_detailed_pages,
])

researcher = LlmAgent(
    name="researcher",
    model=get_shared_llm(),
    instruction=prompts.RESEARCHER_INSTRUCTION,
    tools=researcher_tools,
    include_contents="none",
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
    before_tool_callback=[callbacks.cap_tool_calls, callbacks.track_tool_start],
    # harvest_page_references was previously defined but never actually registered
    # here -- a real bug, same class as finalize_answer's own (see that callback's
    # docstring): images get_images pulls in were never getting merged into their
    # citation entries, so the citation/page-reference panel could miss images Deep
    # Search actually used to answer a question.
    after_tool_callback=[callbacks.track_tool_end, callbacks.harvest_citations, callbacks.harvest_page_references],
    # Guarded to no-op on every mid-research tool-calling turn (see its own
    # docstring) -- only touches the final, tool-call-free answer turn: strips
    # leaked thought-only content and normalizes/tracks the citation ref sequence
    # so runtime.py's _ordered_citations can return citations[] in the SAME order
    # the answer's own markers use.
    after_model_callback=callbacks.finalize_answer,
)

root_agent = LlmAgent(
    name="root_agent",
    model=get_shared_llm(),
    instruction=prompts.ROOT_INSTRUCTION,
    # skip_summarization=True -- root passes through the researcher's answer verbatim
    # (see ROOT_INSTRUCTION). AgentTool isolates researcher's internal state/history
    # from root's own persisted conversation history.
    tools=[AgentTool(agent=researcher, skip_summarization=True)],
    before_agent_callback=callbacks.track_agent_start,
    after_agent_callback=callbacks.track_agent_end,
    # Cap query_pipeline to 1 call closes a real failure mode: on compound questions,
    # the model sometimes called query_pipeline TWICE itself (once per topic), and ADK
    # ran both as fully isolated sessions. root_agent then concatenated both answers
    # with skip_summarization=True — no merge, no transition. Same mechanism as the
    # per-tool caps below.
    before_tool_callback=callbacks.cap_tool_calls,
)
