"""Minimal answer generation for the /chat endpoint -- one Ollama chat call over
already-retrieved context, not the future ADK Retriever->Relevance->Answer agent loop
(that's still "not started" per edenview_progress.md). Same `ollama` call
shape as edenview_ingestion/chunking/contextual.py's `_generate_one`, just synchronous
since a chat turn is one call, not hundreds batched across a document.
"""

from __future__ import annotations

from typing import Optional

import ollama

from edenview_ingestion.settings import get_chat_num_ctx, get_ollama_keep_alive, model_supports_capability

from .models import RetrievalHit

_PROMPT_TEMPLATE = (
    "You are a helpful assistant answering questions using only the numbered context "
    "snippets below, retrieved from the user's own documents. Cite every claim with "
    "the snippet number(s) it came from, like [1] or [2][3]. If the context doesn't "
    "contain the answer, say so plainly instead of guessing.\n\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)

_REWORD_PROMPT_TEMPLATE = (
    "Conversation so far:\n{history}\n\n"
    "The user's latest message below may be a follow-up that only makes sense in "
    "light of the conversation above (e.g. it uses pronouns like \"it\"/\"that\", or "
    "refers to something already discussed, e.g. \"what about the second one\"). "
    "Rewrite it as a single, standalone question suitable for searching a document "
    "database on its own -- if it's already standalone, return it completely "
    "unchanged. Reply with ONLY the rewritten question, nothing else, no preamble.\n\n"
    "Latest message: {query}\n\n"
    "Standalone question:"
)


def _format_context(hits: list[RetrievalHit]) -> str:
    return "\n\n".join(f"[{i}] {hit.context_text}" for i, hit in enumerate(hits, start=1))


def _format_history(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def generate_answer(
    query: str,
    hits: list[RetrievalHit],
    model: str,
    ollama_host: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> tuple[str, Optional[str]]:
    """`hits` must be non-empty -- callers should short-circuit to a canned "no
    relevant information" response themselves rather than calling this with [].

    `history`, if given, is prior turns of this same chat session (plain
    {"role": "user"|"assistant", "content": ...} dicts, oldest first, NOT including
    the current turn) sent ahead of the context+question prompt -- lets the model
    resolve follow-ups ("what about the second one") against earlier turns instead
    of treating every call as a fresh, context-free question. Caller is responsible
    for capping how far back this goes (see api/routers/chat.py's _MAX_HISTORY_TURNS)
    since this is a single-call chat API with no session-level context management of
    its own, unlike the agentic tier's ADK session state.

    Returns (answer, thinking). thinking is Ollama's own native reasoning trace
    (message.thinking, only requested via think=True when model_supports_capability()
    confirms the configured model actually supports "thinking" -- see that function's
    docstring) -- kept SEPARATE from the answer, not discarded and not left off by
    default. Confirmed by direct side-by-side reproduction on this exact kind of
    compound, partially-unanswerable-from-context question: with thinking enabled,
    the model worked through each retrieved snippet individually and correctly
    answered "the context doesn't cover this" when that was true; with thinking
    disabled (an earlier, wrong fix for a since-superseded empty-response bug -- see
    get_chat_num_ctx's docstring), the exact same prompt instead fabricated a
    plausible-sounding but entirely fictional legal citation. For a RAG system,
    grounded refusal beats confident hallucination -- thinking stays on wherever the
    model supports it."""
    prompt = _PROMPT_TEMPLATE.format(context=_format_context(hits), query=query)
    messages = [*(history or []), {"role": "user", "content": prompt}]
    client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()
    kwargs = {"think": True} if model_supports_capability(model, "thinking", ollama_host) else {}
    response = client.chat(
        model=model,
        messages=messages,
        keep_alive=get_ollama_keep_alive(),
        options={"num_ctx": get_chat_num_ctx()},
        **kwargs,
    )
    thinking = getattr(response["message"], "thinking", None)
    return response["message"]["content"].strip(), thinking


def reword_query_for_retrieval(
    query: str, history: list[dict], model: str, ollama_host: Optional[str] = None
) -> str:
    """Vector/sparse search has no notion of conversation -- search()/search_db()
    embed and match on the literal string they're given, so a raw follow-up like
    "what about the second one" retrieves nothing meaningful even though
    generate_answer()'s own `history` param lets the model make sense of the
    follow-up once context IS retrieved. This closes that gap: one quick extra
    Ollama call (only made when there IS prior history -- a session's first turn is
    always already standalone, skip the extra latency) that rewrites the follow-up
    into a standalone question before it's used for retrieval. The reworded query is
    used ONLY for retrieval; the ORIGINAL `query` still goes to generate_answer() so
    the final answer responds to what the user actually typed, in their own words."""
    if not history:
        return query
    prompt = _REWORD_PROMPT_TEMPLATE.format(history=_format_history(history), query=query)
    client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()
    # think=False here (not gated like generate_answer) -- this is a mechanical
    # rewrite task with one right-shaped answer, not a judgment call where reasoning
    # improves grounding, so there's no quality reason to pay the extra latency.
    # False is confirmed safe to pass to both thinking and non-thinking models
    # (see model_supports_capability's docstring), unlike True, so no gating needed.
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        keep_alive=get_ollama_keep_alive(),
        options={"num_ctx": get_chat_num_ctx()},
        think=False,
    )
    reworded = response["message"]["content"].strip()
    return reworded or query
