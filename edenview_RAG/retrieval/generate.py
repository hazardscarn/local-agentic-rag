"""Minimal answer generation for the /chat endpoint -- one Ollama chat call over
already-retrieved context, not the future ADK Retriever->Relevance->Answer agent loop
(that's still "not started" per edenview_progress.md). Same `ollama` call
shape as edenview_ingestion/chunking/contextual.py's `_generate_one`, just synchronous
since a chat turn is one call, not hundreds batched across a document.
"""

from __future__ import annotations

from typing import Optional

import ollama

from edenview_ingestion.settings import get_ollama_keep_alive

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


def _format_context(hits: list[RetrievalHit]) -> str:
    return "\n\n".join(f"[{i}] {hit.context_text}" for i, hit in enumerate(hits, start=1))


def generate_answer(
    query: str, hits: list[RetrievalHit], model: str, ollama_host: Optional[str] = None
) -> str:
    """`hits` must be non-empty -- callers should short-circuit to a canned "no
    relevant information" response themselves rather than calling this with []."""
    prompt = _PROMPT_TEMPLATE.format(context=_format_context(hits), query=query)
    client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}], keep_alive=get_ollama_keep_alive())
    return response["message"]["content"].strip()
