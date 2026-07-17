"""Internal-only Pydantic schemas for the agentic RAG loop (not exposed over the API --
see api/schemas.py for the request/response shapes callers see)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReframeOutput(BaseModel):
    """Output of the reframe step (no tools, output_schema-only per ADK's own
    guidance not to mix output_schema with tools reliably on non-Gemini models).

    `queries` rewrites the user's question for retrieval quality, and splits it into
    multiple entries *only* if the question genuinely contains more than one distinct
    ask -- a single-topic question must come back as a length-1 list, not padded out.
    See prompts.py::REFRAME_INSTRUCTION for the exact conditional-split wording."""

    queries: list[str] = Field(min_length=1, max_length=4)
