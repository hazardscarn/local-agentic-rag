"""Typed exceptions so callers never see a raw tokenizer/Ollama traceback -- mirrors
docling_parsing/errors.py's approach."""

from __future__ import annotations


class ChunkingError(Exception):
    """Base class for every error raised by the chunking package."""


class ContextGenerationError(ChunkingError):
    """The contextual chunker's Ollama call failed outright for a chunk (after retries)."""
