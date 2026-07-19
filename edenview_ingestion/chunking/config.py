"""Per-strategy configuration objects. Model names are never hardcoded here -- they're
read once from config.yaml (project root) via edenview_ingestion.settings, so changing
which tokenizer or Ollama model a strategy uses is a one-line edit to that file. Numeric
defaults (chunk size, token budgets) aren't models and stay as plain Python defaults;
tuning those is explicitly deferred until an embedding model is selected (see
edenview_plan.md build order: chunking before embedding/Qdrant)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator

from edenview_ingestion.settings import get_model, get_ollama_host

# Used purely as a stand-in tokenizer so HybridChunker has something token-aware to
# split against -- not an embedding model itself. Every HybridChunker-based strategy
# reads this same default; see config.yaml for the actual value.
DEFAULT_TOKENIZER_MODEL = get_model("tokenizer")


class RecursiveOverlapConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50

    @model_validator(mode="after")
    def _check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )
        return self


class HybridDoclingConfig(BaseModel):
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    max_tokens: Optional[int] = None  # None -> derived from the tokenizer's own model_max_length
    merge_peers: bool = True  # merge undersized adjacent chunks from the same section


class ParentChildConfig(BaseModel):
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    child_max_tokens: int = 180
    parent_max_tokens: int = 2000

    @model_validator(mode="after")
    def _check_budget(self):
        if self.child_max_tokens >= self.parent_max_tokens:
            raise ValueError(
                f"child_max_tokens ({self.child_max_tokens}) must be smaller than "
                f"parent_max_tokens ({self.parent_max_tokens})"
            )
        return self


class ContextualConfig(BaseModel):
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    max_tokens: Optional[int] = None
    ollama_model: str = get_model("contextual_llm")
    ollama_host: Optional[str] = get_ollama_host()
    concurrency: int = 8
    cache_dir: str = str(Path.cwd() / ".edenview_tmp" / "contextual_cache")
    prompt_template: str = (
        "You are helping build a retrieval system for a document.\n\n"
        "Document section: {headings}\n\n"
        "Chunk text:\n{text}\n\n"
        "Write a single short sentence (max 30 words) that situates this chunk within "
        "the document and would help a search system find it. Do not restate the "
        "content -- explain where it fits. Answer with the sentence only."
    )
