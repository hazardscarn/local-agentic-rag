"""Shared HuggingFaceTokenizer construction for every HybridChunker-based strategy
(hybrid_docling, parent_child, contextual). Centralized so the default tokenizer model
only needs to change in one place once an embedding model is actually chosen -- see
config.py's DEFAULT_TOKENIZER_MODEL docstring.

Loaded tokenizers are cached process-wide by (model, max_tokens): AutoTokenizer.from_pretrained
hits either the network or the local HF cache on disk, so it's worth not repeating per
chunker call within the same ingestion run.
"""

from __future__ import annotations

from typing import Optional

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

_TOKENIZER_CACHE: dict[tuple[str, Optional[int]], HuggingFaceTokenizer] = {}


def get_tokenizer(model: str, max_tokens: int | None = None) -> HuggingFaceTokenizer:
    key = (model, max_tokens)
    cached = _TOKENIZER_CACHE.get(key)
    if cached is not None:
        return cached

    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    kwargs = {"tokenizer": hf_tokenizer}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    tokenizer = HuggingFaceTokenizer(**kwargs)

    _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer
