"""Cross-encoder reranking via FastEmbed (ONNX/CPU, no GPU cost -- same story as
chunking's/vectorstore's sparse BM25 model). Scores each (query, chunk) pair on a
comparable absolute-ish scale, which is what makes merging candidates from *multiple*
Qdrant collections sound -- raw RRF fusion scores from search.py are only meaningfully
comparable within one collection's own query, not across separate ones."""

from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

_MODEL_CACHE: dict[str, TextCrossEncoder] = {}


def _get_model(model_name: str) -> TextCrossEncoder:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = TextCrossEncoder(model_name=model_name)
    return _MODEL_CACHE[model_name]


def rerank(query: str, documents: list[str], model_name: str) -> list[float]:
    """One score per document, same order as `documents` -- higher is more relevant.
    Caller is responsible for sorting/truncating."""
    if not documents:
        return []
    model = _get_model(model_name)
    return list(model.rerank(query, documents))
