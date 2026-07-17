"""Dense (Ollama) + sparse (FastEmbed BM25) embedding -- model names read from
config.yaml via edenview_ingestion.settings, never hardcoded here. Same split as the
proven ingest/shared.py: dense vectors come from Ollama (no local CUDA/torch install
needed), sparse/BM25 term weights come from FastEmbed, ONNX-based and CPU-only."""

from __future__ import annotations

import ollama
from fastembed import SparseTextEmbedding

from edenview_ingestion.settings import get_model, get_ollama_host, get_ollama_keep_alive

_SPARSE_MODEL: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        _SPARSE_MODEL = SparseTextEmbedding(model_name=get_model("sparse_embedding"))
    return _SPARSE_MODEL


# Ollama's llama.cpp runner has two separate limits: the model's actual context
# window (n_ctx, e.g. 4096 for bge-m3 as loaded) and a lower-by-default "physical
# batch size" (num_batch, its own default is 2048) -- an internal compute-batching
# knob, not a real capacity ceiling. A single chunk landing between those two numbers
# (confirmed by reproduction: real chunks at 2133-3877 tokens, all comfortably under
# the 4096 context window) gets rejected by the *batch* limit even though the model
# itself could handle it fine. Passing num_batch here -- matching bge-m3's own
# context window, not guessing higher -- removes that artificial gap entirely,
# without touching chunking at all (Docling's HybridChunker already has a known
# upstream limitation where a large/complex table can't always be split under its
# token budget, so capping chunk size defensively would still leave this exposed).
_EMBED_NUM_BATCH = 4096


def embed_dense(texts: list[str], model: str | None = None) -> list[list[float]]:
    """`model` overrides config.yaml's dense_embedding -- used by
    edenview_RAG/retrieval/search.py to embed each collection's query with that
    collection's own stored embedding_model rather than whatever the current global
    config happens to be, so switching models doesn't silently break search over
    collections built with the old one."""
    host = get_ollama_host()
    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.embed(
        model=model or get_model("dense_embedding"),
        input=texts,
        keep_alive=get_ollama_keep_alive(),
        options={"num_batch": _EMBED_NUM_BATCH},
    )
    return response["embeddings"]


def detect_dense_embedding_dim(model: str, ollama_host: str | None = None) -> int:
    """Probes a dense embedding model's actual output length with one throwaway
    embed call -- used by PUT /system/config so dense_embedding_dim is derived from
    whatever dense_embedding is set to, instead of being a second, independently
    editable field that can silently drift out of sync with it."""
    host = ollama_host if ollama_host is not None else get_ollama_host()
    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.embed(model=model, input=["dimension probe"])
    return len(response["embeddings"][0])


def embed_sparse(texts: list[str]) -> list[dict]:
    """Returns one {"indices": [...], "values": [...]} dict per text -- raw term
    frequencies. Qdrant's Modifier.IDF (set on the collection, see collections.py)
    completes the actual BM25 scoring as points are added."""
    model = _get_sparse_model()
    results = []
    for emb in model.embed(texts):
        results.append({"indices": emb.indices.tolist(), "values": emb.values.tolist()})
    return results


def embed_texts(texts: list[str], batch_size: int = 32) -> list[dict]:
    """One {"dense": [...], "sparse": {...}} dict per text, batched."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        dense_vecs = embed_dense(batch)
        sparse_vecs = embed_sparse(batch)
        for dense, sparse in zip(dense_vecs, sparse_vecs):
            results.append({"dense": dense, "sparse": sparse})
    return results
