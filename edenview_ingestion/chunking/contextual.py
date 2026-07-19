"""S4 -- contextual chunking (Anthropic-style). One Ollama call per chunk.

Base chunks come from the same HybridChunker pass as hybrid_docling.py (full doc-item
provenance, so image linking works identically). On top of that, each chunk gets a short
LLM-generated sentence describing where it sits in the document, prepended to what gets
embedded -- `embed_text = f"{context}\\n\\n{chunker.contextualize(doc_chunk)}"`. `text`
stays the raw chunk content; the enriched string is never shown to a user, only embedded.

Context strings are cached to disk keyed by a hash of the chunk's own text, not by
position -- so changing chunk-size/tokenizer settings only regenerates the chunks whose
content actually changed, and a cache file is shared safely even if unrelated chunker
config changes on a later run. A generation call that fails (after retries) degrades to
no context for that one chunk rather than failing the whole run -- flaky local LLM calls
on a couple of chunks out of hundreds shouldn't lose all the others' work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import ollama
from docling.chunking import HybridChunker

from edenview_ingestion.docling_parsing import ExtractionBundle
from edenview_ingestion.settings import get_ollama_keep_alive

from ._linking import attach_images
from ._provenance import first_item_provenance
from ._tokenizer import get_tokenizer
from .config import ContextualConfig
from .models import Chunk, make_chunk_id

STRATEGY = "contextual"


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, file_hash: str) -> Path:
    return Path(cache_dir) / f"{file_hash}.json"


def _load_cache(cache_dir: str, file_hash: str) -> dict[str, str]:
    path = _cache_path(cache_dir, file_hash)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache_dir: str, file_hash: str, cache: dict[str, str]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, file_hash)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


async def _generate_one(client: ollama.AsyncClient, prompt: str, model: str, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            response = await client.chat(
                model=model, messages=[{"role": "user", "content": prompt}], keep_alive=get_ollama_keep_alive()
            )
            return response["message"]["content"].strip()
        except Exception:
            if attempt == retries:
                return None
    return None


async def _generate_contexts(
    doc_chunks: list, config: ContextualConfig, cache: dict[str, str]
) -> list[str | None]:
    client = ollama.AsyncClient(host=config.ollama_host) if config.ollama_host else ollama.AsyncClient()
    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[str | None] = [None] * len(doc_chunks)
    content_hashes = [_content_hash(dc.text) for dc in doc_chunks]

    async def _worker(index: int) -> None:
        content_hash = content_hashes[index]
        cached = cache.get(content_hash)
        if cached is not None:
            results[index] = cached
            return
        doc_chunk = doc_chunks[index]
        prompt = config.prompt_template.format(
            headings=" / ".join(doc_chunk.meta.headings or []) or "(no heading)", text=doc_chunk.text
        )
        async with semaphore:
            context = await _generate_one(client, prompt, config.ollama_model)
        if context is not None:
            results[index] = context
            cache[content_hash] = context

    await asyncio.gather(*(_worker(i) for i in range(len(doc_chunks))))
    return results


def chunk(bundle: ExtractionBundle, config: ContextualConfig = ContextualConfig()) -> list[Chunk]:
    tokenizer = get_tokenizer(config.tokenizer_model, config.max_tokens)
    chunker = HybridChunker(tokenizer=tokenizer)
    doc_chunks = list(chunker.chunk(bundle.document))

    file_hash = bundle.metadata.file_hash
    cache = _load_cache(config.cache_dir, file_hash)
    contexts = asyncio.run(_generate_contexts(doc_chunks, config, cache))
    _save_cache(config.cache_dir, file_hash, cache)

    failed = sum(1 for c in contexts if c is None)
    if failed:
        print(f"[{STRATEGY}] {failed}/{len(doc_chunks)} chunks got no context after retries -- "
              f"embedding without context for those.")

    chunks: list[Chunk] = []
    for i, (doc_chunk, context) in enumerate(zip(doc_chunks, contexts)):
        serialized = chunker.contextualize(doc_chunk)
        embed_text = f"{context}\n\n{serialized}" if context else serialized
        page_no, bbox = first_item_provenance(doc_chunk.meta.doc_items, bundle.document)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(file_hash, STRATEGY, i),
                chunk_index=i,
                text=doc_chunk.text,
                embed_text=embed_text,
                strategy=STRATEGY,
                doc_stem=bundle.doc_stem,
                file_hash=file_hash,
                page_no=page_no,
                bbox=bbox,
                headings=list(doc_chunk.meta.headings or []),
                doc_item_refs=[it.self_ref for it in doc_chunk.meta.doc_items],
                context=context,
            )
        )

    attach_images(chunks, bundle)
    return chunks
