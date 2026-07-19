"""S3 -- parent-child (small-to-big) retrieval. No LLM calls.

HybridChunker only runs at the child token budget -- it needs the actual
DoclingDocument to walk, so there's no way to hand it "one parent's text" and get back
doc-item-aware children from that substring alone. Instead: chunk once at child
granularity (full self_ref provenance per child, same as hybrid_docling.py), then group
consecutive children into parent buckets by cumulative token budget. A parent's
doc_item_refs is just the union of its children's, so image linking works identically
for both levels.

Returns both "parent" and "child" Chunk objects in one list, tagged via `Chunk.kind`.
Only children are meant to be embedded/searched -- parents are a docstore swapped in at
query time (same pattern as the old ingest/s2_parent_child.py), which the pipeline layer
decides, not this module.
"""

from __future__ import annotations

from docling.chunking import HybridChunker

from edenview_ingestion.docling_parsing import ExtractionBundle

from ._linking import attach_images
from ._provenance import first_item_provenance
from ._table_serializer import MarkdownTableSerializerProvider
from ._tokenizer import get_tokenizer
from .config import ParentChildConfig
from .models import Chunk, make_chunk_id

STRATEGY = "parent_child"


def _group_into_parents(doc_chunks, tokenizer, parent_max_tokens: int) -> list[list]:
    groups: list[list] = []
    current: list = []
    current_tokens = 0

    for doc_chunk in doc_chunks:
        tokens = tokenizer.count_tokens(doc_chunk.text)
        if current and current_tokens + tokens > parent_max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(doc_chunk)
        current_tokens += tokens

    if current:
        groups.append(current)
    return groups


def chunk(bundle: ExtractionBundle, config: ParentChildConfig = ParentChildConfig()) -> list[Chunk]:
    tokenizer = get_tokenizer(config.tokenizer_model, config.child_max_tokens)
    # serializer_provider=MarkdownTableSerializerProvider() -- see that module's
    # docstring, same reasoning as hybrid_docling.py's identical change.
    chunker = HybridChunker(tokenizer=tokenizer, serializer_provider=MarkdownTableSerializerProvider())

    child_doc_chunks = list(chunker.chunk(bundle.document))
    groups = _group_into_parents(child_doc_chunks, tokenizer, config.parent_max_tokens)

    chunks: list[Chunk] = []
    child_index = 0

    for parent_index, group in enumerate(groups):
        parent_refs = [ref for dc in group for ref in (it.self_ref for it in dc.meta.doc_items)]
        parent_text = "\n\n".join(dc.text for dc in group)
        parent_id = make_chunk_id(bundle.metadata.file_hash, f"{STRATEGY}_parent", parent_index)
        parent_page_no, parent_bbox = first_item_provenance(group[0].meta.doc_items, bundle.document)

        chunks.append(
            Chunk(
                chunk_id=parent_id,
                chunk_index=parent_index,
                text=parent_text,
                embed_text=parent_text,
                strategy=STRATEGY,
                kind="parent",
                doc_stem=bundle.doc_stem,
                file_hash=bundle.metadata.file_hash,
                page_no=parent_page_no,
                bbox=parent_bbox,
                headings=list(group[0].meta.headings or []),
                doc_item_refs=parent_refs,
            )
        )

        for doc_chunk in group:
            child_refs = [it.self_ref for it in doc_chunk.meta.doc_items]
            child_page_no, child_bbox = first_item_provenance(doc_chunk.meta.doc_items, bundle.document)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(bundle.metadata.file_hash, f"{STRATEGY}_child", child_index),
                    chunk_index=child_index,
                    text=doc_chunk.text,
                    embed_text=chunker.contextualize(doc_chunk),
                    strategy=STRATEGY,
                    kind="child",
                    doc_stem=bundle.doc_stem,
                    file_hash=bundle.metadata.file_hash,
                    page_no=child_page_no,
                    bbox=child_bbox,
                    headings=list(doc_chunk.meta.headings or []),
                    doc_item_refs=child_refs,
                    parent_id=parent_id,
                )
            )
            child_index += 1

    attach_images(chunks, bundle)
    return chunks
