"""S2 -- Docling's HybridChunker (tokenization-aware, structure-aware). No LLM calls.

Operates on the DoclingDocument directly rather than a markdown export, so heading
hierarchy, table boundaries, and reading order all drive the split -- see
https://docling-project.github.io/docling/examples/hybrid_chunking.ipynb. `chunk.text`
is stored as the display text; `chunker.contextualize(chunk)` (which prefixes the
heading path) is what actually gets embedded, per that notebook's own pattern.
"""

from __future__ import annotations

from docling.chunking import HybridChunker

from edenview_ingestion.docling_parsing import ExtractionBundle

from ._linking import attach_images
from ._provenance import first_item_provenance
from ._table_serializer import MarkdownTableSerializerProvider
from ._tokenizer import get_tokenizer
from .config import HybridDoclingConfig
from .models import Chunk, make_chunk_id

STRATEGY = "hybrid_docling"


def chunk(bundle: ExtractionBundle, config: HybridDoclingConfig = HybridDoclingConfig()) -> list[Chunk]:
    tokenizer = get_tokenizer(config.tokenizer_model, config.max_tokens)
    # serializer_provider=MarkdownTableSerializerProvider() -- see that module's
    # docstring: Docling's own default table serializer here separates a table's
    # header row from its data, confirmed to produce chunk text where real numbers
    # survive but which year/column they belong to doesn't.
    chunker = HybridChunker(
        tokenizer=tokenizer, merge_peers=config.merge_peers, serializer_provider=MarkdownTableSerializerProvider()
    )

    chunks: list[Chunk] = []
    for i, doc_chunk in enumerate(chunker.chunk(bundle.document)):
        doc_item_refs = [it.self_ref for it in doc_chunk.meta.doc_items]
        page_no, bbox = first_item_provenance(doc_chunk.meta.doc_items, bundle.document)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(bundle.metadata.file_hash, STRATEGY, i),
                chunk_index=i,
                text=doc_chunk.text,
                embed_text=chunker.contextualize(doc_chunk),
                strategy=STRATEGY,
                doc_stem=bundle.doc_stem,
                file_hash=bundle.metadata.file_hash,
                page_no=page_no,
                bbox=bbox,
                headings=list(doc_chunk.meta.headings or []),
                doc_item_refs=doc_item_refs,
            )
        )

    attach_images(chunks, bundle)
    return chunks
