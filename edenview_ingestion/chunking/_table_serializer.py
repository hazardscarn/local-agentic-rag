"""Custom docling HybridChunker serializer_provider -- overrides Docling's default
TripletTableSerializer, confirmed by direct reproduction (against a real cached
DoclingDocument, a SEC 10-K financial table) to sever a table's column headers from
its data: the real cell values (e.g. "136,827") survive, but the "2024"/"2023" header
that says which column each one belongs to gets serialized separately and replaced
with bare positional indices in the actual chunk text ("Research and development (1),
3 = 136,827" -- nothing ties index 3 back to a year). MarkdownTableSerializer instead
reads each cell straight from `item.data.grid` and renders a real header row directly
above the data (same underlying source `table.export_to_markdown()` reads --
docling_parsing/tables.py already proved that path correct for these same tables), so
header and value stay attached in one place.

Shared by hybrid_docling.py and parent_child.py -- both build a plain
`HybridChunker(tokenizer=...)` today, inheriting Docling's default serializer."""

from __future__ import annotations

from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.serializer.base import BaseDocSerializer, BaseTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc.document import DoclingDocument


class _MarkdownTableChunkingSerializer(ChunkingDocSerializer):
    table_serializer: BaseTableSerializer = MarkdownTableSerializer()


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Pass to `HybridChunker(serializer_provider=MarkdownTableSerializerProvider())`."""

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return _MarkdownTableChunkingSerializer(doc=doc)
