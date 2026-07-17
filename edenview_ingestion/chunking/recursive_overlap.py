"""S1 -- recursive character splitting with overlap (the basic baseline). No LLM calls.

Same splitting algorithm as the old ingest/s1_overlap.py (LangChain's
RecursiveCharacterTextSplitter: tries paragraph -> sentence -> word -> character
separators in order), but built from the DoclingDocument's own items instead of
`export_to_markdown()`. That's what lets this strategy carry `doc_item_refs` like the
other three -- assembling the text ourselves means every character offset can be traced
back to the self_ref it came from, which `_linking.attach_images()` needs to resolve
images. A plain markdown-export string has no such mapping.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from edenview_ingestion.docling_parsing import ExtractionBundle
from docling_core.types.doc.document import PictureItem, TableItem, TextItem

from ._linking import attach_images
from .config import RecursiveOverlapConfig
from .models import Chunk, make_chunk_id

STRATEGY = "recursive_overlap"

# (start, end, self_ref, page_no) -- half-open character ranges into the assembled text
# below, in the order items were appended.
_OffsetIndex = list[tuple[int, int, str, "int | None"]]


def _assemble_text(bundle: ExtractionBundle) -> tuple[str, _OffsetIndex]:
    doc = bundle.document
    parts: list[str] = []
    offsets: _OffsetIndex = []
    cursor = 0

    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc)
        elif isinstance(item, PictureItem):
            continue  # no text of its own; linked in via _linking.attach_images
        elif isinstance(item, TextItem):
            text = item.text
        else:
            continue
        if not text:
            continue

        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_no = item.prov[0].page_no if item.prov else None
        offsets.append((start, cursor, item.self_ref, page_no))

    return "".join(parts), offsets


def _overlapping(offsets: _OffsetIndex, start: int, end: int) -> _OffsetIndex:
    return [entry for entry in offsets if entry[0] < end and entry[1] > start]


def chunk(bundle: ExtractionBundle, config: RecursiveOverlapConfig = RecursiveOverlapConfig()) -> list[Chunk]:
    text, offsets = _assemble_text(bundle)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, add_start_index=True
    )
    docs = splitter.create_documents([text])

    chunks: list[Chunk] = []
    for i, d in enumerate(docs):
        start = d.metadata.get("start_index", 0)
        end = start + len(d.page_content)
        overlapping = _overlapping(offsets, start, end)
        page_no = overlapping[0][3] if overlapping else None
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(bundle.metadata.file_hash, STRATEGY, i),
                chunk_index=i,
                text=d.page_content,
                embed_text=d.page_content,
                strategy=STRATEGY,
                doc_stem=bundle.doc_stem,
                file_hash=bundle.metadata.file_hash,
                page_no=page_no,
                doc_item_refs=[entry[2] for entry in overlapping],
            )
        )

    attach_images(chunks, bundle)
    return chunks
