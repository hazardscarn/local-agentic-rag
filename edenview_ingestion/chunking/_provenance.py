"""Shared page_no + bbox extraction for HybridChunker-based strategies
(hybrid_docling, parent_child, contextual) -- all three key grounding off a chunk's
first doc_item's provenance, the same approximation already used for page_no alone
when a chunk spans multiple items. bbox is normalized to a 0..1, top-left-origin box
(Docling's own documented visual-grounding call chain) so it stays correct no matter
what resolution a page later gets rendered at -- the frontend/renderer just scales it
against whatever pixel dimensions it actually rendered, never against Docling's own
absolute point-coordinates.

recursive_overlap.py deliberately does NOT use this -- its chunks are built from
concatenated multi-item text spans with no single well-defined source region, so
grounding for that strategy stays page-number-only, no bbox."""

from __future__ import annotations

from docling_core.types.doc.document import DoclingDocument

Bbox = tuple[float, float, float, float]


def first_item_provenance(doc_items, document: DoclingDocument) -> tuple[int | None, Bbox | None]:
    if not doc_items or not doc_items[0].prov:
        return None, None
    prov = doc_items[0].prov[0]
    page_no = prov.page_no
    page = document.pages.get(page_no)
    if page is None:
        return page_no, None
    bbox = prov.bbox.to_top_left_origin(page_height=page.size.height).normalized(page.size)
    return page_no, bbox.as_tuple()
