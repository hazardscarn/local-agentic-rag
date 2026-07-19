"""Visual grounding: renders one page of a previously-ingested PDF on demand, with an
optional highlight box, so a chat citation can show exactly where it came from.
`pypdfium2` (already a dependency -- it's the PDF backend
edenview_ingestion/docling_parsing/extractor.py already uses as
`PyPdfiumDocumentBackend`) renders the page; renders only the one page actually
requested, not every page up front -- see pipeline.py's `_preserve_original_pdf()`
for where the original file this reads comes from."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import ImageDraw

from edenview_ingestion.settings import get_documents_dir

router = APIRouter(tags=["documents"])

_RENDER_SCALE = 2.0
_HIGHLIGHT_COLOR = "#ff5c33"
_HIGHLIGHT_WIDTH = 4


def _resolve_original(file_hash: str) -> Path:
    if "/" in file_hash or "\\" in file_hash or ".." in file_hash:
        raise HTTPException(400, "Invalid file_hash")
    matches = list((get_documents_dir() / "originals").glob(f"{file_hash}.*"))
    if not matches:
        raise HTTPException(
            404,
            f"No original PDF kept for file_hash {file_hash!r} -- either this document predates visual "
            "grounding, or its source format isn't PDF.",
        )
    return matches[0]


def _parse_bbox(bbox: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if bbox is None:
        return None
    try:
        parts = tuple(float(p) for p in bbox.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be 4 comma-separated floats: l,t,r,b") from None
    if len(parts) != 4:
        raise HTTPException(400, "bbox must be 4 comma-separated floats: l,t,r,b")
    return parts  # type: ignore[return-value]


@router.get("/documents/{file_hash}/pages/{page_no}")
def render_page(file_hash: str, page_no: int, bbox: Optional[str] = Query(default=None)):
    path = _resolve_original(file_hash)
    box = _parse_bbox(bbox)

    pdf = pdfium.PdfDocument(str(path))
    if page_no < 1 or page_no > len(pdf):
        raise HTTPException(404, f"Page {page_no} out of range (document has {len(pdf)} pages)")

    page = pdf.get_page(page_no - 1)  # Docling's page_no is 1-indexed, pdfium's is 0-indexed
    bitmap = page.render(scale=_RENDER_SCALE)
    img = bitmap.to_pil()

    if box is not None:
        w, h = img.size
        l, t, r, b = box
        draw = ImageDraw.Draw(img)
        draw.rectangle((l * w, t * h, r * w, b * h), outline=_HIGHLIGHT_COLOR, width=_HIGHLIGHT_WIDTH)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
