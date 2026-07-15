"""Output data models produced by `DoclingExtractor.extract()`."""

from __future__ import annotations

from typing import Any, Optional

from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, ConfigDict


class PageConfidence(BaseModel):
    page_no: int
    grade: str  # QualityGrade value: poor/fair/good/excellent/unspecified


class DocumentMetadata(BaseModel):
    """Document-level facts an ingestion job needs for status tracking and citations --
    not the content itself, which lives on `ExtractionBundle.document`."""

    source_path: str
    doc_stem: str
    file_hash: str
    input_format: str
    num_pages: int
    status: str  # ConversionStatus value: success/partial_success/failure/...
    doc_grade: str  # document-level confidence grade
    page_confidence: list[PageConfidence]
    errors: list[dict[str, Any]]
    # Pages present in doc.pages but absent from confidence.pages -- Docling never
    # scored them, which means they never actually got processed (see notebook section 9).
    unscored_pages: list[int]


class TableRecord(BaseModel):
    table_id: str  # self_ref, e.g. "#/tables/0"
    page_no: Optional[int]
    bbox: Optional[tuple[float, float, float, float]]
    caption: Optional[str]
    num_rows: int
    num_cols: int
    markdown: str  # the copy that gets inlined into the chunk text stream
    csv_path: Optional[str]
    image_path: Optional[str] = None  # rendered crop, filled in when image generation is on


class PictureRecord(BaseModel):
    picture_id: str  # self_ref, e.g. "#/pictures/0"
    page_no: Optional[int]
    bbox: Optional[tuple[float, float, float, float]]
    caption: Optional[str]
    classification_label: Optional[str]
    classification_confidence: Optional[float]
    image_path: Optional[str]
    # self_refs of the nearest surrounding text items, in reading order -- how a
    # retrieved text chunk can pull this image in by metadata without a vision embedder.
    linked_text_refs: list[str]


class ExtractionBundle(BaseModel):
    """Everything one `extract()` call produces: the parsed document itself plus the
    metadata/table/picture records derived from it. There's deliberately no whole-page
    image list -- only picture/table crops are saved, since nothing else uses full page
    renders and saving one file per page for every document doesn't earn its keep."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: DoclingDocument
    doc_stem: str
    cache_dir: str
    metadata: DocumentMetadata
    tables: list[TableRecord]
    pictures: list[PictureRecord]
