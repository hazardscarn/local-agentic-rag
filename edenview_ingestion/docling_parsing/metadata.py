"""Builds `DocumentMetadata` from a Docling `ConversionResult` -- status, confidence,
and errors, so an ingestion job can flag "needs review" instead of silently shipping a
partial document (see notebook section 9, "Confidence scores" and the PARTIAL_SUCCESS
robustness callout)."""

from __future__ import annotations

from docling.datamodel.document import ConversionResult

from .models import DocumentMetadata, PageConfidence


def build_document_metadata(result: ConversionResult, doc_stem: str) -> DocumentMetadata:
    doc = result.document
    confidence = result.confidence
    input_doc = result.input

    page_confidence = [
        PageConfidence(page_no=page_no, grade=scores.mean_grade.value)
        for page_no, scores in confidence.pages.items()
    ]
    # A page missing from confidence.pages entirely was never scored, which means it
    # never actually got processed -- distinct from a page that scored POOR.
    unscored_pages = sorted(set(doc.pages.keys()) - set(confidence.pages.keys()))

    errors = [
        {
            "component_type": e.component_type.value,
            "module_name": e.module_name,
            "message": e.error_message,
        }
        for e in result.errors
    ]

    return DocumentMetadata(
        source_path=str(input_doc.file),
        doc_stem=doc_stem,
        file_hash=input_doc.document_hash,
        input_format=input_doc.format.value,
        num_pages=input_doc.page_count,
        status=result.status.value,
        doc_grade=confidence.mean_grade.value,
        page_confidence=page_confidence,
        errors=errors,
        unscored_pages=unscored_pages,
    )
