"""Custom-schema extraction: given a document and a caller-supplied schema (a string,
dict, or Pydantic model/instance), returns structured JSON per page.

This wraps Docling's own `DocumentExtractor` -- a vision-language model (`NuExtract-2B`
by default, `Granite Vision` as the alternative) that rasterizes pages and reads
structured fields off them. Per the confirmed design, this is the *only* place in
`edenview_ingestion` that runs a VLM: everywhere else (parsing, tables, images,
metadata) is non-generative. Custom-schema extraction stays on Docling's own model
rather than being routed through a separately chosen LLM.

Only PDF and image inputs are supported -- Docling's extraction pipeline only has
default backends configured for those two formats (it works by rendering pages to
images and running a vision model over them), so it doesn't extend to the DOCX/HTML/
CSV/etc. backends `extractor.py` handles for parsing.

First use downloads the selected model's weights from Hugging Face (several GB for
NuExtract-2B) to the local HF cache and runs entirely locally after that -- no data
leaves the machine, but expect a one-time download plus real per-page inference cost.

Requires `pip install qwen-vl-utils` for the default NuExtract-2B model (it's built on
the Qwen-VL architecture and needs it for image preprocessing) -- not in the base
requirements.txt since this whole path is optional/heavy and lazily imported; only
install it once you actually want to run custom-schema extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, Union

from docling.datamodel.extraction import ExtractedPageData, ExtractionTemplateType
from docling.exceptions import ConversionError

from .errors import ExtractionFailedError, UnsupportedFormatError

SchemaExtractionModel = Literal["nuextract", "granite_vision"]

# What Docling's extraction pipeline actually has default backends for (PDF + these
# image types). Checked upfront so a wrong-format input fails fast with a clear message
# instead of loading the VLM first and failing deep inside Docling's own conversion
# loop -- ConversionError there could equally mean "wrong format" or "the model itself
# failed" (e.g. a missing dependency), so it can't be trusted alone to mean bad format.
_SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def extract_schema(
    source: Union[str, Path],
    template: ExtractionTemplateType,
    *,
    model: SchemaExtractionModel = "nuextract",
    page_range: Optional[tuple[int, int]] = None,
    max_num_pages: Optional[int] = None,
    max_file_size: Optional[int] = None,
) -> list[ExtractedPageData]:
    """Extract fields matching `template` from `source` (a PDF or image file).

    `template` follows Docling's own extraction API: a JSON-shaped string
    (`'{"total": "float"}'`), a dict, a Pydantic model class, or a populated Pydantic
    model instance (its values become defaults for fields the document doesn't
    override). See docling docs "Information extraction" for the full worked example.

    `page_range` limits which pages actually get run through the VLM -- worth using on
    anything more than a few pages, since this path pays real per-page inference cost
    (unlike the non-generative parsing side). `max_num_pages`/`max_file_size` reject the
    document outright instead of truncating it; default to no limit, same as Docling's
    own `DocumentExtractor.extract()`.
    """
    # Imported lazily: only pulls in the VLM/transformers machinery if this function
    # is actually called, keeping the base parsing path (extractor.py) lightweight.
    from docling.backend.image_backend import ImageDocumentBackend
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ExtractionPromptStyle, VlmExtractionPipelineOptions
    from docling.datamodel.vlm_model_specs import (
        GRANITE_VISION_4_1_TRANSFORMERS,
        NU_EXTRACT_2B_TRANSFORMERS,
    )
    from docling.document_extractor import DocumentExtractor, ExtractionFormatOption
    from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline

    source = Path(source)
    if not source.exists():
        raise UnsupportedFormatError(f"Source file does not exist: {source}", source=source)
    if source.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"{source} -- schema extraction only supports PDF and image files "
            f"({sorted(_SUPPORTED_SUFFIXES)}), got {source.suffix!r}",
            source=source,
        )

    if model == "nuextract":
        vlm_options = NU_EXTRACT_2B_TRANSFORMERS
        prompt_style = ExtractionPromptStyle.NUEXTRACT
    else:
        vlm_options = GRANITE_VISION_4_1_TRANSFORMERS
        prompt_style = ExtractionPromptStyle.GRANITE_VISION

    pipeline_options = VlmExtractionPipelineOptions(
        vlm_options=vlm_options, extraction_prompt_style=prompt_style
    )
    extractor = DocumentExtractor(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        extraction_format_options={
            InputFormat.PDF: ExtractionFormatOption(
                pipeline_cls=ExtractionVlmPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: ExtractionFormatOption(
                pipeline_cls=ExtractionVlmPipeline,
                backend=ImageDocumentBackend,
                pipeline_options=pipeline_options,
            ),
        },
    )

    try:
        result = extractor.extract(
            source=str(source),
            template=template,
            page_range=page_range or (1, sys.maxsize),
            max_num_pages=max_num_pages or sys.maxsize,
            max_file_size=max_file_size or sys.maxsize,
        )
    except ConversionError as e:
        # Format was already validated above, so a ConversionError here means the
        # conversion itself failed (e.g. a missing model dependency, a corrupt page) --
        # not a format problem.
        raise ExtractionFailedError(f"Schema extraction failed for {source}: {e}", source=source) from e

    if result.status.value == "failure":
        raise ExtractionFailedError(f"Schema extraction failed for {source}: {result.errors}", source=source)

    return result.pages
