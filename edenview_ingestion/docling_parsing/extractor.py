"""The single entrypoint: `DoclingExtractor.extract(source) -> ExtractionBundle`.

Wraps Docling's `DocumentConverter` with every knob from `ExtractionConfig`, prevents a
known crash in the default PDF backend rather than only detecting it after the fact, and
translates Docling's own exceptions into this module's typed errors. Built primarily to
be called inline by an ingestion pipeline (extract -> chunk -> embed -> write), so the
`ExtractionBundle` it returns is the main product -- the on-disk cache under
`StorageConfig.base_dir` (a temp folder under the project directory by default, not a
permanent user-level location) is opt-in via `extract(..., persist=True)`, not automatic.
Picture/table image files are the one exception: they're written whenever image
generation is on, regardless of `persist`, since metadata needs a real file to point at.

Use `DoclingExtractor` as a context manager (or call `.cleanup()` directly) once a run is
done and anything worth keeping has been copied elsewhere -- that removes the whole
`StorageConfig.base_dir` tree:

    with DoclingExtractor(config) as extractor:
        bundle = extractor.extract(path)
        ...  # copy whichever picture/table files the pipeline actually needs
    # base_dir removed here
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from . import _bootstrap  # noqa: F401 -- must run before the docling imports below

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument

from .config import ExtractionConfig
from .errors import ExtractionFailedError, UnsupportedFormatError
from .images import build_picture_records, build_table_image_paths
from .metadata import build_document_metadata
from .models import ExtractionBundle
from .tables import build_table_records


def _build_pdf_pipeline_options(config: ExtractionConfig) -> PdfPipelineOptions:
    """Options shared by PDF and single-image inputs -- both go through Docling's
    paginated pipeline, per the converter's own default format_options."""
    opts = PdfPipelineOptions()
    opts.do_ocr = config.do_ocr
    if config.ocr_options is not None:
        opts.ocr_options = config.ocr_options
    opts.do_table_structure = config.do_table_structure
    opts.table_structure_options = TableStructureOptions(
        do_cell_matching=config.do_cell_matching, mode=config.table_mode
    )
    # get_image() (used for picture/table crops) crops from the rendered page bitmap,
    # so Docling has to rasterize pages whenever crops are wanted -- but nothing in this
    # module saves those full-page renders as files, only the crops taken from them.
    opts.generate_page_images = config.generate_picture_images
    opts.generate_picture_images = config.generate_picture_images
    opts.images_scale = config.images_scale
    opts.do_picture_classification = config.do_picture_classification
    opts.do_code_enrichment = config.do_code_enrichment
    opts.do_formula_enrichment = config.do_formula_enrichment
    opts.accelerator_options = config.accelerator_options()
    if config.artifacts_path is not None:
        opts.artifacts_path = config.artifacts_path
    return opts


def _build_converter(config: ExtractionConfig) -> DocumentConverter:
    pdf_options = _build_pdf_pipeline_options(config)
    return DocumentConverter(
        format_options={
            # The default docling-parse C++ backend accumulates memory per page and can
            # crash with std::bad_alloc past ~page 20 on larger PDFs (known upstream issue,
            # docling-project/docling-parse#227 -- reproduced deliberately in
            # docling_functionalities.ipynb section 9). pypdfium2 runs with constant memory,
            # so it replaces the default backend outright instead of only handling the
            # crash after the fact.
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options, backend=PyPdfiumDocumentBackend),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_options),
        }
        # Every other Docling-supported format (DOCX, PPTX, XLSX, HTML, CSV, Markdown,
        # EPUB, XML variants, ...) is left on the converter's own defaults -- those
        # backends parse native structure directly and don't take OCR/table/image options.
    )


class DoclingExtractor:
    """Parses any Docling-supported document into a `DoclingDocument` plus table/
    picture metadata. Unsupported or corrupt inputs raise `UnsupportedFormatError`/
    `ExtractionFailedError` -- never a raw Docling traceback."""

    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()
        self._converter = _build_converter(self.config)

    def __enter__(self) -> "DoclingExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the entire `StorageConfig.base_dir` tree for this extractor. Call this
        once anything worth keeping (e.g. specific picture/table files) has already been
        copied out -- there's no undo."""
        shutil.rmtree(self.config.storage.base_dir, ignore_errors=True)

    def extract(self, source: str | Path, *, persist: bool = False) -> ExtractionBundle:
        """Parse `source` and return an `ExtractionBundle`.

        `persist` controls only the re-parse-skip cache (`doc.json`/`doc.md`/
        `manifest.json`) -- pass `True` for documents expensive enough that avoiding a
        re-parse is worth it (a 100+ page filing takes minutes; a two-page memo doesn't).
        A cache from an earlier `persist=True` call is still used regardless of this
        call's value -- `persist` only decides whether *this* call writes one.

        Picture/table image files are unaffected by `persist`: they're durable ingestion
        output referenced by metadata (there's no vision embedder to retrieve them any
        other way), not a cache, so they're written whenever image generation is on in
        the config, every time.
        """
        source = Path(source)
        if not source.exists():
            raise UnsupportedFormatError(f"Source file does not exist: {source}", source=source)

        doc_stem = source.stem
        storage = self.config.storage
        cache_dir = storage.doc_dir(doc_stem)
        manifest_path = f"{cache_dir}/manifest.json"
        doc_json_path = f"{cache_dir}/doc.json"

        if os.path.exists(manifest_path) and os.path.exists(doc_json_path):
            return self._load_from_cache(doc_json_path, manifest_path)

        result = self._convert(source)
        doc = result.document

        metadata = build_document_metadata(result, doc_stem)
        if metadata.status == ConversionStatus.FAILURE.value:
            raise ExtractionFailedError(
                f"Docling conversion failed for {source}: {metadata.errors}", source=source
            )

        images_dir = storage.images_dir(doc_stem)
        tables_dir = storage.tables_dir(doc_stem) if self.config.save_table_csv else None

        tables = build_table_records(doc, tables_dir)
        if self.config.save_table_crops:
            table_images = build_table_image_paths(doc, images_dir)
            for table in tables:
                table.image_path = table_images.get(table.table_id)

        pictures = build_picture_records(doc, images_dir, exclude_labels=self.config.picture_exclude_labels)

        bundle = ExtractionBundle(
            document=doc,
            doc_stem=doc_stem,
            cache_dir=cache_dir,
            metadata=metadata,
            tables=tables,
            pictures=pictures,
        )

        if persist:
            os.makedirs(cache_dir, exist_ok=True)
            with open(doc_json_path, "w", encoding="utf-8") as f:
                f.write(doc.model_dump_json())
            with open(f"{cache_dir}/doc.md", "w", encoding="utf-8") as f:
                f.write(doc.export_to_markdown())
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(bundle.model_dump(mode="json", exclude={"document"}), ensure_ascii=False))
        return bundle

    def _convert(self, source: Path):
        try:
            return self._converter.convert(
                str(source),
                page_range=self.config.page_range or (1, sys.maxsize),
                max_num_pages=self.config.max_num_pages or sys.maxsize,
                max_file_size=self.config.max_file_size or sys.maxsize,
            )
        except ConversionError as e:
            raise UnsupportedFormatError(str(e), source=source) from e

    def _load_from_cache(self, doc_json_path: str, manifest_path: str) -> ExtractionBundle:
        with open(doc_json_path, encoding="utf-8") as f:
            doc = DoclingDocument.model_validate_json(f.read())
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        return ExtractionBundle(document=doc, **manifest)
