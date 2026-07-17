"""Configuration objects for the Docling extraction pipeline and its local storage.

`ExtractionConfig` exposes every pipeline knob demonstrated in the extraction survey
(`docling_exp/docling_functionalities.ipynb`) directly, rather than narrowing them down --
the goal is that everything Docling can do stays reachable through this module now, not
just the options a first pass happened to need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import OcrOptions, TableFormerMode
from pydantic import BaseModel, ConfigDict, Field

from edenview_ingestion.settings import get_num_threads, get_page_batch_size

# docling-project/DocumentFigureClassifier-v2.5's 26 labels, minus the ones with no
# retrieval value for RAG -- decorative/administrative marks, never the document's
# actual content. Everything else (charts, plots, diagrams, tables, maps, photographs,
# screenshots, "other", ...) is kept, since a classifier can't always tell whether e.g.
# a photograph or a full-page image is actually meaningful.
DEFAULT_PICTURE_EXCLUDE_LABELS = frozenset(
    {
        "logo",
        "icon",
        "signature",
        "stamp",
        "qr_code",
        "bar_code",
        "page_thumbnail",
        "crossword_puzzle",
        "music",
    }
)


class StorageConfig(BaseModel):
    """Where picture/table crops (and, if `persist=True` was passed to `extract()`, the
    parse cache) get written.

    Defaults to a temp folder under the current project directory (`.edenview_tmp/`,
    gitignored) rather than a permanent user-level location -- this module is called
    99% of the time as one step of an inline RAG ingestion run, not as something whose
    output should accumulate indefinitely on its own. Call `DoclingExtractor.cleanup()`
    (or use it as a context manager) once you're done with a run to remove it.

    When artifacts genuinely need to persist (e.g. colocated with wherever the caller's
    vector DB stores its data), override `base_dir` to point there instead -- this
    module takes no position on that path; it's the ingestion pipeline's decision.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_dir: str = Field(default_factory=lambda: str(Path.cwd() / ".edenview_tmp"))

    def doc_dir(self, doc_stem: str) -> str:
        return f"{self.base_dir}/cache/{doc_stem}"

    def images_dir(self, doc_stem: str) -> str:
        return f"{self.doc_dir(doc_stem)}/images"

    def tables_dir(self, doc_stem: str) -> str:
        return f"{self.doc_dir(doc_stem)}/tables"


class ExtractionConfig(BaseModel):
    """Pipeline configuration passed straight through to Docling's own option objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Page-window controls -- truncate (page_range) vs. reject outright (the other two)
    page_range: Optional[tuple[int, int]] = None
    max_num_pages: Optional[int] = None
    max_file_size: Optional[int] = None

    # OCR
    do_ocr: bool = True
    ocr_options: Optional[OcrOptions] = None  # None -> Docling's own auto-selected engine

    # Tables -- markdown (for inlining into chunk text) and the DataFrame (for row/col
    # counts) are always produced in-memory on the TableRecord regardless of the two
    # flags below; those only control whether anything gets written to disk, since the
    # markdown copy already carries the table's content into the chunk stream.
    do_table_structure: bool = True
    table_mode: TableFormerMode = TableFormerMode.ACCURATE
    do_cell_matching: bool = True
    save_table_csv: bool = False
    save_table_crops: bool = False

    # Picture/table crops -- saved to disk since there's no vision embedder to retrieve
    # them any other way. Whole-page images are never saved as files (nothing uses
    # them), even though Docling has to rasterize each page internally to produce a
    # crop -- see extractor.py.
    generate_picture_images: bool = False
    images_scale: float = 2.0

    # Non-generative enrichments (classifiers/encoders, not VLMs)
    do_picture_classification: bool = False
    do_code_enrichment: bool = False
    do_formula_enrichment: bool = False

    # Pictures classified into one of these labels are skipped entirely (no
    # PictureRecord, no saved crop) -- noise for RAG retrieval, not the document's
    # actual content. Only takes effect when do_picture_classification is on; without a
    # label to check, nothing gets filtered.
    picture_exclude_labels: frozenset[str] = DEFAULT_PICTURE_EXCLUDE_LABELS

    # Performance / offline operation. num_threads and page_batch_size both come from
    # edenview_ingestion.settings, which auto-detects a sensible value for whatever
    # machine this package actually runs on (see get_num_threads()) unless a user has
    # set an explicit override in Settings -> Performance -- Edenview is meant to be
    # installed and run by anyone, not tuned to one dev machine's core count or GPU, so
    # a hardcoded number here would under-use a bigger machine and doesn't help a
    # smaller one either. accelerator_device=AUTO already lets Docling itself pick
    # CUDA/MPS/CPU per-machine; a user with a CUDA-capable GPU gets it automatically
    # once they've installed a CUDA-enabled torch build for their own hardware (an
    # install-time choice -- see scripts/install_torch.py -- not something this package
    # should hardcode into requirements.txt).
    accelerator_device: AcceleratorDevice = AcceleratorDevice.AUTO
    num_threads: int = Field(default_factory=get_num_threads)
    # Not a Docling PdfPipelineOptions field -- a process-wide Docling setting
    # (docling.datamodel.settings.settings.perf.page_batch_size) applied once per
    # extraction call in extractor.py, since Docling has nowhere to take it per-call.
    page_batch_size: int = Field(default_factory=get_page_batch_size)
    artifacts_path: Optional[str] = None

    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def preview(cls, num_pages: int = 5, **overrides) -> "ExtractionConfig":
        """Fast, low-fidelity pass over the first few pages -- the RAG preview path
        (`POST /rag/preview` in edenview_plan.md)."""
        defaults = dict(
            page_range=(1, num_pages),
            table_mode=TableFormerMode.FAST,
            generate_picture_images=False,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def full(cls, **overrides) -> "ExtractionConfig":
        """Full-fidelity pass over the whole document -- the real ingestion path."""
        defaults = dict(
            table_mode=TableFormerMode.ACCURATE,
            generate_picture_images=True,
            do_picture_classification=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    def accelerator_options(self) -> AcceleratorOptions:
        return AcceleratorOptions(num_threads=self.num_threads, device=self.accelerator_device)
