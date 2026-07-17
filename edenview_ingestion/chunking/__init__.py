from edenview_ingestion.docling_parsing import _bootstrap  # noqa: F401 -- must run before the docling imports below

from . import contextual, hybrid_docling, image_descriptions, parent_child, recursive_overlap
from .config import ContextualConfig, HybridDoclingConfig, ParentChildConfig, RecursiveOverlapConfig
from .errors import ChunkingError, ContextGenerationError
from .image_descriptions import generate_image_description_chunks
from .models import Chunk, ChunkImage

# Strategy name -> chunk(bundle, config) callable. The pipeline layer dispatches on this
# by name (pulled from the DuckDB catalog), so callers never need to know which module a
# strategy lives in. Adding a new strategy later is just a new module implementing the
# same `(ExtractionBundle, config) -> list[Chunk]` shape, plus one line here.
#
# image_descriptions isn't in here -- it's not a base splitting strategy, it's a
# composable addition on top of any of the four (see image_descriptions.py). The
# pipeline layer calls generate_image_description_chunks(bundle) separately and
# concatenates the result onto whichever base strategy's output it's using.
CHUNKERS = {
    recursive_overlap.STRATEGY: recursive_overlap.chunk,
    hybrid_docling.STRATEGY: hybrid_docling.chunk,
    parent_child.STRATEGY: parent_child.chunk,
    contextual.STRATEGY: contextual.chunk,
}

__all__ = [
    "Chunk",
    "ChunkImage",
    "ChunkingError",
    "ContextGenerationError",
    "RecursiveOverlapConfig",
    "HybridDoclingConfig",
    "ParentChildConfig",
    "ContextualConfig",
    "CHUNKERS",
    "generate_image_description_chunks",
]
