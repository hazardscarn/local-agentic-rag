"""Verification script for the chunking package -- runs all 4 base strategies plus the
image-description addition against a real sample document and checks structural
invariants (non-empty output, deterministic chunk IDs, valid parent/child links, image
linkage). Not pytest -- a standalone runnable script matching this repo's existing
demo-script convention (see edenview_ingestion/docling_parsing/scripts/test_extraction.py).

Exits 0 if every check passed (skips don't count as failures -- e.g. the
image-description check skips itself if picture_description_llm isn't pulled yet), 1 if
anything genuinely failed.

Usage:
    python -m test.chunking.verify_chunking
"""

from __future__ import annotations

import sys

# edenview_ingestion must be imported before any docling.* import (its __init__ flips
# settings.inference.compile_torch_models=False before Docling bakes in torch.compile
# for the picture classifier -- see docling_parsing/_bootstrap.py). Importing it first
# here, even though the names below come from its submodules, keeps that guarantee
# regardless of import order after this point.
import edenview_ingestion  # noqa: F401
from docling.datamodel.pipeline_options import TableFormerMode

from edenview_ingestion.chunking import CHUNKERS, generate_image_description_chunks
from edenview_ingestion.docling_parsing import (
    DoclingExtractor,
    ExtractionConfig,
    StorageConfig,
    generate_picture_descriptions,
)
from edenview_ingestion.docling_parsing.errors import DoclingParsingError

SAMPLE_PDF = "data/sample_files/covid-19-risk-factors-Japan.pdf"
STORAGE_DIR = "test/chunking/cache"

_passed = 0
_failed = 0
_skipped = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def skip(name: str, reason: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  SKIP  {name}  ({reason})")


def verify_base_strategies() -> None:
    config = ExtractionConfig(
        page_range=(1, 5),
        table_mode=TableFormerMode.FAST,
        generate_picture_images=True,
        do_picture_classification=True,
        storage=StorageConfig(base_dir=STORAGE_DIR),
    )
    extractor = DoclingExtractor(config)
    bundle = extractor.extract(SAMPLE_PDF)
    print(
        f"\nExtracted {bundle.doc_stem}: pages={bundle.metadata.num_pages} "
        f"tables={len(bundle.tables)} pictures={len(bundle.pictures)}"
    )

    for strategy, chunk_fn in CHUNKERS.items():
        print(f"\n[{strategy}]")
        chunks = chunk_fn(bundle)
        check(f"{strategy}: produces chunks", len(chunks) > 0, f"got {len(chunks)}")

        ids = [c.chunk_id for c in chunks]
        check(f"{strategy}: chunk_ids unique", len(ids) == len(set(ids)))

        rerun = chunk_fn(bundle)
        check(f"{strategy}: deterministic chunk_ids across reruns", [c.chunk_id for c in rerun] == ids)

        check(f"{strategy}: every chunk has embed_text", all(c.embed_text for c in chunks))
        check(f"{strategy}: doc_item_refs populated on every chunk", all(c.doc_item_refs for c in chunks))

        if strategy == "parent_child":
            parent_ids = {c.chunk_id for c in chunks if c.kind == "parent"}
            children = [c for c in chunks if c.kind == "child"]
            check("parent_child: has both parents and children", len(parent_ids) > 0 and len(children) > 0)
            check("parent_child: every child's parent_id resolves", all(c.parent_id in parent_ids for c in children))

        if strategy == "contextual":
            with_context = sum(1 for c in chunks if c.context)
            print(f"  {with_context}/{len(chunks)} chunks got an LLM-generated context")
            check("contextual: at least one chunk got context (Ollama reachable)", with_context > 0)

        images_linked = sum(1 for c in chunks if c.images)
        print(f"  {images_linked}/{len(chunks)} chunks linked to an image")

    extractor.cleanup()


def verify_image_descriptions() -> None:
    print("\n[image_description]")
    config = ExtractionConfig(
        page_range=(1, 5),
        table_mode=TableFormerMode.FAST,
        generate_picture_images=True,
        do_picture_classification=True,
        storage=StorageConfig(base_dir=STORAGE_DIR),
    )
    extractor = DoclingExtractor(config)
    try:
        bundle = extractor.extract(SAMPLE_PDF)
    except DoclingParsingError as e:
        skip(
            "image_description: extraction",
            f"{type(e).__name__}: {e}",
        )
        extractor.cleanup()
        return

    # Separate step, not a Docling pipeline option -- see
    # docling_parsing/picture_description.py for why (Docling's own
    # do_picture_description is broken for reasoning-capable local VLMs).
    generate_picture_descriptions(bundle.pictures)

    described = [p for p in bundle.pictures if p.description]
    if not described:
        skip(
            "image_description: any picture got a description",
            "picture_description_llm likely not pulled -- run `ollama pull <config.yaml's picture_description_llm>`",
        )
        extractor.cleanup()
        return

    chunks = generate_image_description_chunks(bundle)
    check("image_description: one chunk per described picture", len(chunks) == len(described))
    check("image_description: every chunk has non-empty text", all(c.text for c in chunks))
    check(
        "image_description: every chunk tagged kind=image_description",
        all(c.kind == "image_description" for c in chunks),
    )
    extractor.cleanup()


def main() -> int:
    verify_base_strategies()
    verify_image_descriptions()

    print(f"\n{'=' * 60}\n{_passed} passed, {_failed} failed, {_skipped} skipped\n{'=' * 60}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
