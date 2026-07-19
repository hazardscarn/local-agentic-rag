"""Manual test/demo script for the chunking package.

Runs DoclingExtractor on a file, then runs one or more chunking strategies over the
resulting ExtractionBundle, printing chunk counts, a couple of sample chunks, and image
linkage stats per strategy. Mirrors docling_parsing/scripts/test_extraction.py's
conventions (argparse, --keep, --storage-dir).

"contextual" is excluded from the default strategy list since it needs a running Ollama
server with the configured model pulled -- pass it explicitly via --strategies to include it.

Usage:
    python -m edenview_ingestion.chunking.scripts.test_chunking <path> [options]

Examples:
    # All strategies except contextual, first 5 pages, with image crops
    python -m edenview_ingestion.chunking.scripts.test_chunking data/sample_files/covid-19-risk-factors-Japan.pdf --pages 5 --keep

    # Just one strategy
    python -m edenview_ingestion.chunking.scripts.test_chunking data/sample_files/covid-19-risk-factors-Japan.pdf --strategies hybrid_docling

    # Include contextual (needs `ollama pull qwen3:4b` and Ollama running)
    python -m edenview_ingestion.chunking.scripts.test_chunking data/sample_files/wc-2026-regulations.pdf --pages 5 --strategies recursive_overlap,contextual
"""

from __future__ import annotations

import argparse
import sys
import time

# edenview_ingestion must be imported before any docling.* import -- see the matching
# comment in test/chunking/verify_chunking.py.
import edenview_ingestion  # noqa: F401
from docling.datamodel.pipeline_options import TableFormerMode

from edenview_ingestion.chunking import CHUNKERS
from edenview_ingestion.docling_parsing import DoclingExtractor, ExtractionConfig, StorageConfig
from edenview_ingestion.docling_parsing.errors import DoclingParsingError

DEFAULT_STRATEGIES = ["recursive_overlap", "hybrid_docling", "parent_child"]


def _parse_pages(value: str) -> tuple[int, int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return (int(start), int(end))
    return (1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to the document to chunk (any Docling-supported format)")
    parser.add_argument("--pages", type=_parse_pages, default=None, metavar="N|N-M")
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help=f"Comma-separated strategy names, or 'all'. Default: {','.join(DEFAULT_STRATEGIES)} "
        "(contextual excluded by default -- needs Ollama running).",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Don't remove the temp workspace afterwards (default: cleaned up)"
    )
    parser.add_argument("--storage-dir", default="test/chunking", help="Where picture/table crops get written.")
    return parser.parse_args()


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    args = parse_args()
    strategies = list(CHUNKERS.keys()) if args.strategies == "all" else args.strategies.split(",")
    unknown = [s for s in strategies if s not in CHUNKERS]
    if unknown:
        print(f"Unknown strategies: {unknown}. Available: {list(CHUNKERS.keys())}")
        return 1

    config = ExtractionConfig(
        page_range=args.pages,
        table_mode=TableFormerMode.FAST,
        generate_picture_images=True,
        do_picture_classification=True,
        storage=StorageConfig(base_dir=args.storage_dir),
    )

    print(f"Extracting: {args.path}")
    extractor = DoclingExtractor(config)
    try:
        bundle = extractor.extract(args.path)
    except DoclingParsingError as e:
        print(f"\n[{type(e).__name__}] {e}")
        extractor.cleanup()
        return 1

    print(f"doc_stem={bundle.doc_stem} pages={bundle.metadata.num_pages} "
          f"tables={len(bundle.tables)} pictures={len(bundle.pictures)}")

    for strategy in strategies:
        _print_header(f"Strategy: {strategy}")
        t0 = time.time()
        try:
            chunks = CHUNKERS[strategy](bundle)
        except Exception as e:
            print(f"[{type(e).__name__}] {e}")
            continue
        elapsed = time.time() - t0

        embedded = [c for c in chunks if c.kind != "parent"]
        with_images = [c for c in chunks if c.images]
        print(f"chunks={len(chunks)} (embeddable={len(embedded)})  with_images={len(with_images)}  "
              f"elapsed={elapsed:.1f}s")
        if embedded:
            sizes = [len(c.embed_text) for c in embedded]
            print(f"embed_text length: min={min(sizes)} max={max(sizes)} avg={sum(sizes) / len(sizes):.0f}")

        for c in chunks[:2]:
            preview = c.text[:150].replace("\n", " ")
            print(f"\n- chunk_id={c.chunk_id[:8]}... kind={c.kind} page={c.page_no} "
                  f"headings={c.headings} images={[i.image_path for i in c.images]}")
            print(f"  text: {preview!r}")
            if c.context:
                print(f"  context: {c.context!r}")

    if args.keep:
        print(f"\ntemp workspace kept at {config.storage.base_dir}")
    else:
        extractor.cleanup()
        print("\ntemp workspace removed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
