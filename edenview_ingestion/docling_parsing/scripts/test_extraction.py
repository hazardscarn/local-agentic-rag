"""Manual test/demo script for `docling_parsing`.

Takes a file path the same way a real caller would -- a UI file-browse action ultimately
hands the backend a local path (or an uploaded file saved to a temp path first), so this
just accepts a path on the command line and runs it through DoclingExtractor, printing
everything the module extracted: status/confidence, tables, and pictures.

By default the temp workspace (picture/table crop files, and the parse cache if
--persist was used) is removed at the end of the run, same as real inline usage would.
Pass --keep to leave it on disk so you can inspect the saved files.

Usage:
    python -m edenview_ingestion.docling_parsing.scripts.test_extraction <path> [options]

Examples:
    # Full capability pass (OCR, tables, crops, classification) on a small file
    python -m edenview_ingestion.docling_parsing.scripts.test_extraction data/sample_files/covid-19-risk-factors-Japan.pdf --keep

    # Fast text-only pass, first 5 pages, no crops
    python -m edenview_ingestion.docling_parsing.scripts.test_extraction data/sample_files/wc-2026-regulations.pdf --pages 5 --no-images

    # Just pages 5 through 10, or a single page (10-10)
    python -m edenview_ingestion.docling_parsing.scripts.test_extraction data/sample_files/wc-2026-regulations.pdf --pages 5-10

    # Persist the parse cache (worth it for large/slow documents) and keep it around
    python -m edenview_ingestion.docling_parsing.scripts.test_extraction data/sample_files/vistra-20260331.pdf --persist --keep

    # Feed it something Docling can't parse, to see the graceful error path
    python -m edenview_ingestion.docling_parsing.scripts.test_extraction requirements.txt
"""

from __future__ import annotations

import argparse
import sys
import time

from docling.datamodel.pipeline_options import TableFormerMode

from edenview_ingestion.docling_parsing import DoclingExtractor, ExtractionConfig, StorageConfig
from edenview_ingestion.docling_parsing.errors import DoclingParsingError


def _parse_pages(value: str) -> tuple[int, int]:
    """"N" -> first N pages (1, N); "N-M" -> an explicit range (N, M); a single page
    is just "N-N"."""
    if "-" in value:
        start, end = value.split("-", 1)
        return (int(start), int(end))
    return (1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to the document to extract (any Docling-supported format)")
    parser.add_argument(
        "--pages",
        type=_parse_pages,
        default=None,
        metavar="N|N-M",
        help='Limit to the first N pages ("--pages 10"), an explicit range '
        '("--pages 5-10"), or a single page ("--pages 7-7"). Default: no limit.',
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (default: on)")
    parser.add_argument(
        "--table-mode", choices=["fast", "accurate"], default="accurate", help="TableFormer mode (default: accurate)"
    )
    parser.add_argument("--no-images", action="store_true", help="Skip picture/table crop generation")
    parser.add_argument("--no-classify", action="store_true", help="Skip picture classification")
    parser.add_argument(
        "--persist", action="store_true", help="Write the parse cache to disk (default: in-memory only)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="Don't remove the temp workspace afterwards (default: cleaned up)"
    )
    parser.add_argument(
        "--storage-dir",
        default="test/docling-parsing",
        help="Where picture/table files (and the parse cache, if --persist) get written. "
        "Default: test/docling-parsing (gitignored). Pass the same folder a real vector "
        "DB would use to see how that colocation would look.",
    )
    return parser.parse_args()


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    args = parse_args()

    config = ExtractionConfig(
        page_range=args.pages,
        do_ocr=not args.no_ocr,
        table_mode=TableFormerMode.FAST if args.table_mode == "fast" else TableFormerMode.ACCURATE,
        generate_picture_images=not args.no_images,
        do_picture_classification=not args.no_images and not args.no_classify,
        storage=StorageConfig(base_dir=args.storage_dir),
    )

    print(f"Extracting: {args.path}")
    print(f"Config: {config.model_dump(exclude={'storage'})}")
    print(f"Storage: {config.storage.base_dir}")

    extractor = DoclingExtractor(config)
    t0 = time.time()
    try:
        bundle = extractor.extract(args.path, persist=args.persist)
    except DoclingParsingError as e:
        print(f"\n[{type(e).__name__}] {e}")
        if not args.keep:
            extractor.cleanup()
        return 1
    elapsed = time.time() - t0

    _print_header("Metadata")
    m = bundle.metadata
    print(f"doc_stem:        {bundle.doc_stem}")
    print(f"format:          {m.input_format}")
    print(f"pages:           {m.num_pages}")
    print(f"status:          {m.status}")
    print(f"doc_grade:       {m.doc_grade}")
    print(f"unscored_pages:  {m.unscored_pages}")
    print(f"errors:          {m.errors}")
    print(f"cache_dir:       {bundle.cache_dir}")
    print(f"elapsed:         {elapsed:.1f}s")

    _print_header(f"Tables ({len(bundle.tables)})")
    for t in bundle.tables[:5]:
        print(f"- {t.table_id}  page={t.page_no}  {t.num_rows}x{t.num_cols}  caption={t.caption!r}")
        print(f"  csv={t.csv_path}  image={t.image_path}")
    if len(bundle.tables) > 5:
        print(f"  ... and {len(bundle.tables) - 5} more")

    _print_header(f"Pictures ({len(bundle.pictures)})")
    for p in bundle.pictures[:5]:
        print(
            f"- {p.picture_id}  page={p.page_no}  "
            f"class={p.classification_label}({p.classification_confidence})  linked_text={p.linked_text_refs}"
        )
        print(f"  image={p.image_path}")
    if len(bundle.pictures) > 5:
        print(f"  ... and {len(bundle.pictures) - 5} more")

    # --persist implies keeping the workspace -- writing the cache and then deleting it
    # in the same run would make --persist pointless.
    if args.keep or args.persist:
        print(f"\ntemp workspace kept at {config.storage.base_dir}")
    else:
        extractor.cleanup()
        print("\ntemp workspace removed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
