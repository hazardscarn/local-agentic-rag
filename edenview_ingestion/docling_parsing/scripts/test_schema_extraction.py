"""Manual test/demo script for `docling_parsing.schema_extraction` (custom-schema
extraction -- the one place in this package that runs a VLM).

Give it a PDF or image and a schema (as a JSON string or a JSON file), and it prints
back the structured fields Docling's DocumentExtractor pulled out, per page. Heavier
than the parsing-side test script: first run downloads the selected model's weights
from Hugging Face (several GB for the default NuExtract-2B) and each page takes real
inference time -- not something to run casually or in a loop.

Usage:
    python -m edenview_ingestion.docling_parsing.scripts.test_schema_extraction <path> [options]

Examples:
    # Inline JSON-string template (Docling's own "string template" form)
    python -m edenview_ingestion.docling_parsing.scripts.test_schema_extraction data/sample_files/invoice.pdf --template "{\\"bill_no\\": \\"string\\", \\"total\\": \\"float\\"}"

    # A larger schema kept in a file instead
    python -m edenview_ingestion.docling_parsing.scripts.test_schema_extraction data/sample_files/filing.pdf --template-file schema.json

    # Use Granite Vision instead of the default NuExtract-2B
    python -m edenview_ingestion.docling_parsing.scripts.test_schema_extraction data/sample_files/invoice.pdf --template "{\\"total\\": \\"float\\"}" --model granite_vision

    # No --template given -- runs a generic demo schema just to prove the path works
    python -m edenview_ingestion.docling_parsing.scripts.test_schema_extraction data/sample_files/invoice.pdf

    # Limit to specific pages -- worth doing on anything more than a few pages, since
    # each page costs real VLM inference time
    python -m edenview_ingestion.docling_parsing.scripts.test_schema_extraction data/sample_files/filing.pdf --template-file schema.json --pages 3-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from edenview_ingestion.docling_parsing.errors import DoclingParsingError
from edenview_ingestion.docling_parsing.schema_extraction import extract_schema

DEFAULT_TEMPLATE = '{"title": "string", "summary": "string"}'


def _parse_pages(value: str) -> tuple[int, int]:
    """"N" -> first N pages (1, N); "N-M" -> an explicit range (N, M); a single page
    is just "N-N"."""
    if "-" in value:
        start, end = value.split("-", 1)
        return (int(start), int(end))
    return (1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to a PDF or image file (the only two formats this path supports)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--template", help='JSON-shaped schema string, e.g. \'{"total": "float"}\'')
    group.add_argument("--template-file", help="Path to a JSON file with a (possibly nested) schema dict")
    parser.add_argument(
        "--model", choices=["nuextract", "granite_vision"], default="nuextract", help="Default: nuextract"
    )
    parser.add_argument(
        "--pages",
        type=_parse_pages,
        default=None,
        metavar="N|N-M",
        help='Limit to the first N pages ("--pages 10"), an explicit range '
        '("--pages 5-10"), or a single page ("--pages 7-7"). Default: no limit -- '
        "worth setting on anything more than a few pages given the per-page VLM cost.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.template_file:
        with open(args.template_file, encoding="utf-8") as f:
            template = json.load(f)
    elif args.template:
        template = args.template
    else:
        template = DEFAULT_TEMPLATE
        print(f"No --template/--template-file given -- using a generic demo schema: {template}")

    print(f"Extracting structured fields from: {args.path}")
    print(f"Model: {args.model} (first run downloads weights from Hugging Face -- can take a while)")
    print(f"Template: {template}")
    print(f"Pages: {args.pages or 'all'}")

    t0 = time.time()
    try:
        pages = extract_schema(args.path, template, model=args.model, page_range=args.pages)
    except DoclingParsingError as e:
        print(f"\n[{type(e).__name__}] {e}")
        return 1
    elapsed = time.time() - t0

    print(f"\nElapsed: {elapsed:.1f}s -- {len(pages)} page(s)")
    for page in pages:
        print(f"\n--- page {page.page_no} ---")
        print(json.dumps(page.extracted_data, indent=2, ensure_ascii=False))
        if page.errors:
            print("errors:", page.errors)

    return 0


if __name__ == "__main__":
    sys.exit(main())
