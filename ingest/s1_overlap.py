"""S1 — overlap chunking (baseline). No LLM calls.

Usage:
    python ingest/s1_overlap.py --pdf data/raw_sources/kerala_fiscal_health_2026.pdf --space kerala_finance
"""

import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import (
    create_collection,
    embed_texts,
    get_docling_markdown,
    get_qdrant_client,
    make_point,
    make_point_id,
    upsert_points,
)

STRATEGY = "s1_overlap"


def ingest(pdf_path: str, space: str, chunk_size: int = 512, chunk_overlap: int = 50):
    markdown, stem = get_docling_markdown(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.create_documents([markdown])
    print(f"[{stem}] Split into {len(chunks)} chunks")

    texts = [c.page_content for c in chunks]
    vectors = embed_texts(texts)

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        point_id = make_point_id(space, STRATEGY, stem, i)
        payload = {
            "chunk_index": i,
            "strategy": STRATEGY,
            "source_doc": stem,
            "char_start": chunk.metadata.get("start_index", 0),
        }
        points.append(make_point(point_id, chunk.page_content, vec, payload))

    client = get_qdrant_client()
    collection = create_collection(client, space, STRATEGY)
    upsert_points(client, collection, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to source PDF")
    parser.add_argument("--space", required=True, help="Logical RAG space name, e.g. kerala_finance")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()
    ingest(args.pdf, args.space, args.chunk_size, args.chunk_overlap)
