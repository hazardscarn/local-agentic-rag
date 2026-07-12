"""S2 — parent document retriever (small-to-big). No LLM calls.

Only child chunks are embedded and stored in Qdrant. Parent chunks live in a
JSON docstore on disk, keyed by parent_id, merged across documents/runs.

Usage:
    python ingest/s2_parent_child.py --pdf data/raw_sources/kerala_fiscal_health_2026.pdf --space kerala_finance
"""

import argparse
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import (
    DOCSTORE_DIR,
    create_collection,
    embed_texts,
    get_docling_markdown,
    get_qdrant_client,
    load_docstore,
    make_point,
    make_point_id,
    save_docstore,
    upsert_points,
)

STRATEGY = "s2_parent_child"

# Stable namespace for deriving deterministic parent IDs (separate from Qdrant point IDs).
_PARENT_ID_NAMESPACE = uuid.UUID("b4f2e3d5-6c7d-4b8e-9f0a-1b2c3d4e5f6a")


def _parent_id(space: str, stem: str, parent_index: int) -> str:
    return str(uuid.uuid5(_PARENT_ID_NAMESPACE, f"{space}|{stem}|{parent_index}"))


def ingest(
    pdf_path: str,
    space: str,
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 128,
):
    markdown, stem = get_docling_markdown(pdf_path)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
    parents = parent_splitter.create_documents([markdown])
    print(f"[{stem}] Split into {len(parents)} parent chunks")

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=0)

    docstore = load_docstore(space)
    child_texts = []
    child_meta = []  # (parent_id, child_index_within_doc)

    child_counter = 0
    for parent_index, parent in enumerate(parents):
        pid = _parent_id(space, stem, parent_index)
        docstore[pid] = parent.page_content

        for child in child_splitter.create_documents([parent.page_content]):
            child_texts.append(child.page_content)
            child_meta.append((pid, child_counter))
            child_counter += 1

    print(f"[{stem}] Split into {len(child_texts)} child chunks")

    save_docstore(space, docstore)
    print(f"[{stem}] Saved {len(docstore)} parent chunks to {DOCSTORE_DIR / f'{space}.json'}")

    vectors = embed_texts(child_texts)

    points = []
    for (pid, child_index), text, vec in zip(child_meta, child_texts, vectors):
        point_id = make_point_id(space, STRATEGY, stem, child_index)
        payload = {
            "parent_id": pid,
            "strategy": STRATEGY,
            "source_doc": stem,
        }
        points.append(make_point(point_id, text, vec, payload))

    client = get_qdrant_client()
    collection = create_collection(client, space, STRATEGY)
    upsert_points(client, collection, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to source PDF")
    parser.add_argument("--space", required=True, help="Logical RAG space name, e.g. kerala_finance")
    parser.add_argument("--parent-chunk-size", type=int, default=1024)
    parser.add_argument("--child-chunk-size", type=int, default=128)
    args = parser.parse_args()
    ingest(args.pdf, args.space, args.parent_chunk_size, args.child_chunk_size)
