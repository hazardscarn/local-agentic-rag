"""Hybrid (dense + sparse, RRF-fused) retrieval against a space+strategy Qdrant collection.

For s2_parent_child, the child chunk is what gets matched, but the parent chunk text
(looked up from the on-disk docstore) is what gets returned — small-to-big retrieval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingest"))

from qdrant_client import models

from shared import collection_name, embed_dense, embed_sparse, get_qdrant_client, load_docstore


def search(space: str, strategy: str, query_text: str, top_k: int = 5, fetch_k: int = 20) -> list[dict]:
    client = get_qdrant_client()
    collection = collection_name(space, strategy)

    dense_vec = embed_dense([query_text])[0]
    sparse_vec = embed_sparse([query_text])[0]

    result = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=fetch_k),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"]),
                using="sparse",
                limit=fetch_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    docstore = load_docstore(space) if strategy == "s2_parent_child" else None

    hits = []
    for point in result.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        if docstore is not None:
            text = docstore.get(payload.get("parent_id"), text)
        hits.append(
            {
                "score": point.score,
                "text": text,
                "source_doc": payload.get("source_doc"),
                "strategy": strategy,
            }
        )
    return hits
