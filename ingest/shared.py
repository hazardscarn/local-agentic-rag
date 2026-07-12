"""Shared building blocks for all ingestion strategies: Docling parsing (cached
per document), bge-m3 dense embedding via Ollama, BM25 sparse embedding via
FastEmbed, and Qdrant collection helpers.

Collections are namespaced by "space" (a logical RAG project, e.g. "kerala_finance")
so multiple documents can be ingested incrementally into the same collection per
strategy: collection name = f"{space}_{strategy}" (e.g. "kerala_finance_s1_overlap").
"""

import json
import uuid
from pathlib import Path

import ollama
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_db")
DOCSTORE_DIR = PROJECT_ROOT / "s2_docstore"
DENSE_MODEL = "bge-m3"
DENSE_SIZE = 1024
SPARSE_MODEL = "Qdrant/bm25"

# Stable namespace for deriving deterministic point IDs — re-ingesting the same
# document/strategy/chunk-index overwrites the same Qdrant point instead of duplicating it.
_POINT_ID_NAMESPACE = uuid.UUID("a3f1e2d4-5b6c-4a7d-8e9f-0a1b2c3d4e5f")

_SPARSE_MODEL = None  # lazy-loaded singleton, stays resident across a script run


def doc_stem(pdf_path: str) -> str:
    """Stable identifier for a source document, derived from its filename."""
    return Path(pdf_path).stem


def get_docling_doc(pdf_path: str) -> tuple[DoclingDocument, str]:
    """Parse a PDF with Docling, or load from cache if already parsed.

    Returns (DoclingDocument, doc_stem). Cache lives under cache/<doc_stem>/.
    """
    stem = doc_stem(pdf_path)
    doc_cache_dir = CACHE_DIR / stem
    cache_json = doc_cache_dir / "doc.json"
    cache_md = doc_cache_dir / "doc.md"

    if cache_json.exists():
        print(f"[{stem}] Loading cached Docling document...")
        with open(cache_json, encoding="utf-8") as f:
            return DoclingDocument.model_validate_json(f.read()), stem

    print(f"[{stem}] Parsing PDF with Docling (this may take a few minutes)...")
    pipeline_options = PdfPipelineOptions()
    # These reports are born-digital (selectable text), so OCR is unneeded.
    pipeline_options.do_ocr = False
    pipeline_options.generate_page_images = False
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device="cpu")
    # The default docling-parse C++ backend accumulates memory per page and crashes
    # with std::bad_alloc partway through longer documents regardless of OCR/image
    # settings (known upstream bug: docling-project/docling-parse#227). pypdfium2
    # runs with constant memory, so it's used as the PDF backend here instead.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document

    doc_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_json, "w", encoding="utf-8") as f:
        f.write(doc.model_dump_json())
    with open(cache_md, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())

    print(f"[{stem}] Cached to {cache_json} and {cache_md}")
    return doc, stem


def get_docling_markdown(pdf_path: str) -> tuple[str, str]:
    """Convenience wrapper: get cached Markdown export for a document."""
    stem = doc_stem(pdf_path)
    cache_md = CACHE_DIR / stem / "doc.md"
    if not cache_md.exists():
        get_docling_doc(pdf_path)  # populates the cache
    return cache_md.read_text(encoding="utf-8"), stem


def _get_sparse_model() -> SparseTextEmbedding:
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        print(f"Loading {SPARSE_MODEL} sparse embedding model...")
        _SPARSE_MODEL = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _SPARSE_MODEL


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense embeddings via Ollama's bge-m3."""
    response = ollama.embed(model=DENSE_MODEL, input=texts)
    return response["embeddings"]


def embed_sparse(texts: list[str]) -> list[dict]:
    """BM25 sparse embeddings via FastEmbed. Returns raw term frequencies —
    Qdrant's Modifier.IDF (set on the collection) completes the BM25 scoring."""
    model = _get_sparse_model()
    results = []
    for emb in model.embed(texts):
        results.append(
            {
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist(),
            }
        )
    return results


def embed_texts(texts: list[str], batch_size: int = 32) -> list[dict]:
    """Embed a list of texts with dense (Ollama bge-m3) + sparse (FastEmbed BM25).

    Returns one dict per text: {"dense": [...], "sparse": {"indices": [...], "values": [...]}}
    """
    from tqdm import tqdm

    results = []
    batches = range(0, len(texts), batch_size)
    for i in tqdm(batches, desc="Embedding", unit="batch", total=(len(texts) + batch_size - 1) // batch_size):
        batch = texts[i : i + batch_size]
        dense_vecs = embed_dense(batch)
        sparse_vecs = embed_sparse(batch)
        for dense, sparse in zip(dense_vecs, sparse_vecs):
            results.append({"dense": dense, "sparse": sparse})
    return results


def get_qdrant_client() -> QdrantClient:
    """Local path client — creates ./qdrant_db/ folder on first run."""
    return QdrantClient(path=QDRANT_PATH)


def collection_name(space: str, strategy: str) -> str:
    return f"{space}_{strategy}"


def create_collection(client: QdrantClient, space: str, strategy: str) -> str:
    """Create the collection for this space+strategy if it doesn't already exist."""
    name = collection_name(space, strategy)
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        print(f"Collection '{name}' already exists — skipping creation")
        return name
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"Created collection: {name}")
    return name


def make_point_id(space: str, strategy: str, stem: str, index: int) -> str:
    """Deterministic point ID so re-ingesting the same doc/strategy overwrites
    rather than duplicates."""
    key = f"{space}|{strategy}|{stem}|{index}"
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, key))


def make_point(point_id: str, text: str, vectors: dict, payload: dict) -> PointStruct:
    """Build a Qdrant PointStruct from text, embed output, and metadata payload."""
    return PointStruct(
        id=point_id,
        vector={
            "dense": vectors["dense"],
            "sparse": SparseVector(
                indices=vectors["sparse"]["indices"],
                values=vectors["sparse"]["values"],
            ),
        },
        payload={"text": text, **payload},
    )


def upsert_points(client: QdrantClient, collection: str, points: list[PointStruct], batch_size: int = 100):
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
    print(f"Upserted {len(points)} points into '{collection}'")


def load_docstore(space: str) -> dict:
    """Load the S2 parent-chunk docstore for a space (parent_id -> text). Empty if none yet."""
    path = DOCSTORE_DIR / f"{space}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_docstore(space: str, store: dict):
    DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCSTORE_DIR / f"{space}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
