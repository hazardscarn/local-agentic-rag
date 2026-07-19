"""Resolves `Chunk.doc_item_refs` against an `ExtractionBundle`'s pictures/tables to
populate `Chunk.images` -- the join that lets a retrieved chunk pull its images back in.

Two ways a picture ends up linked to a chunk:
  - direct: the picture's own self_ref is itself in the chunk's doc_item_refs (the
    picture literally falls inside the chunk's span -- only possible for the
    HybridChunker-based strategies, since only they carry real doc-item provenance).
  - proximate: one of the chunk's doc_item_refs is a text item that the picture already
    named as a nearest neighbour (`PictureRecord.linked_text_refs`, built at extraction
    time in docling_parsing/images.py).
Tables are only linked directly -- a table's own markdown is already inlined into
whichever chunk contains it, so proximity linking would just be noise.
"""

from __future__ import annotations

from edenview_ingestion.docling_parsing import ExtractionBundle

from .models import Chunk, ChunkImage


def attach_images(chunks: list[Chunk], bundle: ExtractionBundle) -> None:
    """Mutates each chunk's `images` list in place."""
    picture_by_ref = {p.picture_id: p for p in bundle.pictures if p.image_path}
    table_by_ref = {t.table_id: t for t in bundle.tables if t.image_path}

    proximity: dict[str, list] = {}
    for picture in bundle.pictures:
        if not picture.image_path:
            continue
        for ref in picture.linked_text_refs:
            proximity.setdefault(ref, []).append(picture)

    for chunk in chunks:
        seen: set[str] = set()
        images: list[ChunkImage] = []

        for ref in chunk.doc_item_refs:
            direct_picture = picture_by_ref.get(ref)
            if direct_picture is not None and direct_picture.picture_id not in seen:
                seen.add(direct_picture.picture_id)
                images.append(
                    ChunkImage(
                        picture_id=direct_picture.picture_id,
                        image_path=direct_picture.image_path,
                        caption=direct_picture.caption,
                        page_no=direct_picture.page_no,
                        kind="picture",
                    )
                )

            table = table_by_ref.get(ref)
            if table is not None and table.table_id not in seen:
                seen.add(table.table_id)
                images.append(
                    ChunkImage(
                        picture_id=table.table_id,
                        image_path=table.image_path,
                        caption=table.caption,
                        page_no=table.page_no,
                        kind="table",
                    )
                )

            for nearby in proximity.get(ref, []):
                if nearby.picture_id in seen:
                    continue
                seen.add(nearby.picture_id)
                images.append(
                    ChunkImage(
                        picture_id=nearby.picture_id,
                        image_path=nearby.image_path,
                        caption=nearby.caption,
                        page_no=nearby.page_no,
                        kind="picture",
                    )
                )

        chunk.images = images
