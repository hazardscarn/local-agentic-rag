"""Image-description chunks -- an addition composable on top of any of the 4 base
strategies, not a competing 5th one (see conversation: this exists because a chunk of
prose near a figure won't retrieve for a query that's really about the figure's content
unless the figure's own content is itself an embeddable, retrievable chunk).

Reads `PictureRecord.description`, populated by
`docling_parsing.generate_picture_descriptions()` (a separate post-extraction step, not
a Docling pipeline option -- see that module's docstring for why) using the config.yaml
`picture_description_llm` over local Ollama. No LLM calls happen here -- this module
only turns already-generated descriptions into standalone, embeddable Chunk objects. If
that step was never run against the bundle, every picture's `description` is None and
this returns an empty list.

Each description becomes its own chunk (not appended into a neighboring text chunk) so
it has its own retrieval surface -- blending it into whatever chunk happens to be
nearby would dilute that chunk's embedding and risk still missing a sharp
image-specific query.
"""

from __future__ import annotations

from edenview_ingestion.docling_parsing import ExtractionBundle

from .models import Chunk, ChunkImage, make_chunk_id

STRATEGY = "image_description"


def generate_image_description_chunks(bundle: ExtractionBundle) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for picture in bundle.pictures:
        if not picture.description:
            continue

        images = (
            [
                ChunkImage(
                    picture_id=picture.picture_id,
                    image_path=picture.image_path,
                    caption=picture.caption,
                    page_no=picture.page_no,
                    kind="picture",
                )
            ]
            if picture.image_path
            else []
        )

        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(bundle.metadata.file_hash, STRATEGY, index),
                chunk_index=index,
                text=picture.description,
                embed_text=picture.description,
                strategy=STRATEGY,
                kind="image_description",
                doc_stem=bundle.doc_stem,
                file_hash=bundle.metadata.file_hash,
                page_no=picture.page_no,
                doc_item_refs=[picture.picture_id],
                images=images,
            )
        )
        index += 1

    return chunks
