"""Bitmap export: picture/table crops only, saved to disk (never embedded -- there's no
vision embedder in this stack, so images are made retrievable through metadata instead).
Whole-page renders are never saved as files -- Docling still has to rasterize each page
internally to produce a crop (`generate_page_images` stays on under the hood, see
`extractor.py`), but nothing here writes a page-N.png; only the actual picture/table
regions are worth a file. Each picture also gets its classification label (if enrichment
was on, folded into the filename too) and the self_refs of its nearest surrounding text,
so a retrieved text chunk can pull the image in by metadata lookup."""

from __future__ import annotations

import os
import re

from docling_core.types.doc.document import DoclingDocument, PictureDescriptionData, PictureItem, TextItem

from .models import PictureRecord


def _picture_description(picture: PictureItem) -> str | None:
    """Reads `picture.meta.description.text` -- the current API. `picture.annotations`
    (iterated as a fallback) is deprecated in this docling-core version and, confirmed
    by inspection, the picture-description enrichment stage writes straight to `meta`
    without populating it, so the fallback alone silently returned nothing."""
    if picture.meta is not None and picture.meta.description is not None:
        return picture.meta.description.text or None
    for annotation in picture.annotations:
        if isinstance(annotation, PictureDescriptionData):
            return annotation.text or None
    return None


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def build_table_image_paths(doc: DoclingDocument, images_dir: str) -> dict[str, str]:
    """table_id (self_ref) -> saved crop path, for tables that have a rendered image.
    Opt-in (see ExtractionConfig.save_table_crops) -- most callers don't need a table
    rendered as an image on top of its CSV/markdown."""
    paths: dict[str, str] = {}
    for i, table in enumerate(doc.tables):
        image = table.get_image(doc)
        if image is None:
            continue
        os.makedirs(images_dir, exist_ok=True)
        prov = table.prov[0] if table.prov else None
        suffix = f"-p{prov.page_no}" if prov else ""
        path = f"{images_dir}/table-{i}{suffix}.png"
        image.save(path)
        paths[table.self_ref] = path
    return paths


def _nearest_text_refs(ordered_items: list, picture_index: int, window: int = 1) -> list[str]:
    """self_refs of the `window` nearest text items on each side of a picture, in
    reading order -- the link between a figure and the prose it's discussed in."""
    refs: list[str] = []

    before = [it for it in ordered_items[:picture_index] if isinstance(it, TextItem)]
    refs.extend(it.self_ref for it in before[-window:])

    after = [it for it in ordered_items[picture_index + 1 :] if isinstance(it, TextItem)]
    refs.extend(it.self_ref for it in after[:window])

    return refs


def build_picture_records(
    doc: DoclingDocument, images_dir: str, exclude_labels: frozenset[str] = frozenset()
) -> list[PictureRecord]:
    """`exclude_labels` drops pictures classified into any of those labels entirely (no
    record, no saved crop) -- e.g. logos/icons/signatures, noise for RAG retrieval. Only
    has an effect when picture classification produced a label at all."""
    ordered_items = [item for item, _level in doc.iterate_items()]
    picture_positions = [i for i, item in enumerate(ordered_items) if isinstance(item, PictureItem)]

    records: list[PictureRecord] = []
    for i, picture_index in enumerate(picture_positions):
        picture = ordered_items[picture_index]

        classification_label = classification_confidence = None
        if picture.meta is not None and picture.meta.classification is not None:
            main = picture.meta.classification.get_main_prediction()
            classification_label = main.class_name
            classification_confidence = main.confidence

        if classification_label in exclude_labels:
            continue

        prov = picture.prov[0] if picture.prov else None

        image_path = None
        image = picture.get_image(doc)
        if image is not None:
            os.makedirs(images_dir, exist_ok=True)
            page_part = f"-p{prov.page_no}" if prov else ""
            label_part = f"-{_slug(classification_label)}" if classification_label else ""
            image_path = f"{images_dir}/picture-{i}{page_part}{label_part}.png"
            image.save(image_path)

        records.append(
            PictureRecord(
                picture_id=picture.self_ref,
                page_no=prov.page_no if prov else None,
                bbox=prov.bbox.as_tuple() if prov else None,
                caption=picture.caption_text(doc) or None,
                classification_label=classification_label,
                classification_confidence=classification_confidence,
                image_path=image_path,
                linked_text_refs=_nearest_text_refs(ordered_items, picture_index),
                description=_picture_description(picture),
            )
        )
    return records
