"""Table export: every `TableItem` becomes a `TableRecord` carrying a Markdown string
(for inlining into the chunk text stream so semantic/BM25 search can find table content
directly -- always produced, in-memory) and, optionally, a CSV file for exact structured
lookup. No HTML export -- CSV + Markdown cover structured lookup and chunk-text inlining
respectively, and a third format alongside them is dead weight. No table image crops
here either -- see `images.py`'s `build_table_image_paths`, which is opt-in."""

from __future__ import annotations

import os

from docling_core.types.doc.document import DoclingDocument

from .models import TableRecord


def build_table_records(doc: DoclingDocument, tables_dir: str | None) -> list[TableRecord]:
    """`tables_dir` truthy -> also write each table to a CSV file under it; `None` ->
    markdown/dataframe stats still get computed (needed for the record either way), just
    nothing is written to disk."""
    records: list[TableRecord] = []
    for i, table in enumerate(doc.tables):
        df = table.export_to_dataframe(doc=doc)
        markdown = table.export_to_markdown(doc=doc)
        prov = table.prov[0] if table.prov else None
        page_no = prov.page_no if prov else None

        csv_path = None
        if tables_dir:
            os.makedirs(tables_dir, exist_ok=True)
            suffix = f"-p{page_no}" if page_no is not None else ""
            csv_path = f"{tables_dir}/table-{i}{suffix}.csv"
            df.to_csv(csv_path, index=False)

        records.append(
            TableRecord(
                table_id=table.self_ref,
                page_no=page_no,
                bbox=prov.bbox.as_tuple() if prov else None,
                caption=table.caption_text(doc) or None,
                num_rows=df.shape[0],
                num_cols=df.shape[1],
                markdown=markdown,
                csv_path=csv_path,
            )
        )
    return records
