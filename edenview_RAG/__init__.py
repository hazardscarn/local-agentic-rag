from edenview_ingestion.docling_parsing import _bootstrap  # noqa: F401 -- must run before any docling.* import,
# even though edenview_RAG doesn't touch docling directly today -- see
# edenview_ingestion/docling_parsing/_bootstrap.py and edenview_ingestion/__init__.py
# for why this has to be first regardless of which package a caller imports first.
