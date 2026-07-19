from edenview_ingestion.docling_parsing import _bootstrap  # noqa: F401 -- must run before any docling.* import,
# regardless of which router (or future domain package -- data agent, ADK agents) gets
# imported first. See edenview_ingestion/docling_parsing/_bootstrap.py.
