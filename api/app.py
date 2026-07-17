"""FastAPI app -- run with:
    PYTHONPATH=. uvicorn api.app:app --reload
Then open http://localhost:8000/docs for interactive Swagger UI testing.

Lives at the project root (not nested under edenview_RAG) since this is meant to be the
single app every domain package mounts routers onto -- edenview_RAG's retrieval routers
today, the future tabular data agent and ADK agents' routers alongside them later, all
still one FastAPI app / one running process (see edenview_progress.md's
single-process deployment model decision -- unchanged by this, still one process, just
one shared app definition instead of nesting it inside whichever domain happened to get
built first).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import catalog as catalog_router
from .routers import chat as chat_router
from .routers import config as config_router
from .routers import documents as documents_router
from .routers import files as files_router
from .routers import ingest as ingest_router
from .routers import search as search_router
from .routers import system as system_router

app = FastAPI(title="Edenview RAG API")

# Local single-user app, not exposed beyond localhost by default -- wide open CORS is
# fine here and gets out of the way of a frontend running on a different dev port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog_router.router)
app.include_router(ingest_router.router)
app.include_router(search_router.router)
app.include_router(files_router.router)
app.include_router(system_router.router)
app.include_router(config_router.router)
app.include_router(chat_router.router)
app.include_router(documents_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}
