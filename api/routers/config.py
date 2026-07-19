"""Model-settings read/write -- thin wrapper over edenview_ingestion.settings'
get_all_model_settings()/update_model_settings(), which persist to config.yaml on
disk (via ruamel.yaml, preserving its comments) rather than a session-only override."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from edenview_ingestion import settings
from edenview_ingestion.vectorstore.embedding import detect_dense_embedding_dim

from ..schemas import (
    ModelSettings,
    PerformanceSettings,
    UpdateModelSettingsRequest,
    UpdateModelSettingsResponse,
    UpdatePerformanceRequest,
    UpdateWorkspaceRequest,
    WorkspaceBrowseResponse,
    WorkspaceSettings,
)

router = APIRouter(tags=["config"])

# Keys that are looked up fresh on every call (edenview_ingestion.vectorstore.embedding's
# embed_dense()/get_ollama_host()) versus keys baked into a Pydantic class-level default
# at import time (RetrievalConfig.reranker_model, HybridDoclingConfig/ContextualConfig's
# tokenizer_model/ollama_model) or a module-global singleton (embedding.py's
# _SPARSE_MODEL) -- traced through the actual call sites, not assumed. Only the former
# apply without restarting the API server.
RESTART_REQUIRED_KEYS = {
    "tokenizer",
    "sparse_embedding",
    "contextual_llm",
    "picture_description_llm",
    "reranker",
}


@router.get("/system/config", response_model=ModelSettings)
def get_config():
    return settings.get_all_model_settings()


@router.put("/system/config", response_model=UpdateModelSettingsResponse)
def update_config(body: UpdateModelSettingsRequest):
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "No fields provided to update")

    # dense_embedding_dim is derived from dense_embedding, never independently
    # editable -- otherwise the two can silently drift apart (wrong Qdrant vector
    # size, or worse, a same-dimension coincidence that mixes embedding spaces with
    # no error at all). If the caller changed dense_embedding without also
    # explicitly setting dense_embedding_dim, probe the new model's real dimension.
    if "dense_embedding" in updates and "dense_embedding_dim" not in updates:
        try:
            updates["dense_embedding_dim"] = detect_dense_embedding_dim(
                updates["dense_embedding"], ollama_host=updates.get("ollama_host")
            )
        except Exception as e:
            raise HTTPException(
                400, f"Could not reach Ollama to detect {updates['dense_embedding']!r}'s dimension: {e}"
            ) from e

    try:
        updated = settings.update_model_settings(updates)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    restarts = sorted(RESTART_REQUIRED_KEYS & set(updates))
    return UpdateModelSettingsResponse(updated=updated, restart_required=restarts)


def _get_performance_settings() -> PerformanceSettings:
    raw = settings.load_config().get("performance", {})
    return PerformanceSettings(
        num_threads=settings.get_num_threads(),
        page_batch_size=settings.get_page_batch_size(),
        max_concurrent_extractions=settings.get_max_concurrent_extractions(),
        num_threads_is_auto=raw.get("num_threads") is None,
        page_batch_size_is_auto=raw.get("page_batch_size") is None,
        max_concurrent_extractions_is_auto=raw.get("max_concurrent_extractions") is None,
    )


@router.get("/system/performance", response_model=PerformanceSettings)
def get_performance():
    return _get_performance_settings()


@router.put("/system/performance", response_model=PerformanceSettings)
def update_performance(body: UpdatePerformanceRequest):
    """Every field is independently optional -- omitting one leaves it as whatever it
    already was (auto or a prior override); explicitly passing null for a field that
    was already overridden reverts it to auto-detecting instead."""
    updates = body.model_dump(exclude_unset=True)
    if "num_threads" in updates:
        settings.update_num_threads(updates["num_threads"])
    if "page_batch_size" in updates:
        settings.update_page_batch_size(updates["page_batch_size"])
    if "max_concurrent_extractions" in updates:
        settings.update_max_concurrent_extractions(updates["max_concurrent_extractions"])
    return _get_performance_settings()


@router.post("/system/workspace/browse", response_model=WorkspaceBrowseResponse)
def browse_workspace_folder():
    """Opens a native OS folder-picker dialog on this machine via tkinter (bundled
    with Python) -- viable specifically because Edenview is a local, single-user
    app: the API server and the browser tab hitting it always run on the same
    machine, so the dialog this pops up is the user's own desktop, not some
    unrelated remote server's screen. Blocks this request's worker thread until the
    user picks a folder or cancels -- fine, FastAPI runs sync `def` routes in
    Starlette's threadpool, not the event loop."""
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title="Choose Edenview's workspace folder")
    finally:
        root.destroy()
    return WorkspaceBrowseResponse(path=path or None)


@router.get("/system/workspace", response_model=WorkspaceSettings)
def get_workspace():
    root = settings.load_config().get("workspace", {}).get("root", "edenview_data")
    return WorkspaceSettings(root=root, resolved_path=str(settings.get_workspace_root()))


@router.put("/system/workspace", response_model=WorkspaceSettings)
def update_workspace(body: UpdateWorkspaceRequest):
    """Always requires an API server restart to take effect -- the DuckDB connection
    and Qdrant client are opened once per process and cached as singletons (see
    catalog/connection.py's get_connection(), vectorstore/client.py's get_client()),
    against whatever path was resolved the first time each was called. Does not move
    any existing data (see config.yaml's `workspace:` comment)."""
    if not body.root.strip():
        raise HTTPException(400, "Workspace root cannot be empty")
    settings.update_workspace_root(body.root.strip())
    return get_workspace()
