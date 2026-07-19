"""System-info endpoint -- thin wrapper over edenview_ingestion.system_inspector, for a
future model-selection UI to know what's actually viable to run on this machine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from edenview_ingestion import catalog
from edenview_ingestion.system_inspector import (
    OllamaModelCapabilities,
    SystemSpecs,
    get_system_specs,
    list_ollama_models_with_capabilities,
    unload_all_ollama_models,
    unload_ollama_model,
)

from ..schemas import ClearStaleJobsResponse, UnloadAllModelsResponse, UnloadModelRequest

router = APIRouter(tags=["system"])


@router.get("/system/info", response_model=SystemSpecs)
def system_info():
    return get_system_specs()


@router.get("/system/ollama/models", response_model=list[OllamaModelCapabilities])
def ollama_models_with_capabilities():
    """Every pulled model's name/size/capabilities -- for filtering a model-selection
    dropdown to only options that will actually work (e.g. the agentic RAG pipeline's
    agent model needs "tools"). Not part of /system/info since that's polled every
    4s/30s by the sidebar/system monitor -- see
    list_ollama_models_with_capabilities()'s own docstring."""
    return list_ollama_models_with_capabilities()


@router.post("/system/ollama/unload", status_code=204)
def ollama_unload(body: UnloadModelRequest):
    try:
        unload_ollama_model(body.model)
    except Exception as e:
        raise HTTPException(502, f"Failed to unload {body.model!r}: {e}") from e


@router.post("/system/ollama/unload-all", response_model=UnloadAllModelsResponse)
def ollama_unload_all():
    """Frees the RAM/VRAM every currently-loaded Ollama model is holding, in one call --
    the bulk counterpart to /system/ollama/unload, for a single "reset everything"
    action instead of unloading models used by embedding, chat, contextual chunking,
    picture description, or the agent one at a time. Always safe to call, including
    with nothing loaded (returns an empty list, not an error). Only reclaims memory
    held *idle* -- Ollama has no cancel endpoint, so a call actively streaming right
    now still finishes on its own; see unload_all_ollama_models()'s docstring."""
    return UnloadAllModelsResponse(models_unloaded=unload_all_ollama_models())


@router.post("/system/jobs/clear-stale", response_model=ClearStaleJobsResponse)
def clear_stale_jobs():
    """Marks every "queued"/"running" ingestion job left behind by a backend that
    crashed or was restarted mid-job as "error" -- see
    edenview_ingestion/catalog/crud.py's clear_stale_jobs(), the single source of
    truth this shares with scripts/fresh_start.py. Always safe to call, including
    with nothing actually stale (returns an empty result, not an error) -- this only
    touches job status rows, never real documents/collections/chat data."""
    cleared = catalog.crud.clear_stale_jobs()
    return ClearStaleJobsResponse(
        cleared_count=len(cleared),
        cleared_filenames=[j.filename or j.doc_id or j.job_id for j in cleared],
    )
