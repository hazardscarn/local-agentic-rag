"""System-info endpoint -- thin wrapper over edenview_ingestion.system_inspector, for a
future model-selection UI to know what's actually viable to run on this machine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from edenview_ingestion.system_inspector import SystemSpecs, get_system_specs, unload_ollama_model

from ..schemas import UnloadModelRequest

router = APIRouter(tags=["system"])


@router.get("/system/info", response_model=SystemSpecs)
def system_info():
    return get_system_specs()


@router.post("/system/ollama/unload", status_code=204)
def ollama_unload(body: UnloadModelRequest):
    try:
        unload_ollama_model(body.model)
    except Exception as e:
        raise HTTPException(502, f"Failed to unload {body.model!r}: {e}") from e
