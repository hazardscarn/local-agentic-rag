"""Detects the local machine's specs -- OS, CPU, RAM, GPU/VRAM, and which Ollama models
are actually pulled -- so the app can recommend which models are realistically viable to
run, matching edenview_plan.md's SystemInspector concept.

GPU detection is cross-platform but NVIDIA/Apple Silicon only:
  - NVIDIA: `nvidia-smi` subprocess (not pynvml) -- no extra binary dependency, works on
    Windows/Linux if the driver's installed, gracefully returns [] (not an error) on a
    machine with no NVIDIA GPU or no driver.
  - Apple Silicon: unified memory, no separate VRAM pool the way NVIDIA has one -- its
    usable "GPU memory" is bounded by total system RAM, reported as such rather than
    forced into the vram_total_mb field.
  - AMD (ROCm) / Intel GPUs: not implemented -- rare for this audience's local ML
    workloads. Falls through to no GPU detected, not an error.

Lives under edenview_ingestion (not its own package) for now -- this is a small,
dependency-light utility, and edenview_plan.md's eventual edenview-core split can move
it there without this module's own code changing.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from functools import lru_cache
from typing import Optional

import ollama
import psutil
from pydantic import BaseModel

from edenview_ingestion.settings import get_ollama_host


class GPUInfo(BaseModel):
    name: str
    vendor: str  # "nvidia" | "apple" | "unknown"
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    unified_memory: bool = False  # True for Apple Silicon -- vram_total_mb stays None


class OllamaModelInfo(BaseModel):
    name: str
    size_gb: float


class OllamaInfo(BaseModel):
    available: bool
    host: str
    models: list[OllamaModelInfo] = []
    error: Optional[str] = None


class LoadedModelInfo(BaseModel):
    """One entry from `ollama ps` -- a model currently resident in memory, taking up
    RAM/VRAM whether or not it's actively serving a request right now."""

    name: str
    size_gb: float
    size_vram_gb: float
    expires_at: Optional[datetime] = None  # when Ollama will auto-evict it if idle


class TorchAccelerationInfo(BaseModel):
    """Whether Docling's own models (layout analysis, OCR, table structure, picture
    classification -- all torch-based) can actually use this machine's GPU, as
    distinct from whether a GPU merely exists (`gpus` above) -- a machine can have an
    NVIDIA GPU (`gpus` non-empty) while still running torch CPU-only if the installed
    torch build doesn't have CUDA support, which is a very real and easy-to-miss gap.
    See scripts/install_torch.py for the one-time setup step that closes it."""

    installed: bool
    device: Optional[str] = None  # "cuda" | "mps" | "cpu" -- None if torch isn't installed at all
    gpu_name: Optional[str] = None  # torch's own view of the active CUDA device, if device == "cuda"


class SystemSpecs(BaseModel):
    platform: str  # "Windows" | "Linux" | "Darwin"
    platform_release: str
    architecture: str  # e.g. "AMD64", "arm64"
    cpu_cores_physical: Optional[int]
    cpu_cores_logical: Optional[int]
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[GPUInfo]
    ollama: OllamaInfo
    loaded_models: list[LoadedModelInfo] = []
    torch_acceleration: TorchAccelerationInfo


def _detect_nvidia_gpus() -> list[GPUInfo]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, total_mb, free_mb = parts
        try:
            gpus.append(
                GPUInfo(name=name, vendor="nvidia", vram_total_mb=int(total_mb), vram_free_mb=int(free_mb))
            )
        except ValueError:
            continue
    return gpus


def _detect_apple_silicon_gpu() -> list[GPUInfo]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        )
        chip_name = result.stdout.strip() or "Apple Silicon"
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        chip_name = "Apple Silicon"
    return [GPUInfo(name=f"{chip_name} (Metal, unified memory)", vendor="apple", unified_memory=True)]


def _detect_gpus() -> list[GPUInfo]:
    nvidia = _detect_nvidia_gpus()
    if nvidia:
        return nvidia
    return _detect_apple_silicon_gpu()


def _detect_ollama() -> OllamaInfo:
    host = get_ollama_host() or "http://localhost:11434"
    try:
        client = ollama.Client(host=host)
        response = client.list()
        models = [
            OllamaModelInfo(name=m.model, size_gb=round((m.size or 0) / (1024**3), 2)) for m in response.models
        ]
        return OllamaInfo(available=True, host=host, models=models)
    except Exception as e:
        return OllamaInfo(available=False, host=host, models=[], error=str(e))


def get_loaded_ollama_models() -> list[LoadedModelInfo]:
    """Currently-loaded Ollama models -- these sit in RAM/VRAM until Ollama evicts
    them (on their own `expires_at`, or immediately via unload_ollama_model() below),
    regardless of whether a request is in flight. Returns [] if Ollama isn't
    reachable, same as _detect_ollama()'s own graceful degradation."""
    host = get_ollama_host() or "http://localhost:11434"
    try:
        response = ollama.Client(host=host).ps()
    except Exception:
        return []
    return [
        LoadedModelInfo(
            name=m.model or m.name or "unknown",
            size_gb=round((m.size or 0) / (1024**3), 2),
            size_vram_gb=round((m.size_vram or 0) / (1024**3), 2),
            expires_at=m.expires_at,
        )
        for m in response.models
    ]


def unload_ollama_model(model: str) -> None:
    """Immediately evicts `model` from memory instead of waiting for its keep_alive
    timeout -- Ollama's own documented pattern: an empty-prompt generate() call with
    keep_alive=0 unloads without doing any real generation work."""
    host = get_ollama_host() or "http://localhost:11434"
    ollama.Client(host=host).generate(model=model, prompt="", keep_alive=0)


@lru_cache(maxsize=1)
def _detect_torch_acceleration() -> TorchAccelerationInfo:
    """Cached -- which device torch actually uses is fixed for the lifetime of this
    process (set by whichever wheel got installed, checked once at import time), not
    something that needs re-probing on every call. Uncached, `import torch` alone costs
    several seconds -- get_system_specs() backs /system/info, which the sidebar system
    monitor polls every ~4s, so paying that cost per call made the whole app feel
    sluggish (confirmed: /system/info went from sub-second to ~4.7s consistently,
    every call, before this was cached)."""
    try:
        import torch
    except ImportError:
        return TorchAccelerationInfo(installed=False)

    if torch.cuda.is_available():
        return TorchAccelerationInfo(installed=True, device="cuda", gpu_name=torch.cuda.get_device_name(0))
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return TorchAccelerationInfo(installed=True, device="mps")
    return TorchAccelerationInfo(installed=True, device="cpu")


def get_system_specs() -> SystemSpecs:
    vm = psutil.virtual_memory()
    return SystemSpecs(
        platform=platform.system(),
        platform_release=platform.release(),
        architecture=platform.machine(),
        cpu_cores_physical=psutil.cpu_count(logical=False),
        cpu_cores_logical=psutil.cpu_count(logical=True),
        ram_total_gb=round(vm.total / (1024**3), 1),
        ram_available_gb=round(vm.available / (1024**3), 1),
        gpus=_detect_gpus(),
        ollama=_detect_ollama(),
        loaded_models=get_loaded_ollama_models(),
        torch_acceleration=_detect_torch_acceleration(),
    )
