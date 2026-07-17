"""Detects whether this machine has a usable NVIDIA GPU and, if so, swaps the CPU-only
torch build that `pip install -r requirements.txt` already installed for a CUDA-enabled
build of the *exact same version* -- falling back to leaving the CPU build in place if
no CUDA build actually works here. Run this once during setup, *after*
`pip install -r requirements.txt` (torch must already be installed for this to have
anything to compare against).

Why this exists: Docling's own models (layout analysis, OCR, table structure
recognition, picture classification) are torch-based and run dramatically faster on a
CUDA GPU than on CPU. But *which* torch wheel to install depends entirely on this
specific machine's hardware and driver -- something a static requirements.txt can't
express, and something Edenview (meant to run on any enterprise user's own machine, not
tuned to one dev machine) shouldn't hardcode either.

Deliberately NOT built on a third-party auto-detection tool (e.g. light-the-torch):
that one works by monkey-patching pip's own internals, which pip's maintainers
explicitly don't support doing and which has broken before on a pip upgrade (pip 22.3).
Everything here is a plain `pip install <pkg>==<version> --index-url ...` call --
ordinary, documented pip behavior, the same command format pytorch.org's own
get-started page generates. Slower to write, nothing to go stale out from under us.

Safety:
  - Pins the CUDA install to the *exact version* requirements.txt already resolved,
    never an unpinned `torch` -- an unpinned install could silently pull a newer/older
    torch than what docling/torchvision/transformers actually depend on.
  - Verifies each attempt by actually importing torch and checking
    torch.cuda.is_available() afterward, not just trusting `pip install` exited 0 --
    a wheel can install cleanly and still not find a working CUDA runtime.
  - If no CUDA channel actually works, explicitly reinstalls the original pinned
    version from the default index, restoring a known-good CPU state rather than
    leaving torch in whatever partial state a failed attempt left behind.

Usage:
    python scripts/install_torch.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover -- Python <3.8, not a supported target here
    import importlib_metadata  # type: ignore[no-redef]

# CUDA wheel channels PyTorch currently publishes at https://download.pytorch.org/whl/,
# newest first -- cu130/cu128 confirmed current directly from Docling's own GPU setup
# docs (docling-project.github.io/docling/getting_started/rtx/), the rest kept as
# fallbacks for older drivers. This list *will* go stale as PyTorch adds new channels
# and drops old ones -- check that page (or https://pytorch.org/get-started/locally/)
# for the current set if every channel below fails on a machine that clearly has a
# recent NVIDIA GPU.
CUDA_WHEEL_CHANNELS = ["cu130", "cu128", "cu126", "cu121", "cu118"]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _detect_nvidia_gpu() -> bool:
    """True if `nvidia-smi` is on PATH and reports at least one GPU -- the standard
    driver-level check for an NVIDIA GPU that works without torch/CUDA already
    installed."""
    if shutil.which("nvidia-smi") is None:
        return False
    result = _run(["nvidia-smi", "-L"])
    return result.returncode == 0 and "GPU" in result.stdout


def _installed_torch_version() -> str | None:
    try:
        return importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        return None


def _install(version: str, index_url: str | None) -> bool:
    # --force-reinstall is required here, not optional: pip compares only the version
    # *string* when deciding whether a requirement is "already satisfied" -- swapping
    # from a CPU build to a CUDA build of the exact same torch version (or between two
    # different CUDA channels) would otherwise print "Requirement already satisfied"
    # and silently do nothing, since torch==2.x.y is "installed" either way.
    #
    # Deliberately NOT using --no-deps: on Linux, CUDA torch wheels depend on separate
    # nvidia-cublas-cu12/nvidia-cudnn-cu12/etc. runtime packages that --no-deps would
    # skip, leaving torch.cuda.is_available() False despite the wheel installing
    # cleanly (Windows CUDA wheels typically bundle these instead, so this wouldn't
    # show up testing on Windows alone -- exactly the kind of cross-platform gap this
    # package can't afford). Heavier (may reinstall numpy/sympy/etc. too) but correct
    # on every platform.
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", f"torch=={version}"]
    if index_url:
        cmd += ["--index-url", index_url]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def _cuda_actually_works() -> bool:
    result = _run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"])
    return result.returncode == 0 and result.stdout.strip() == "True"


def main() -> None:
    original_version = _installed_torch_version()
    if original_version is None:
        print(
            "torch isn't installed yet -- run `pip install -r requirements.txt` first, then run this "
            "script to (if this machine has an NVIDIA GPU) swap in a CUDA-enabled build of the same "
            "torch version. Nothing to do yet."
        )
        return

    if not _detect_nvidia_gpu():
        print(
            f"No NVIDIA GPU detected (nvidia-smi not found, or found no GPU) -- keeping the installed "
            f"CPU build of torch (version {original_version}). Nothing to change."
        )
        return

    print(f"NVIDIA GPU detected. Currently installed: torch {original_version} (CPU). Trying CUDA builds of the same version, newest channel first.")
    for channel in CUDA_WHEEL_CHANNELS:
        index_url = f"https://download.pytorch.org/whl/{channel}"
        print(f"\n--- Trying {channel} ---")
        if _install(original_version, index_url) and _cuda_actually_works():
            print(f"\nDone: torch {original_version} reinstalled with {channel}, and torch.cuda.is_available() is True.")
            return
        print(f"{channel} didn't give a working CUDA build on this machine -- trying the next one.")

    print(
        f"\nCouldn't get a working CUDA build of torch {original_version} on this machine despite "
        f"detecting an NVIDIA GPU -- restoring the original CPU build so nothing is left half-installed. "
        f"Extraction will still work correctly, just without GPU acceleration. See "
        f"https://pytorch.org/get-started/locally/ if you want to troubleshoot GPU support manually "
        f"(driver version, CUDA toolkit version, etc.)."
    )
    _install(original_version, None)


if __name__ == "__main__":
    main()
