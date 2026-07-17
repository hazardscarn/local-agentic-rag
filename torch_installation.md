# Installing torch (CPU vs. GPU)

Docling's own extraction models — layout analysis, OCR, table structure recognition,
picture classification — are all `torch`-based, and run dramatically faster on an
NVIDIA GPU than on CPU. This is a separate, one-time setup step, not something baked
into `requirements.txt` — here's why, and what to actually run.

## Run this after installing requirements

```
pip install -r requirements.txt
python scripts/install_torch.py
```

That's it. The script detects whether this machine has an NVIDIA GPU and, if so,
swaps in a CUDA-enabled build of the exact same torch version `requirements.txt`
just installed. If there's no NVIDIA GPU, it leaves the CPU build alone and tells you
so. Safe to re-run any time (e.g. after a driver update, or if you want to double-check
it's still working).

## Why this can't just live in requirements.txt

`requirements.txt` is a static file — pip has no way to say "if this machine has an
NVIDIA GPU, install the CUDA build; otherwise install the CPU build" from a plain
requirements list. Which torch wheel is correct depends entirely on *this specific
machine's* hardware and driver, which is exactly the kind of thing this project avoids
hardcoding (see `edenview_target_audience` in this project's notes — it's meant to run
on whatever machine an enterprise user installs it on, not tuned to one dev machine).

This is also exactly how [Docling's own GPU setup
docs](https://docling-project.github.io/docling/getting_started/rtx/) describe it:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

`scripts/install_torch.py` automates the "which command do I actually need to run"
decision Docling's docs otherwise leave to you.

## Why not a tool like `light-the-torch`?

There's a published PyPI package (`light-the-torch`) that does auto-detection like
this. It was deliberately not used here: it works by monkey-patching pip's own
internals, which pip's maintainers explicitly say isn't supported and might break
without warning — and it has, in fact, broken before on a pip upgrade (pip 22.3). For a
package meant to install cleanly on arbitrary future machines with arbitrary future pip
versions, that's a real long-term risk. `scripts/install_torch.py` only ever calls
plain, documented `pip install <pkg>==<version> --index-url ...` — the same command
format pytorch.org's own install-matrix page generates — so there's nothing
undocumented for a future pip release to break.

## What the script actually does

1. Checks for an NVIDIA GPU via `nvidia-smi` (works without torch installed at all).
2. No GPU found → leaves whatever `requirements.txt` already installed (the CPU build)
   alone.
3. GPU found → notes the *exact* torch version already installed, then tries
   reinstalling that same version from PyTorch's CUDA wheel channels, newest first
   (`cu130`, `cu128`, `cu126`, `cu121`, `cu118`). Each attempt is verified by actually
   importing torch and checking `torch.cuda.is_available()` afterward — a wheel can
   install cleanly and still not find a working CUDA runtime on this machine (driver
   too old, etc.), so "pip install succeeded" alone isn't good enough.
4. If every CUDA channel fails, it explicitly reinstalls the original CPU version, so
   you're never left with torch half-installed or in an unknown state.

It always pins to the version `requirements.txt` already resolved rather than
installing an unpinned `torch` — otherwise a GPU-equipped machine could silently end up
with a *different* torch version than a CPU-only machine, which risks breaking
compatibility with whatever version `docling`/`transformers` actually depend on.

## Checking whether it worked

Open the app's **Settings** page — the "This machine" card shows an "Extraction
acceleration" line (`CUDA (<gpu name>)`, `Apple Silicon (MPS)`, or `CPU only`). If a
GPU is detected but acceleration still shows `CPU only`, a red warning banner appears
telling you to run the script above.

From the command line, the same check:

```
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

- **Script says "torch isn't installed yet"** — run `pip install -r requirements.txt`
  first; the script needs an existing torch install to know which exact version to
  match on the CUDA side.
- **GPU detected, but every CUDA channel fails** — usually means the installed NVIDIA
  driver is too old for any currently-published PyTorch CUDA wheel. Check
  `nvidia-smi`'s own reported "CUDA Version" (its *driver's* max supported version) and
  compare against <https://pytorch.org/get-started/locally/> for what's current; you
  may need a driver update.
- **The CUDA channel list in the script looks stale** — PyTorch adds and drops wheel
  channels over time. Check <https://pytorch.org/get-started/locally/> or
  [Docling's own GPU page](https://docling-project.github.io/docling/getting_started/rtx/)
  for the current set and update `CUDA_WHEEL_CHANNELS` in `scripts/install_torch.py`.
