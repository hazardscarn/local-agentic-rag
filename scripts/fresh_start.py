"""Local dev-environment reset: kills any stray dev-server processes bound to
this project's ports and clears out ingestion jobs left "queued"/"running" by
a server that was killed or crashed mid-job (DuckDB has no way to know a job's
process died -- its status row just stays wherever it was, forever, until
something else corrects it). Meant to be run whenever `npm run dev` needs a
guaranteed clean restart, not something left running as part of any normal
request path.

Why this exists: this project's dev backend has no --reload (Python code
changes always need a full restart -- see edenview-ui/package.json's
dev:backend script), and `concurrently` does not reliably kill sibling
processes when one of `npm run dev`'s two commands exits or is interrupted --
a real, repeatedly observed pattern in this project's own development, not a
hypothetical. A half-killed dev server leaves an orphaned process still
holding a port (blocking the next `npm run dev`) and/or an ingestion job stuck
at "running"/"queued" forever (nothing ever calls
catalog.crud.complete_job() for it once its process is gone).

What this does NOT touch: any process not bound to one of DEV_PORTS below (so
your IDE's language server, Jupyter kernels, Ollama, etc. are never touched),
and any ingestion job that's already "done"/"error".

Usage:
    python scripts/fresh_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psutil

# So `from edenview_ingestion import catalog` resolves when run as a plain
# script (`python scripts/fresh_start.py`) rather than `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# edenview-ui's dev script: frontend on 3000 (Next.js falls back to 3001+ if
# 3000 is already held by a stray process from a previous crash), backend on
# 8000 -- see edenview-ui/package.json's dev:frontend/dev:backend scripts.
DEV_PORTS = {3000, 3001, 8000}


def kill_dev_server_processes() -> list[int]:
    """Kills whatever process (if any) is listening on each of DEV_PORTS.
    Matches by port, not by process name/command line, so it can't
    accidentally hit an unrelated python.exe/node.exe (e.g. a Jupyter kernel
    or another project's dev server) that just happens to share a name."""
    killed: list[int] = []
    for conn in psutil.net_connections(kind="inet"):
        if conn.status != psutil.CONN_LISTEN or conn.laddr.port not in DEV_PORTS:
            continue
        if conn.pid is None or conn.pid in killed:
            continue
        try:
            proc = psutil.Process(conn.pid)
            name = proc.name()
            proc.kill()
            proc.wait(timeout=5)
            killed.append(conn.pid)
            print(f"Killed {name} (pid {conn.pid}) listening on port {conn.laddr.port}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            print(f"Could not kill pid {conn.pid} on port {conn.laddr.port}: {e}")
    if not killed:
        print("No dev-server processes found on ports", sorted(DEV_PORTS))
    return killed


def clear_stale_ingestion_jobs() -> int:
    """Thin wrapper over catalog.crud.clear_stale_jobs() -- the single source
    of truth this script shares with the POST /system/jobs/clear-stale API
    route, so the two never drift. Safe to call with no backend running at
    all (opens its own DuckDB connection via catalog.crud's default)."""
    from edenview_ingestion import catalog

    stale = catalog.crud.clear_stale_jobs()
    for job in stale:
        label = job.filename or job.doc_id or job.job_id
        print(f"Marked stale job {job.job_id} ({label}) as error (was {job.status!r})")
    if not stale:
        print("No stale queued/running ingestion jobs found.")
    return len(stale)


def main() -> None:
    print("=== Killing dev-server processes ===")
    kill_dev_server_processes()
    print("\n=== Clearing stale ingestion jobs ===")
    clear_stale_ingestion_jobs()
    print("\nDone -- safe to run `npm run dev` again.")


if __name__ == "__main__":
    main()
