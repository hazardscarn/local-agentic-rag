"""Loads `config.yaml` (project root) -- the single place every model name and service
connection used across the ingestion stack is declared. `docling_parsing`, `chunking`,
`catalog`, and `vectorstore` all read through this module rather than hardcoding a model
name or connection detail, so changing which tokenizer, LLM, Qdrant host, or catalog
file path is used is a one-line edit to config.yaml, not a code change.

Loaded once per process (`lru_cache`) -- restart the process to pick up an edited
config.yaml, same as any other Python config-at-import-time pattern in this codebase.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import ollama
import yaml
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Model keys any caller is allowed to read/write via get_all_model_settings() /
# update_model_settings() -- kept as an explicit allowlist (not "whatever's under
# models:") so a typo'd key from an API request fails loudly instead of silently
# adding a stray field to config.yaml.
MODEL_KEYS = (
    "tokenizer",
    "dense_embedding",
    "dense_embedding_dim",
    "sparse_embedding",
    "contextual_llm",
    "picture_description_llm",
    "chat_llm",
    "reranker",
)


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(name: str) -> str:
    """`name` is a key under `models:` in config.yaml, e.g. "tokenizer", "contextual_llm",
    "picture_description_llm"."""
    try:
        return load_config()["models"][name]
    except KeyError:
        raise KeyError(
            f"No model named {name!r} in config.yaml's `models:` section ({CONFIG_PATH})"
        ) from None


def get_ollama_host() -> Optional[str]:
    return load_config().get("ollama", {}).get("host")


def get_ollama_keep_alive() -> str:
    """How long a model stays loaded after its last call before Ollama's scheduler
    evicts it -- passed explicitly on every Ollama call this app makes instead of
    relying on Ollama's own 5-minute default. See config.yaml's `ollama.keep_alive`
    comment for why 30m, not -1."""
    return str(load_config().get("ollama", {}).get("keep_alive", "30m"))


def get_chat_num_ctx() -> int:
    """Ollama's request-time num_ctx for the /chat endpoint's chat_llm -- see
    config.yaml's `models.chat_num_ctx` comment for the real bug this fixes (a
    retrieved-context prompt silently truncated to Ollama's ~2048 default, leaving
    no room for the model to actually answer) and why the default is generous
    (32768, not a smaller value) -- generate_answer() deliberately keeps a
    thinking-capable model's reasoning on, and that needs real headroom too."""
    return int(load_config().get("models", {}).get("chat_num_ctx", 32768))


def get_dense_embedding_dim() -> int:
    return int(load_config()["models"]["dense_embedding_dim"])


def get_agent_model() -> str:
    """Raw read of config.yaml's `agent.model` -- NOT the same thing as
    edenview_RAG.agentic_rag.config.get_agent_model_name(), which this module can't
    import (that package imports settings.py, not the other way around). This is
    just the on-disk value, for the Settings API to read/write; the agentic_rag
    package's own getter is what anything actually building the agent tree uses."""
    return load_config()["agent"]["model"]


def get_agent_vision_model() -> Optional[str]:
    """Raw read of config.yaml's `agent.vision_model` -- None means "unset", NOT
    "resolved to agent_model". The fallback-to-agent_model-if-vision-capable logic
    lives in edenview_RAG.agentic_rag.config.get_vision_model(), which this raw
    getter deliberately does not replicate (same reasoning as get_agent_model())."""
    return load_config().get("agent", {}).get("vision_model")


# API-facing flat key -> its real name under config.yaml's `agent:` section.
AGENT_KEY_MAP = {
    "agent_model": "model",
    "agent_vision_model": "vision_model",
}

# Every dense_embedding model this app knows a matching HuggingFace tokenizer for --
# keyed by the Ollama model name's family (the part before ":", so a size/tag suffix
# like ":567m" or ":latest" doesn't need its own entry as long as that family shares
# one tokenizer across sizes -- true for every family below). Verified directly, not
# guessed: each of these downloads and loads successfully via transformers'
# AutoTokenizer.from_pretrained (google/embeddinggemma-300m was tried and excluded --
# it's a gated HF repo requiring separate license acceptance + an HF token, too
# fragile to auto-download here even though Ollama reports the model itself as
# embedding-capable). Chunk token-budget sizing (HybridChunker) has to match what the
# embedding model itself actually tokenizes, so tokenizer is never an independent user
# choice -- see api/routers/config.py's update_config(), which derives it from
# whichever dense_embedding is selected via this map, and rejects any dense_embedding
# this map doesn't cover rather than leaving a stale/mismatched tokenizer in place.
EMBEDDING_TOKENIZER_MAP: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "granite-embedding": "ibm-granite/granite-embedding-278m-multilingual",
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
}


def tokenizer_for_dense_embedding(dense_embedding: str) -> Optional[str]:
    """Looks up EMBEDDING_TOKENIZER_MAP by model family (everything before the first
    ':' in an Ollama model name, e.g. "bge-m3" from "bge-m3:567m") -- None if this
    dense_embedding isn't one of the models this app has a verified tokenizer for."""
    family = dense_embedding.split(":", 1)[0]
    return EMBEDDING_TOKENIZER_MAP.get(family)


@lru_cache(maxsize=8)
def model_supports_capability(model: str, capability: str, ollama_host: Optional[str] = None) -> bool:
    """Checks a model's actual reported capabilities via Ollama's own `/api/show`
    (e.g. `capabilities: ["completion", "tools", "vision", "thinking"]`) rather than
    assuming -- confirmed directly, more than once, that requesting a capability a
    model doesn't have is a hard failure, not a graceful no-op (e.g. `think=True`
    against a non-thinking model raises a real `400 "<model> does not support
    thinking"` from Ollama). Shared by both edenview_RAG.retrieval (Simple RAG's
    thinking gate) and edenview_RAG.agentic_rag (tool-calling/vision gates) so there's
    one place this check lives, not two near-duplicates. Treats an unreachable/unknown
    model as lacking the capability rather than guessing."""
    client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()
    try:
        info = client.show(model)
    except Exception:
        return False
    capabilities = info.get("capabilities") if isinstance(info, dict) else getattr(info, "capabilities", None)
    return capability in (capabilities or [])


def _auto_num_threads() -> int:
    """`cpu_count - 2` (never below 1) -- leaves headroom for the OS and everything
    else running on whatever machine this package happens to be installed on, rather
    than claiming every core. Not tuned to any one dev machine."""
    return max((os.cpu_count() or 4) - 2, 1)


def get_num_threads() -> int:
    """How many threads Docling's own pipeline uses per extraction. Auto-detected from
    this machine by default (see _auto_num_threads()) -- config.yaml's
    `performance.num_threads` overrides it with an explicit value if set (editable from
    Settings -> Performance, not just by hand-editing config.yaml)."""
    raw = load_config().get("performance", {}).get("num_threads")
    return int(raw) if raw is not None else _auto_num_threads()


def get_page_batch_size() -> int:
    """How many pages of a single document Docling processes in one batch during its
    own internal pipeline (its own default is 4) -- a process-wide Docling setting
    (docling.datamodel.settings.settings.perf.page_batch_size), not something
    per-document; see docling_parsing/extractor.py for where this gets applied.
    Higher can mean faster throughput at the cost of more memory held at once --
    config.yaml's `performance.page_batch_size` (default 4, Docling's own default)
    overrides it, editable from Settings -> Performance."""
    raw = load_config().get("performance", {}).get("page_batch_size")
    return int(raw) if raw is not None else 4


def get_max_concurrent_extractions() -> int:
    """How many documents can run Docling extraction (the actual resource-hungry step
    -- loads layout/OCR/table-structure/picture-classification models into memory,
    contends for CPU threads, and if a GPU is present, that one shared GPU too) at the
    same time. Deliberately a flat, conservative default (4) rather than scaled by CPU
    core count the way num_threads is -- more cores doesn't mean more *concurrent*
    extractions are safe, since RAM and a single shared GPU are the actual limiting
    resources on most machines, and neither scales with core count. Confirmed the hard
    way: 22 files ingested at once on a 16-core/16GB/one-GPU machine ran every
    extraction simultaneously with no limit, RAM fell to ~4.6GB free and the one GPU
    was oversubscribed 22 ways, making the whole batch far slower than running a few
    at a time would have -- 4 lets a handful of documents actually finish and free
    their resources instead of every single one crawling in lockstep. Requires a
    backend restart to take effect -- unlike num_threads/page_batch_size (re-read
    fresh per extraction), the limiter enforcing this is sized once at process start,
    see pipeline.py's _EXTRACTION_SEMAPHORE."""
    raw = load_config().get("performance", {}).get("max_concurrent_extractions")
    return int(raw) if raw is not None else 4


def update_max_concurrent_extractions(value: Optional[int]) -> int:
    _update_performance_setting("max_concurrent_extractions", value)
    return get_max_concurrent_extractions()


def _update_performance_setting(key: str, value) -> None:
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with open(CONFIG_PATH, encoding="utf-8") as f:
        doc = yaml_rt.load(f)
    doc.setdefault("performance", {})[key] = value
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml_rt.dump(doc, f)
    load_config.cache_clear()


def update_num_threads(value: Optional[int]) -> int:
    """`value=None` reverts to auto-detecting from this machine's CPU count on every
    call (see _auto_num_threads()) instead of pinning a specific number."""
    _update_performance_setting("num_threads", value)
    return get_num_threads()


def update_page_batch_size(value: Optional[int]) -> int:
    """`value=None` reverts to Docling's own default (4)."""
    _update_performance_setting("page_batch_size", value)
    return get_page_batch_size()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_workspace_root() -> Path:
    """Absolute path to the single folder everything Edenview builds locally lives
    under -- the embedded Qdrant store, the DuckDB catalog, and permanent per-document
    storage each live in a fixed subfolder of this (see get_qdrant_path()/
    get_catalog_path()/get_documents_dir() below). Editable from Settings -> Workspace
    (api/routers/config.py's /system/workspace); see config.yaml's `workspace:` comment
    for why this shouldn't be changed casually once real data exists under it."""
    raw = load_config().get("workspace", {}).get("root", "edenview_data")
    return _resolve_path(raw)


def update_workspace_root(new_root: str) -> Path:
    """Writes `workspace.root` into config.yaml (ruamel round-trip, same pattern as
    update_model_settings()) and returns the newly resolved absolute path. Does NOT
    move any existing qdrant_db/catalog.duckdb/documents data, and does not take
    effect in this process until restarted -- get_connection()/get_client() already
    cache their DuckDB connection/Qdrant client as module-level singletons for the
    process's lifetime, opened against whatever get_catalog_path()/get_qdrant_path()
    returned the first time they were called."""
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with open(CONFIG_PATH, encoding="utf-8") as f:
        doc = yaml_rt.load(f)

    doc.setdefault("workspace", {})["root"] = new_root

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml_rt.dump(doc, f)

    load_config.cache_clear()
    return get_workspace_root()


def get_qdrant_path() -> Path:
    """Absolute path to the local embedded Qdrant store (`QdrantClient(path=...)`), no
    server/Docker involved -- a fixed subfolder of get_workspace_root()."""
    return get_workspace_root() / "qdrant_db"


def get_catalog_path() -> Path:
    """Absolute path to the DuckDB catalog file -- a fixed subfolder of
    get_workspace_root()."""
    return get_workspace_root() / "catalog.duckdb"


def get_documents_dir() -> Path:
    """Permanent per-document storage root (parse cache + picture/table crops), keyed
    by doc_stem -- a fixed subfolder of get_workspace_root(). Not the docling_parsing
    module's default temp dir; see pipeline.py's ingest_document()."""
    return get_workspace_root() / "documents"


def get_all_model_settings() -> dict:
    """Every key in MODEL_KEYS plus the Ollama host/keep_alive and the agent.* keys,
    for a Settings UI to render as a form. Reads through the cached load_config(),
    same as every other getter here."""
    config = load_config()
    settings = {key: config["models"][key] for key in MODEL_KEYS}
    settings["ollama_host"] = get_ollama_host()
    settings["ollama_keep_alive"] = get_ollama_keep_alive()
    settings["agent_model"] = get_agent_model()
    settings["agent_vision_model"] = get_agent_vision_model()
    return settings


def update_model_settings(updates: dict) -> dict:
    """Merges `updates` (a subset of MODEL_KEYS, plus optionally "ollama_host"/
    "ollama_keep_alive"/AGENT_KEY_MAP's keys) into config.yaml on disk and returns
    the full settings afterward.

    Uses ruamel.yaml's round-trip mode instead of pyyaml -- config.yaml carries an
    explanatory comment above nearly every key, and yaml.safe_dump would silently
    throw all of that away on a plain read-modify-write. ruamel preserves comments,
    key order, and formatting for anything it didn't touch.

    Clears load_config()'s lru_cache afterward so the next read (in this same
    process) picks up the change -- though see api/routers/config.py's
    RESTART_REQUIRED_KEYS for which settings are actually consumed fresh per call
    versus baked into a module-level default (or an import-time check, for the
    agent.* keys) at import time.
    """
    special_cased = {"ollama_host", "ollama_keep_alive"} | set(AGENT_KEY_MAP)
    unknown = set(updates) - set(MODEL_KEYS) - special_cased
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with open(CONFIG_PATH, encoding="utf-8") as f:
        doc = yaml_rt.load(f)

    for key, value in updates.items():
        if key == "ollama_host":
            doc["ollama"]["host"] = value
        elif key == "ollama_keep_alive":
            doc["ollama"]["keep_alive"] = value
        elif key in AGENT_KEY_MAP:
            doc.setdefault("agent", {})[AGENT_KEY_MAP[key]] = value
        else:
            doc["models"][key] = value

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml_rt.dump(doc, f)

    load_config.cache_clear()
    return get_all_model_settings()
