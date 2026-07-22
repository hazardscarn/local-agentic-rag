"""Configuration for the ADK-based agentic RAG pipeline. Model name/iteration defaults
are never hardcoded -- read from config.yaml's `agent:` section via
edenview_ingestion.settings, same pattern as edenview_RAG/retrieval/config.py.

Also sets OLLAMA_API_BASE at import time -- LiteLLM's `ollama_chat/*` provider reads
this env var directly (independent of any `api_base=` kwarg passed to LiteLlm(...)), so
it needs to be set once before any LiteLlm model object is constructed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

from edenview_ingestion.settings import get_ollama_host, load_config, model_supports_capability

from .errors import AgenticRAGError

os.environ.setdefault("OLLAMA_API_BASE", get_ollama_host() or "http://localhost:11434")


def get_agent_model_name() -> str:
    try:
        return load_config()["agent"]["model"]
    except KeyError:
        raise KeyError("No `agent.model` set in config.yaml -- see the `agent:` section") from None


def get_agent_num_ctx() -> int:
    """Ollama's actual runtime context window -- see config.yaml's `agent.num_ctx`
    comment for why this must be set explicitly (Ollama's own default of ~2048
    silently truncates large retrieved-context prompts regardless of the model's own
    much larger max context)."""
    return int(load_config().get("agent", {}).get("num_ctx", 16384))


def get_vision_model() -> Optional[str]:
    """Optional, independently-settable model for get_answer_from_images's direct-
    ollama call -- see config.yaml's `agent.vision_model` comment. Unset falls back
    to `agent.model` if it's vision-capable, else None (the tool is simply not
    registered -- see require_tool_calling_model's sibling gating in agent.py)."""
    configured = load_config().get("agent", {}).get("vision_model")
    if configured:
        return configured
    model = get_agent_model_name()
    return model if model_supports_capability(model, "vision", get_ollama_host()) else None


def require_tool_calling_model() -> None:
    """Called once at agent-tree build time (agent.py, module import) -- fails
    loudly and immediately if the configured `agent.model` can't tool-call, rather
    than the tree silently building and failing confusingly mid-run. Unlike vision,
    there's no reasonable degraded mode here: every LLM node in this pipeline relies
    on native tool-calling."""
    model = get_agent_model_name()
    if not model_supports_capability(model, "tools", get_ollama_host()):
        raise AgenticRAGError(
            f"agent.model {model!r} does not report tool-calling support "
            "(checked via `ollama show`'s capabilities list) -- the agentic RAG "
            "pipeline requires a tool-calling-capable model. Pick a different "
            "agent.model in config.yaml."
        )


def _register_ollama_model_with_litellm(model: str) -> None:
    """Not a monkeypatch -- this is LiteLLM's own public, documented configuration
    API (`litellm.register_model`) for telling it a custom/unlisted model supports
    native tool-calling. LiteLLM's `ollama_chat` provider decides whether to use
    Ollama's real native tool-calling (tools=[...] request param + message.tool_calls
    response field) or a legacy fallback (inject the function schema into the prompt
    as text, request format="json", then json.loads() the response's plain `content`
    as the function call) by checking its OWN bundled model registry
    (`litellm.get_model_info`) -- NOT by checking the live Ollama server's actual
    reported capabilities. Every model this project uses (qwen3.5:*, granite4.1:*) is
    too new/rare to be in that bundled registry (confirmed directly against litellm
    1.84.0's own bundled `litellm.model_cost` registry -- the only `ollama/qwen*`
    entry is an unrelated cloud SKU, `ollama/qwen3-coder:480b-cloud`, and there's no
    `ollama/granite*` entry at all), so litellm would otherwise silently take the
    legacy path, which breaks against real tool-calling responses. Registering the
    model here forces litellm onto the
    correct native tools/tool_calls path instead. Idempotent -- safe to call every
    time get_shared_llm() builds its (cached) instance."""
    import litellm

    litellm.register_model(
        {f"ollama/{model}": {"supports_function_calling": True, "litellm_provider": "ollama", "mode": "chat"}}
    )


@lru_cache(maxsize=1)
def get_shared_llm() -> LiteLlm:
    """The ONE LiteLlm instance every tool-calling/synthesis agent in the pipeline
    shares -- never a second/smaller model for Eval, never a separate content model.
    Matters on a 6GB VRAM budget where bge-m3 (dense embeddings) is already Ollama-
    resident at query time -- a second concurrently-resident LLM risks Ollama
    swapping models between pipeline steps, which can cost more wall-clock time than
    it saves (see get_reword_llm()'s docstring for why that node still reuses this
    same underlying model, just with thinking disabled, rather than a second model).
    The one deliberate exception is `agent.vision_model`, used only inside
    get_answer_from_images's own direct `ollama.Client()` call (tools.py) -- that
    call bypasses this shared LiteLlm/ADK model entirely, so it isn't part of this
    cache.

    ADK's own Ollama docs warn: use the "ollama_chat/" provider prefix, not bare
    "ollama/" -- the latter "can result in unexpected behaviors such as infinite tool
    call loops and ignoring previous context."."""
    model = get_agent_model_name()
    _register_ollama_model_with_litellm(model)
    return LiteLlm(model=f"ollama_chat/{model}", num_ctx=get_agent_num_ctx(), max_tokens=16000)


class RetrievalScope(BaseModel):
    """What a request-level ChatRequest resolves into -- passed as initial/refreshed
    ADK session state under the "scope" key, never as a tool function argument (the
    agent tree is built once and reused across every HTTP request; only session
    state varies per call)."""

    collection_names: Optional[list[str]] = None
    db_name: Optional[str] = None
    file_hashes: Optional[list[str]] = None
    strategy: Optional[str] = None
    top_k: int = 5
    use_reranker: bool = True
