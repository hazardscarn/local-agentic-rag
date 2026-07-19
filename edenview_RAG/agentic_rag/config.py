"""Configuration for the ADK-based agentic RAG loop. Model name/effort defaults are
never hardcoded -- read from config.yaml's `agent:` section via edenview_ingestion.settings,
same pattern as edenview_RAG/retrieval/config.py.

Also sets OLLAMA_API_BASE at import time -- LiteLLM's `ollama_chat/*` provider reads
this env var directly (independent of any `api_base=` kwarg passed to LiteLlm(...)), so
it needs to be set once before any LiteLlm model object is constructed. Mirrors
retrieval/config.py's own import-time DEFAULT_RERANKER_MODEL = get_model("reranker")
pattern -- resolve config once at import, not per-call.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

import ollama
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

from edenview_ingestion.settings import get_ollama_host, load_config

Effort = Literal["low", "medium", "high"]

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


_LITELLM_OLLAMA_REASONING_BUG_PATCHED = False


def _patch_litellm_ollama_reasoning_content_bug() -> None:
    """Works around a real, reproduced upstream litellm bug (litellm==1.77.7):
    `litellm.llms.ollama.chat.transformation._extract_reasoning_content()` does
    `message["content"]` without checking the key exists, raising `KeyError:
    'content'` while building a follow-up request whose conversation history
    includes a prior assistant turn that has `reasoning_content` but no `content` key
    at all -- exactly what a "thinking"-capable model's tool-call-only turn (no plain
    text, just a function call) produces, i.e. a completely normal turn for any
    multi-step tool-calling agent. Patched at the *module* level
    (`transformation.py`'s own namespace), not `common_utils` where the function is
    defined -- `transformation.py` imports it by name (`from ... import
    _extract_reasoning_content`), which binds its own local reference that patching
    the defining module's attribute would not affect. Idempotent."""
    global _LITELLM_OLLAMA_REASONING_BUG_PATCHED
    if _LITELLM_OLLAMA_REASONING_BUG_PATCHED:
        return

    import litellm.llms.ollama.chat.transformation as _ollama_transformation

    _original = _ollama_transformation._extract_reasoning_content

    def _safe_extract_reasoning_content(message: dict):
        if "reasoning_content" in message or "reasoning" in message:
            content = message.get("content")
            key = "reasoning_content" if "reasoning_content" in message else "reasoning"
            return message[key], content
        return _original(message)

    _ollama_transformation._extract_reasoning_content = _safe_extract_reasoning_content
    _LITELLM_OLLAMA_REASONING_BUG_PATCHED = True


_LITELLM_OLLAMA_IMAGE_PREFIX_BUG_PATCHED = False


def _patch_litellm_ollama_image_data_uri_bug() -> None:
    """Works around a real, reproduced upstream litellm bug (litellm==1.77.7):
    `litellm.llms.ollama.chat.transformation`'s image handling
    (`extract_images_from_message`, in `common_utils.py`) pulls each image's
    `image_url["url"]` value and appends it to Ollama's `images` field VERBATIM --
    including a `data:image/png;base64,` data-URI prefix, if that's the form ADK's
    own genai->litellm message conversion produced (which it does, confirmed
    directly). Ollama's own `/api/chat` expects `images` to be PURE base64 with no
    prefix, and rejects anything else outright: reproduced directly as `{"error":
    "illegal base64 data at input byte 4"}` -- byte 4 is exactly where a "data:"
    prefix's colon sits, confirming this precise cause (not a base64-encoding bug
    elsewhere). This is the mechanism `tools.inspect_image` +
    `callbacks.inject_pending_images` (the image-into-context feature) depends on
    entirely -- without this patch, ANY retrieved-image inspection request to a local
    Ollama vision model fails outright. Patched at the *module* level
    (`transformation.py`'s own namespace, same reasoning as the reasoning_content
    patch above -- it imports the function by name, so patching `common_utils`'s
    attribute wouldn't affect the already-bound reference here). Idempotent."""
    global _LITELLM_OLLAMA_IMAGE_PREFIX_BUG_PATCHED
    if _LITELLM_OLLAMA_IMAGE_PREFIX_BUG_PATCHED:
        return

    import litellm.llms.ollama.chat.transformation as _ollama_transformation

    _original = _ollama_transformation.extract_images_from_message

    def _strip_data_uri_prefix(images: list[str]) -> list[str]:
        stripped = []
        for image in images:
            if isinstance(image, str) and image.startswith("data:") and "," in image:
                image = image.split(",", 1)[1]
            stripped.append(image)
        return stripped

    def _safe_extract_images_from_message(message):
        return _strip_data_uri_prefix(_original(message))

    _ollama_transformation.extract_images_from_message = _safe_extract_images_from_message
    _LITELLM_OLLAMA_IMAGE_PREFIX_BUG_PATCHED = True


def _register_ollama_model_with_litellm(model: str) -> None:
    """LiteLLM's ollama_chat provider decides whether to use Ollama's real native
    tool-calling (tools=[...] request param + message.tool_calls response field) or a
    legacy fallback (inject the function schema into the prompt as text, request
    format="json", then json.loads() the response's plain `content` as the function
    call) by checking its OWN bundled model registry (`litellm.get_model_info`) --
    NOT by checking the live Ollama server's actual reported capabilities. Every
    model this project uses (qwen3.5:*, qwen3-vl:*, granite4.1:*) is too new/rare to
    be in that bundled registry, so litellm silently took the legacy path -- which
    broke with a real, reproduced failure (`litellm.exceptions.APIConnectionError:
    Expecting value: line 1 column 1` from trying to json.loads() an empty response
    content string, since the model was actually responding with real tool_calls that
    the legacy path never looks at). Registering the model here forces litellm onto
    the correct native tools/tool_calls path instead. Idempotent -- safe to call
    every time get_shared_llm() builds its (cached) instance."""
    import litellm

    litellm.register_model(
        {f"ollama/{model}": {"supports_function_calling": True, "litellm_provider": "ollama", "mode": "chat"}}
    )


@lru_cache(maxsize=1)
def get_shared_llm() -> LiteLlm:
    """The ONE LiteLlm instance every agent across every tier shares -- never a
    second/smaller model for the critic, never a separate vision model. Matters on a
    6GB VRAM budget where bge-m3 (dense embeddings) is already Ollama-resident at
    query time. Lives here (not agent.py) so both agent.py and subagent.py can import
    it without an agent.py <-> subagent.py circular import.

    ADK's own Ollama docs warn: use the "ollama_chat/" provider prefix, not bare
    "ollama/" -- the latter "can result in unexpected behaviors such as infinite tool
    call loops and ignoring previous context."."""
    model = get_agent_model_name()
    _register_ollama_model_with_litellm(model)
    _patch_litellm_ollama_reasoning_content_bug()
    _patch_litellm_ollama_image_data_uri_bug()
    return LiteLlm(model=f"ollama_chat/{model}", num_ctx=get_agent_num_ctx())


def get_default_effort() -> Effort:
    return load_config().get("agent", {}).get("default_effort", "high")


def get_max_iterations(effort: Effort) -> int:
    """`effort="low"` has no loop at all, so this is only ever called for
    "medium"/"high" -- defaults (2/4) match config.yaml's own comments if the key is
    somehow missing."""
    defaults = {"medium": 2, "high": 4}
    configured = load_config().get("agent", {}).get("max_iterations", {})
    return int(configured.get(effort, defaults[effort]))


@lru_cache(maxsize=8)
def model_supports_vision(model: str) -> bool:
    """Ollama's `show` now reports a `capabilities` list (e.g.
    ["completion", "vision", "tools", "thinking"]) -- confirmed directly against a
    running Ollama instance, not assumed. Used to decide whether the "high" tier's
    image-inspection tool/callback gets registered at all for the configured
    agent.model. Treats an unreachable/unknown model as non-multimodal rather than
    guessing."""
    client = ollama.Client(host=get_ollama_host()) if get_ollama_host() else ollama.Client()
    try:
        info = client.show(model)
    except Exception:
        return False
    capabilities = getattr(info, "capabilities", None) or []
    return "vision" in capabilities


class RetrievalScope(BaseModel):
    """What a request-level ChatRequest resolves into -- passed as initial/refreshed
    ADK session state under the "scope" key, never as a tool function argument (the
    three agent trees are built once and reused across every HTTP request; only
    session state varies per call)."""

    collection_names: Optional[list[str]] = None
    db_name: Optional[str] = None
    file_hashes: Optional[list[str]] = None
    strategy: Optional[str] = None
    top_k: int = 5
    use_reranker: bool = True
