"""Generates descriptions for already-retained pictures by calling Ollama's *native*
API directly on their saved crop files -- deliberately NOT via Docling's own
do_picture_description/PictureDescriptionApiOptions enrichment.

Root cause of why that path doesn't work here, confirmed by direct reproduction (not
guessed): Docling's picture-description stage calls a *reasoning-capable* local VLM
(e.g. qwen3-vl) through Ollama's OpenAI-compatible `/v1/chat/completions` endpoint,
which surfaces the model's real answer in a separate `reasoning` field -- but Docling's
response parser (`docling.utils.api_image_request._extract_generated_text`) only reads
`message.content`, which is left empty. This is a known issue pattern, not specific to
this codebase (see e.g. docling-project/docling discussions #2581, #2434, and the same
content/reasoning-field split reported against other OpenAI-compat consumers of Ollama
reasoning models). Verified: calling `api_image_request()` directly with the exact same
image/params Docling uses returns a correct description; going through Docling's actual
enrichment pipeline on the same picture returns empty every time.

Ollama's *native* chat API (used here, via the `ollama` package's Client.chat(), not the
OpenAI-compat HTTP shim) does not have this problem -- it correctly separates
`message.content` (the final answer) from `message.thinking` (the reasoning trace).
Verified working end-to-end against a real extracted crop.

This only ever runs on pictures already in `ExtractionBundle.pictures` -- i.e. after
`images.build_picture_records()` has already dropped logos/icons/etc. via
`picture_exclude_labels`, so there's no separate classification-gating logic needed
here; every picture this module sees is one already worth describing.
"""

from __future__ import annotations

import ollama

from edenview_ingestion.settings import get_model, get_ollama_host, get_ollama_keep_alive

from .models import PictureRecord

DEFAULT_PROMPT = "Describe this image in 1-2 sentences, focused on what information it conveys."


def generate_picture_descriptions(
    pictures: list[PictureRecord], prompt: str = DEFAULT_PROMPT
) -> None:
    """Mutates each PictureRecord.description in place. Skips pictures that already
    have a description (idempotent re-run) or have no saved crop file to describe.
    A per-picture failure (Ollama unreachable, model missing, etc.) logs and leaves
    that picture's description as None rather than aborting the rest -- same
    graceful-degradation approach as the contextual chunker's per-chunk LLM calls."""
    host = get_ollama_host()
    client = ollama.Client(host=host) if host else ollama.Client()
    model = get_model("picture_description_llm")

    for picture in pictures:
        if picture.description or not picture.image_path:
            continue
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [picture.image_path]}],
                keep_alive=get_ollama_keep_alive(),
            )
            text = (response["message"].get("content") or "").strip()
            picture.description = text or None
        except Exception as e:
            print(f"[picture_description] Failed to describe {picture.picture_id}: {e}")
