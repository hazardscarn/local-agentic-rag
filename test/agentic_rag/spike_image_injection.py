"""Narrow spike test (per the build plan's Step 5) for the ONE link in the
image-into-context chain not confirmed by ADK's docs or any codelab found during
research: does LiteLLM's `ollama_chat` provider actually forward an
artifact-loaded `types.Part(inline_data=...)` injected into `llm_request.contents`
by a `before_model_callback` into Ollama's native `images: [...]` chat field, for a
real local model (not Gemini, which is what the reference codelab used)?

Uses a synthetic image with unambiguous, unmemorizable text (a random code) so a
correct answer can only come from the model actually reading the injected image, not
from pretrained knowledge -- a clean pass/fail signal independent of any real
document's unknown content.

Usage:
    PYTHONPATH=. venv/Scripts/python.exe test/agentic_rag/spike_image_injection.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from google.adk.tools.tool_context import ToolContext  # noqa: E402 -- module-level so get_type_hints() can resolve it

TEST_IMAGE_PATH = Path(__file__).parent.parent.parent / "scratch_test_image.png"
SECRET_CODE = "OCTOPUS99"


async def main() -> int:
    from google.adk.agents import LlmAgent
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from edenview_RAG.agentic_rag.config import get_shared_llm

    if not TEST_IMAGE_PATH.exists():
        print(f"Test image not found at {TEST_IMAGE_PATH} -- create it first.")
        return 1

    async def show_test_image(tool_context: ToolContext) -> dict:
        """Call this to see the test image."""
        data = TEST_IMAGE_PATH.read_bytes()
        part = types.Part.from_bytes(data=data, mime_type="image/png")
        await tool_context.save_artifact(filename="test_image.png", artifact=part)
        return {"status": "ok", "tool_response_artifact_id": "test_image.png"}

    async def inject_pending_image(callback_context, llm_request) -> None:
        for content in llm_request.contents:
            if not content.parts:
                continue
            modified = []
            for part in content.parts:
                modified.append(part)
                fr = getattr(part, "function_response", None)
                if fr and fr.name == "show_test_image" and isinstance(fr.response, dict):
                    artifact_id = fr.response.get("tool_response_artifact_id")
                    if artifact_id:
                        image_part = await callback_context.load_artifact(filename=artifact_id)
                        if image_part:
                            modified.append(image_part)
            content.parts = modified
        return None

    agent = LlmAgent(
        name="image_spike",
        model=get_shared_llm(),
        instruction=(
            "Call show_test_image exactly once, then report exactly what text you see "
            "written in the image, verbatim."
        ),
        tools=[show_test_image],
        before_model_callback=inject_pending_image,
    )
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session_id = str(uuid.uuid4())
    await session_service.create_session(app_name="spike", user_id="local", session_id=session_id, state={})
    runner = Runner(agent=agent, app_name="spike", session_service=session_service, artifact_service=artifact_service)

    final_text = ""
    async for event in runner.run_async(
        user_id="local",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="What does the test image show?")]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if p.text]
            if texts:
                final_text = "".join(texts)

    print("MODEL RESPONSE:", repr(final_text))
    if SECRET_CODE in final_text:
        print(f"\nPASS -- model correctly read the injected image's text ({SECRET_CODE!r} found in response).")
        return 0
    else:
        print(f"\nFAIL -- expected {SECRET_CODE!r} in the model's response, image injection did not work as expected.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
