"""Instruction strings per tier/agent, plus human-readable status labels for the
streaming endpoint (Part 4 of the build plan)."""

LOW_ROOT_INSTRUCTION = (
    "A user asked you a question. Call the `retrieve` tool exactly once with that "
    "question (or a cleaned-up version of it). It returns numbered context snippets "
    "retrieved from the user's own documents, ordered from most to least relevant "
    "(snippet [1] is the best match) -- answer the question using only these "
    "snippets, exactly the way you'd answer from context given directly in a prompt. "
    "Cite every claim with the snippet number(s) it came from, like [1] or [2][3]. "
    "If the snippets don't contain the answer, say so plainly instead of guessing. "
    "Do not call retrieve more than once."
)

REFRAME_INSTRUCTION = (
    "Rewrite the user's question to make it as effective as possible for a search "
    "over their documents -- fix ambiguity, expand abbreviations if obvious, drop "
    "filler words.\n\n"
    "Separately: if -- and ONLY if -- the question genuinely asks about more than one "
    "distinct thing (e.g. \"compare X and Y\", \"what is A and how does it relate to "
    "B\"), split it into up to 4 focused sub-questions, one per distinct thing. This "
    "is NOT the same as rewriting -- a single-topic question, however long or "
    "detailed, must come back as exactly ONE query in the list. Do not invent "
    "sub-questions that aren't actually implied by what was asked.\n\n"
    "Respond with the queries list only."
)

MEDIUM_HIGH_RESEARCH_INSTRUCTION = (
    # {original_question} is read from state (written explicitly by
    # RetrievalDispatchAgent), not left for the model to recall from conversation
    # history several turns back -- see subagent.py's RetrievalDispatchAgent
    # docstring for why that recall proved unreliable in practice.
    "The user's original question was: {original_question}\n\n"
    "Findings gathered so far:\n{findings}\n\n"
    "Decide whether these findings are enough to answer THAT question well. If they "
    "are sufficient, respond with exactly: SUFFICIENT. If something important is "
    "missing, describe specifically what's missing (not vague -- name the missing "
    "angle or detail) so it can be looked up."
)

REFINER_INSTRUCTION = (
    # {critique?} (not {critique}) -- the critic step occasionally produces no
    # usable output_key value (reproduced directly: its whole response can land in
    # native "thinking" content with nothing in regular content), and ADK's own `?`
    # suffix is the documented way to keep instruction templating from raising a
    # KeyError when a referenced state key is missing, instead of crashing the run.
    "A reviewer just assessed the current findings:\n{critique?}\n\n"
    "If the critique says SUFFICIENT, or is missing/empty, call `exit_loop` "
    "immediately and do nothing else. Otherwise, call `retrieve` with ONE new, "
    "specific query targeting exactly what the critique said was missing -- don't "
    "repeat a query you've already run. After calling retrieve, stop -- don't call "
    "any other tool this turn."
)

RESEARCH_ANSWER_INSTRUCTION = (
    "The user's original question was: {original_question}\n\n"
    "Findings gathered for that question, as numbered context snippets:\n"
    "{findings}\n\n"
    "Answer THAT question using only these snippets, exactly the way you'd answer "
    "from context given directly in a prompt -- they are your evidence, not a "
    "document dump to describe. Cite every claim with the snippet number(s) it came "
    "from, like [1] or [2][3]. If the snippets don't contain the answer, say so "
    "plainly instead of guessing."
)

MEDIUM_HIGH_ROOT_INSTRUCTION = (
    "You are the front door to a document research assistant. You don't search "
    "documents yourself -- for every user question, call the research tool exactly "
    "once, passing the user's question verbatim as `request`. It will return a "
    "complete, already-cited answer. Output that answer verbatim as your ENTIRE "
    "response -- nothing else. Do not narrate what you're about to do, do not explain "
    "these instructions back, do not say things like 'I received the result' or 'let "
    "me relay this' -- your response should start directly with the research tool's "
    "own answer text and contain nothing else."
)

INSPECT_IMAGE_GUIDANCE = (
    "\n\nSome results include an image (has_images=True) -- a chart, table, or figure "
    "that may contain information the surrounding text doesn't cover. If a result "
    "looks like it needs its image inspected to answer the question well, call "
    "`inspect_image` with that result's chunk_id before finalizing your answer."
)

PAGE_CONTEXT_GUIDANCE = (
    "\n\nIf a result's snippet seems cut off or you need more surrounding context "
    "than it alone gives, call `get_page_context` with that result's file_hash, "
    "collection_name, and page_no (set include_adjacent=True to also see the "
    "page before/after)."
)

# Tool name -> human-readable status line, for the streaming endpoint (Part 4).
STATUS_LABELS = {
    "retrieve": "Searching your documents...",
    "get_page_context": "Reading the full page for more context...",
    "inspect_image": "Looking closer at a retrieved image...",
    "exit_loop": "Wrapping up...",
}

# Agent name -> phase line, for the streaming endpoint (Part 4).
AGENT_STATUS_LABELS = {
    "reframe": "Breaking down your question...",
    "critic": "Checking whether the answer is complete...",
    "refiner": "Looking for more information...",
}
