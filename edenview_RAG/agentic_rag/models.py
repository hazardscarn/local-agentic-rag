"""Internal-only Pydantic schemas for the agentic RAG loop (not exposed over the API --
see api/schemas.py for the request/response shapes callers see)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DecomposedQuestions(BaseModel):
    """Output of the one-time `decompose` step, run ONCE per turn before any
    searching starts -- NOT re-run per loop iteration. Splits the user's original
    question into standalone sub-questions only if it genuinely asks about more than
    one distinct thing; a single-topic question comes back as a length-1 list. This
    is the ONLY place topic-counting/splitting happens -- see prompts.py's module
    docstring for why conflating this with per-iteration rephrasing (the old design)
    caused reworder to re-decide the split on every pass and diverge."""

    questions: list[str] = Field(max_length=4)


class RewordedQueries(BaseModel):
    """Output of the reworder step, which runs once per iteration of
    `subquestion_loop` -- one loop instance PER decomposed sub-question (see
    agent.py's SubquestionOrchestrator). Its only job is rephrasing `{
    current_subquestion}` for better corpus retrieval -- it never splits or invents
    new sub-questions (that already happened once, upstream, in `decompose`), hence
    max_length=1: at most one alternate phrasing of the SAME fixed sub-question.
    An empty list is valid: reworder only produces a query at all if this is the
    first attempt at this sub-question, or the prior Eval verdict for it said
    needs_requery -- see prompts.py::REWORDER_INSTRUCTION for the exact wording."""

    queries: list[str] = Field(max_length=1)


class EvalOutput(BaseModel):
    """Output of the eval step -- a CRAG-style lightweight grader over the findings
    gathered so far for the current sub-question. `sufficient` is what
    SubquestionLoop (agent.py) reads to decide whether to end that sub-question's
    loop; `needs_requery`/`needs_deep_search` are what SubquestionLoop's own control
    flow reads to decide whether to actually invoke reworder/deep_search at all on
    the NEXT iteration (both were previously always-invoked LLM steps -- now they're
    only called when their corresponding flag is true, see agent.py's module
    docstring) -- both flags can be true at once."""

    sufficient: bool
    needs_requery: bool = False
    needs_deep_search: bool = False
    reason: str
