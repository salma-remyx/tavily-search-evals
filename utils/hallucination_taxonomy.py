"""Span-severity taxonomy for grounding failures.

The baseline :class:`~utils.span_grounding_check.SpanGroundingChecker`
collapses every unsupported span of an answer into a single binary
"ungrounded" outcome and one ``hallucination_score``. The paper's detector
does more than that: it assigns each span a *category*. A span can either
**contradict** the retrieved evidence (it asserts something the evidence
denies -- an actively wrong claim) or merely be **unverifiable** (it is not
supported by the evidence, but nothing contradicts it either). The paper
tracks these separately because a contradicted span is a strictly more
severe failure than an unsupported one, and it flags a whole response as
hallucinated whenever *any* span is contradicted.

This module keeps that taxonomy at full fidelity and layers it on top of
the existing detector: it takes the spans the grounding step already flagged
as unsupported and splits them into ``CONTRADICTED`` vs ``UNVERIFIABLE``,
then reports a separate ``contradiction_score`` alongside the baseline's
``hallucination_score``. It runs only when the grounding step found
something, so a fully grounded answer costs no extra work.

Adapted (Mode 2) from "Beyond Document Grounding: Span-Level Hallucination
Detection over Code, Tool Output, and Documents" (arXiv:2607.00895). The
substituted auxiliary component is the *detector backbone*: the paper
fine-tunes a Qwen3.5-2B token-classifier that emits per-span categories,
while here the categories come from a zero-shot LLM judge over this repo's
existing ``ChatOpenAI(...).with_structured_output(...)`` path -- the same
detector family the paper evaluates and the same substitution the baseline
grounding checker already makes. The paper's separate training/benchmark
framework and its code / tool-output modalities are intentionally out of
scope for this prose-answer SimpleQA loop.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .span_grounding_check import SpanGroundingChecker

logger = logging.getLogger(__name__)


class SpanLabel(str, Enum):
    """The paper's span-severity taxonomy.

    ``SUPPORTED`` is included for completeness (a grounded span), but only
    ``CONTRADICTED`` and ``UNVERIFIABLE`` are ever assigned here, because the
    taxonomy pass only sees spans the grounding step already deemed
    unsupported.
    """

    SUPPORTED = "supported"
    UNVERIFIABLE = "unverifiable"
    CONTRADICTED = "contradicted"


class SpanCategory(BaseModel):
    """One labelled span emitted by the categorising judge."""

    text: str
    label: SpanLabel = SpanLabel.UNVERIFIABLE


class SpanCategories(BaseModel):
    """Structured output schema for the categorising judge."""

    spans: List[SpanCategory] = Field(default_factory=list)


class SpanTaxonomyClassifier:
    """Split already-detected ungrounded spans into contradicted vs unverifiable.

    The LLM only assigns a *category* to each span; the character offsets are
    inherited verbatim from the grounding step and the per-class scores are
    computed deterministically (see :func:`summarize_taxonomy`), so the
    output is reproducible and unit-testable independent of the judge.
    """

    CLASSIFY_PROMPT = """
    You are a strict grounding auditor operating in span-classification mode.

    You are given a retrieved CONTEXT, a QUESTION, and a list of SPANS taken
    verbatim from a model answer. Every span has ALREADY been judged as not
    directly supported by the CONTEXT. Your only job is to assign each span
    one of two severity labels:

    - "contradicted": the CONTEXT asserts something that conflicts with the
      span (e.g. the span says "Berlin" but the context says "Paris", or the
      span negates / changes a number, name, date, or fact the context
      states). The span is actively wrong given the evidence.
    - "unverifiable": the CONTEXT neither supports nor contradicts the span.
      There is simply no evidence either way.

    ## Rules (non-negotiable):
    - Return ONLY the structured object. Do not add any prose.
    - Copy each span's "text" VERBATIM from the input list.
    - Prefer "contradicted" only when the context genuinely conflicts with the
      span; when in doubt between the two, use "unverifiable".

    ## CONTEXT:
    {context}

    ## QUESTION:
    {question}

    ## SPANS (one per line):
    {spans}
    """.strip()

    def __init__(
        self,
        llm_model: str = "gpt-4.1",
        temperature: float = 0.0,
        structured_llm: Optional[object] = None,
    ):
        """Initialize the classifier.

        Args:
            llm_model: Model used for the zero-shot categorising judge.
            temperature: Sampling temperature for the judge (0 for determinism).
            structured_llm: Optional injected ``with_structured_output`` runnable
                (test seam). When omitted, one is built lazily from
                ``ChatOpenAI`` on first use so a fully grounded answer -- which
                never triggers a categorisation call -- needs no LLM at all.
        """
        self._llm_model = llm_model
        self._temperature = temperature
        self._structured_llm = structured_llm

    def _get_llm(self):
        if self._structured_llm is None:
            self._structured_llm = ChatOpenAI(
                model=self._llm_model, temperature=self._temperature
            ).with_structured_output(SpanCategories)
        return self._structured_llm

    def classify(
        self,
        context: str,
        question: str,
        answer: str,
        ungrounded_spans: List[Dict],
    ) -> Dict:
        """Assign a severity label to each already-detected ungrounded span.

        Args:
            context: The retrieved evidence the answer was drawn from.
            question: The original user query.
            answer: The extracted predicted answer (used only to score coverage).
            ungrounded_spans: Located span dicts (``text``/``start``/``end``)
                produced by :meth:`SpanGroundingChecker.check`.

        Returns:
            The taxonomy summary (see :func:`summarize_taxonomy`). When no
            spans are ungrounded, no LLM call is made and everything is empty.
            When the judge call fails, spans fail open to ``UNVERIFIABLE`` (the
            less-severe class) so an infra failure never inflates the
            contradiction signal.
        """
        if not ungrounded_spans:
            return summarize_taxonomy(answer, [])

        prompt = self.CLASSIFY_PROMPT.format(
            context=context or "",
            question=question or "",
            spans="\n".join(f"- {s['text']}" for s in ungrounded_spans),
        )

        label_by_text: Dict[str, SpanLabel] = {}
        try:
            result = self._get_llm().invoke([{"role": "user", "content": prompt}])
            for cat in getattr(result, "spans", None) or []:
                label = cat.label if isinstance(cat.label, SpanLabel) else SpanLabel(cat.label)
                # A span the detector already flagged can never be SUPPORTED here.
                if label == SpanLabel.SUPPORTED:
                    label = SpanLabel.UNVERIFIABLE
                label_by_text[cat.text] = label
        except Exception as e:  # noqa: BLE001 - infra failure must not break the run
            logger.error("Span taxonomy classification failed for '%s': %s", question, e)

        labelled = [
            {
                "text": span["text"],
                "start": span["start"],
                "end": span["end"],
                "label": label_by_text.get(span["text"], SpanLabel.UNVERIFIABLE),
            }
            for span in ungrounded_spans
        ]
        return summarize_taxonomy(answer, labelled)


def summarize_taxonomy(answer: str, labelled_spans: List[Dict]) -> Dict:
    """Reduce labelled ungrounded spans to per-class scores and a flag.

    Deterministic core, shared by tests and the classifier. ``contradiction_score``
    and ``unverifiable_score`` are the fractions of the answer's characters
    covered by contradicted / unverifiable spans respectively, computed with
    the same overlap-merging as ``hallucination_score``. ``hallucinated`` is
    the paper's response-level rule: True whenever *any* span is ungrounded
    (contradicted or unverifiable), not a ratio threshold.
    """
    contradicted = [s for s in labelled_spans if s["label"] == SpanLabel.CONTRADICTED]
    unverifiable = [s for s in labelled_spans if s["label"] == SpanLabel.UNVERIFIABLE]
    return {
        "contradiction_score": round(SpanGroundingChecker._coverage(answer, contradicted), 3),
        "unverifiable_score": round(SpanGroundingChecker._coverage(answer, unverifiable), 3),
        "contradicted_spans": contradicted,
        "unverifiable_spans": unverifiable,
        "hallucinated": bool(contradicted or unverifiable),
    }
