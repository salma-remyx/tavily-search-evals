"""Span-level answer-grounding check.

Complements the existing CorrectnessEvaluator (which compares a predicted
answer against a gold reference) with a *grounding* dimension: given the
retrieved context, the question, and the extracted answer, flag the spans
of the answer that are NOT supported by the context and reduce them to a
single ``hallucination_score`` in ``[0, 1]``.

Adapted (Mode 2) from "Beyond Document Grounding: Span-Level Hallucination
Detection over Code, Tool Output, and Documents" (arXiv:2607.00895). The
paper's core mechanism -- span-level hallucination detection over
``(context, question, answer)`` producing character-labelled spans and a
score -- is kept at full fidelity. The auxiliary component that is
substituted is the *detector backbone*: the paper fine-tunes a Qwen3.5-2B
detector, while we use a zero-shot LLM judge through this repo's existing
``ChatOpenAI(...).with_structured_output(...)`` path (the same pattern
``CorrectnessEvaluator`` already uses). The paper itself evaluates zero-shot
LLM judges as one detector family, so this is a within-method substitution
rather than a collapse to a naive baseline. The paper's separate
benchmark/training framework is intentionally out of scope -- evaluation is
wired into this repo's SimpleQA loop instead.
"""

import logging
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UngroundedSpans(BaseModel):
    """Structured output schema for the grounding judge.

    ``ungrounded_spans`` holds verbatim substrings of the answer that the
    judge could not find in (or infer from) the retrieved context.
    """

    ungrounded_spans: List[str] = Field(default_factory=list)


class SpanGroundingChecker:
    """Detect answer spans that are not grounded in the retrieved context.

    The LLM only *labels* which verbatim substrings of the answer are
    unsupported; the character offsets and the ``hallucination_score`` are
    computed deterministically from those labels (see ``_locate`` /
    ``_coverage``). This keeps the score reproducible and unit-testable
    independent of LLM offset fidelity.
    """

    GROUNDING_PROMPT = """
    You are a strict grounding auditor operating in span-labelling mode.

    You are given a retrieved CONTEXT, a QUESTION, and a model ANSWER. Your
    job is to identify every span of the ANSWER that is NOT supported by the
    CONTEXT -- i.e. factual content in the answer that cannot be located in,
    or directly inferred from, the context. These are the "ungrounded" /
    hallucinated spans.

    ## Rules (non-negotiable):
    - Return ONLY the structured object. Do not add any prose.
    - Each span MUST be copied VERBATIM from the ANSWER (identical
      characters, capitalization, and punctuation). Never paraphrase.
    - Copy the minimal phrase that is unsupported; do not include surrounding
      text that IS supported by the context.
    - A span is ungrounded only if the context does not support it. If the
      context neither confirms nor denies a claim but the claim is a specific
      factual assertion (a number, name, date, place), treat it as ungrounded.
    - If every part of the answer is supported by the context, return an empty
      list of spans.

    ## CONTEXT:
    {context}

    ## QUESTION:
    {question}

    ## ANSWER:
    {answer}
    """.strip()

    def __init__(
        self,
        llm_model: str = "gpt-4.1",
        temperature: float = 0.0,
        structured_llm: Optional[object] = None,
    ):
        """Initialize the checker.

        Args:
            llm_model: Model used for the zero-shot grounding judge.
            temperature: Sampling temperature for the judge (0 for determinism).
            structured_llm: Optional injected ``with_structured_output`` runnable
                (test seam). When omitted, one is built from ``ChatOpenAI``.
        """
        self.structured_llm = structured_llm or ChatOpenAI(
            model=llm_model, temperature=temperature
        ).with_structured_output(UngroundedSpans)

    def check(self, context: str, question: str, answer: str) -> Dict:
        """Score how much of ``answer`` is ungrounded w.r.t. ``context``.

        Args:
            context: The retrieved evidence the answer was supposed to be
                drawn from (e.g. the post-processed search result).
            question: The original user query.
            answer: The extracted predicted answer to audit.

        Returns:
            A dict with:
              - ``hallucination_score``: float in [0, 1]; fraction of the
                answer's characters covered by ungrounded spans (0 = fully
                grounded). ``0.0`` is also returned when the answer is empty
                or the judge call fails (logged), so an infra failure never
                penalises a provider.
              - ``ungrounded_spans``: list of ``{text, start, end, grounded}``
                dicts located in the answer.
              - ``grounded``: True iff no ungrounded span was detected.
        """
        if not answer or not str(answer).strip():
            return self._empty_result()

        prompt = self.GROUNDING_PROMPT.format(
            context=context or "", question=question or "", answer=answer
        )

        try:
            detected = self.structured_llm.invoke(
                [{"role": "user", "content": prompt}]
            )
            span_texts = getattr(detected, "ungrounded_spans", None) or []
        except Exception as e:  # noqa: BLE001 - infra failure must not break the run
            logger.error("Grounding check failed for query '%s': %s", question, e)
            return self._empty_result(error=str(e))

        spans = self._locate(answer, span_texts)
        score = self._coverage(answer, spans)
        return {
            "hallucination_score": round(score, 3),
            "ungrounded_spans": spans,
            "grounded": score == 0.0,
        }

    @staticmethod
    def _locate(answer: str, span_texts: List[str]) -> List[Dict]:
        """Locate each verbatim span text inside the answer by first match.

        Spans the judge did not copy verbatim (and so cannot be found) are
        silently dropped -- they cannot be reliably labelled.
        """
        located: List[Dict] = []
        for text in span_texts or []:
            if not text:
                continue
            start = answer.find(text)
            if start == -1:
                continue
            located.append(
                {"text": text, "start": start, "end": start + len(text), "grounded": False}
            )
        return located

    @staticmethod
    def _coverage(answer: str, located: List[Dict]) -> float:
        """Fraction of ``answer`` characters covered by the located spans.

        Overlapping spans are merged first so the score never double-counts.
        """
        if not answer:
            return 0.0
        intervals = sorted((s["start"], s["end"]) for s in located)
        merged: List[List[int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        covered = sum(end - start for start, end in merged)
        return min(1.0, covered / len(answer))

    @staticmethod
    def _empty_result(error: Optional[str] = None) -> Dict:
        result = {
            "hallucination_score": 0.0,
            "ungrounded_spans": [],
            "grounded": True,
        }
        if error:
            result["error"] = error
        return result
