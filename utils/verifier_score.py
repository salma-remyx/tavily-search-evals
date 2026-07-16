"""Continuous answer-verification scoring via logit expectation.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391v1).

The repo's ``CorrectnessEvaluator`` is a *standard LM judge*: it prompts the
model for a single grading token (A/B/C = CORRECT/INCORRECT/NOT_ATTEMPTED)
and takes the ``argmax`` of that token, collapsing to a discrete 1.0/0.0
score. LLM-as-a-Verifier's core insight is that the *distribution* over
those same scoring tokens already carries fine-grained information: instead
of the argmax, compute the **expectation over the scoring-token logits** to
get a continuous correctness score in [0, 1]. This is the paper's
"score granularity" scaling axis, and it is the mechanism implemented here
at full fidelity (softmax over the A/B/C logprobs -> expected score).

This is a Mode 2 (adapted port). The following auxiliary axes from the paper
are intentionally out of scope because they require infrastructure this repo
does not host:

  - **Repeated evaluation** (temperature resampling for variance reduction):
    a single greedy call with ``top_logprobs`` already exposes the full
    scoring-token distribution, so the resampling axis adds little here.
  - **Criteria decomposition**, the **cost-efficient candidate-ranking
    algorithm**, **dense RL feedback**, and the **Claude Code extension**:
    these assume a multi-criteria judge, a candidate-selection pipeline, an
    RL trainer, and a host agent respectively -- none of which exist in this
    evaluation framework.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# A/B/C grading scheme, mirroring evaluators.correctness_evaluator.
GRADE_LABELS = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
# Per-grade score contribution. Correctness probability == P(A); B and C both
# map to 0.0, matching CorrectnessEvaluator's ``score == 1.0 if grade == "A"``.
GRADE_SCORES = {"A": 1.0, "B": 0.0, "C": 0.0}

# Compact A/B/C grader. Same rubric semantics as CorrectnessEvaluator but kept
# short so the first generated token is reliably the grade letter.
_GRADER_TEMPLATE = """\
You are grading a predicted answer against a reference (gold) answer.

Grade the predicted answer as exactly one of:
- A: CORRECT -- fully contains the important information in the reference, \
no contradictions. Only semantic meaning matters; capitalization, punctuation, \
grammar, and order do not. Hedging is allowed if the gold target is included.
- B: INCORRECT -- contains a factual statement that contradicts the reference.
- C: NOT_ATTEMPTED -- does not include the gold target's information and does \
not contradict it (e.g. "I don't know", too vague).

Question: {question}
Reference answer: {reference_answer}
Predicted answer: {predicted_answer}

Reply with only the single letter A, B, or C.
"""


def _entry_field(entry: Any, field: str) -> Any:
    """Read a field from a logprobs entry that may be a dict or an object."""
    if isinstance(entry, dict):
        return entry.get(field)
    return getattr(entry, field, None)


def _grade_of(token: Any) -> Optional[str]:
    """Normalize a scoring token to an A/B/C grade, or None if it is not one."""
    if not isinstance(token, str):
        return None
    normalized = token.strip().upper()
    return normalized if normalized in GRADE_LABELS else None


def extract_grade_logprobs(logprobs: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Collect the best (least-negative) logprob per grade token.

    Walks the chosen tokens and their ``top_logprobs`` alternatives from a
    langchain ``response_metadata["logprobs"]`` blob and keeps the highest
    logprob seen for each of A/B/C. Token text is stripped/upper-cased so a
    leading-space variant such as ``" A"`` still maps to ``"A"``.

    Returns a (possibly empty) mapping of grade -> natural-log probability.
    """
    best: Dict[str, float] = {}
    content = (logprobs or {}).get("content") or []
    for entry in content:
        if entry is None:
            continue
        candidates = []
        chosen_token = _entry_field(entry, "token")
        chosen_logprob = _entry_field(entry, "logprob")
        if chosen_token is not None and chosen_logprob is not None:
            candidates.append((chosen_token, chosen_logprob))
        for alt in (_entry_field(entry, "top_logprobs") or []):
            alt_token = _entry_field(alt, "token")
            alt_logprob = _entry_field(alt, "logprob")
            if alt_token is not None and alt_logprob is not None:
                candidates.append((alt_token, alt_logprob))
        for token, logprob in candidates:
            grade = _grade_of(token)
            if grade is None or not isinstance(logprob, (int, float)):
                continue
            if grade not in best or logprob > best[grade]:
                best[grade] = float(logprob)
    return best


def expected_score(
    grade_logprobs: Dict[str, float],
    grade_scores: Optional[Dict[str, float]] = None,
) -> Tuple[float, str, Dict[str, float]]:
    """Compute the LLM-as-a-Verifier continuous score.

    Softmax-normalizes the supplied grade logprobs (restricted to the known
    grades) and returns the expected score ``sum(grade_prob * grade_score)``.

    Args:
        grade_logprobs: mapping of grade token (e.g. ``"A"``) to its natural
            log-probability under the judge model.
        grade_scores: per-grade score contributions (default ``GRADE_SCORES``).

    Returns:
        ``(expected_score, argmax_grade, {grade: probability})``. When no
        scoring token was recovered, returns ``(0.0, "C", {})`` -- treating an
        unparseable judgment as not-attempted, matching CorrectnessEvaluator's
        fallback.
    """
    grade_scores = grade_scores or GRADE_SCORES
    known = {g: lp for g, lp in grade_logprobs.items() if g in grade_scores}
    if not known:
        return 0.0, "C", {}
    max_lp = max(known.values())
    exps = {g: math.exp(lp - max_lp) for g, lp in known.items()}
    total = sum(exps.values())
    probs = {g: (e / total if total > 0 else 0.0) for g, e in exps.items()}
    score = sum(probs[g] * grade_scores[g] for g in probs)
    argmax_grade = max(probs, key=probs.get)
    return score, argmax_grade, probs


class VerifierScorer:
    """Score answer correctness continuously via scoring-token logprobs.

    Uses the same A/B/C grading scheme as ``CorrectnessEvaluator`` but reads
    the distribution over the grade token rather than its argmax, producing a
    continuous score (the probability the answer is correct).
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        top_logprobs: int = 20,
        grade_scores: Optional[Dict[str, float]] = None,
    ):
        self.grade_scores = grade_scores or GRADE_SCORES
        # Plain generation (no structured output) so we can read raw token
        # logprobs; cap at one token since the grade is a single letter.
        self._llm = ChatOpenAI(model=model_name, temperature=temperature).bind(
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=1,
        )

    async def score(
        self,
        question: str,
        predicted_answer: str,
        reference_answer: str,
    ) -> Dict[str, Any]:
        """Return a continuous correctness score for one predicted answer.

        Returns a dict with ``verifier_score`` (float in [0, 1]),
        ``verifier_grade`` (label), and ``verifier_probs`` (per-grade
        probability).
        """
        prompt = _GRADER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
        )
        message = await self._llm.ainvoke([{"role": "user", "content": prompt}])
        logprobs = (getattr(message, "response_metadata", None) or {}).get("logprobs") or {}
        grade_logprobs = extract_grade_logprobs(logprobs)
        cont_score, grade, probs = expected_score(grade_logprobs, self.grade_scores)
        if not probs:
            logger.warning("Verifier recovered no A/B/C scoring tokens; defaulting to 0.0")
        return {
            "verifier_score": round(cont_score, 4),
            "verifier_grade": GRADE_LABELS.get(grade, grade),
            "verifier_probs": {g: round(p, 4) for g, p in probs.items()},
        }
