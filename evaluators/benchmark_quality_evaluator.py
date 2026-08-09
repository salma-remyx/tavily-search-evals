"""Reference-free benchmark-quality auditor.

Adapted from "Benchmarking the Benchmarks: Evaluating Benchmarks for
Conversational Agents" (arxiv:2608.06329). The paper proposes using LLM
judges to assess a benchmark along three axes -- consistency, complexity,
and policy coverage -- without ground-truth labels, producing actionable
diagnostics of weaknesses in the test set itself (rather than grading
predicted answers).

This module ports the paper's two LLM-judge axes (consistency, complexity)
at full fidelity and substitutes the paper's policy-coverage axis -- which
requires a hand-defined policy taxonomy this repo does not carry -- with a
parameter-free coverage proxy computed from lexical / answer-shape
diversity of the question set. The paper's validation apparatus (human
annotations, controlled quality-degrading perturbations, cross-domain
correlation) is intentionally out of scope here and belongs in a
downstream evaluation.

It consumes the loaded ``examples`` list (the test set) the same way the
existing :class:`evaluators.correctness_evaluator.CorrectnessEvaluator`
consumes answers -- via ``langchain_openai.ChatOpenAI`` with structured
output -- so it slots into the same evaluation pipeline.
"""

import random
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()


class BenchmarkQualityGrade(BaseModel):
    """Schema for a single item's reference-free quality judgement."""

    consistency_score: Annotated[
        int,
        "consistency score from 1 (ambiguous or unanswerable as written) to 5 (fully unambiguous)",
    ]
    complexity_score: Annotated[
        int,
        "complexity score from 1 (trivially answerable, near-zero search value) to 5 (requires substantial retrieval or multi-step reasoning)",
    ]
    issue: Annotated[
        str,
        "short diagnostic tag: 'none' for a well-formed non-trivial item, else one of 'ambiguous', 'unanswerable', 'trivial', 'underspecified', or another concise label naming the dominant weakness",
    ]


@dataclass
class BenchmarkQualityConfig:
    """Configuration for the benchmark-quality audit."""

    model_name: str = "gpt-4.1"
    temperature: float = 0.0
    # How many items to send to the judge. Judging every item is unnecessary
    # for estimating benchmark quality; a reproducible sample keeps it cheap.
    sample_size: int = 50


# Reference-free judge prompt: the judge never sees a predicted answer.
JUDGE_TEMPLATE = """
You are auditing the quality of a single benchmark item: a question paired
with its reference answer. You do NOT see any predicted answer -- judge the
item on its own merits. This is a reference-free quality audit.

Score the item on two axes (each 1-5) and emit a short diagnostic tag.

CONSISTENCY (is the task well-formed and unambiguous?):
  5 - Completely unambiguous; a single determinable answer; the reference answer fits exactly.
  3 - Mostly clear but with minor ambiguity (e.g. a slightly underspecified entity or time window).
  1 - Ambiguous, internally inconsistent, or unanswerable as written; the reference answer does not resolve it.

COMPLEXITY (how much retrieval or reasoning does it demand?):
  5 - Requires multi-step reasoning or synthesizing several retrieved facts.
  3 - Requires a single non-trivial retrieval / lookup.
  1 - Trivially answerable from common knowledge or a closed-book guess.

Issue tag (concise): "none" for a well-formed, non-trivial item; otherwise
one of "ambiguous", "unanswerable", "trivial", "underspecified", or another
single short label that names the dominant weakness.

Item:
Question: {question}
Reference answer: {reference_answer}

Return only the structured scores.
""".strip()


class BenchmarkQualityEvaluator:
    """Reference-free LLM-judge auditor for benchmark (test-set) quality.

    Adapted from arxiv:2608.06329: consistency and complexity are scored by
    an LLM judge at full fidelity; policy coverage is approximated by a
    parameter-free coverage proxy (see :meth:`_coverage_proxy`).
    """

    def __init__(self, config: BenchmarkQualityConfig = BenchmarkQualityConfig()):
        """Initialize the auditor with configuration."""
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
        ).with_structured_output(BenchmarkQualityGrade)

    def _judge(self, item: Dict[str, Any]) -> BenchmarkQualityGrade:
        """Score a single item with the LLM judge."""
        prompt = JUDGE_TEMPLATE.format(
            question=item.get("question", ""),
            reference_answer=item.get("answer", ""),
        )
        return self.llm.invoke([{"role": "user", "content": prompt}])

    def audit(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Audit a list of benchmark items and return aggregate diagnostics.

        Each item is a dict with ``question`` and ``answer`` keys -- the same
        shape produced by ``utils.prepare_examples``. The coverage proxy is
        computed over the full item list; judging is applied to a
        reproducible sample to bound cost.
        """
        items = list(items)
        total = len(items)
        if total == 0:
            return self._empty_report()

        sample_size = self.config.sample_size
        if sample_size is not None and 0 < sample_size < total:
            # Deterministic sample: the audit must be reproducible across runs.
            to_judge = random.Random(0).sample(items, sample_size)
        else:
            to_judge = items
            sample_size = total

        grades: List[BenchmarkQualityGrade] = []
        for item in to_judge:
            try:
                grades.append(self._judge(item))
            except Exception as exc:  # keep auditing the rest of the set
                _logger().warning("benchmark audit: judge failed on item %r: %s",
                                  item.get("index"), exc)

        n_judged = len(grades)
        consistencies = [g.consistency_score for g in grades]
        complexities = [g.complexity_score for g in grades]
        issues = [(g.issue or "none").strip().lower() for g in grades]
        issue_counts = Counter(issues)
        flagged = sum(1 for tag in issues if tag != "none")

        return {
            "n_total": total,
            "n_audited": sample_size,
            "n_judged": n_judged,
            "judge_failures": sample_size - n_judged,
            "mean_consistency": round(sum(consistencies) / n_judged, 3) if n_judged else 0.0,
            "mean_complexity": round(sum(complexities) / n_judged, 3) if n_judged else 0.0,
            "flagged_ratio": round(flagged / n_judged, 3) if n_judged else 0.0,
            "issue_counts": dict(issue_counts),
            "coverage": self._coverage_proxy(items),
            "model": self.config.model_name,
        }

    def _coverage_proxy(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Parameter-free proxy for the paper's policy-coverage axis.

        Without a hand-defined policy taxonomy, approximate "does the
        benchmark cover a varied space" via question uniqueness, answer
        uniqueness, and lexical (type-token) diversity.
        """
        questions = [str(it.get("question", "")).lower() for it in items]
        answers = [str(it.get("answer", "")).lower() for it in items]
        total = len(items) or 1

        unique_question_ratio = len(set(questions)) / total
        unique_answer_ratio = len(set(answers)) / total

        tokens = [tok for q in questions for tok in q.split()]
        lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0.0

        coverage_score = (unique_question_ratio + unique_answer_ratio + lexical_diversity) / 3
        return {
            "unique_question_ratio": round(unique_question_ratio, 3),
            "unique_answer_ratio": round(unique_answer_ratio, 3),
            "lexical_diversity": round(lexical_diversity, 3),
            "coverage_score": round(coverage_score, 3),
        }

    def _empty_report(self) -> Dict[str, Any]:
        return {
            "n_total": 0,
            "n_audited": 0,
            "n_judged": 0,
            "judge_failures": 0,
            "mean_consistency": 0.0,
            "mean_complexity": 0.0,
            "flagged_ratio": 0.0,
            "issue_counts": {},
            "coverage": {
                "unique_question_ratio": 0.0,
                "unique_answer_ratio": 0.0,
                "lexical_diversity": 0.0,
                "coverage_score": 0.0,
            },
            "model": self.config.model_name,
        }

    @property
    def evaluation_name(self) -> str:
        """Name of this evaluator."""
        return "benchmark_quality_evaluator"

    @property
    def evaluation_description(self) -> str:
        """Description of what this evaluator does."""
        return (
            "Reference-free LLM-judge audit of benchmark consistency, "
            "complexity, and coverage."
        )


def _logger():
    """Lazy module-level logger (matches the repo's logging convention)."""
    import logging

    return logging.getLogger(__name__)
