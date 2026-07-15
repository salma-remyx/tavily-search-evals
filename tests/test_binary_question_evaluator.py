"""Unit tests for the binary-question evaluator (BINEVAL adaptation).

Covers the pure aggregation core and the LLM-backed ``evaluate`` path with
the judge mocked out (no network access required).
"""

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.binary_question_evaluator import (  # noqa: E402
    DEFAULT_SIMPLEQA_QUESTIONS,
    BinaryQuestion,
    BinaryQuestionConfig,
    BinaryQuestionEvaluator,
    aggregate_verdicts,
)


def test_default_questions_are_well_formed():
    ids = [q.id for q in DEFAULT_SIMPLEQA_QUESTIONS]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    assert len(DEFAULT_SIMPLEQA_QUESTIONS) >= 3
    for question in DEFAULT_SIMPLEQA_QUESTIONS:
        assert question.polarity in ("yes_good", "yes_bad")


def test_aggregate_yes_good_vs_yes_bad_polarity():
    # yes_good said-yes -> good; yes_bad said-yes -> bad.
    questions = [
        BinaryQuestion("a", "dim", "a?", "yes_good"),
        BinaryQuestion("b", "dim", "b?", "yes_bad"),
    ]
    res = aggregate_verdicts(questions, {"a": True, "b": True})
    assert res["score"] == 0.5
    assert res["dimensions"]["dim"] == 0.5
    assert res["verdicts"] == {"a": True, "b": True}


def test_aggregate_missing_verdict_defaults_to_no():
    questions = [BinaryQuestion("a", "dim", "a?", "yes_good")]
    res = aggregate_verdicts(questions, {})
    assert res["score"] == 0.0
    assert res["verdicts"] == {"a": False}


def test_aggregate_groups_by_dimension():
    questions = [
        BinaryQuestion("a", "fact", "a?", "yes_good"),
        BinaryQuestion("b", "fact", "b?", "yes_good"),
        BinaryQuestion("c", "style", "c?", "yes_good"),
    ]
    res = aggregate_verdicts(questions, {"a": True, "b": False, "c": True})
    assert res["dimensions"] == {"fact": 0.5, "style": 1.0}
    assert res["score"] == round(2 / 3, 3)


class _FakeLLM:
    """Stand-in for the structured-output judge; returns canned verdicts."""

    def __init__(self, verdicts):
        # verdicts: list of (question_id, answer) tuples
        self._verdicts = verdicts

    def invoke(self, messages):
        return SimpleNamespace(
            verdicts=[
                SimpleNamespace(question_id=qid, answer=ans)
                for qid, ans in self._verdicts
            ]
        )


def _patched_evaluator(verdicts):
    evaluator = BinaryQuestionEvaluator(BinaryQuestionConfig())
    evaluator.llm = _FakeLLM(verdicts)
    return evaluator


def test_evaluate_aggregates_full_verdicts():
    evaluator = _patched_evaluator(
        [
            ("contains_key_info", True),
            ("contradicts_reference", False),
            ("addresses_question", True),
            ("fully_answers", True),
            ("is_concise", True),
        ]
    )
    res = asyncio.run(
        evaluator.evaluate(
            question="What is the capital of France?",
            predicted_answer="Paris",
            reference_answer="Paris",
        )
    )
    assert res["score"] == 1.0
    assert res["dimensions"]["factual_consistency"] == 1.0
    # raw per-question verdicts preserved for transparency / debugging
    assert res["verdicts"]["contains_key_info"] is True
    assert res["verdicts"]["contradicts_reference"] is False


def test_evaluate_handles_partial_judge_response():
    evaluator = _patched_evaluator([("contains_key_info", True)])
    res = asyncio.run(
        evaluator.evaluate(question="q", predicted_answer="a", reference_answer="r")
    )
    # only 1 of 5 answered; the rest default to "no", so score < 1.0
    assert res["score"] < 1.0
    assert res["verdicts"]["contains_key_info"] is True
    assert res["verdicts"]["is_concise"] is False
