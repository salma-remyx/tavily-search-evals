"""Integration tests for the vital-nugget recall evaluator.

These import from existing, non-new modules (``utils`` EvaluationType and
``evaluators.correctness_evaluator``) to prove the new metric plugs into the
repo's existing SimpleQA answer-grading contract, then exercise it end to end
on the offline (heuristic) path. No search provider or OpenAI call is made.
"""

import asyncio

from utils import EvaluationType  # existing, non-new module in utils/
from evaluators.correctness_evaluator import (  # existing, non-new evaluator
    CorrectnessConfig,
    CorrectnessEvaluator,
)

from evaluators.vital_nugget_recall import (
    VitalNuggetRecallEvaluator,
    extract_nuggets_heuristic,
    score_result_row,
    score_vital_nugget_recall)


def test_heuristic_nugget_split():
    nuggets = extract_nuggets_heuristic("Malia Obama and Sasha Obama")
    assert "Malia Obama" in nuggets
    assert "Sasha Obama" in nuggets
    # A single-fact answer with no seams stays a single nugget.
    assert extract_nuggets_heuristic("Michio Sugeno") == ["Michio Sugeno"]


def test_full_coverage_scores_one():
    result = score_vital_nugget_recall(
        reference_answer="Malia Obama and Sasha Obama",
        predicted_answer="sasha and malia obama",
    )
    assert result["recall"] == 1.0
    assert result["strict_recall"] == 1.0
    assert result["value"] == "FULL_COVERAGE"
    assert result["total"] == 2 and result["covered"] == 2


def test_partial_coverage_strict_recall_zero():
    # Missing one of two nuggets: recall is fractional, strict recall drops.
    result = score_vital_nugget_recall(
        reference_answer="Malia Obama and Sasha Obama",
        predicted_answer="Malia Obama",
    )
    assert result["recall"] == 0.5
    assert result["strict_recall"] == 0.0
    assert result["value"] == "PARTIAL_COVERAGE"


def test_no_coverage():
    result = score_vital_nugget_recall(
        reference_answer="San Francisco, California",
        predicted_answer="I don't know.",
    )
    assert result["recall"] == 0.0
    assert result["value"] == "NO_COVERAGE"


def test_evaluator_matches_correctness_contract():
    # The new evaluator shares CorrectnessEvaluator's evaluate() shape and
    # returns the same required keys ("score", "value").
    assert hasattr(CorrectnessEvaluator, "evaluate")
    assert CorrectnessConfig().model_name == "gpt-4.1"

    evaluator = VitalNuggetRecallEvaluator()
    result = asyncio.run(evaluator.evaluate(
        inputs={"question": "Who are Obama's children?"},
        outputs={"answer": "Sasha and Malia Obama"},
        reference_outputs={"answer": "Malia Obama and Sasha Obama"},
    ))
    assert "score" in result and "value" in result
    assert result["score"] == 1.0
    assert result["value"] == "FULL_COVERAGE"


def test_scores_existing_simpleqa_result_row():
    # score_result_row consumes the schema emitted by evaluate_provider_simple_qa
    # in run_evaluation.py, gated to the SimpleQA evaluation type.
    assert EvaluationType.SIMPLEQA.value == "simpleqa"

    row = {
        "index": 0,
        "question": "Name Obama's children.",
        "reference_answer": "Malia Obama and Sasha Obama",
        "predicted_answer": "Malia, Sasha, and Susan Obama",
        "is_correct": False,  # binary grader flags the hallucinated "Susan"
        "grade": "INCORRECT",
    }
    scored = score_result_row(row)
    # Both gold nuggets (Malia, Sasha) are covered even though the binary grade
    # is INCORRECT: vital-nugget recall surfaces the partial signal the binary
    # grade collapses away.
    assert scored["total"] == 2
    assert scored["covered"] == 2
    assert scored["recall"] == 1.0
