"""Integration tests for the multi-run consistency audit.

These exercise ``utils.multirun_consistency.compute_run_consistency`` on
per-example result rows shaped exactly like the ones the existing SimpleQA loop
(``evaluate_provider_simple_qa``) already produces, with the grade vocabulary
pinned to the real ``CorrectnessEvaluator`` the pipeline uses. This proves the
audit consumes the pipeline's actual data contract, not a self-invented one.

Adapted from arXiv:2608.06202 ("What Current AI Benchmarks Leave Unmeasured"),
whose central finding is that repeated runs of the same prompt disagree on up to
21% of prompts -- exactly the signal ``compute_run_consistency`` surfaces.
"""

import csv
import os

import pytest

# Imported from a NON-NEW module in evaluators/ -- this is the evaluator the
# SimpleQA loop calls, so its grade vocabulary defines the rows we feed below.
from evaluators.correctness_evaluator import CorrectnessEvaluator
from utils.multirun_consistency import compute_run_consistency, save_consistency_summary

# Pin the synthetic grade strings to the evaluator the pipeline actually uses:
# CorrectnessEvaluator.evaluate() returns {"score": 1.0|0.0, "value": <grade>},
# and evaluate_provider_simple_qa stores that value as "grade" and score==1.0
# as "is_correct". Assert the vocabulary matches before building rows.
_EVALUATOR_TEMPLATE = CorrectnessEvaluator.OPENAI_GRADER_TEMPLATE
for _token in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED"):
    assert _token in _EVALUATOR_TEMPLATE, f"evaluator vocabulary drifted: missing {_token}"


def _row(index, grade, answer):
    """Build a per-example result row the way evaluate_provider_simple_qa does.

    grade is evaluator.evaluate()['value']; is_correct is score == 1.0, which
    the evaluator emits iff the grade is CORRECT.
    """
    return {
        "index": index,
        "question": f"question {index}",
        "reference_answer": "reference",
        "predicted_answer": answer,
        "is_correct": grade == "CORRECT",
        "grade": grade,
        "token_count": 0,
        "token_avg": 0,
    }


def test_all_runs_agree_is_fully_consistent():
    # Three runs, every example graded identically each time.
    runs = [
        [_row(0, "CORRECT", "Paris"), _row(1, "INCORRECT", "London")],
        [_row(0, "CORRECT", "Paris"), _row(1, "INCORRECT", "London")],
        [_row(0, "CORRECT", "Paris"), _row(1, "INCORRECT", "London")],
    ]

    metrics = compute_run_consistency(runs)

    assert metrics["n_runs"] == 3
    assert metrics["total_examples"] == 2
    # Paper's headline signal: 0% inconsistent when runs agree.
    assert metrics["grade_consistency_rate"] == 1.0
    assert metrics["inconsistent_count"] == 0
    assert metrics["inconsistent_rate"] == 0.0
    # Accuracy is stable across runs -> zero variance.
    assert metrics["accuracy_per_run"] == [0.5, 0.5, 0.5]
    assert metrics["accuracy_mean"] == 0.5
    assert metrics["accuracy_std"] == 0.0
    # Identical wording run-to-run -> perfect text similarity.
    assert metrics["answer_similarity_mean"] == 1.0
    assert metrics["abstention_consistency_rate"] == 1.0


def test_grade_disagreement_surfaces_inconsistency():
    # The paper found repeated runs disagree on up to 21% of prompts; here run 2
    # flips the grade on example 1, so 1 of 2 examples is inconsistent (50%).
    runs = [
        [_row(0, "CORRECT", "Paris"), _row(1, "CORRECT", "4")],
        [_row(0, "CORRECT", "Paris"), _row(1, "INCORRECT", "5")],
    ]

    metrics = compute_run_consistency(runs)

    assert metrics["grade_consistency_rate"] == 0.5
    assert metrics["inconsistent_count"] == 1
    assert metrics["inconsistent_rate"] == 0.5
    # Accuracy swings between runs -> nonzero stability std.
    assert metrics["accuracy_per_run"] == [1.0, 0.5]
    assert metrics["accuracy_std"] > 0.0
    # Example 1's wording diverged ("4" vs "5") -> mean similarity < 1.0.
    assert metrics["answer_similarity_mean"] < 1.0


def test_accuracy_variance_across_runs():
    # Run accuracies 1.0, 0.0, 0.5 -> mean 0.5, std of [0.5, 0.5, 0.0].
    runs = [
        [_row(0, "CORRECT", "a"), _row(1, "CORRECT", "b")],
        [_row(0, "INCORRECT", "a"), _row(1, "INCORRECT", "b")],
        [_row(0, "CORRECT", "a"), _row(1, "INCORRECT", "b")],
    ]

    metrics = compute_run_consistency(runs)

    assert metrics["accuracy_per_run"] == [1.0, 0.0, 0.5]
    assert metrics["accuracy_mean"] == pytest.approx(0.5, abs=1e-6)


def test_abstention_inconsistency_is_detected():
    # Example 0 abstains on run 1 but not run 2 -> abstention inconsistent.
    runs = [
        [_row(0, "NOT_ATTEMPTED", "I don't know")],
        [_row(0, "CORRECT", "Paris")],
    ]

    metrics = compute_run_consistency(runs)

    assert metrics["abstention_consistency_rate"] == 0.0
    # Grade also disagrees here, so it is flagged inconsistent too.
    assert metrics["inconsistent_count"] == 1


def test_single_run_is_trivially_consistent():
    # runs == 1 (the repo's default): nothing to disagree with.
    metrics = compute_run_consistency([[_row(0, "CORRECT", "Paris")]])

    assert metrics["n_runs"] == 1
    assert metrics["grade_consistency_rate"] == 1.0
    assert metrics["accuracy_std"] == 0.0


def test_empty_input_is_safe():
    metrics = compute_run_consistency([])

    assert metrics["n_runs"] == 0
    assert metrics["grade_consistency_rate"] == 0.0
    assert metrics["total_examples"] == 0


def test_save_consistency_summary_writes_csv(tmp_path):
    runs = [
        [_row(0, "CORRECT", "Paris"), _row(1, "INCORRECT", "London")],
        [_row(0, "CORRECT", "Paris"), _row(1, "CORRECT", "London")],
    ]
    metrics = compute_run_consistency(runs)

    summary_path = save_consistency_summary({"tavily": metrics}, str(tmp_path))

    assert os.path.basename(summary_path) == "multirun_consistency.csv"
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "tavily"
    assert row["n_runs"] == "2"
    assert row["total_examples"] == "2"
    # 1 of 2 examples flipped grade across runs -> 50% inconsistent.
    assert float(row["inconsistent_rate"]) == 0.5
    for column in (
        "accuracy_mean",
        "accuracy_std",
        "grade_consistency_rate",
        "answer_similarity_mean",
        "abstention_consistency_rate",
        "timestamp",
    ):
        assert column in row
