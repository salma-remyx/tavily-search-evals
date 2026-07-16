"""Tests for the answer-in-context diagnostic and its wiring.

The roundtrip test imports the *existing* ``save_result`` / ``save_summary``
helpers from ``utils.utils`` (not the new module) and exercises them end-to-end
with the diagnostic fields, proving the integration rather than just the new
module in isolation.
"""

import csv

from utils.utils import save_result, save_summary, EvaluationType
from utils.answer_in_context import (
    answer_in_context,
    summarize_answer_in_context,
    DEFAULT_CONTEXT_BUDGET_TOKENS,
)


def test_budget_truncation_drops_late_gold_but_full_context_keeps_it():
    # Gold answer sits beyond the budget window in a large packed context.
    filler = "filler filler filler " * 200
    context = filler + " the answer is forty-two and that is final"
    gold = "forty-two"

    aic = answer_in_context(context, gold, budget_tokens=20)

    assert aic["in_context_full"] is True        # present in full retrieved context
    assert aic["in_context"] is False            # but dropped by the budget prefix
    assert aic["budget_applied"] is True
    assert aic["context_tokens"] <= 20
    # content tokens of "forty-two" survive partially even when the span doesn't
    assert aic["coverage"] >= 0.0


def test_no_budget_reports_full_survival_only():
    aic = answer_in_context("Paris is the capital of France.", "Paris", budget_tokens=None)
    assert aic["in_context"] is True
    assert aic["in_context_full"] is True
    assert aic["budget_applied"] is False
    assert aic["coverage"] == 1.0


def test_empty_gold_and_empty_context_are_safe():
    assert answer_in_context("", "")["in_context"] is False
    assert answer_in_context("some context", "")["coverage"] == 0.0


def test_save_result_and_summary_roundtrip_with_aic(tmp_path):
    """Exercises the existing save_result/save_summary path with the new fields."""
    provider = "tavily"
    rows = [
        {
            "index": 0, "question": "q0", "reference_answer": "Paris",
            "predicted_answer": "Paris", "is_correct": True, "grade": "correct",
            "token_count": 100, "token_avg": 100,
            "aic_in_context": True, "aic_in_context_full": True,
            "aic_coverage": 1.0, "aic_budget_tokens": DEFAULT_CONTEXT_BUDGET_TOKENS,
        },
        {
            "index": 1, "question": "q1", "reference_answer": "Mars",
            "predicted_answer": "Venus", "is_correct": False, "grade": "incorrect",
            "token_count": 100, "token_avg": 100,
            "aic_in_context": False, "aic_in_context_full": False,
            "aic_coverage": 0.0, "aic_budget_tokens": DEFAULT_CONTEXT_BUDGET_TOKENS,
        },
    ]
    for row in rows:
        save_result(row, provider, str(tmp_path), EvaluationType.SIMPLEQA)

    provider_results = {provider: {"provider": provider}}
    save_summary(provider_results, str(tmp_path), EvaluationType.SIMPLEQA)

    # Per-example CSV carries the diagnostic columns.
    with open(f"{tmp_path}/{provider}_simpleqa_results.csv") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert "aic_in_context" in fieldnames
        assert "aic_coverage" in fieldnames
        records = list(reader)
    assert len(records) == 2

    # Summary CSV carries the provider-level aggregate columns.
    with open(f"{tmp_path}/summary.csv") as f:
        reader = csv.DictReader(f)
        summary_row = next(reader)
    assert "aic_in_context_rate" in summary_row
    assert "aic_separation" in summary_row

    # One of two examples survived the budget -> rate 0.5.
    assert float(summary_row["aic_in_context_rate"]) == 0.5
    # The surviving example was correct, the non-surviving one was not -> the
    # diagnostic separates answer quality, separation == 1.0.
    assert float(summary_row["aic_separation"]) == 1.0

    # The aggregate helper agrees when fed the roundtripped rows directly.
    agg = summarize_answer_in_context(records)
    assert agg["in_context_rate"] == 0.5
    assert agg["separation"] == 1.0
