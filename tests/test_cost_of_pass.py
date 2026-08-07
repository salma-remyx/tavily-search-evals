"""Tests for the cost-of-pass economic metric and its summary wiring.

Exercises the public ``utils.utils.save_summary`` call site (a non-new module)
to prove the cost-of-pass column is wired into summary.csv, plus unit tests on
``utils.cost_of_pass`` itself.
"""

import csv
import os

import pandas as pd
import pytest

from utils.utils import EvaluationType, save_summary
from utils.cost_of_pass import (
    cost_of_pass,
    frontier_cost_of_pass,
    price_per_token,
)


def _write_provider_results(output_dir, provider_name, rows):
    """Write a fake ``{provider}_simpleqa_results.csv`` like the pipeline does."""
    path = os.path.join(output_dir, f"{provider_name}_simpleqa_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_cost_of_pass_matches_paper_formula():
    # 1M tokens @ $2/M (gpt-4.1) = $2.0 total; 3 correct of 5 attempts.
    # cost_of_pass = per_attempt_cost / accuracy = (2.0/5) / (3/5) = 2.0/3.
    cop = cost_of_pass(total_tokens=1_000_000, total_attempts=5, correct_count=3,
                       model="gpt-4.1")
    assert cop == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_cost_of_pass_infinite_when_never_correct():
    cop = cost_of_pass(total_tokens=1_000_000, total_attempts=5, correct_count=0,
                       model="gpt-4.1")
    assert cop == float("inf")


def test_cost_of_pass_scales_with_price():
    base = cost_of_pass(1_000_000, 5, 3, model="gpt-4.1")
    mini = cost_of_pass(1_000_000, 5, 3, model="gpt-4.1-mini")
    # mini is 5x cheaper per token than 4.1, so its cost-of-pass is 5x lower.
    assert mini == pytest.approx(base / 5.0, rel=1e-9)
    assert price_per_token("gpt-4.1") == 2.0e-6
    assert price_per_token("gpt-4.1-mini") == 0.4e-6


def test_price_per_token_falls_back_for_unknown_model():
    # Unknown token_model falls back to the gpt-4.1 default so the metric
    # is always defined rather than raising.
    assert price_per_token("some-unknown-model") == price_per_token("gpt-4.1")
    assert price_per_token(None) == price_per_token("gpt-4.1")


def test_frontier_cost_of_pass_picks_cheapest():
    costs = {"tavily": 0.5, "exa": 2.0, "brave": float("inf")}
    provider, cost = frontier_cost_of_pass(costs)
    assert provider == "tavily"
    assert cost == 0.5


def test_frontier_cost_of_pass_all_off_frontier():
    provider, cost = frontier_cost_of_pass({"a": float("inf"), "b": float("inf")})
    assert provider is None
    assert cost == float("inf")


def test_save_summary_writes_cost_of_pass_column(tmp_path):
    """Integration: save_summary (existing module) emits cost_of_pass in summary.csv."""
    _write_provider_results(
        str(tmp_path),
        "tavily",
        rows={
            "index": [0, 1, 2, 3, 4],
            "is_correct": [True, True, True, False, False],
            "token_count": [200_000, 200_000, 200_000, 200_000, 200_000],
        },
    )

    # provider_results only needs the provider key for SimpleQA; metrics are
    # recomputed by save_summary from the per-result CSV.
    provider_results = {"tavily": {}}
    save_summary(provider_results, str(tmp_path), EvaluationType.SIMPLEQA,
                 token_model="gpt-4.1")

    with open(os.path.join(str(tmp_path), "summary.csv")) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "tavily"
    assert row["accuracy"] == str(round(3 / 5, 3))
    assert row["correct_count"] == "3"
    assert row["total_count"] == "5"
    assert row["total_tokens"] == "1000000"
    # 1M tokens * $2/M / 3 correct = $0.666667 per correct answer.
    assert row["cost_of_pass"] == str(round(2.0 / 3.0, 6))


def test_save_summary_cost_of_pass_inf_when_no_correct(tmp_path):
    """A provider with zero correct answers gets an unbounded cost-of-pass."""
    _write_provider_results(
        str(tmp_path),
        "exa",
        rows={
            "index": [0, 1],
            "is_correct": [False, False],
            "token_count": [100, 100],
        },
    )
    save_summary({"exa": {}}, str(tmp_path), EvaluationType.SIMPLEQA,
                 token_model="gpt-4.1")

    with open(os.path.join(str(tmp_path), "summary.csv")) as f:
        row = list(csv.DictReader(f))[0]
    assert row["correct_count"] == "0"
    assert float(row["cost_of_pass"]) == float("inf")
