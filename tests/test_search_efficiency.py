"""Tests for the search-efficiency (Pareto-AUC) metric and its wiring.

These tests import the existing evaluation utilities -- ``utils.utils.save_result``
and ``utils.token_utils.get_token_stats`` -- and exercise the efficiency report
against the real on-disk result schema those utilities write. This proves the
metric integrates with the pipeline's actual output rather than operating on
synthetic shapes.
"""

import csv
import os

from utils.token_utils import get_token_stats
from utils.utils import EvaluationType, save_result
from utils.search_efficiency import (
    ProviderPoint,
    compute_efficiency_report,
    load_provider_points,
    pareto_auc,
    pareto_frontier,
)


def _write_provider_rows(output_dir, provider, rows):
    """Drive the pipeline's own save_result to write real-schema CSV rows."""
    for row in rows:
        save_result(row, provider, output_dir, EvaluationType.SIMPLEQA)


def test_pareto_frontier_excludes_dominated_providers():
    points = [
        ProviderPoint("cheap", quality=0.9, cost=100),
        ProviderPoint("pricey", quality=0.9, cost=200),  # dominated by "cheap"
        ProviderPoint("best", quality=0.99, cost=400),
    ]
    front = pareto_frontier(points)
    names = [p.name for p in front]
    assert "cheap" in names
    assert "best" in names
    assert "pricey" not in names
    assert front == sorted(front, key=lambda p: p.cost)


def test_auc_rewards_cheaper_quality():
    # Same quality, one provider cheaper -> a more efficient set than the costly one alone.
    both = [ProviderPoint("cheap", 0.9, 100), ProviderPoint("pricey", 0.9, 200)]
    costly_only = [ProviderPoint("pricey", 0.9, 200)]
    assert pareto_auc(both) > pareto_auc(costly_only)
    assert 0.0 <= pareto_auc(both) <= 1.0


def test_cost_axis_uses_token_utils():
    # The cost axis is mean per-query tokens, aggregated by the existing token_utils.
    total, avg = get_token_stats([100, 200])
    assert (total, avg) == (300, 150.0)


def test_compute_efficiency_report_reads_real_pipeline_output(tmp_path):
    out = tmp_path / "simpleqa_run"
    out.mkdir()
    _write_provider_rows(
        str(out),
        "tavily",
        [
            {"index": 0, "question": "q0", "reference_answer": "a", "predicted_answer": "a",
             "is_correct": True, "grade": "correct", "token_count": 100, "token_avg": 100},
            {"index": 1, "question": "q1", "reference_answer": "b", "predicted_answer": "b",
             "is_correct": True, "grade": "correct", "token_count": 200, "token_avg": 150},
        ],
    )
    _write_provider_rows(
        str(out),
        "exa",
        [
            {"index": 0, "question": "q0", "reference_answer": "a", "predicted_answer": "a",
             "is_correct": True, "grade": "correct", "token_count": 500, "token_avg": 500},
            {"index": 1, "question": "q1", "reference_answer": "b", "predicted_answer": "x",
             "is_correct": False, "grade": "incorrect", "token_count": 500, "token_avg": 500},
        ],
    )

    points = load_provider_points(str(out), EvaluationType.SIMPLEQA)
    by_name = {p.name: p for p in points}
    assert set(by_name) == {"tavily", "exa"}
    assert by_name["tavily"].quality == 1.0          # 2/2 correct
    assert by_name["tavily"].cost == 150.0           # mean(100, 200)
    assert by_name["exa"].quality == 0.5             # 1/2 correct
    assert by_name["exa"].cost == 500.0              # mean(500, 500)

    report = compute_efficiency_report(str(out), EvaluationType.SIMPLEQA)
    assert "tavily" in report["frontier"]
    assert "exa" not in report["frontier"]           # dominated: lower quality AND higher cost
    assert os.path.exists(report["report_path"])

    rows = {r["provider"]: r for r in report["providers"]}
    assert rows["tavily"]["efficiency_rank"] == 1
    assert rows["tavily"]["on_pareto_frontier"] is True
    assert rows["exa"]["on_pareto_frontier"] is False

    # The written report carries every provider with the expected schema.
    with open(report["report_path"]) as csvfile:
        written = list(csv.DictReader(csvfile))
    assert {r["provider"] for r in written} == {"tavily", "exa"}
    assert set(written[0]) == {
        "provider", "quality", "cost", "on_pareto_frontier", "efficiency_rank",
    }
