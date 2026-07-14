"""Tests for the Capability Frontier analysis and its save_summary wiring.

These tests go through the public surface:
  * ``compute_capability_frontier`` from the new ``utils.capability_frontier``
    module (pure unit of the frontier math), and
  * ``save_summary`` from the existing ``utils`` package -- the call site that
    was wired to emit the oracle row into ``summary.csv``.

Drive ``save_summary`` through the same on-disk result CSVs the real
pipeline writes (via ``utils.save_result``), so the integration test exercises
the exact read-back + frontier path the benchmark run takes.
"""

import csv
import os

from utils import EvaluationType, save_summary
from utils.capability_frontier import compute_capability_frontier

# Matches the fieldnames written by utils.save_result for SimpleQA.
SIMPLEQA_FIELDS = [
    "index", "question", "reference_answer", "predicted_answer",
    "is_correct", "grade", "token_count", "token_avg",
]


def _write_provider_results(output_dir, provider, rows):
    """Write a per-provider SimpleQA results CSV exactly as save_result would."""
    path = os.path.join(output_dir, f"{provider}_{EvaluationType.SIMPLEQA.value}_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIMPLEQA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SIMPLEQA_FIELDS})
    return path


def test_compute_frontier_quantifies_untapped_potential():
    # Provider A gets Q0,Q1 right; Provider B gets Q1,Q2 right.
    # Best single provider = 2/3. Oracle covers Q0 (A), Q2 (B), Q1 (both) = 3/3.
    per_query = {
        "alpha": [
            {"index": 0, "is_correct": True},
            {"index": 1, "is_correct": True},
            {"index": 2, "is_correct": False},
        ],
        "beta": [
            {"index": 0, "is_correct": False},
            {"index": 1, "is_correct": True},
            {"index": 2, "is_correct": True},
        ],
    }
    frontier = compute_capability_frontier(per_query)

    assert frontier["providers_compared"] == ["alpha", "beta"]
    assert frontier["total_queries"] == 3
    assert frontier["oracle_correct_count"] == 3
    assert frontier["oracle_accuracy"] == 1.0
    assert frontier["best_single_provider"] in ("alpha", "beta")
    assert frontier["best_single_provider_accuracy"] == round(2 / 3, 4)
    assert frontier["frontier_gap"] == round(1.0 - 2 / 3, 4)
    # Best error rate 1/3 -> oracle error rate 0 => 100% reduction.
    assert frontier["error_rate_reduction"] == 1.0


def test_compute_frontier_handles_string_flags_and_disjoint_sets():
    # CSV read-back yields string "True"/"False"; the frontier must coerce.
    per_query = {
        "alpha": [{"index": 0, "is_correct": "True"}],
        "beta": [{"index": 0, "is_correct": "False"}],
    }
    frontier = compute_capability_frontier(per_query)
    assert frontier["oracle_accuracy"] == 1.0
    assert frontier["best_single_provider"] == "alpha"


def test_save_summary_emits_oracle_frontier_row(tmp_path):
    # Drive the real save_summary wiring: alpha nails Q0, beta nails Q1 ->
    # no single provider is perfect but the oracle covers every query.
    _write_provider_results(tmp_path, "alpha", [
        {"index": 0, "is_correct": True, "grade": "correct"},
        {"index": 1, "is_correct": False, "grade": "incorrect"},
    ])
    _write_provider_results(tmp_path, "beta", [
        {"index": 0, "is_correct": False, "grade": "incorrect"},
        {"index": 1, "is_correct": True, "grade": "correct"},
    ])

    save_summary(
        {"alpha": {"accuracy": 0.5}, "beta": {"accuracy": 0.5}},
        str(tmp_path),
        EvaluationType.SIMPLEQA,
    )

    with open(os.path.join(tmp_path, "summary.csv")) as f:
        rows = list(csv.DictReader(f))

    providers = {r["provider"] for r in rows}
    assert {"alpha", "beta", "capability_frontier (oracle)"}.issubset(providers)

    frontier_row = next(r for r in rows if r["provider"] == "capability_frontier (oracle)")
    assert float(frontier_row["accuracy"]) == 1.0
    assert int(frontier_row["correct_count"]) == 2
    assert int(frontier_row["total_count"]) == 2
