"""
Integration tests for utils.rag_metric_agreement.

These drive the real ``save_result`` (from the existing utils package) to write
SimpleQA result rows in the on-disk CSV contract, read them back, and run the
metric-vs-human agreement analysis over them -- proving the new capability
wires into the existing evaluation contract rather than only self-testing.
"""

import pandas as pd

from utils import EvaluationType, save_result
from utils.rag_metric_agreement import (
    claim_recall,
    evaluate_metric_agreement,
    pearson_r,
    score_results,
)

_PROVIDER = "tavily"


def _write_rows(tmp_path, rows):
    """Write rows through the real save_result saver and read the CSV back."""
    for row in rows:
        save_result(row, _PROVIDER, str(tmp_path), EvaluationType.SIMPLEQA)
    df = pd.read_csv(tmp_path / f"{_PROVIDER}_simpleqa_results.csv")
    return df.to_dict("records")


def _row(index, ref, pred, correct):
    return {
        "index": index,
        "question": f"question {index}",
        "reference_answer": ref,
        "predicted_answer": pred,
        "is_correct": correct,
        "grade": "CORRECT" if correct else "INCORRECT",
        "token_count": 10,
        "token_avg": 10,
    }


def test_recall_tracks_human_judgment(tmp_path):
    rows = [
        _row(0, "Paris is the capital of France.", "Paris is the capital of France.", True),
        _row(1, "Jupiter is the largest planet.", "Saturn is the largest planet.", False),
        _row(2, "The speed of light is fast.", "Light travels very fast indeed.", False),
        _row(3, "William Shakespeare wrote Hamlet.", "William Shakespeare wrote Hamlet.", True),
    ]
    records = _write_rows(tmp_path, rows)
    human_labels = [1.0, 0.0, 0.0, 1.0]  # aligned with the rows above

    report = evaluate_metric_agreement(records, human_labels)

    assert report["recall"]["n"] == 4.0
    # Recall should move with the human correctness labels (positive correlation).
    assert report["recall"]["pearson"] > 0.5
    assert report["recall"]["spearman"] > 0.5


def test_correct_answer_recovers_more_reference_than_wrong_answer():
    correct = claim_recall(
        "Paris is the capital of France.", "Paris is the capital of France."
    )
    wrong = claim_recall(
        "Paris is the capital of France.", "Saturn is the largest planet."
    )
    assert correct > wrong
    assert correct == 1.0


def test_constant_metric_has_zero_correlation():
    # No variance on the metric side -> correlation is defined as 0.0.
    assert pearson_r([0.5, 0.5, 0.5], [0.0, 1.0, 0.0]) == 0.0


def test_faithfulness_needs_context():
    row = _row(0, "Paris", "Paris", True)
    with_context = score_results([row], context_map={0: "Paris France"})
    without_context = score_results([row])
    assert with_context[0]["faithfulness"] is not None
    assert with_context[0]["faithfulness"] > 0.0
    # Without retrieved context, faithfulness is unavailable (not fabricated).
    assert without_context[0]["faithfulness"] is None
