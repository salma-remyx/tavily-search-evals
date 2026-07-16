"""Tests for the continuous LLM-as-a-Verifier scorer and its wiring.

The core mechanism (expectation over scoring-token logits) is exercised via
the pure helpers in ``utils.verifier_score``. The integration test imports the
pre-existing ``utils.utils`` module and confirms the continuous score flows
through the SimpleQA results CSV, exercising the ``verifier_score`` fieldname
wired into ``save_result``.
"""

import math

import pandas as pd

from utils.utils import EvaluationType, save_result  # NON-NEW module in utils/
from utils.verifier_score import expected_score, extract_grade_logprobs


def test_expected_score_matches_discrete_grade_at_extremes():
    # Peaked-correct -> ~1.0 with argmax A, consistent with discrete is_correct=True.
    score, grade, _ = expected_score(
        {"A": math.log(0.98), "B": math.log(0.01), "C": math.log(0.01)}
    )
    assert grade == "A"
    assert abs(score - 0.98) < 1e-6

    # Peaked-incorrect -> ~0.0 with argmax B, consistent with discrete is_correct=False.
    score, grade, _ = expected_score(
        {"A": math.log(0.01), "B": math.log(0.97), "C": math.log(0.02)}
    )
    assert grade == "B"
    assert score < 0.05


def test_expected_score_is_continuous_between_extremes():
    # An uncertain judge lands strictly between 0 and 1 -- the signal argmax drops.
    score, grade, probs = expected_score(
        {"A": math.log(0.6), "B": math.log(0.3), "C": math.log(0.1)}
    )
    assert grade == "A"
    assert 0.55 < score < 0.65
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_expected_score_empty_falls_back_to_not_attempted():
    score, grade, probs = expected_score({})
    assert score == 0.0
    assert grade == "C"
    assert probs == {}


def test_extract_grade_logprobs_normalizes_leading_space_and_reads_top_logprobs():
    # Mirrors the langchain response_metadata["logprobs"]["content"] shape.
    blob = {
        "content": [
            {
                "token": " A",
                "logprob": -0.02,
                "top_logprobs": [
                    {"token": " A", "logprob": -0.02},
                    {"token": "B", "logprob": -3.9},
                    {"token": "C", "logprob": -5.6},
                ],
            }
        ]
    }
    grade_logprobs = extract_grade_logprobs(blob)
    assert set(grade_logprobs) == {"A", "B", "C"}
    assert grade_logprobs["A"] == -0.02  # leading-space " A" collapses to "A"


def test_verifier_score_persists_through_save_result(tmp_path):
    # The continuous score produced by the verifier (via its pure expected_score
    # mechanism) must round-trip through the existing SimpleQA results CSV,
    # exercising the verifier_score fieldname wired into utils.utils.save_result.
    score, _, _ = expected_score(
        {"A": math.log(0.9), "B": math.log(0.07), "C": math.log(0.03)}
    )
    result = {
        "index": 0,
        "question": "q",
        "reference_answer": "gold",
        "predicted_answer": "pred",
        "is_correct": True,
        "grade": "CORRECT",
        "verifier_score": score,
        "token_count": 0,
        "token_avg": 0,
    }
    save_result(result, "tavily", str(tmp_path), EvaluationType.SIMPLEQA)

    df = pd.read_csv(tmp_path / "tavily_simpleqa_results.csv")
    assert "verifier_score" in df.columns
    assert abs(df.loc[0, "verifier_score"] - score) < 1e-6
