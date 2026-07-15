"""Integration tests for the binary-question (BINEVAL) result persistence.

These import from the EXISTING (non-new) ``utils.utils`` persistence layer to
prove the BINEVAL fields produced alongside the correctness grade in
``run_evaluation.evaluate_provider_simple_qa`` flow through ``save_result`` /
``save_summary`` and land in the SimpleQA per-example and summary CSVs.
"""

import json
import os

import pandas as pd

from utils.utils import EvaluationType, save_result, save_summary


def _result(**overrides):
    base = dict(
        index=0,
        question="What is the capital of France?",
        reference_answer="Paris",
        predicted_answer="Paris",
        is_correct=True,
        grade="CORRECT",
        token_count=10,
        token_avg=10,
    )
    base.update(overrides)
    return base


def test_save_result_persists_bineval_columns(tmp_path):
    output_dir = str(tmp_path)
    save_result(
        _result(
            bineval_score=0.8,
            bineval_dimensions=json.dumps(
                {"factual_consistency": 0.5, "relevance": 1.0}
            ),
            bineval_verdicts=json.dumps(
                {"contains_key_info": True, "contradicts_reference": False}
            ),
        ),
        "tavily",
        output_dir,
        EvaluationType.SIMPLEQA,
    )

    df = pd.read_csv(os.path.join(output_dir, "tavily_simpleqa_results.csv"))
    assert {
        "bineval_score",
        "bineval_dimensions",
        "bineval_verdicts",
    }.issubset(df.columns)
    assert float(df.loc[0, "bineval_score"]) == 0.8
    assert json.loads(df.loc[0, "bineval_dimensions"])["relevance"] == 1.0
    assert json.loads(df.loc[0, "bineval_verdicts"])["contains_key_info"] is True


def test_save_result_writes_blank_when_bineval_unavailable(tmp_path):
    # When the binary-question judge could not run (score is None), the result
    # must still persist, leaving the optional columns blank rather than erroring.
    output_dir = str(tmp_path)
    save_result(_result(bineval_score=None), "exa", output_dir, EvaluationType.SIMPLEQA)

    df = pd.read_csv(os.path.join(output_dir, "exa_simpleqa_results.csv"))
    assert "bineval_score" in df.columns
    assert pd.isna(df.loc[0, "bineval_score"])


def test_save_summary_aggregates_mean_bineval_score(tmp_path):
    provider = "brave"
    output_dir = str(tmp_path)
    rows = [
        _result(
            index=0,
            is_correct=True,
            grade="CORRECT",
            bineval_score=0.8,
            bineval_dimensions="{}",
            bineval_verdicts="{}",
        ),
        _result(
            index=1,
            is_correct=False,
            grade="INCORRECT",
            bineval_score=0.6,
            bineval_dimensions="{}",
            bineval_verdicts="{}",
        ),
    ]
    for row in rows:
        save_result(row, provider, output_dir, EvaluationType.SIMPLEQA)

    save_summary({provider: {}}, output_dir, EvaluationType.SIMPLEQA)

    summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
    assert "bineval_score" in summary.columns
    assert float(summary.loc[0, "bineval_score"]) == 0.7
