"""Tests for the CollabEval sparse evaluation estimator and its wiring.

The first test exercises the integration through the existing public surface
(``utils.utils.save_summary``) to prove the call-site edit actually invokes the
new capability. The remaining tests cover the core estimator directly.
"""

import os

import numpy as np
import pandas as pd

from utils.utils import save_summary, EvaluationType
from utils.sparse_eval_estimator import (
    build_score_matrix,
    collab_eval_estimate,
    low_rank_reconstruct,
    prediction_powered_estimate,
)


def _write_provider_results(output_dir, provider, correctness):
    """Write a SimpleQA results CSV matching the pipeline's on-disk format."""
    records = [
        {
            "index": idx,
            "question": f"q{idx}",
            "reference_answer": "ref",
            "predicted_answer": "pred",
            "is_correct": bool(correct),
            "grade": "correct" if correct else "incorrect",
            "token_count": 0,
            "token_avg": 0,
        }
        for idx, correct in correctness.items()
    ]
    pd.DataFrame(records).to_csv(
        os.path.join(output_dir, f"{provider}_simpleqa_results.csv"), index=False
    )


def _synthetic_providers(seed=42, n_questions=80):
    """Generate a clearly low-rank provider x question correctness matrix.

    Correctness is driven by shared per-question difficulty and a per-provider
    skill level, so the matrix has strong low-rank structure that CollabEval's
    reconstruction can exploit.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.random(n_questions)
    skill = {
        "tavily": 0.9,
        "exa": 0.75,
        "brave": 0.6,
        "serper": 0.8,
        "perplexity_search": 0.85,
    }
    correctness = {}
    for provider, level in skill.items():
        prob = np.clip(level * (1 - 0.6 * difficulty), 0.05, 0.95)
        correctness[provider] = {
            i: bool(rng.random() < prob[i]) for i in range(n_questions)
        }
    return correctness


def test_save_summary_runs_collab_eval(tmp_path):
    """The save_summary call-site edit must produce collabeval_summary.csv.

    This imports from the existing ``utils.utils`` module and drives the wiring
    edit end-to-end (the integration the orchestrator gates on).
    """
    correctness = _synthetic_providers()
    for provider, scores in correctness.items():
        _write_provider_results(str(tmp_path), provider, scores)

    # save_summary is the integration point; the provider_results dict only
    # needs the provider names as keys for the SIMPLEQA summary + CollabEval.
    save_summary(
        {provider: {} for provider in correctness},
        str(tmp_path),
        EvaluationType.SIMPLEQA,
    )

    # The main summary must still be produced (pipeline unaffected) ...
    assert (tmp_path / "summary.csv").exists()
    # ... and the CollabEval analysis must land alongside it.
    collab_path = tmp_path / "collabeval_summary.csv"
    assert collab_path.exists()
    df = pd.read_csv(collab_path)

    expected_cols = {
        "provider", "estimated_accuracy", "ci_lower", "ci_upper", "ci_width",
        "naive_accuracy", "naive_ci_width", "full_accuracy", "n_observed",
    }
    assert expected_cols.issubset(set(df.columns))
    assert set(correctness).issubset(set(df["provider"]))
    # Estimates are valid accuracies.
    assert ((df["estimated_accuracy"] >= 0) & (df["estimated_accuracy"] <= 1)).all()
    # Each provider was evaluated from a sparse subset, not the full matrix.
    assert (df["n_observed"] < len(next(iter(correctness.values())))).all()
    # CollabEval's control-variate CI is never wider than the naive baseline
    # wherever the baseline has positive width.
    baselined = df[df["naive_ci_width"] > 0]
    assert (baselined["ci_width"] <= baselined["naive_ci_width"] + 1e-9).all()
    # And the low-rank reconstruction keeps the estimates close to the truth.
    assert (df["estimated_accuracy"] - df["full_accuracy"]).abs().mean() < 0.2


def test_build_score_matrix_aligns_on_index(tmp_path):
    _write_provider_results(str(tmp_path), "alpha", {0: True, 1: False, 2: True})
    _write_provider_results(str(tmp_path), "beta", {0: False, 2: True})  # missing q1
    providers, matrix = build_score_matrix(str(tmp_path), ["alpha", "beta"])
    assert providers == ["alpha", "beta"]
    assert matrix.shape == (2, 3)
    assert np.isnan(matrix[1, 1])  # beta did not annotate question 1
    assert matrix[0, 0] == 1.0 and matrix[1, 0] == 0.0


def test_low_rank_reconstruct_fills_missing():
    matrix = np.array(
        [[1.0, 1.0, np.nan, 1.0], [0.0, np.nan, 0.0, 0.0]],
    )
    recon = low_rank_reconstruct(matrix, rank=1)
    assert recon.shape == matrix.shape
    # Reconstructed values for the observed cells track the truth reasonably.
    assert recon[0, 0] > recon[1, 0]


def test_prediction_powered_estimate_is_unbiased_and_tighter():
    rng = np.random.default_rng(0)
    n = 200
    truth_prob = rng.uniform(0.2, 0.9, size=n)
    truth = (rng.random(n) < truth_prob).astype(float)
    # A predictor that tracks the true probability (a good control variate).
    pred = truth_prob.copy()
    observed = np.ones(n, dtype=bool)
    est = prediction_powered_estimate(truth, pred, observed)
    # Unbiased: the PPI point estimate recovers the true mean.
    assert abs(est["estimate"] - truth.mean()) < 0.05
    # Tighter: residual variance is smaller than raw-score variance.
    assert est["ci_width"] < est["naive_ci_width"]
    assert est["ci_lower"] < est["estimate"] < est["ci_upper"]
    assert est["n_observed"] == n


def test_collab_eval_estimate_is_accurate_and_tighter():
    correctness = _synthetic_providers(seed=1)
    providers = list(correctness)
    n_questions = len(next(iter(correctness.values())))
    matrix = np.array(
        [[1.0 if correctness[p][q] else 0.0 for q in range(n_questions)] for p in providers]
    )
    estimates = collab_eval_estimate(
        matrix, density=0.3, rank=4, rng=np.random.default_rng(0)
    )
    estimates = [e for e in estimates if e["naive_ci_width"] > 0]
    # The low-rank reconstruction keeps the sparse estimates close to the truth.
    abs_err = np.mean([abs(e["estimate"] - e["true_mean"]) for e in estimates])
    assert abs_err < 0.15
    # The PPI control variate yields a tighter interval than the naive baseline
    # on average (CollabEval's efficiency claim at a fixed annotation budget).
    mean_ci = np.mean([e["ci_width"] for e in estimates])
    mean_naive = np.mean([e["naive_ci_width"] for e in estimates])
    assert mean_ci < mean_naive
