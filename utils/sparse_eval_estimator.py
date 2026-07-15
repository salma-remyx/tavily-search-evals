"""Sparse evaluation estimation via low-rank matrix completion.

Adapted from CollabEval (Collaborative Model Evaluation via Matrix Completion,
arxiv:2607.05046v1). The core mechanism is ported at full fidelity: a low-rank
reconstruction of the provider x question correctness matrix is used as a
control variate inside a prediction-powered-inference (PPI) estimator, which
yields an *unbiased* estimate of each provider's mean accuracy together with a
statistically valid confidence interval that is tighter than the naive
sample-mean baseline at the same annotation budget.

Target-native substitutions (vs. the paper):
  - The paper's cross-model benchmark harness is replaced by this repo's own
    SimpleQA provider x question correctness matrix (read from the per-provider
    results CSVs that ``save_summary`` already aggregates).
  - Binary 0/1 correctness scores stand in for the paper's continuous judge
    scores; the PPI estimator is identical, only the score type differs.

The module also exposes a simulation harness (``collab_eval_estimate``) that
masks a dense correctness matrix down to a fraction ``p`` of annotated cells,
reconstructs it at low rank, and reports the PPI estimate, its CI, and the
naive baseline against the full-matrix ground truth -- the efficiency-gain
experiment suggested by the paper for an existing benchmark.
"""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Normal quantiles for the common confidence levels (avoids a scipy dependency).
_Z_TABLE = {0.90: 1.64485, 0.95: 1.95996, 0.99: 2.57583}


def _z_score(confidence: float) -> float:
    return _Z_TABLE.get(confidence, 1.95996)


def build_score_matrix(
    output_dir: str,
    provider_names: List[str],
    suffix: str = "simpleqa_results.csv",
) -> Tuple[List[str], np.ndarray]:
    """Build the M (providers) x N (questions) correctness matrix.

    Reads each provider's per-question results CSV and aligns them on the
    ``index`` column, producing a float matrix of 0/1 correctness scores with
    ``NaN`` marking (provider, question) cells that were never annotated.

    Returns:
        (providers, matrix) where ``providers`` is the list of providers that
        yielded usable data (in input order) and ``matrix`` has one row each.
    """
    column_index: Dict[int, int] = {}  # question index -> column position
    rows: Dict[str, Dict[int, float]] = {}
    for provider in provider_names:
        path = os.path.join(output_dir, f"{provider}_{suffix}")
        if not os.path.exists(path):
            logger.warning("[collab_eval] missing %s; skipping provider", path)
            continue
        df = pd.read_csv(path)
        if "index" not in df.columns or "is_correct" not in df.columns:
            continue
        row: Dict[int, float] = {}
        for _, record in df.iterrows():
            question = int(record["index"])
            column_index.setdefault(question, len(column_index))
            row[question] = 1.0 if bool(record["is_correct"]) else 0.0
        if row:
            rows[provider] = row

    providers = list(rows.keys())
    n_cols = len(column_index)
    if not providers or n_cols == 0:
        raise ValueError("No usable provider correctness data found for CollabEval")

    matrix = np.full((len(providers), n_cols), np.nan)
    for i, provider in enumerate(providers):
        for question, value in rows[provider].items():
            matrix[i, column_index[question]] = value
    return providers, matrix


def low_rank_reconstruct(
    matrix: np.ndarray,
    rank: int = 5,
    n_iter: int = 25,
    tol: float = 1e-4,
) -> np.ndarray:
    """Low-rank reconstruction of a partially-observed score matrix.

    Iterative SVD imputation (soft-impute style): missing entries are filled
    with the current column means, the matrix is truncated to ``rank`` via the
    SVD, and the procedure repeats until the reconstruction stabilizes. The
    returned surface is the low-rank prediction used as the PPI control
    variate -- it provides a value for *every* (provider, question) cell,
    including unannotated ones.
    """
    observed = ~np.isnan(matrix)
    # Seed unobserved cells with per-column means (global mean fallback).
    finite = np.isfinite(matrix)
    col_mean = np.full(matrix.shape[1], float(np.nanmean(matrix)) if finite.any() else 0.0)
    per_col = np.array([np.nanmean(matrix[:, j]) if np.any(observed[:, j]) else col_mean[j]
                        for j in range(matrix.shape[1])])
    filled = np.where(observed, matrix, per_col)
    previous = filled.copy()
    for _ in range(n_iter):
        u, sigma, vt = np.linalg.svd(filled, full_matrices=False)
        k = max(1, min(rank, len(sigma)))
        approx = (u[:, :k] * sigma[:k]) @ vt[:k, :]
        filled = np.where(observed, matrix, approx)
        scale = np.linalg.norm(previous) + 1e-9
        if np.linalg.norm(filled - previous) / scale < tol:
            break
        previous = filled.copy()

    u, sigma, vt = np.linalg.svd(filled, full_matrices=False)
    k = max(1, min(rank, len(sigma)))
    return (u[:, :k] * sigma[:k]) @ vt[:k, :]


def prediction_powered_estimate(
    true_row: np.ndarray,
    pred_row: np.ndarray,
    observed: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Prediction-powered (PPI) mean estimator using a control variate.

    Given a provider's true scores on the observed subset and the low-rank
    prediction across all questions, estimate the full-matrix mean accuracy::

        theta_pp = mean(true_obs - pred_obs) + mean(pred_all)

    This is unbiased for the true mean and has lower variance than the naive
    sample mean whenever the low-rank prediction correlates with the truth.
    Returns the point estimate, its confidence interval, the naive baseline,
    and the number of observed cells.
    """
    truth = true_row[observed].astype(float)
    pred_obs = pred_row[observed].astype(float)
    n = int(observed.sum())
    if n == 0:
        raise ValueError("No observed entries for PPI estimate")

    residual = truth - pred_obs
    estimate = float(np.mean(residual) + np.mean(pred_row))
    naive_estimate = float(np.mean(truth))

    var_pp = float(np.var(residual, ddof=1)) / n if n > 1 else 0.0
    var_naive = float(np.var(truth, ddof=1)) / n if n > 1 else 0.0
    z = _z_score(confidence)
    return {
        "estimate": estimate,
        "ci_lower": estimate - z * np.sqrt(var_pp),
        "ci_upper": estimate + z * np.sqrt(var_pp),
        "ci_width": 2 * z * np.sqrt(var_pp),
        "naive_estimate": naive_estimate,
        "naive_ci_width": 2 * z * np.sqrt(var_naive),
        "n_observed": n,
    }


def collab_eval_estimate(
    matrix: np.ndarray,
    density: float = 0.2,
    rank: int = 5,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Run CollabEval estimation on a correctness matrix.

    Treats the supplied (dense) matrix as ground truth, simulates a sparse
    evaluation in which each provider has only ``density`` of its questions
    annotated, reconstructs the full matrix at low rank, and produces a
    prediction-powered estimate of each provider's mean accuracy with a
    confidence interval. Each row also carries the naive sample-mean baseline
    and the full-matrix truth for comparison.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    n_providers, n_questions = matrix.shape
    truth_mean = np.nanmean(matrix, axis=1)
    observed_full = ~np.isnan(matrix)

    masked = np.full_like(matrix, np.nan, dtype=float)
    for i in range(n_providers):
        idx = np.where(observed_full[i])[0]
        if idx.size == 0:
            continue
        keep = rng.choice(idx, size=max(1, int(round(density * idx.size))), replace=False)
        masked[i, keep] = matrix[i, keep]

    prediction = low_rank_reconstruct(masked, rank=rank)

    rows: List[Dict] = []
    for i in range(n_providers):
        observed_i = ~np.isnan(masked[i])
        if not observed_i.any():
            continue
        estimate = prediction_powered_estimate(masked[i], prediction[i], observed_i, confidence)
        estimate["true_mean"] = float(truth_mean[i])
        estimate["density"] = density
        rows.append(estimate)
    return rows


def run_collab_eval(
    output_dir: str,
    provider_names: List[str],
    density: float = 0.2,
    rank: int = 5,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Build the matrix from provider CSVs, run CollabEval, persist results.

    Writes ``collabeval_summary.csv`` next to ``summary.csv`` with per-provider
    PPI estimates, their CIs, the naive baseline, and the full-matrix accuracy.
    Returns the rows plus an aggregate summary (PPI MSE vs. truth and the mean
    CI-width reduction relative to the naive baseline).
    """
    providers, matrix = build_score_matrix(output_dir, provider_names)
    estimates = collab_eval_estimate(
        matrix,
        density=density,
        rank=rank,
        confidence=confidence,
        rng=np.random.default_rng(seed),
    )

    rows: List[Dict] = []
    squared_error = 0.0
    ci_reduction = 0.0
    for provider, estimate in zip(providers, estimates):
        rows.append({
            "provider": provider,
            "estimated_accuracy": round(estimate["estimate"], 4),
            "ci_lower": round(estimate["ci_lower"], 4),
            "ci_upper": round(estimate["ci_upper"], 4),
            "ci_width": round(estimate["ci_width"], 4),
            "naive_accuracy": round(estimate["naive_estimate"], 4),
            "naive_ci_width": round(estimate["naive_ci_width"], 4),
            "full_accuracy": round(estimate["true_mean"], 4),
            "n_observed": estimate["n_observed"],
        })
        squared_error += (estimate["estimate"] - estimate["true_mean"]) ** 2
        if estimate["naive_ci_width"] > 0:
            ci_reduction += 1 - estimate["ci_width"] / estimate["naive_ci_width"]

    summary_path = os.path.join(output_dir, "collabeval_summary.csv")
    fieldnames = [
        "provider", "estimated_accuracy", "ci_lower", "ci_upper", "ci_width",
        "naive_accuracy", "naive_ci_width", "full_accuracy", "n_observed",
    ]
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    summary = {
        "mse_pp": round(squared_error / count, 6) if count else 0.0,
        "mean_ci_reduction": round(ci_reduction / count, 4) if count else 0.0,
        "density": density,
        "n_providers": count,
        "n_questions": int(matrix.shape[1]),
        "summary_path": summary_path,
    }
    logger.info(
        "[collab_eval] density=%.2f providers=%d questions=%d | PPI MSE=%.6f "
        "mean CI reduction vs naive=%.1f%% -> %s",
        density, count, matrix.shape[1], summary["mse_pp"],
        100.0 * summary["mean_ci_reduction"], summary_path,
    )
    return {"providers": rows, "summary": summary}
