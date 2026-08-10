"""IRT-based ability estimation from a binary correctness matrix.

Adapted from PromptEval (Bastounis et al., 2024), "Efficient multi-prompt
evaluation of LLMs" (arXiv:2405.17202). PromptEval models a
models x prompts binary correctness matrix with Item Response Theory (the
1PL Rasch model) and uses Bayesian inference (PyMC) to obtain posterior
ability distributions, then runs an active-learning loop to pick the most
informative prompts to evaluate next.

This module keeps PromptEval's *core mechanism* — the Rasch IRT model of the
correctness matrix, with one ability parameter per provider and one
difficulty parameter per prompt — at full fidelity. Two *auxiliary*
components are substituted with dependency-free equivalents suited to this
repo:

  * Bayesian MCMC (PyMC) for posterior ability distributions and credible
    intervals  ->  Joint Maximum Likelihood (JML) with Fisher-information
    standard errors as approximate confidence intervals.
  * The greedy active-learning acquisition loop  ->  the classical IRT
    Fisher-information ranking, which selects the prompts whose difficulty
    sits nearest the abilities being compared (the maximally discriminating
    items), serving the same efficiency goal as PromptEval's selection.

These substitutions preserve PromptEval's two deliverables —
uncertainty-aware ability comparison and an efficiency-oriented informative
subset — without adding a heavy inference dependency.

The module is intentionally pure-stdlib (``csv`` + ``math``) so it has no
third-party requirements of its own; it reads the same per-provider result
CSVs that ``save_summary`` already writes.
"""

import csv
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def read_provider_correctness(csv_path: str) -> Dict[int, int]:
    """Read a SimpleQA provider result CSV into ``{item_index: 0|1}``.

    Rows whose ``grade`` is ``ERROR`` or whose ``is_correct`` is missing are
    skipped: an error row is not a real correctness judgement and including
    it would bias the item difficulty.
    """
    observations: Dict[int, int] = {}
    if not os.path.exists(csv_path):
        return observations
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grade = (row.get("grade") or "").strip()
            if grade == "ERROR":
                continue
            index_raw = (row.get("index") or "").strip()
            value_raw = (row.get("is_correct") or "").strip().lower()
            if not index_raw or value_raw not in ("true", "false", "1", "0"):
                continue
            try:
                index = int(float(index_raw))
            except ValueError:
                continue
            observations[index] = 1 if value_raw in ("true", "1") else 0
    return observations


def fit_rasch(
    observations: Dict[str, Dict[int, int]],
    n_iter: int = 3000,
    lr: float = 0.1,
    ridge: float = 1.0,
    tol: float = 1e-5,
) -> Optional[Dict]:
    """Fit a 1PL Rasch model by Joint Maximum Likelihood.

    Args:
        observations: ``{provider: {item_index: 0|1}}``. Missing
            (provider, item) pairs are skipped, so unequal coverage and
            ERROR rows are handled gracefully.
        n_iter: maximum gradient-ascent steps.
        lr: gradient-ascent learning rate.
        ridge: L2 penalty on abilities and difficulties; guards against
            parameter separation (an item every provider gets right/wrong)
            which would otherwise drive difficulty to +/-infinity.
        tol: stop early when the log-likelihood gain drops below this.

    Returns:
        ``None`` if the matrix is too small to fit. Otherwise a dict::

            {
                "providers": [provider, ...],
                "items": [item_index, ...],
                "abilities": {provider: float},          # mean-anchored to 0
                "difficulties": {item_index: float},
                "ability_se": {provider: float},         # Fisher-info std error
                "difficulty_se": {item_index: float},
                "ability_ci": {provider: (low, high)},   # ~95% interval
                "n_providers": int,
                "n_items": int,
                "n_observations": int,
            }
    """
    providers = list(observations.keys())
    items = sorted({idx for obs in observations.values() for idx in obs})
    if len(providers) < 1 or len(items) < 3:
        logger.info(
            "Skipping Rasch fit: need >=1 provider and >=3 items "
            "(got %d providers, %d items).", len(providers), len(items),
        )
        return None

    theta = {p: 0.0 for p in providers}
    beta = {i: 0.0 for i in items}

    # Flatten observations into a list for the inner loop.
    pairs: List[Tuple[str, int, int]] = [
        (p, i, val)
        for p in providers
        for i, val in observations[p].items()
        if i in beta
    ]

    prev_ll: Optional[float] = None
    for _ in range(n_iter):
        grad_theta = {p: 0.0 for p in providers}
        grad_beta = {i: 0.0 for i in items}
        ll = 0.0
        info_theta = {p: 0.0 for p in providers}
        info_beta = {i: 0.0 for i in items}
        for p, i, val in pairs:
            p_correct = _sigmoid(theta[p] - beta[i])
            err = val - p_correct
            grad_theta[p] += err
            grad_beta[i] -= err
            info_theta[p] += p_correct * (1.0 - p_correct)
            info_beta[i] += p_correct * (1.0 - p_correct)
            # log-likelihood of a Bernoulli, clipped to avoid log(0).
            ll += val * math.log(p_correct + 1e-12) + \
                (1 - val) * math.log(1.0 - p_correct + 1e-12)

        # L2 (ridge) toward zero to regularize separation.
        for p in providers:
            grad_theta[p] -= ridge * theta[p]
        for i in items:
            grad_beta[i] -= ridge * beta[i]

        for p in providers:
            theta[p] += lr * grad_theta[p]
        for i in items:
            beta[i] += lr * grad_beta[i]

        # Anchor the (otherwise unidentifiable) origin: mean ability = 0.
        mean_theta = sum(theta.values()) / len(theta)
        for p in providers:
            theta[p] -= mean_theta

        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Standard errors from Fisher information (+ ridge guard).
    ability_se = {
        p: 1.0 / math.sqrt(info_theta[p] + ridge) for p in providers
    }
    difficulty_se = {
        i: 1.0 / math.sqrt(info_beta[i] + ridge) for i in items
    }
    z95 = 1.96
    ability_ci = {
        p: (theta[p] - z95 * ability_se[p], theta[p] + z95 * ability_se[p])
        for p in providers
    }

    return {
        "providers": providers,
        "items": items,
        "abilities": theta,
        "difficulties": beta,
        "ability_se": ability_se,
        "difficulty_se": difficulty_se,
        "ability_ci": ability_ci,
        "n_providers": len(providers),
        "n_items": len(items),
        "n_observations": len(pairs),
    }


def rank_informative_items(fit: Dict, providers: Optional[List[str]] = None) -> List[Dict]:
    """Rank items by total Fisher information across the given providers.

    For the Rasch model an item's Fisher information about an ability is
    ``p(1-p)``, maximized when the item's difficulty matches that ability.
    Summing across providers selects the items that discriminate across the
    *ability range being compared* — the same efficiency goal as PromptEval's
    active-learning acquisition, obtained without its iterative retraining
    loop. Easy items (everyone correct) and hard items (everyone wrong)
    carry little information and sink to the bottom of the ranking.

    Returns a list of ``{index, fisher_info, difficulty}`` sorted by
    information descending.
    """
    providers = providers or fit["providers"]
    theta = fit["abilities"]
    beta = fit["difficulties"]
    ranked: List[Dict] = []
    for i in fit["items"]:
        info = 0.0
        for p in providers:
            p_correct = _sigmoid(theta.get(p, 0.0) - beta[i])
            info += p_correct * (1.0 - p_correct)
        ranked.append({"index": i, "fisher_info": info, "difficulty": beta[i]})
    ranked.sort(key=lambda r: r["fisher_info"], reverse=True)
    return ranked


def estimate_abilities(
    provider_names: List[str],
    output_dir: str,
    result_suffix: str = "simpleqa_results.csv",
    subset_fraction: float = 0.3,
) -> Optional[Dict]:
    """End-to-end hook: read per-provider CSVs, fit Rasch, write artifacts.

    Writes ``irt_ability.csv`` (per-provider ability + SE + CI alongside raw
    accuracy) and ``informative_subset.csv`` (prompts ranked by Fisher
    information, with the top ``subset_fraction`` flagged as the recommended
    priority for a cheaper re-run) into ``output_dir``.

    Returns the fit dict, or ``None`` if there was too little data.
    """
    observations: Dict[str, Dict[int, int]] = {}
    raw_accuracy: Dict[str, Tuple[int, int]] = {}
    for provider in provider_names:
        csv_path = os.path.join(output_dir, f"{provider}_{result_suffix}")
        obs = read_provider_correctness(csv_path)
        if not obs:
            continue
        observations[provider] = obs
        raw_accuracy[provider] = (sum(obs.values()), len(obs))

    fit = fit_rasch(observations)
    if fit is None:
        return None

    # irt_ability.csv: ability + uncertainty next to raw accuracy.
    ability_file = os.path.join(output_dir, "irt_ability.csv")
    with open(ability_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "provider", "ability", "ability_se", "ci_low", "ci_high",
            "raw_accuracy", "correct_count", "observed_count",
        ])
        for provider in fit["providers"]:
            correct, total = raw_accuracy.get(provider, (0, 0))
            acc = round(correct / total, 4) if total else 0.0
            low, high = fit["ability_ci"][provider]
            writer.writerow([
                provider,
                round(fit["abilities"][provider], 4),
                round(fit["ability_se"][provider], 4),
                round(low, 4),
                round(high, 4),
                acc,
                correct,
                total,
            ])

    # informative_subset.csv: prompts ranked by Fisher information.
    ranked = rank_informative_items(fit)
    n_recommend = max(1, round(len(ranked) * subset_fraction)) if ranked else 0
    subset_file = os.path.join(output_dir, "informative_subset.csv")
    with open(subset_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "index", "fisher_info", "difficulty", "recommended"])
        for rank, item in enumerate(ranked, start=1):
            writer.writerow([
                rank,
                item["index"],
                round(item["fisher_info"], 4),
                round(item["difficulty"], 4),
                "true" if rank <= n_recommend else "false",
            ])

    logger.info(
        "Estimated IRT abilities for %d providers across %d prompts "
        "(%d judgements). Wrote %s and %s.",
        fit["n_providers"], fit["n_items"], fit["n_observations"],
        ability_file, subset_file,
    )
    return fit
