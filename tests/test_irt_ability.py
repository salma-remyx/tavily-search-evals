"""Integration test for the PromptEval-style IRT ability hook in save_summary.

Exercises the wiring end-to-end through the existing (non-new) module
``utils.utils.save_summary``: it writes the same per-provider SimpleQA result
CSVs the real evaluation pipeline produces, invokes ``save_summary``, and
asserts the IRT artifacts are produced and behave correctly.
"""

import csv
import os

from utils.utils import save_summary, EvaluationType


SIMPLEQA_COLUMNS = [
    "index", "question", "reference_answer", "predicted_answer",
    "is_correct", "grade", "token_count", "token_avg",
]

# Planted correctness with a clear Rasch structure: alpha (high ability) gets
# everything except the two hardest items, gamma (low ability) only the easy
# ones. Index -> {provider: correct?}. There are easy (all-correct), medium
# (discriminating) and hard (mostly-wrong) items.
MATRIX = {
    "alpha":   [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "beta":    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "gamma":   [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
}


def _write_provider_results(output_dir, provider, vector):
    """Write a per-provider SimpleQA results CSV in save_result's format."""
    path = os.path.join(output_dir, f"{provider}_simpleqa_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIMPLEQA_COLUMNS)
        writer.writeheader()
        for idx, correct in enumerate(vector):
            writer.writerow({
                "index": idx,
                "question": f"q{idx}",
                "reference_answer": "a",
                "predicted_answer": "pred",
                "is_correct": bool(correct),
                "grade": "correct" if correct else "incorrect",
                "token_count": 0,
                "token_avg": 0,
            })
    return path


def test_save_summary_produces_irt_artifacts(tmp_path):
    output_dir = str(tmp_path)
    for provider, vector in MATRIX.items():
        _write_provider_results(output_dir, provider, vector)

    provider_results = {provider: {} for provider in MATRIX}
    save_summary(provider_results, output_dir, EvaluationType.SIMPLEQA)

    # summary.csv is the existing behavior; IRT artifacts are the new hook.
    assert os.path.exists(os.path.join(output_dir, "summary.csv"))
    ability_path = os.path.join(output_dir, "irt_ability.csv")
    subset_path = os.path.join(output_dir, "informative_subset.csv")
    assert os.path.exists(ability_path)
    assert os.path.exists(subset_path)

    # Parse the ability estimates.
    with open(ability_path) as f:
        rows = list(csv.DictReader(f))
    abilities = {r["provider"]: float(r["ability"]) for r in rows}
    se = {r["provider"]: float(r["ability_se"]) for r in rows}
    ci = {r["provider"]: (float(r["ci_low"]), float(r["ci_high"])) for r in rows}

    # The Rasch fit must recover the planted ability ordering, and the
    # approximate CI must bracket the point estimate.
    assert abilities["alpha"] > abilities["beta"] > abilities["gamma"]
    for provider in MATRIX:
        assert se[provider] > 0
        low, high = ci[provider]
        assert low <= abilities[provider] <= high

    # Informative subset ranks discriminating items above all-correct items.
    with open(subset_path) as f:
        ranked = list(csv.DictReader(f))
    assert len(ranked) == next(len(v) for v in MATRIX.values())
    assert int(ranked[0]["rank"]) == 1
    # Items 0-2 are correct for every provider (no information); they must
    # rank strictly below the discriminating medium items (e.g. index 3).
    rank_of = {int(r["index"]): int(r["rank"]) for r in ranked}
    assert rank_of[3] < rank_of[0]


def test_save_summary_skips_irt_for_document_relevance(tmp_path):
    """The hook is gated to SimpleQA; document_relevance must not write it."""
    output_dir = str(tmp_path)
    # save_summary reads each provider's result CSV unconditionally (for the
    # example count), so provide one even though the branch ignores it.
    with open(os.path.join(output_dir, "tavily_document_relevance_results.csv"),
              "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "question", "token_count", "token_avg", "grade"])
        writer.writeheader()
        writer.writerow({"index": 0, "question": "q", "token_count": 0,
                         "token_avg": 0, "grade": "completed"})
    provider_results = {"tavily": {"relevant_docs_percentage": 80.0,
                                   "relevant_docs": 8, "total_docs": 10,
                                   "app_name": "x"}}
    save_summary(provider_results, output_dir, EvaluationType.DOCUMENT_RELEVANCE)
    assert not os.path.exists(os.path.join(output_dir, "irt_ability.csv"))
    assert os.path.exists(os.path.join(output_dir, "summary.csv"))


def test_fit_rasch_directly_round_trips():
    """The capability module's core fitter: ability tracks raw accuracy."""
    from utils.irt_ability import fit_rasch

    observations = {provider: {i: v for i, v in enumerate(vec)}
                    for provider, vec in MATRIX.items()}
    fit = fit_rasch(observations)
    assert fit is not None
    abilities = fit["abilities"]
    assert abilities["alpha"] > abilities["beta"] > abilities["gamma"]
