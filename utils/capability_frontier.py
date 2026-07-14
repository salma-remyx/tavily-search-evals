"""Capability Frontier analysis for multi-provider search evaluation.

Inspired by "The Capability Frontier: Benchmarks Miss 82% of Model Performance"
(arXiv:2606.26836). That paper shows that reporting one model's accuracy on
one run systematically understates the *collective* capability of a model set:
an oracle that retains the best answer across models for each query reaches a
substantially higher ceiling -- the "Capability Frontier" -- than any single
model, and the gap widens with the diversity (topic entropy) of the workload.

This module ports that core insight to this repo's search-provider evaluation.
The framework already runs every provider over the same benchmark (SimpleQA)
and records a per-query correctness label (`is_correct`). Treating each search
provider as the analog of one of the paper's models, we compute an oracle
ceiling: for each query, the frontier is correct if *any* provider answered it
correctly. Comparing that ceiling to the best single provider quantifies the
untapped collective potential of the provider suite.

Adapted port (Mode 3 -- inspired experiment):
  * The paper's "models + sampled generations" are mapped onto the repo's
    "search providers" (one generation each). The optimal-selection oracle is
    unchanged; only the unit of selection differs.

Intentionally out of scope (auxiliary components the paper's data does not
exist for in this repo):
  * Cost-Pareto frontier -- the repo records no per-provider cost signal, so
    there is no cost axis to form a Pareto frontier over.
  * "Max over noisy samples" bias correction -- one deterministic run per
    provider; the per-query grade is a single label, not a noisy sample to
    take a maximum over.
  * Topic-entropy probabilistic simulation -- an illustrative aside in the
    paper, not a metric this evaluation tracks.
"""

from typing import Dict, List, Mapping


def _is_correct(value) -> bool:
    """Coerce a stored correctness flag to a strict bool.

    Results are read back from CSV, so the flag may arrive as a Python bool,
    a numpy bool, the string "True"/"False", or be missing entirely.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def compute_capability_frontier(
    per_query_results: Mapping[str, List[Mapping]],
    correctness_field: str = "is_correct",
    index_field: str = "index",
) -> Dict:
    """Compute the Capability Frontier (oracle ceiling) across providers.

    Args:
        per_query_results: Maps provider name -> list of per-query result
            dicts. Each dict must carry ``index_field`` (the shared query id)
            and a boolean-ish ``correctness_field``. This matches the per-query
            rows that ``utils.save_result`` writes for the SimpleQA benchmark.
        correctness_field: Field holding the per-query correctness flag.
        index_field: Field identifying the same query across providers.

    Returns:
        A dict describing the frontier ceiling, the best single provider, and
        the untapped-potential gap. Returns zeroed values for empty input.
    """
    providers = [p for p, rows in per_query_results.items() if rows]

    # Distinct query set across every provider (queries need not be identical).
    all_indices = {
        row.get(index_field)
        for rows in per_query_results.values()
        for row in rows
        if row.get(index_field) is not None
    }
    query_count = len(all_indices)

    # Per-provider accuracy, re-derived from the per-query labels so the
    # frontier and the single-provider baseline share one source of truth.
    provider_accuracy: Dict[str, float] = {}
    for provider, rows in per_query_results.items():
        scored = [r for r in rows if r.get(index_field) is not None]
        if not scored:
            continue
        correct = sum(1 for r in scored if _is_correct(r.get(correctness_field)))
        provider_accuracy[provider] = correct / len(scored)

    # Oracle / frontier: a query is covered if ANY provider answered it right.
    covered = {
        row.get(index_field)
        for rows in per_query_results.values()
        for row in rows
        if _is_correct(row.get(correctness_field)) and row.get(index_field) is not None
    }
    oracle_correct = len(covered)
    oracle_accuracy = oracle_correct / query_count if query_count else 0.0

    best_provider = max(provider_accuracy, key=provider_accuracy.get) if provider_accuracy else None
    best_accuracy = provider_accuracy.get(best_provider, 0.0) if best_provider else 0.0

    frontier_gap = oracle_accuracy - best_accuracy
    best_error = 1.0 - best_accuracy
    oracle_error = 1.0 - oracle_accuracy
    # Paper's headline framing: relative error-rate reduction from letting the
    # oracle correct the single best provider's misses.
    error_rate_reduction = (
        (best_error - oracle_error) / best_error if best_error > 0 else 0.0
    )

    return {
        "providers_compared": sorted(providers),
        "total_queries": query_count,
        "oracle_correct_count": oracle_correct,
        "oracle_accuracy": round(oracle_accuracy, 4),
        "best_single_provider": best_provider,
        "best_single_provider_accuracy": round(best_accuracy, 4),
        "frontier_gap": round(frontier_gap, 4),
        "error_rate_reduction": round(error_rate_reduction, 4),
    }
