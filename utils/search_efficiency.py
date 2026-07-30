"""Search-efficiency scoring for search-provider evaluations.

Adapted from "Efficiency Matters in Autonomous Research" (arXiv:2607.24647),
which argues that autonomous-research systems should be judged by the
*efficiency* of reaching a quality outcome -- the area under the curve (AUC)
of the quality-vs-cost Pareto frontier -- and not only by final outcome
quality.

This module applies that insight to this repo's own problem: comparing web
search providers. The evaluation pipeline already records, per provider, both
outcome quality (SimpleQA accuracy / document-relevance percentage) and
resource cost (token usage, via ``utils.token_utils``). Each provider is
treated as a point in (cost, quality) space; we build the Pareto frontier of
providers that are not beaten on both axes and score the whole provider set
with a normalized AUC of that frontier. A provider set that reaches high
quality cheaply scores near 1.0; one that burns tokens for little quality
scores low.

Intentional scope: the paper also proposes "fluid search", an adaptive
portfolio bandit over a forest of search processes. This repo hosts no such
search-algorithm forest, so that controller is out of scope here -- we port the
*efficiency metric* (the paper's measurable contribution), not the search
controller. This is a Mode 3 (inspired-experiment) adaptation.
"""

import argparse
import csv
import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .token_utils import get_token_stats
from .utils import EvaluationType

logger = logging.getLogger(__name__)

__all__ = [
    "ProviderPoint",
    "pareto_frontier",
    "pareto_auc",
    "load_provider_points",
    "compute_efficiency_report",
]


@dataclass
class ProviderPoint:
    """A provider placed in (cost, quality) space.

    Attributes:
        name: Provider identifier (the CSV filename stem).
        quality: Outcome score in [0, 1] (accuracy / relevance percentage).
        cost: Resource consumed per unit of work (mean tokens per query).
    """

    name: str
    quality: float
    cost: float


def _is_dominated(cost: float, quality: float, pairs: List[Tuple[float, float]]) -> bool:
    """True if some other point matches/beats us on both axes and is strict on one.

    Quality is maximized and cost is minimized.
    """
    for other_cost, other_quality in pairs:
        if other_cost <= cost and other_quality >= quality:
            if other_cost < cost or other_quality > quality:
                return True
    return False


def pareto_frontier(points: List[ProviderPoint]) -> List[ProviderPoint]:
    """Return the non-dominated providers, sorted by cost ascending.

    A provider is on the frontier when no other provider is both cheaper (or
    equal) and higher-quality (or equal) with at least one strict improvement.
    """
    pairs = [(p.cost, p.quality) for p in points]
    frontier = [p for p in points if not _is_dominated(p.cost, p.quality, pairs)]
    frontier.sort(key=lambda p: p.cost)
    return frontier


def pareto_auc(points: List[ProviderPoint]) -> float:
    """Normalized area under the quality-vs-cost Pareto frontier, in [0, 1].

    Costs and qualities are scaled by the maxima across the candidate set so the
    score is comparable across runs. The frontier curve is anchored at the
    origin (zero budget yields zero quality), extended to the full cost range at
    the best quality the frontier reaches, then integrated with the trapezoidal
    rule. The score is *comparative*: it is most informative with several
    providers (a lone provider normalizes to its own maxima and scores 0.5).
    """
    if not points:
        return 0.0
    max_cost = max(p.cost for p in points)
    max_quality = max(p.quality for p in points)
    if max_cost <= 0 or max_quality <= 0:
        return 0.0

    frontier = pareto_frontier(points)
    # Origin anchor -> normalized frontier points (already cost-sorted).
    curve: List[Tuple[float, float]] = [(0.0, 0.0)]
    for p in frontier:
        curve.append((p.cost / max_cost, p.quality / max_quality))
    # Hold the best reached quality out to the full normalized cost range.
    if curve[-1][0] < 1.0:
        curve.append((1.0, curve[-1][1]))

    auc = 0.0
    for (cost_prev, quality_prev), (cost_curr, quality_curr) in zip(curve, curve[1:]):
        auc += (cost_curr - cost_prev) * (quality_prev + quality_curr) / 2.0
    return round(auc, 6)


def _truthy(series: pd.Series) -> pd.Series:
    """Coerce a possibly-string is_correct column to a boolean series."""
    return (
        series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "correct"])
    )


def load_provider_points(
    output_dir: str,
    evaluation_type: EvaluationType = EvaluationType.SIMPLEQA,
) -> List[ProviderPoint]:
    """Read a run's results directory into (cost, quality) points per provider.

    Cost is the mean per-query token count from each provider's results CSV,
    aggregated via ``utils.token_utils.get_token_stats``. For SimpleQA, quality
    is the fraction of correct rows read from the same per-row CSV; for the
    document-relevance benchmark it is ``relevant_docs_percentage`` from
    ``summary.csv``.
    """
    suffix = evaluation_type.value
    pattern = os.path.join(output_dir, f"*_{suffix}_results.csv")

    qualities: Dict[str, float] = {}
    if evaluation_type == EvaluationType.DOCUMENT_RELEVANCE:
        summary_path = os.path.join(output_dir, "summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            for _, row in summary.iterrows():
                qualities[row["provider"]] = float(row.get("relevant_docs_percentage", 0.0))

    points: List[ProviderPoint] = []
    for csv_path in sorted(glob.glob(pattern)):
        provider = os.path.basename(csv_path).removesuffix(f"_{suffix}_results.csv")
        df = pd.read_csv(csv_path)
        if df.empty or "token_count" not in df.columns:
            logger.warning("Skipping %s: no token_count rows", csv_path)
            continue

        token_counts = [int(t) for t in df["token_count"].dropna().tolist()]
        _, mean_cost = get_token_stats(token_counts)

        if evaluation_type == EvaluationType.SIMPLEQA and "is_correct" in df.columns:
            correct = _truthy(df["is_correct"])
            quality = float(correct.sum()) / len(correct) if len(correct) else 0.0
        else:
            quality = qualities.get(provider, 0.0)

        points.append(ProviderPoint(name=provider, quality=quality, cost=mean_cost))

    return points


def compute_efficiency_report(
    output_dir: str,
    evaluation_type: EvaluationType = EvaluationType.SIMPLEQA,
    write: bool = True,
) -> Dict[str, object]:
    """Score a completed run's provider set by search efficiency.

    Reads the run's per-provider result CSVs, builds the (cost, quality) Pareto
    frontier, computes the normalized AUC (the efficiency headline number), and
    -- unless ``write`` is False -- writes ``efficiency_report.csv`` alongside
    the existing ``summary.csv``.

    Returns a dict with the global AUC, the frontier provider names, and
    per-provider rows (``provider``, ``quality``, ``cost``,
    ``on_pareto_frontier``, ``efficiency_rank`` where 1 is the cheapest
    non-dominated provider).
    """
    points = load_provider_points(output_dir, evaluation_type)
    if not points:
        logger.warning("No provider points found under %s", output_dir)
        return {"search_efficiency_auc": 0.0, "frontier": [], "providers": []}

    frontier = pareto_frontier(points)
    frontier_names = {p.name for p in frontier}
    auc = pareto_auc(points)

    # Frontier providers first (cheapest first), then dominated (cheapest first).
    ranked = sorted(points, key=lambda p: (p.name not in frontier_names, p.cost))
    providers: List[Dict[str, object]] = []
    for rank, p in enumerate(ranked, start=1):
        providers.append(
            {
                "provider": p.name,
                "quality": round(p.quality, 6),
                "cost": round(p.cost, 6),
                "on_pareto_frontier": p.name in frontier_names,
                "efficiency_rank": rank,
            }
        )

    report_path = os.path.join(output_dir, "efficiency_report.csv")
    if write:
        with open(report_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "provider",
                    "quality",
                    "cost",
                    "on_pareto_frontier",
                    "efficiency_rank",
                ],
            )
            writer.writeheader()
            for row in providers:
                writer.writerow(row)
        logger.info(
            "Search-efficiency AUC=%.4f; wrote %s (frontier: %s)",
            auc,
            report_path,
            ", ".join(sorted(frontier_names)) or "none",
        )

    return {
        "search_efficiency_auc": auc,
        "frontier": sorted(frontier_names),
        "providers": providers,
        "report_path": report_path if write else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: score a run's providers by search efficiency."""
    parser = argparse.ArgumentParser(
        description="Score a run's providers by search efficiency (Pareto AUC).",
    )
    parser.add_argument(
        "output_dir",
        help="A results directory containing per-provider *_results.csv files",
    )
    parser.add_argument(
        "--evaluation_type",
        choices=[
            EvaluationType.SIMPLEQA.value,
            EvaluationType.DOCUMENT_RELEVANCE.value,
        ],
        default=EvaluationType.SIMPLEQA.value,
    )
    args = parser.parse_args(argv)

    report = compute_efficiency_report(args.output_dir, EvaluationType(args.evaluation_type))
    print(f"Search-efficiency AUC: {report['search_efficiency_auc']}")
    print(f"Pareto frontier: {', '.join(report['frontier']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
