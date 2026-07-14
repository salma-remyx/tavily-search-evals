"""
Assess how well RAG metrics track human judgment over a SimpleQA results CSV.

Reads a per-provider SimpleQA results CSV (the contract written by
``save_result``) plus a human-labels CSV, scores each row with parameter-free
RAGChecker-style recall/faithfulness metrics, and reports Pearson / Spearman
correlation and threshold agreement against the human labels.

This is the analysis from "Evaluating RAG Metrics in Applied Contexts"
(arXiv:2607.07302): the value of a RAG metric is how well it agrees with human
judgment, not its raw score. See ``utils/rag_metric_agreement.py`` for the
scoring and substitution notes.

Usage:
    python run_metric_agreement.py \
        --results_csv results/simpleqa/.../tavily_simpleqa_results.csv \
        --human_labels_csv human_labels.csv \
        [--context_csv retrieved_context.csv]

human_labels.csv columns: index,human_score
retrieved_context.csv columns: index,context
"""

import argparse
import logging

import pandas as pd

from utils.rag_metric_agreement import evaluate_metric_agreement

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_indexed_map(csv_path: str, value_col: str) -> dict:
    df = pd.read_csv(csv_path)
    return {int(row["index"]): row[value_col] for _, row in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate RAG metrics with human judgments over SimpleQA results."
    )
    parser.add_argument(
        "--results_csv", required=True,
        help="A per-provider {provider}_simpleqa_results.csv produced by run_evaluation.py.",
    )
    parser.add_argument(
        "--human_labels_csv", required=True,
        help="CSV with columns index,human_score aligned to the results rows.",
    )
    parser.add_argument(
        "--context_csv", default=None,
        help="Optional CSV with columns index,context to enable the faithfulness metric.",
    )
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv).to_dict("records")
    human_map = _load_indexed_map(args.human_labels_csv, "human_score")
    human_labels = [float(human_map.get(int(r["index"]), 0.0)) for r in results]

    context_map = None
    if args.context_csv:
        context_map = _load_indexed_map(args.context_csv, "context")

    report = evaluate_metric_agreement(results, human_labels, context_map=context_map)

    print("\n===== RAG METRIC vs HUMAN AGREEMENT =====")
    print(f"Results: {args.results_csv}  (n={len(results)} rows)")
    print("-" * 45)
    for metric, stats in report.items():
        print(f"{metric}:")
        for key, value in stats.items():
            print(f"    {key:<12}: {value}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
