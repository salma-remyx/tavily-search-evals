"""Stress-test document relevance under controlled poor-quality evidence.

Runnable wiring for :mod:`utils.evidence_stress`. It composes the repo's
existing data/output utilities with the DeepStress-derived stress capability to
turn the static document-relevance score into a robustness curve.

DeepStress (arXiv:2607.13920) replaces a search agent's retrieval module with a
controlled synthetic environment and dials the frequency of challenging evidence
(trustworthiness / relevance / factuality). Here the controlled environment is
built from each dataset example's known relevant passage (``answer_context``):
we inject distractors at controlled frequencies and measure how the relevance
signal degrades. No search API keys are required, so this runs end-to-end in CI.

Example::

    python stress_eval.py --sample 50
    python stress_eval.py --noise_ratios 0.0,0.2,0.4 --dimensions relevance factuality
"""

import argparse
import csv
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Dict, List

# Existing repo utilities -- composing these is the integration point.
from utils import EvaluationType
from utils.evidence_stress import (
    ALL_DIMENSIONS,
    DEFAULT_NOISE_RATIOS,
    stress_sweep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_DATASET = "datasets/document_relevance_dynamic_test_set.json"

_DIMENSION_MAP = {d.value: d for d in ALL_DIMENSIONS}


def _load_examples(dataset_path: str) -> List[Dict]:
    """Load dataset examples, keeping the per-example relevant passage.

    The repo's existing ``load_document_relevance_eval_data`` loader drops
    ``answer_context`` (it only needs question/answer), but the stress test needs
    that passage as the controlled relevant document, so we read the JSON here.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)
    items = data["dataset"] if isinstance(data, dict) and "dataset" in data else data
    examples = []
    for item in items:
        answer = (item.get("answer") or "").strip()
        context = (item.get("answer_context") or "").strip()
        if not answer or not context:
            continue
        examples.append(
            {
                "index": len(examples),
                "question": item.get("question", ""),
                "answer": answer,
                "answer_context": context,
            }
        )
    return examples


def _controlled_documents(answer_context: str) -> List[str]:
    """Segment a relevant passage into sentence-level documents.

    DeepStress replaces retrieval with a controlled environment; here the known
    relevant passage is that environment. Segmenting it into sentences gives a
    realistic multi-document relevant set (mirroring ``max_results`` retrieved
    docs) so the controlled-frequency sweep produces a smooth robustness curve
    rather than a single all-or-nothing step. No content is fabricated.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_context) if s.strip()]
    return sentences or [answer_context]


def run_stress_eval(
    dataset_path: str,
    sample: int,
    noise_ratios: List[float],
    dimensions,
    seed: int,
    output_dir: str,
):
    examples = _load_examples(dataset_path)
    if sample and sample < len(examples):
        examples = random.Random(seed).sample(examples, sample)
    logger.info(
        "Stress-testing %d examples across noise ratios %s", len(examples), noise_ratios
    )

    per_example = []
    for ex in examples:
        # The controlled relevant document set, segmented from the known passage.
        sweep = stress_sweep(
            _controlled_documents(ex["answer_context"]),
            ex["question"],
            ex["answer"],
            noise_ratios=noise_ratios,
            dimensions=dimensions,
            seed=seed,
        )
        per_example.append({"index": ex["index"], "question": ex["question"], "sweep": sweep})

    # Aggregate each noise ratio across examples (mean of the per-example stats).
    aggregated = []
    for r in noise_ratios:
        rows = [ex["sweep"][i] for ex in per_example for i, s in enumerate(ex["sweep"]) if s["noise_ratio"] == r]
        if not rows:
            continue
        n = len(rows)
        aggregated.append(
            {
                "noise_ratio": r,
                "baseline_relevant_pct": round(sum(x["baseline_relevant_pct"] for x in rows) / n, 2),
                "stress_relevant_pct": round(sum(x["stress_relevant_pct"] for x in rows) / n, 2),
                "degradation_pp": round(sum(x["degradation_pp"] for x in rows) / n, 2),
                "robustness": round(sum(x["robustness"] for x in rows) / n, 3),
                "examples": n,
            }
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(output_dir, f"{EvaluationType.DOCUMENT_RELEVANCE.value}_stress", ts)
    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise_ratio",
                "baseline_relevant_pct",
                "stress_relevant_pct",
                "degradation_pp",
                "robustness",
                "examples",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregated)

    _print_report(aggregated, len(examples), dimensions, summary_path)
    return aggregated


def _print_report(aggregated, n_examples, dimensions, summary_path):
    print("\n===== DOCUMENT RELEVANCE STRESS RESULTS =====")
    print(f"Examples: {n_examples} | Dimensions: {[d.value for d in dimensions]}")
    print("-------------------------------------------------------------")
    print(f"{'noise':>6} | {'baseline%':>10} | {'stress%':>9} | {'degrad(pp)':>11} | {'robustness':>11}")
    print("-------------------------------------------------------------")
    for row in aggregated:
        print(
            f"{row['noise_ratio']:>6.1f} | {row['baseline_relevant_pct']:>10.2f} | "
            f"{row['stress_relevant_pct']:>9.2f} | {row['degradation_pp']:>11.2f} | "
            f"{row['robustness']:>11.3f}"
        )
    print("-------------------------------------------------------------")
    print(f"Summary written to {summary_path}\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Stress-test document relevance (DeepStress).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to the document-relevance dataset JSON")
    parser.add_argument("--sample", type=int, default=50, help="Number of examples to stress-test (0 = all)")
    parser.add_argument("--noise_ratios", default=",".join(str(r) for r in DEFAULT_NOISE_RATIOS), help="Comma-separated challenging-evidence frequencies")
    parser.add_argument("--dimensions", default=",".join(d.value for d in ALL_DIMENSIONS), help="Comma-separated dimensions: relevance,trustworthiness,factuality")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic sampling/injection")
    parser.add_argument("--output_dir", default="results", help="Directory to save stress results")
    return parser.parse_args()


def main():
    args = _parse_args()
    noise_ratios = [float(r) for r in args.noise_ratios.split(",") if r.strip() != ""]
    dims = []
    for name in args.dimensions.split(","):
        name = name.strip()
        if name and name in _DIMENSION_MAP:
            dims.append(_DIMENSION_MAP[name])
    dimensions = dims or list(ALL_DIMENSIONS)
    sample = args.sample if args.sample and args.sample > 0 else 0
    run_stress_eval(
        dataset_path=args.dataset,
        sample=sample,
        noise_ratios=noise_ratios,
        dimensions=dimensions,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
