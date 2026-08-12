"""Multi-run consistency metrics for the SimpleQA pipeline.

Adapted from: "What Current AI Benchmarks Leave Unmeasured: Modality, Search,
Citations, and Implications (for Safety Evaluations)" (arXiv:2608.06202).

That paper audits the assumptions behind LLM benchmark evaluations and shows
that reporting only a single-run accuracy number hides important behavioral
variation: repeated runs of the *same* prompt produced inconsistent responses
on up to 21% of prompts, and accuracy swung by several points across runs and
search conditions. Its central, self-contained recommendation is to run each
prompt N times and report grade-consistency across runs instead of a single
accuracy figure.

This module re-implements that central measurement for this repo's per-provider
SimpleQA loop (which today runs each query once and emits only accuracy):

  - ``compute_run_consistency`` -- the pure core. Given the per-example results
    of N repeated runs (exactly the dicts ``evaluate_provider_simple_qa``
    already produces), it reports grade-consistency, accuracy stability,
    answer-text similarity and abstention consistency across runs.
  - ``measure_provider_consistency`` -- the collector. Runs each example N times
    over the same building blocks as ``evaluate_provider_simple_qa`` and feeds
    the per-run results into ``compute_run_consistency``.
  - ``save_consistency_summary`` -- surfaces the metrics as a CSV next to the
    existing ``summary.csv``.

Mode 2 (adapted port). The paper's full audit compares access modalities
(ChatGPT chat UI vs. OpenAI API) and hand-codes citation grounding and
abstention across them -- infrastructure this repo does not host (a second
access modality, human citation coding, a separate benchmark suite). We keep
the paper's core measurement (grade-consistency across repeated runs) at full
fidelity and substitute the auxiliary surfaces with target-native,
parameter-free proxies: response text similarity uses a ``difflib``
SequenceMatcher ratio instead of a learned embedder, and abstention is detected
from the existing ``NOT_ATTEMPTED`` grade rather than a separate coder. The
paper's separate evaluation framework is cut entirely -- results land in the
existing results directory alongside ``summary.csv``.
"""

import asyncio
import csv
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional


def _default_text_similarity(a: str, b: str) -> float:
    """Parameter-free pairwise answer-text similarity in [0, 1].

    Mode 2 substitution for the paper's learned response-similarity signal:
    a ``difflib.SequenceMatcher`` ratio needs no model, no weights and no extra
    dependency, while still surfacing when repeated runs drift in wording.
    """
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a or "", b or "").ratio()


def compute_run_consistency(
    per_run_results: List[List[Dict[str, Any]]],
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, Any]:
    """Compute multi-run consistency metrics from per-run per-example results.

    Args:
        per_run_results: One entry per run; each entry is the list of
            per-example result dicts produced by the SimpleQA loop -- the same
            shape as the ``results`` list returned by
            ``evaluate_provider_simple_qa`` (each dict carries at least
            ``index``, ``is_correct`` (bool), ``grade`` (str) and
            ``predicted_answer`` (str)).
        similarity_fn: Optional pairwise text-similarity callable. Defaults to
            the parameter-free SequenceMatcher ratio.

    Returns:
        A dict with: ``n_runs``, ``accuracy_per_run``, ``accuracy_mean``,
        ``accuracy_std``, ``grade_consistency_rate`` (fraction of examples whose
        grade was identical across all runs -- the complement of the paper's
        "up to 21% inconsistent"), ``inconsistent_count``, ``inconsistent_rate``,
        ``answer_similarity_mean``, ``abstention_consistency_rate``,
        ``total_examples`` and a ``per_example`` breakdown.
    """
    if similarity_fn is None:
        similarity_fn = _default_text_similarity

    n_runs = len(per_run_results)
    metrics: Dict[str, Any] = {
        "n_runs": n_runs,
        "accuracy_per_run": [],
        "accuracy_mean": 0.0,
        "accuracy_std": 0.0,
        "grade_consistency_rate": 0.0,
        "inconsistent_count": 0,
        "inconsistent_rate": 0.0,
        "answer_similarity_mean": 0.0,
        "abstention_consistency_rate": 0.0,
        "total_examples": 0,
        "per_example": [],
    }
    if n_runs == 0:
        return metrics

    # Accuracy per run (over rows that were actually graded, not ERROR rows).
    accuracy_per_run: List[float] = []
    for run in per_run_results:
        graded = [r for r in run if r.get("grade") != "ERROR"]
        if graded:
            acc = sum(1 for r in graded if r.get("is_correct")) / len(graded)
        else:
            acc = 0.0
        accuracy_per_run.append(round(acc, 3))
    metrics["accuracy_per_run"] = accuracy_per_run

    accuracy_mean = sum(accuracy_per_run) / n_runs
    metrics["accuracy_mean"] = round(accuracy_mean, 3)
    if n_runs > 1:
        variance = sum((a - accuracy_mean) ** 2 for a in accuracy_per_run) / n_runs
        metrics["accuracy_std"] = round(variance ** 0.5, 3)

    # Align each example across runs by its index.
    by_index: Dict[Any, List[Dict[str, Any]]] = {}
    for run in per_run_results:
        for row in run:
            by_index.setdefault(row.get("index"), []).append(row)

    total = len(by_index)
    metrics["total_examples"] = total
    if total == 0:
        return metrics

    consistent = 0
    abstention_consistent = 0
    similarity_scores: List[float] = []
    per_example: List[Dict[str, Any]] = []

    for idx, rows in by_index.items():
        grades = [r.get("grade") for r in rows]
        answers = [str(r.get("predicted_answer", "")) for r in rows]
        corrects = [bool(r.get("is_correct")) for r in rows]

        grade_consistent = len(set(grades)) <= 1
        if grade_consistent:
            consistent += 1

        # Abstention proxied by the NOT_ATTEMPTED grade; consistent when every
        # run of this example agreed on whether it abstained.
        abstained = [g == "NOT_ATTEMPTED" for g in grades]
        if len(set(abstained)) <= 1:
            abstention_consistent += 1

        if len(rows) >= 2:
            pair_sims = [
                similarity_fn(answers[i], answers[j])
                for i in range(len(answers))
                for j in range(i + 1, len(answers))
            ]
            ex_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 1.0
        else:
            ex_sim = 1.0
        similarity_scores.append(ex_sim)

        per_example.append({
            "index": idx,
            "grades": grades,
            "grade_consistent": grade_consistent,
            "is_correct_per_run": corrects,
            "answer_similarity": round(ex_sim, 3),
        })

    metrics["grade_consistency_rate"] = round(consistent / total, 3)
    metrics["inconsistent_count"] = total - consistent
    metrics["inconsistent_rate"] = round((total - consistent) / total, 3)
    metrics["abstention_consistency_rate"] = round(abstention_consistent / total, 3)
    metrics["answer_similarity_mean"] = round(sum(similarity_scores) / total, 3)
    metrics["per_example"] = per_example
    return metrics


def save_consistency_summary(
    consistency_metrics: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> str:
    """Write per-provider multi-run consistency metrics to a CSV.

    Mirrors the role of ``save_summary`` for the consistency audit: one row per
    provider with the behavioral dimensions the paper recommends reporting
    alongside raw accuracy.

    Args:
        consistency_metrics: Mapping of provider name to the metrics dict
            returned by ``compute_run_consistency`` (or by
            ``measure_provider_consistency``, which spreads those metrics into
            its returned result).
        output_dir: Directory to write ``multirun_consistency.csv`` into.

    Returns:
        The path of the written summary file.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "multirun_consistency.csv")
    fieldnames = [
        "provider",
        "n_runs",
        "accuracy_mean",
        "accuracy_std",
        "grade_consistency_rate",
        "inconsistent_rate",
        "inconsistent_count",
        "total_examples",
        "answer_similarity_mean",
        "abstention_consistency_rate",
        "timestamp",
    ]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for provider_name, metrics in consistency_metrics.items():
            writer.writerow({
                "provider": provider_name,
                "n_runs": metrics.get("n_runs", 0),
                "accuracy_mean": metrics.get("accuracy_mean", 0.0),
                "accuracy_std": metrics.get("accuracy_std", 0.0),
                "grade_consistency_rate": metrics.get("grade_consistency_rate", 0.0),
                "inconsistent_rate": metrics.get("inconsistent_rate", 0.0),
                "inconsistent_count": metrics.get("inconsistent_count", 0),
                "total_examples": metrics.get("total_examples", 0),
                "answer_similarity_mean": metrics.get("answer_similarity_mean", 0.0),
                "abstention_consistency_rate": metrics.get("abstention_consistency_rate", 0.0),
                "timestamp": timestamp,
            })
    return summary_file


async def measure_provider_consistency(
    provider_name: str,
    search_handler,
    examples: List[Dict[str, Any]],
    post_processor,
    output_dir: str,
    evaluation_type: "EvaluationType",  # noqa: F821 - resolved at runtime below
    evaluator_model: str = "gpt-4.1",
    batch_size: int = 3,
    n_runs: int = 3,
) -> Dict[str, Any]:
    """Run each example ``n_runs`` times and report grade-consistency.

    Reuses the same building blocks as ``evaluate_provider_simple_qa``
    (``search_handler.search``, ``post_processor.extract_answer`` and
    ``CorrectnessEvaluator``) but, unlike that function, (a) loops each example
    ``n_runs`` times and (b) calls ``save_result`` with explicit ``output_dir``
    and ``evaluation_type`` arguments. Only the first run's per-example rows are
    persisted to the provider results CSV (so ``save_summary`` keeps working);
    every run feeds the consistency audit.

    Heavy dependencies (``CorrectnessEvaluator``, ``save_result``) are imported
    lazily so the pure ``compute_run_consistency`` stays importable without the
    repo's LLM/search stack.

    Returns:
        The consistency metrics spread together with ``per_run_results`` plus a
        representative ``accuracy``/``correct_count``/``total_count``/``results``
        (from run 0), so callers can treat the return value like a normal
        provider result.
    """
    # Lazy imports keep compute_run_consistency dependency-light and avoid any
    # import cycle through the utils/evaluators packages.
    from evaluators.correctness_evaluator import CorrectnessConfig, CorrectnessEvaluator
    from utils.utils import EvaluationType, save_result

    evaluator = CorrectnessEvaluator(CorrectnessConfig(model_name=evaluator_model))
    per_run_results: List[List[Dict[str, Any]]] = []

    async def evaluate_once(example: Dict[str, Any]) -> Dict[str, Any]:
        index = example["index"]
        try:
            query = example["question"]
            reference_answer = example["answer"]

            search_result = await search_handler.search(query)
            original_answer = search_result.get("answer", "")

            is_llm_response = search_handler.is_llm_response
            if is_llm_response:
                search_ans = original_answer
                token_count, token_avg = 0, 0
            else:
                search_ans, token_count, token_avg = await search_handler.post_process(search_result)

            answer = post_processor.extract_answer(
                query=query,
                is_llm_response=is_llm_response,
                search_result=search_ans,
            )
            evaluation_result = await evaluator.evaluate(
                {"question": query},
                {"answer": answer},
                {"answer": reference_answer},
            )
            return {
                "index": index,
                "question": query,
                "reference_answer": reference_answer,
                "predicted_answer": answer,
                "is_correct": evaluation_result["score"] == 1.0,
                "grade": evaluation_result["value"],
                "token_count": token_count if not is_llm_response else 0,
                "token_avg": token_avg if not is_llm_response else 0,
            }
        except Exception as e:  # noqa: BLE001 - mirror the pipeline's per-example resilience
            return {
                "index": index,
                "question": example.get("question", ""),
                "reference_answer": example.get("answer", ""),
                "predicted_answer": "ERROR",
                "is_correct": False,
                "grade": "ERROR",
                "token_count": 0,
                "token_avg": 0,
                "error": str(e),
            }

    for run_idx in range(n_runs):
        run_results: List[Dict[str, Any]] = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            gathered = await asyncio.gather(*(evaluate_once(ex) for ex in batch))
            for row in gathered:
                run_results.append(row)
                # Persist only the first run so save_summary's CSV stays coherent.
                if run_idx == 0:
                    save_result(row, provider_name, output_dir, evaluation_type)
            time.sleep(3.0)  # avoid rate limiting, as in the single-run path
        per_run_results.append(run_results)

    metrics = compute_run_consistency(per_run_results)

    run_zero = per_run_results[0] if per_run_results else []
    graded_zero = [r for r in run_zero if r.get("grade") != "ERROR"]
    correct_zero = sum(1 for r in graded_zero if r.get("is_correct"))
    accuracy = round(correct_zero / len(graded_zero), 3) if graded_zero else 0.0

    return {
        "provider": provider_name,
        "results": run_zero,
        "accuracy": accuracy,
        "correct_count": correct_zero,
        "total_count": len(graded_zero),
        "per_run_results": per_run_results,
        **metrics,
    }
