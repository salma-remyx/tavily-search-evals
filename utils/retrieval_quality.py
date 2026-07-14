"""Reference-free retrieval-quality scoring (eRAG).

Adapted from "Evaluating Retrieval Quality in Retrieval-Augmented
Generation" (Salemi & Zamani, 2024 — https://arxiv.org/abs/2404.13781).

The paper's motivation is that query-document relevance labels correlate
poorly with a RAG system's downstream performance, while full end-to-end
evaluation of every retrieved-document subset is prohibitively expensive.
eRAG proposes a middle path: run the *downstream generator* on each
retrieved document *individually*, turn each document's downstream success
into a binary relevance label, and aggregate those labels with standard
ranked-retrieval metrics (precision@k, MRR, average precision).

That contract maps directly onto this repo's SimpleQA path, which already
supplies eRAG's four ingredients:

* the retrieval list — the documents a handler returns,
* a per-query generator — ``PostProcessor.extract_answer``,
* a binary downstream metric — ``CorrectnessEvaluator`` (score 0/1).

This module is intentionally pure-stdlib: the aggregation metrics are the
paper's result, and they need no external ``pytrec_eval`` dependency for
the small per-query label lists produced here. The heavyweight parts of
the paper (learned rerankers, multi-document subset search) are out of
scope — the value delivered is the reference-free per-document scoring
signal itself.
"""

import asyncio
import re
from typing import Awaitable, Callable, Dict, List, Sequence, Union

# Documents joined by ``ProviderHandler._format_search_results_for_prompt``
# begin each entry with a ``**Document N.**`` marker. We split on that
# boundary to recover the individual documents from an already-formatted
# prompt string without having to thread the raw list through every
# provider handler.
_DOCUMENT_BOUNDARY = re.compile(r"(?=\*\*Document\s+\d+\.\*\*)")

# Async callable that produces an answer from (query, single_document).
GenerateFn = Callable[[str, str], Awaitable[str]]
# Async callable that scores (query, answer, reference_answer) as 0/1.
JudgeFn = Callable[[str, str, str], Awaitable[float]]


def split_formatted_documents(formatted: str) -> List[str]:
    """Recover individual documents from a formatted prompt string.

    ``ProviderHandler._format_search_results_for_prompt`` concatenates
    documents into a single string, each prefixed with ``**Document N.**``.
    This reverses that join so each document can be scored on its own.

    Args:
        formatted: The concatenated document string (or already-split list
            handed straight through by the caller).

    Returns:
        A list of per-document strings with surrounding whitespace trimmed.
    """
    if not formatted:
        return []
    parts = _DOCUMENT_BOUNDARY.split(formatted)
    return [part.strip() for part in parts if part.strip()]


def precision_at_k(labels: Sequence[int], k: int) -> float:
    """Fraction of the top-``k`` documents that are individually relevant."""
    if k <= 0:
        return 0.0
    top = labels[:k]
    return sum(top) / k


def reciprocal_rank(labels: Sequence[int]) -> float:
    """Reciprocal of the rank of the first relevant document (0 if none)."""
    for rank, label in enumerate(labels, start=1):
        if label:
            return 1.0 / rank
    return 0.0


def average_precision(labels: Sequence[int]) -> float:
    """Average precision for a single ranked list of binary labels."""
    num_relevant = sum(labels)
    if num_relevant == 0:
        return 0.0
    hits = 0
    running = 0.0
    for rank, label in enumerate(labels, start=1):
        if label:
            hits += 1
            running += hits / rank
    return running / num_relevant


def aggregate_labels(
    labels: Sequence[int], ks: Sequence[int] = (1, 3, 5)
) -> Dict[str, float]:
    """Aggregate per-document downstream labels into retrieval metrics.

    Args:
        labels: Binary relevance labels in retrieval rank order (rank 1
            first), where 1 means the document alone yielded a correct
            downstream answer.
        ks: Cutoffs to report precision@k for.

    Returns:
        A dict of eRAG retrieval-quality metrics for this query.
    """
    labels = [1 if label else 0 for label in labels]
    metrics: Dict[str, float] = {
        "mrr": round(reciprocal_rank(labels), 4),
        "map": round(average_precision(labels), 4),
        # Fraction of retrieved documents that are individually sufficient;
        # eRAG's simplest set-based signal.
        "erag_precision": round(sum(labels) / len(labels), 4) if labels else 0.0,
        "hit": 1.0 if any(labels) else 0.0,
    }
    for k in ks:
        metrics[f"precision@{k}"] = round(precision_at_k(labels, k), 4)
    return metrics


async def score_documents(
    query: str,
    reference_answer: str,
    documents: Union[str, Sequence[str]],
    generate_fn: GenerateFn,
    judge_fn: JudgeFn,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, object]:
    """Score a retrieval list with the eRAG reference-free procedure.

    Each document is run through ``generate_fn`` on its own, the resulting
    answer is graded by ``judge_fn`` against the reference, and the binary
    per-document outcomes are aggregated into retrieval metrics.

    Args:
        query: The user query.
        reference_answer: The gold answer used by ``judge_fn``.
        documents: Either the list of per-document strings or the single
            formatted prompt string emitted by a provider handler (which is
            split back into documents automatically).
        generate_fn: Async ``(query, document) -> answer``.
        judge_fn: Async ``(query, answer, reference_answer) -> score`` where
            a score ``>= 1.0`` counts the document as relevant.
        ks: Precision cutoffs to report.

    Returns:
        The aggregated metrics plus the raw ``labels`` and document count.
    """
    if isinstance(documents, str):
        documents = split_formatted_documents(documents)

    async def label_one(document: str) -> int:
        answer = await generate_fn(query, document)
        score = await judge_fn(query, answer, reference_answer)
        return 1 if float(score) >= 1.0 else 0

    labels: List[int] = (
        list(await asyncio.gather(*(label_one(doc) for doc in documents)))
        if documents
        else []
    )

    result: Dict[str, object] = aggregate_labels(labels, ks)
    result["labels"] = labels
    result["num_documents"] = len(labels)
    return result


def mean_metrics(per_query: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Average eRAG metrics across queries for a provider-level summary.

    Non-numeric fields (``labels``) are ignored; queries that produced no
    documents (and therefore no eRAG signal) are skipped.
    """
    scored = [m for m in per_query if m and m.get("num_documents")]
    if not scored:
        return {}
    numeric_keys = [
        key
        for key, value in scored[0].items()
        if isinstance(value, (int, float)) and key != "num_documents"
    ]
    summary = {
        key: round(sum(float(m[key]) for m in scored) / len(scored), 4)
        for key in numeric_keys
    }
    summary["queries_scored"] = len(scored)
    return summary
