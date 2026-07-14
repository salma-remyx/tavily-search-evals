"""
RAG metric scoring + agreement analysis for SimpleQA results.

Capability: score each retrieved-answer result with parameter-free
RAGChecker-style RAG metrics (claim recall and answer faithfulness), then
measure how well each metric *tracks human judgment* via correlation. The
point is not the raw metric value but its agreement with human labels -- the
selection signal this framework needs as it layers more RAG-centric metrics
on top of the existing SimpleQA / document-relevance benchmarks.

This consumes the existing SimpleQA result contract written by ``save_result``
(the ``index / question / reference_answer / predicted_answer / is_correct``
rows), so it slots into the same query -> retrieved-doc -> score pipeline the
document-relevance benchmark already uses -- no new data shape.

Adapted from: "Evaluating RAG Metrics in Applied Contexts: An Experiment, Its
Findings and Its Limitations" (arXiv:2607.07302). That paper scores a RAG
system with several metric libraries (Ragas, DeepEval, RAGChecker, Opik) and
correlates each against two human evaluators and recall. This module keeps the
core mechanism -- faithfulness/recall scoring plus correlation-vs-human
analysis -- at full fidelity, and substitutes the auxiliary pieces the repo
does not host:

  * the paper's four external metric libraries and its LLM-based atomic-claim
    extractor are replaced by a single self-contained, parameter-free
    claim/token-coverage implementation (clause split + token overlap);
  * scipy correlation is replaced by an in-line Pearson/Spearman
    implementation (the repo already depends on neither scipy nor numpy for
    this, and avoiding them keeps the module import-light).
"""

import logging
import math
import re
from typing import Dict, List, Optional, Sequence

from .utils import EvaluationType

logger = logging.getLogger(__name__)

# Common function words that should not dominate token-overlap coverage.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
        "being", "of", "to", "in", "on", "for", "and", "or", "but", "as", "at",
        "by", "it", "its", "this", "that", "with", "from", "into", "than",
        "then", "so", "do", "does", "did", "has", "have", "had", "will",
        "would", "can", "could", "should", "may", "might", "i", "you", "he",
        "she", "we", "they", "his", "her", "their", "our", "your", "my", "me",
        "him", "them", "us", "not", "no",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase, keep alphanumeric tokens, drop function words and 1-char noise."""
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _claims(text: str) -> List[set]:
    """Parameter-free proxy for RAGChecker's atomic-claim extraction.

    Splits on clause/sentence boundaries and returns each non-empty clause as a
    set of content tokens. This is the substitution for the paper's LLM-based
    claim extractor -- cheap, deterministic, and dependency-free.
    """
    parts = re.split(r"[.,!?;:\n]+", str(text or ""))
    claims = []
    for part in parts:
        toks = set(_tokenize(part))
        if toks:
            claims.append(toks)
    return claims


def _claim_coverage(claim_tokens: set, text_tokens: set) -> float:
    """Fraction of a claim's content tokens that appear in the supporting text."""
    if not claim_tokens:
        return 0.0
    present = sum(1 for token in claim_tokens if token in text_tokens)
    return present / len(claim_tokens)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def claim_recall(reference: str, response: str) -> float:
    """RAGChecker-style recall: how much of the reference answer is covered by the response.

    Mean, over the reference's claims, of the fraction of each claim's content
    tokens present in the response. Ranges 0..1; higher = more of the gold
    answer recovered.
    """
    ref_claims = _claims(reference)
    if not ref_claims:
        return 0.0
    response_tokens = set(_tokenize(response))
    if not response_tokens:
        return 0.0
    return _mean([_claim_coverage(c, response_tokens) for c in ref_claims])


def answer_faithfulness(response: str, context: Optional[str]) -> Optional[float]:
    """RAGChecker-style faithfulness: how much of the response is supported by retrieved context.

    Mean, over the response's claims, of the fraction of each claim's content
    tokens present in the retrieved context. Returns ``None`` when no context is
    available so the caller can drop the row from the faithfulness correlation
    rather than fabricate a signal.
    """
    response_claims = _claims(response)
    if not response_claims:
        return 0.0
    context_tokens = set(_tokenize(context))
    if not context_tokens:
        return None
    return _mean([_claim_coverage(c, context_tokens) for c in response_claims])


def score_result_row(
    row: Dict, context: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """Score a single SimpleQA result row (the contract written by ``save_result``)."""
    reference = str(row.get("reference_answer", "") or "")
    predicted = str(row.get("predicted_answer", "") or "")
    return {
        "index": row.get("index"),
        "recall": claim_recall(reference, predicted),
        "faithfulness": answer_faithfulness(predicted, context),
    }


def _to_records(results) -> List[Dict]:
    """Accept a list of dicts or anything with a ``to_dict`` (e.g. a DataFrame)."""
    if isinstance(results, list):
        return results
    if hasattr(results, "to_dict"):
        return results.to_dict("records")
    return list(results)


def score_results(
    results, context_map: Optional[Dict] = None
) -> List[Dict[str, Optional[float]]]:
    """Score every result row; ``context_map`` keys on the row ``index`` for faithfulness."""
    context_map = context_map or {}
    records = _to_records(results)
    scored = []
    for row in records:
        ctx = context_map.get(row.get("index"))
        scored.append(score_result_row(row, context=ctx))
    return scored


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation. Returns 0.0 when either side has no variance or n < 2."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    mean_x = _mean(x)
    mean_y = _mean(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return num / (denom_x * denom_y)


def _rankdata(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based), handling ties -- the input to Spearman."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # average of 1-based positions i+1 .. j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson on the ranks."""
    return pearson_r(_rankdata(list(x)), _rankdata(list(y)))


def _binary_agreement(metric_vals: Sequence[float], human_vals: Sequence[float]) -> float:
    """Fraction of rows where the metric and the human label land on the same side of 0.5."""
    if not metric_vals:
        return 0.0
    agree = sum(1 for m, h in zip(metric_vals, human_vals) if (m >= 0.5) == (h >= 0.5))
    return agree / len(metric_vals)


def evaluate_metric_agreement(
    results,
    human_labels: Sequence[float],
    context_map: Optional[Dict] = None,
    evaluation_type: EvaluationType = EvaluationType.SIMPLEQA,
) -> Dict[str, Dict[str, float]]:
    """Score results and report each metric's agreement with human judgment.

    Args:
        results: SimpleQA result rows (list of dicts or a DataFrame) using the
            contract written by ``save_result``.
        human_labels: per-row human score aligned with ``results`` (e.g. 1.0
            correct / 0.0 incorrect, or a graded value in 0..1).
        context_map: optional ``{index: retrieved_context_text}`` enabling the
            faithfulness metric; without it only recall is reported.
        evaluation_type: which result contract these rows follow. The metrics
            here are defined for the SimpleQA contract.

    Returns:
        ``{metric: {n, pearson, spearman, agreement, mean_metric, mean_human}}``.
    """
    if evaluation_type != EvaluationType.SIMPLEQA:
        raise ValueError(
            "rag_metric_agreement is defined for SimpleQA result rows "
            "(reference_answer / predicted_answer); "
            f"got evaluation_type={evaluation_type!r}."
        )

    scored = score_results(results, context_map)
    human = list(human_labels)
    if len(scored) != len(human):
        raise ValueError(
            f"results ({len(scored)}) and human_labels ({len(human)}) must align."
        )

    report: Dict[str, Dict[str, float]] = {}
    for metric in ("recall", "faithfulness"):
        pairs = [
            (s[metric], h)
            for s, h in zip(scored, human)
            if s.get(metric) is not None
        ]
        if len(pairs) < 2:
            report[metric] = {
                "n": float(len(pairs)),
                "pearson": 0.0,
                "spearman": 0.0,
                "agreement": 0.0,
            }
            continue
        metric_vals = [p[0] for p in pairs]
        human_vals = [p[1] for p in pairs]
        report[metric] = {
            "n": float(len(pairs)),
            "pearson": round(pearson_r(metric_vals, human_vals), 4),
            "spearman": round(spearman_rho(metric_vals, human_vals), 4),
            "agreement": round(_binary_agreement(metric_vals, human_vals), 4),
            "mean_metric": round(_mean(metric_vals), 4),
            "mean_human": round(_mean(human_vals), 4),
        }
    return report
