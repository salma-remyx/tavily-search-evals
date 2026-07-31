"""Vital-nugget recall evaluator.

Decomposes a gold (reference) answer into atomic "vital nuggets" and measures
how many of those nuggets are covered by a predicted answer, yielding a
``recall`` score and a ``strict_recall`` flag (all nuggets covered). This is a
finer-grained sibling of :class:`CorrectnessEvaluator`: where the SimpleQA
grader collapses to a single CORRECT / INCORRECT / NOT_ATTEMPTED verdict,
vital-nugget recall shows *which* pieces of the gold answer the retrieval +
answer pipeline actually surfaced.

Adapted from the operations-grounded evaluation in APS-RAG / APS-Bench
("A corrective agentic hybrid RAG and an operations-grounded evaluation for a
scientific facility", arXiv:2607.24663). The paper's "vital-nugget recall" and
"strict vital recall" metrics are kept at full fidelity. The paper's full
hybrid RAG platform (dense + sparse + knowledge-graph fusion, corrective
agentic loop, MCP tooling, cross-encoder reranker) is intentionally out of
scope here: this repository is a search-API evaluation harness, not a RAG
platform, so only the evaluation methodology ports. Nugget extraction and
coverage checking default to deterministic, parameter-free proxies (the paper
uses an LLM for both); an optional LLM path is available for higher fidelity
when an OpenAI API key is configured.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

# Delimiters / connectives that separate atomic nuggets inside a gold answer.
_NUGGET_SPLIT_RE = re.compile(r"\s*(?:,|;|/|:|&|\band\b|\bor\b|–|—)\s*", re.IGNORECASE)
# Connectives handled as their own word boundaries above; bare hyphen is
# intentionally NOT a split point so compound terms (e.g. "well-known") survive.
_NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if not text:
        return ""
    lowered = text.lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_nuggets_heuristic(reference_answer: str) -> List[str]:
    """Split a gold answer into atomic nuggets without any external dependency.

    This is a parameter-free proxy for the paper's LLM nugget extraction: it
    treats delimiter / connective boundaries (commas, semicolons, slashes,
    colons, "and", "or") as nugget seams. When no seam is found the whole
    answer is returned as a single nugget.
    """
    if reference_answer is None:
        return []
    pieces = [p.strip() for p in _NUGGET_SPLIT_RE.split(reference_answer)]
    nuggets = [p for p in pieces if p]
    return nuggets or ([reference_answer.strip()] if reference_answer.strip() else [])


async def extract_nuggets_llm(reference_answer: str, llm_model: str = "gpt-4.1") -> List[str]:
    """Extract atomic nuggets from a gold answer using an LLM (optional).

    Mirrors the paper's LLM nugget extraction for higher fidelity. Requires
    ``langchain_openai`` and a configured ``OPENAI_API_KEY``; the deterministic
    :func:`extract_nuggets_heuristic` is the default and needs neither.
    """
    # Imported lazily so the module (and its offline default path) stays
    # importable without the OpenAI dependency or an API key.
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=llm_model, temperature=0.0)
    prompt = (
        "Decompose the following gold answer into its atomic vital nuggets: "
        "the distinct facts a complete answer must contain. Return one nugget "
        "per line, no numbering or extra text.\n\nGold answer:\n"
        f"{reference_answer}"
    )
    raw = llm.invoke(prompt).content
    return [line.strip(" -*\t") for line in str(raw).splitlines() if line.strip(" -*\t")]


def nugget_covered(nugget: str, predicted_answer: str, jaccard_threshold: float = 0.5) -> bool:
    """Decide whether a nugget is covered by the predicted answer (deterministic).

    A nugget counts as covered when, after normalization, it appears as a
    substring of the prediction, its token set is a subset of the prediction's
    tokens, or its token-overlap (Jaccard) with the prediction meets
    ``jaccard_threshold``.
    """
    n_norm = _normalize(nugget)
    p_norm = _normalize(predicted_answer)
    if not n_norm:
        return True
    if not p_norm:
        return False
    if n_norm in p_norm:
        return True
    n_tokens = set(n_norm.split())
    p_tokens = set(p_norm.split())
    if n_tokens and n_tokens.issubset(p_tokens):
        return True
    if n_tokens:
        union = n_tokens | p_tokens
        jaccard = len(n_tokens & p_tokens) / len(union) if union else 0.0
        if jaccard >= jaccard_threshold:
            return True
    return False


def score_vital_nugget_recall(
    reference_answer: str,
    predicted_answer: str,
    nuggets: Optional[Sequence[str]] = None,
    jaccard_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Score vital-nugget recall of ``predicted_answer`` against ``reference_answer``.

    Args:
        reference_answer: The gold answer (decomposed into nuggets unless given).
        predicted_answer: The answer produced by the search + extraction pipeline.
        nuggets: Pre-extracted vital nuggets. Defaults to the heuristic split of
            the reference answer.
        jaccard_threshold: Token-overlap threshold for fuzzy nugget coverage.

    Returns:
        Dict with ``recall`` (fraction of nuggets covered), ``strict_recall``
        (1.0 iff every nugget is covered), ``covered``/``total`` counts, the
        resolved ``nuggets`` list, a ``coverage`` boolean list, and a ``value``
        label (``FULL_COVERAGE`` / ``PARTIAL_COVERAGE`` / ``NO_COVERAGE``).
    """
    resolved = list(nuggets) if nuggets is not None else extract_nuggets_heuristic(reference_answer)
    coverage = [nugget_covered(n, predicted_answer, jaccard_threshold) for n in resolved]
    total = len(resolved)
    covered = sum(1 for c in coverage if c)
    recall = round(covered / total, 4) if total else 0.0
    strict_recall = 1.0 if (total > 0 and covered == total) else 0.0

    if total == 0 or recall == 0.0:
        value = "NO_COVERAGE"
    elif recall == 1.0:
        value = "FULL_COVERAGE"
    else:
        value = "PARTIAL_COVERAGE"

    return {
        "score": recall,
        "value": value,
        "recall": recall,
        "strict_recall": strict_recall,
        "covered": covered,
        "total": total,
        "nuggets": resolved,
        "coverage": coverage,
    }


def score_result_row(row: Dict[str, Any], jaccard_threshold: float = 0.5) -> Dict[str, Any]:
    """Score an existing SimpleQA result row (as produced by the eval pipeline).

    Operates on the row schema emitted by ``evaluate_provider_simple_qa`` in
    ``run_evaluation.py`` (``reference_answer`` / ``predicted_answer`` keys), so
    vital-nugget recall can be computed over already-saved results without
    re-running any search provider.
    """
    return score_vital_nugget_recall(
        reference_answer=row.get("reference_answer", ""),
        predicted_answer=row.get("predicted_answer", ""),
        jaccard_threshold=jaccard_threshold,
    )


class VitalNuggetRecallEvaluator:
    """Answer-grading evaluator with the same contract as CorrectnessEvaluator.

    Where :class:`CorrectnessEvaluator` returns a binary verdict, this evaluator
    returns vital-nugget recall over the same ``(inputs, outputs,
    reference_outputs)`` dicts, so it drops in alongside the existing SimpleQA
    grader. Defaults to the offline (heuristic) path; set ``use_llm=True`` to
    extract nuggets with an LLM as the paper does.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_model: str = "gpt-4.1",
        jaccard_threshold: float = 0.5,
    ):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.jaccard_threshold = jaccard_threshold

    async def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate vital-nugget recall of a predicted answer.

        Args:
            inputs: Dict containing ``question`` (unused, kept for contract parity).
            outputs: Dict containing ``answer`` (the predicted answer).
            reference_outputs: Dict containing ``answer`` (the gold answer).

        Returns:
            Dict with ``score`` and ``value`` (matching CorrectnessEvaluator)
            plus the detailed recall breakdown.
        """
        predicted_answer = outputs.get("answer", "")
        reference_answer = reference_outputs.get("answer", "")
        if self.use_llm:
            nuggets = await extract_nuggets_llm(reference_answer, self.llm_model)
            result = score_vital_nugget_recall(
                reference_answer, predicted_answer, nuggets=nuggets, jaccard_threshold=self.jaccard_threshold
            )
        else:
            result = score_vital_nugget_recall(
                reference_answer, predicted_answer, jaccard_threshold=self.jaccard_threshold
            )
        return result

    @property
    def evaluation_name(self) -> str:
        return "vital_nugget_recall_evaluator"

    @property
    def evaluation_description(self) -> str:
        return (
            "Measures vital-nugget recall: the fraction of atomic gold-answer "
            "nuggets covered by the predicted answer (strict vital recall when "
            "all are covered)."
        )
