"""
Answer-in-context diagnostic for budget-constrained retrieval evaluation.

Adapted from "What Survives Into Context: A Diagnostic for Budget-Constrained
Multi-Hop RAG and When Submodular Evidence Packing Improves It"
(arXiv:2607.00725v1).

The paper's general contribution is *answer-in-context*: a diagnostic that asks
whether a gold answer survives as a contiguous span in the *packed reader
context* (the text actually shown to the reader under a fixed budget) rather
than merely in the retrieved set. It predicts downstream answer quality better
than document recall/relevance and carries information beyond them -- a gap
this repo currently leaves open, since it checks only Quotient document
relevance and LLM-judged extracted-answer correctness, never whether the gold
answer survives the context the reader actually sees.

This module implements that diagnostic for this repo's packed reader context:
the string a search handler's ``post_process`` returns and feeds to the answer
extractor. The core signal is retained verbatim -- "does the gold answer
survive the budget applied to the packed context, and does that survival
separate correct answers from incorrect ones?" Two auxiliary components are
substituted target-native (Mode 2):

  * The paper's tokenizer / reader budget is replaced by a parameter-free
    whitespace token counter, kept self-consistent (the budget and the reported
    context size use the same counter).
  * The paper's span detector is kept as a strict normalized contiguous match,
    complemented by a content-token coverage score so the signal degrades
    gracefully on paraphrase rather than dropping to zero.

The conditional contribution of the paper -- a learned submodular evidence
packer -- is intentionally out of scope here: this adds the *diagnostic*, not a
new packer. The repo's existing provider-ranked packing is treated as the naive
packer baseline against which the diagnostic is defined.
"""

from typing import Callable, Dict, Iterable, Optional

DEFAULT_CONTEXT_BUDGET_TOKENS = 512
"""Default reader-context budget (whitespace tokens). Binding but not extreme
for typical multi-document web-search contexts -- the regime the paper argues
the diagnostic applies in. Pass ``budget_tokens=None`` to disable truncation."""

_STOPWORDS = frozenset(
    (
        "a an the of to in on at for and or nor but if then else is are was were "
        "be been being this that these those it its as by with from into over "
        "under about against between through during before after above below up "
        "down out off not no"
    ).split()
)


def _tokens(text: str) -> list:
    """Lowercase alphanumeric content tokens (stoplist + length filtered)."""
    out = []
    for raw in (text or "").lower().split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) >= 2 and tok not in _STOPWORDS:
            out.append(tok)
    return out


def _normalize(text: str) -> str:
    """Lowercase + collapsed-whitespace normalization for span matching."""
    return " ".join((text or "").lower().split())


def _count_default(text: str) -> int:
    return len((text or "").split())


def _truncate_to_budget(
    text: str,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> str:
    """Largest front-prefix of ``text`` whose token count fits ``budget_tokens``.

    Models a naive reader-context packer (the paper's baseline): it fills the
    budget from the front -- i.e. the provider's top-ranked (most relevant)
    documents first.
    """
    if budget_tokens is None or count_tokens(text) <= budget_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _span_found(context: str, gold: str) -> bool:
    ng = _normalize(gold)
    return bool(ng) and ng in _normalize(context)


def _coverage(context: str, gold: str) -> float:
    gold_tokens = _tokens(gold)
    if not gold_tokens:
        return 0.0
    context_tokens = set(_tokens(context))
    present = sum(1 for tok in gold_tokens if tok in context_tokens)
    return round(present / len(gold_tokens), 4)


def answer_in_context(
    packed_context: str,
    gold_answer: str,
    budget_tokens: Optional[int] = DEFAULT_CONTEXT_BUDGET_TOKENS,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> Dict[str, object]:
    """Compute the answer-in-context diagnostic for one packed reader context.

    Args:
        packed_context: The reader context as actually fed to the reader (a
            search handler's ``post_process`` output for this repo).
        gold_answer: The reference answer whose survival we test.
        budget_tokens: Reader-context budget in tokens. When set, the primary
            ``in_context`` signal is computed over the budgeted prefix of the
            context (what the reader actually sees); ``in_context_full`` is
            always reported alongside as the recall analog.
        count_tokens: Token counter; defaults to a whitespace heuristic so the
            module has no hard tokenizer dependency.

    Returns:
        Dict with ``in_context`` (budgeted survival), ``in_context_full``
        (full-context survival -- the recall analog the paper argues is
        misleading), ``coverage`` (content-token coverage in the budgeted
        context), ``span_found`` (strict contiguous-span survival in the
        budgeted context), ``budget_applied``, and ``context_tokens``.
    """
    count = count_tokens or _count_default
    context = packed_context or ""
    gold = gold_answer or ""

    in_context_full = _span_found(context, gold)

    if budget_tokens is not None:
        budgeted = _truncate_to_budget(context, budget_tokens, count)
        budget_applied = budgeted != context
    else:
        budgeted = context
        budget_applied = False

    return {
        "in_context": _span_found(budgeted, gold),
        "in_context_full": in_context_full,
        "coverage": _coverage(budgeted, gold),
        "span_found": _span_found(budgeted, gold),
        "budget_applied": budget_applied,
        "context_tokens": count(budgeted),
    }


def _truthy(value) -> bool:
    """Coerce CSV-roundtripped values (bool, str, int) to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def summarize_answer_in_context(results: Iterable[Dict]) -> Dict[str, float]:
    """Aggregate the answer-in-context diagnostic across per-example results.

    Surfaces the paper's headline claim -- that answer-survival-in-context
    *separates* correct answers from incorrect ones better than retrieval -- at
    the provider level, as the gap in accuracy between examples whose gold
    answer survived the budget and those whose did not.

    Args:
        results: Iterable of per-example result dicts (e.g. rows read back from
            a provider's results CSV). Missing fields default to falsy/zero, so
            older result files without the diagnostic columns aggregate cleanly.

    Returns:
        Dict with ``mean_coverage``, ``in_context_rate``,
        ``accuracy_when_in_context``, ``accuracy_when_not_in_context``, and
        ``separation`` (the former minus the latter).
    """
    rows = list(results)
    n = len(rows)
    if n == 0:
        return {
            "mean_coverage": 0.0,
            "in_context_rate": 0.0,
            "accuracy_when_in_context": 0.0,
            "accuracy_when_not_in_context": 0.0,
            "separation": 0.0,
        }

    in_ctx = [_truthy(r.get("aic_in_context")) for r in rows]
    correct = [_truthy(r.get("is_correct")) for r in rows]
    coverages = [float(r.get("aic_coverage", 0.0) or 0.0) for r in rows]

    ic_correct = [c for i, c in zip(in_ctx, correct) if i]
    nic_correct = [c for i, c in zip(in_ctx, correct) if not i]

    acc_ic = round(sum(ic_correct) / len(ic_correct), 4) if ic_correct else 0.0
    acc_nic = round(sum(nic_correct) / len(nic_correct), 4) if nic_correct else 0.0

    return {
        "mean_coverage": round(sum(coverages) / n, 4),
        "in_context_rate": round(sum(in_ctx) / n, 4),
        "accuracy_when_in_context": acc_ic,
        "accuracy_when_not_in_context": acc_nic,
        "separation": round(acc_ic - acc_nic, 4),
    }
