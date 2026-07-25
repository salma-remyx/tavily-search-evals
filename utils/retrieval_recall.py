"""Retrieval-recall scoring and dataset loading for literature search.

Adapted from *LitSearch: A Retrieval Benchmark for Scientific Literature
Search* (Asai et al., 2024; arXiv:2407.18940). LitSearch scores a retrieval
system by recall over a set of ground-truth ("gold") papers per query.

Target-native adaptation (Mode 2): LitSearch matches gold papers against a
*fixed* arxiv corpus by exact paper id. This repo retrieves over the open web
via search-provider APIs, so there is no closed corpus to index into. We
therefore score recall by matching each gold paper against the retrieved
documents through a parameter-free proxy -- normalized-title containment and
arxiv-id overlap -- instead of a learned semantic matcher. The core signal
(fraction of gold papers surfaced) is preserved; the closed-corpus assumption
is intentionally dropped, and the QuotientAI dependency used by the document
relevance benchmark is avoided entirely so the scorer is self-contained.
"""

import json
import logging
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Arxiv ids come in two shapes: modern "2309.05389" (4 digits, dot, 4-5 digits)
# and legacy "cs.CL/0703123" (category, slash, 7 digits). Matched bare so they
# are found in gold identifiers ("arxiv:2309.05389 ...") and document URLs alike.
_ARXIV_ID_RE = re.compile(
    r"("
    r"\d{4}\.\d{4,5}"  # modern: 2309.05389
    r"|"
    r"[a-z\-]+(?:\.[a-z\-]+)?/\d{7}"  # legacy: cs.CL/0703123
    r")",
    re.IGNORECASE,
)
# Matches an "arxiv:ID" / "arxiv/abs/ID" token so it can be stripped when
# deriving a paper's title from a combined "arxiv:ID Title" gold string.
_ARXIV_TOKEN_RE = re.compile(
    r"arxiv[:\s/]*(?:abs/)?(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, drop non-alphanumerics, and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(text).lower())).strip()


def _arxiv_ids(text: str) -> set:
    """Extract normalized arxiv-style ids from arbitrary text.

    Punctuation is removed so "2201.11903" and "cs.cl/0703123" compare stably
    across gold identifiers and document URLs.
    """
    return {
        match.group(1).lower().replace("/", "").replace(".", "")
        for match in _ARXIV_ID_RE.finditer(str(text))
    }


def _title_blob(text: str) -> str:
    """Normalized title with any arxiv id stripped (for substring matching)."""
    return _normalize(_ARXIV_TOKEN_RE.sub(" ", str(text)))


def _doc_fingerprint(doc) -> Tuple[str, set]:
    """Return (normalized text blob, arxiv id set) for a retrieved document.

    Handlers' document-relevance path returns documents as either dicts
    (carrying title/url/content) or their stringified form; both are handled.
    """
    if isinstance(doc, dict):
        parts = [doc.get("title"), doc.get("url"), doc.get("content")]
        blob = " ".join(str(part) for part in parts if part)
    else:
        blob = str(doc)
    return _normalize(blob), _arxiv_ids(blob)


def recall_at_k(
    retrieved_docs: Sequence,
    gold_papers: Sequence[str],
    k: Optional[int] = None,
) -> Dict:
    """Compute recall of gold papers within the top-``k`` retrieved documents.

    A gold paper counts as retrieved if any ranked document shares an arxiv id
    with it, or contains its normalized title as a substring (length >= 8 so a
    trivially short gold blob cannot spuriously match).

    Args:
        retrieved_docs: Ranked retrieved documents (dicts or strings), best first.
        gold_papers: Ground-truth paper identifiers (titles and/or arxiv ids).
        k: Cutoff on the retrieved ranking. ``None`` uses all retrieved docs.

    Returns:
        Dict with ``recall`` in [0, 1], ``matched`` (gold identifiers found),
        ``gold_count``, and the effective ``k`` used.
    """
    ranked = list(retrieved_docs)[:k] if k is not None else list(retrieved_docs)
    fingerprints = [_doc_fingerprint(doc) for doc in ranked]

    matched: List[str] = []
    for gold in gold_papers:
        g_title = _title_blob(gold)
        g_ids = _arxiv_ids(gold)
        hit = False
        if g_ids:
            hit = any(g_ids & d_ids for _blob, d_ids in fingerprints)
        if not hit and len(g_title) >= 8:
            hit = any(g_title in d_blob for d_blob, _ids in fingerprints)
        if hit:
            matched.append(gold)

    gold_count = len(gold_papers)
    recall = round(len(matched) / gold_count, 3) if gold_count else 0.0
    effective_k = k if k is not None else len(ranked)
    return {
        "recall": recall,
        "matched": matched,
        "gold_count": gold_count,
        "k": effective_k,
    }


def load_litsearch_data(
    json_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    random_sample: Optional[int] = None,
) -> pd.DataFrame:
    """Load a LitSearch-style dataset (JSON) into a DataFrame.

    Each item carries a ``question`` and a ``gold_papers`` list (the paper
    identifiers that should be retrieved). Returns a DataFrame with
    ``problem``, ``answer`` (gold joined for compatibility), ``gold_papers``
    (the list), and ``index`` columns, matching the shape the rest of the
    pipeline expects from the other loaders.
    """
    logger.info(f"Loading LitSearch data from JSON file: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset = data["dataset"] if isinstance(data, dict) and "dataset" in data else data
    total = len(dataset)

    if random_sample is not None and random_sample > 0:
        size = min(random_sample, total)
        dataset = random.sample(dataset, size)
        base = 0
    else:
        if end_index is None:
            end_index = total
        start_index = max(0, min(start_index, total - 1))
        end_index = max(start_index + 1, min(end_index, total))
        dataset = dataset[start_index:end_index]
        base = start_index

    rows = []
    for i, item in enumerate(dataset):
        gold = item.get("gold_papers") or item.get("answer_papers") or []
        rows.append({
            "problem": item["question"],
            "answer": " | ".join(gold),
            "gold_papers": list(gold),
            "index": base + i,
        })
    return pd.DataFrame(rows)
