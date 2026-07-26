"""Controlled poor-quality-evidence stress testing for document relevance.

Adapted from *DeepStress: Stress-Testing Deep Search Agents* (arXiv:2607.13920),
which replaces a search agent's retrieval module with a controlled synthetic
environment and dials the *frequency* of challenging evidence along three
dimensions -- trustworthiness, relevance, and factuality -- to measure how
robustly a system handles unreliable information.

This module ports that core mechanism onto this repo's document-relevance
contract (a list of retrieved documents per query). Instead of a live agent we
operate directly on the retrieved-document set: we inject poor-quality
"distractor" documents at a controlled frequency and measure how the relevance
signal degrades. The static ``relevant_docs_percentage`` the repo already
computes becomes a *robustness* measurement.

Mode 2 (adapted port). The DeepStress dimensions and the controlled-frequency
injection are kept at full fidelity. The auxiliary component -- the paper's
learned / QuotientAI relevance judgments -- is replaced by a parameter-free
vocabulary-overlap relevance proxy (see :func:`relevance_proxy`). No external
services or model calls are required, so the stress measurement is runnable in
tests and CI without API keys.
"""

import logging
import random
import re
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# A compact English stoplist keeps the relevance proxy parameter-free. It is
# intentionally generic (no domain words) so it does not bias any dataset.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into over
    under is are was were be been being this that these those it its as not no
    yes which who whom whose what when where why how all any each few more most
    other some such only own same so than too very can will just do does did has
    have had having about above below between through during after before up down
    out off again further here there
    """.split()
)

# Default sweep of challenging-evidence frequencies, mirroring DeepStress's
# "control the frequency of challenging evidence" experiment axis.
DEFAULT_NOISE_RATIOS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# Minimum fraction of the reference vocabulary a document must share to count as
# relevant under the parameter-free proxy.
DEFAULT_RELEVANCE_THRESHOLD = 0.2


class DistractorDimension(str, Enum):
    """Dimensions of poor-quality evidence, named after DeepStress's controls."""

    TRUSTWORTHINESS = "trustworthiness"
    RELEVANCE = "relevance"
    FACTUALITY = "factuality"


ALL_DIMENSIONS: Tuple[DistractorDimension, ...] = (
    DistractorDimension.RELEVANCE,
    DistractorDimension.TRUSTWORTHINESS,
    DistractorDimension.FACTUALITY,
)


def _content_words(text: str) -> Set[str]:
    """Lowercase alphanumeric tokens of length >= 3, minus stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _document_text(document) -> str:
    """Normalize a document (str or dict) to a flat string for scoring."""
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        # The repo's document_relevance path stringifies result dicts holding
        # keys like url/title/content; joining values covers either shape.
        return " ".join(str(v) for v in document.values())
    return str(document)


def make_distractors(
    question: str,
    answer: str,
    dimensions: Sequence[DistractorDimension] = ALL_DIMENSIONS,
) -> List[str]:
    """Build one poor-quality distractor document per requested dimension.

    Each distractor is constructed to be poor along exactly one DeepStress axis
    while being plausible enough to stress a relevance scorer:

    * ``RELEVANCE`` -- off-topic filler with near-zero topical overlap.
    * ``TRUSTWORTHINESS`` -- on-topic but laced with unverifiability markers
      (anonymous / unverified / disputed), so a relevance scorer that only looks
      at topical overlap is fooled.
    * ``FACTUALITY`` -- on-topic but asserts the *wrong* answer (it negates the
      known ground truth), so topical overlap hides a factual defect.

    Distractors are deterministic and parameter-free (no model calls).
    """
    q = (question or "").strip()
    a = (answer or "").strip()
    out: List[str] = []
    for dim in dimensions:
        if dim == DistractorDimension.RELEVANCE:
            out.append(
                "Source: unrelated-archive.example\n"
                "Content: A seasonal guide to companion planting in raised "
                "vegetable beds. Tomatoes pair well with basil, while carrots "
                "and onions help deter common garden pests through the summer."
            )
        elif dim == DistractorDimension.TRUSTWORTHINESS:
            out.append(
                "Source: anonymous-aggregator.example\n"
                f"Content: Unverified posts claim to address the question: \"{q}\". "
                "The reliability of these claims is unknown, the original sources "
                "cannot be confirmed, and several anonymous reviewers dispute the "
                "published conclusions. No peer-reviewed corroboration exists, so "
                "treat everything below with caution."
            )
        elif dim == DistractorDimension.FACTUALITY:
            out.append(
                "Source: opinion-blog.example\n"
                f"Content: Regarding \"{q}\", the commonly published answer that "
                f"\"{a}\" is mistaken. Independent reviewers assert the correct "
                f"answer differs and that \"{a}\" is inaccurate; they instead "
                "promote a conflicting conclusion without supporting evidence."
            )
    return out


def inject_noise(
    documents: Sequence,
    question: str,
    answer: str,
    noise_ratio: float = 0.3,
    dimensions: Sequence[DistractorDimension] = ALL_DIMENSIONS,
    seed: int = 0,
) -> List:
    """Return a copy of ``documents`` with poor-quality evidence injected.

    ``noise_ratio`` is the target *frequency* of challenging evidence in the
    perturbed set (fraction of documents that are distractors), following
    DeepStress's controlled-frequency design. The number of injected distractors
    is ``k = round(noise_ratio * n / (1 - noise_ratio))`` so that ``k / (n + k)
    ~= noise_ratio`` for realistic set sizes; ``k`` is floored at 1 whenever
    stress is requested so the perturbed set is never identical to the baseline.
    Distractors cycle through ``dimensions`` in order, so all requested axes are
    represented once ``k`` reaches the number of dimensions (i.e. at higher
    frequencies), and they are deterministically interleaved with the baseline
    documents using ``seed``.

    The input list is never mutated.
    """
    baseline = list(documents)
    n = len(baseline)
    if n == 0 or noise_ratio <= 0 or not dimensions:
        return baseline

    pool = make_distractors(question, answer, dimensions)
    if not pool:
        return baseline

    if noise_ratio >= 1.0:
        k = max(len(pool), n)
    else:
        k = int(round(noise_ratio * n / (1.0 - noise_ratio)))
        k = max(1, k)

    distractors = [pool[i % len(pool)] for i in range(k)]
    perturbed = baseline + distractors
    random.Random(seed).shuffle(perturbed)
    return perturbed


def relevance_proxy(
    document,
    question: str,
    answer: str,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> bool:
    """Parameter-free relevance judgment (Mode 2 substitution for QuotientAI).

    A document is relevant when it shares at least ``threshold`` of the
    reference vocabulary built from the query and ground-truth answer. This is
    deliberately a *topical* signal only -- it cannot see trustworthiness or
    factuality defects, which is exactly the gap DeepStress exposes: a
    trustworthiness or factuality distractor that is on-topic slips through.
    """
    reference = _content_words(question) | _content_words(answer)
    if not reference:
        # No signal to judge against; do not penalize.
        return bool(_document_text(document).strip())
    doc_words = _content_words(_document_text(document))
    overlap = len(reference & doc_words) / len(reference)
    return overlap >= threshold


def stress_robustness(
    documents: Sequence,
    question: str,
    answer: str,
    noise_ratio: float = 0.3,
    dimensions: Sequence[DistractorDimension] = ALL_DIMENSIONS,
    seed: int = 0,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> Dict:
    """Measure how relevance holds up under one challenging-evidence frequency.

    Returns baseline/stress relevant-document percentages, the percentage-point
    degradation, and a ``robustness`` retention ratio in ``[0, 1]`` (1.0 = no
    relevance lost; 0.0 = all relevance lost). Counts and the applied
    parameters are included so the result is self-describing for downstream
    aggregation.
    """
    docs = list(documents)
    n = len(docs)
    perturbed = inject_noise(docs, question, answer, noise_ratio, dimensions, seed)

    baseline_relevant = sum(
        1 for d in docs if relevance_proxy(d, question, answer, threshold)
    )
    stress_relevant = sum(
        1 for d in perturbed if relevance_proxy(d, question, answer, threshold)
    )
    baseline_pct = (baseline_relevant / n * 100.0) if n else 0.0
    stress_pct = (stress_relevant / len(perturbed) * 100.0) if perturbed else 0.0
    degradation = baseline_pct - stress_pct
    robustness = (stress_pct / baseline_pct) if baseline_pct > 0 else 0.0

    return {
        "noise_ratio": noise_ratio,
        "dimensions": [d.value for d in dimensions],
        "baseline_docs": n,
        "stress_docs": len(perturbed),
        "injected_docs": len(perturbed) - n,
        "baseline_relevant_pct": round(baseline_pct, 2),
        "stress_relevant_pct": round(stress_pct, 2),
        "degradation_pp": round(degradation, 2),
        "robustness": round(max(0.0, min(1.0, robustness)), 3),
        "seed": seed,
    }


def stress_sweep(
    documents: Sequence,
    question: str,
    answer: str,
    noise_ratios: Sequence[float] = DEFAULT_NOISE_RATIOS,
    dimensions: Sequence[DistractorDimension] = ALL_DIMENSIONS,
    seed: int = 0,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> List[Dict]:
    """Robustness curve across increasing challenging-evidence frequencies.

    This is the headline DeepStress result translated to this repo: how the
    relevance signal degrades as the frequency of poor-quality evidence rises.
    Returns one :func:`stress_robustness` dict per noise ratio, in order.
    """
    return [
        stress_robustness(documents, question, answer, r, dimensions, seed, threshold)
        for r in noise_ratios
    ]
