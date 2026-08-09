"""Noise-robustness evaluation for tool-augmented QA.

Adapted from *PredAct-Bench: Benchmarking Tool-Augmented Dialogue under
Controlled Tool Noise* (arxiv:2608.02372). PredAct-Bench's central
operational idea is to inject *statistically controlled noise* into a
tool's output and measure how an agent's downstream reliance / correctness
shifts as the noise grows (its Relative AI-Reliance, RAIR).

This module ports that core mechanism onto this repo's SimpleQA contract.
Here the "tool" is a search API: we take the documents a provider returned,
apply a controlled amount of degradation, re-run answer extraction +
correctness grading on the degraded context, and score how much of the
agent's correctness *survives* the noise. That yields a per-provider
noise-robustness signal this framework did not previously expose.

What is ported (core mechanism, full fidelity):
  - controlled, deterministic noise injection into retrieved documents
    (content truncation, irrelevant-document injection, document drop);
  - a reliance-shift score: among answers the agent got right with the
    clean tool, what fraction stay right under tool noise (the repo analog
    of RAIR).

What is intentionally NOT ported (paper auxiliaries with no call site here):
  - the educational decision-support framing (OULAD / PREDACT-CS datasets);
  - the multi-turn RAIR/RSR trust-calibration framework and human study.
This repo evaluates batch web-search APIs, not classroom agents, so those
are out of scope.
"""

import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# A retrieved document is rendered (see
# ProviderHandler._format_search_results_for_prompt) as:
#     \n**Document N.** Source: <url>\nContent: <text>
# joined across documents by newlines.
_BLOCK_SPLIT_RE = re.compile(r"\*\*Document\s+\d+\.\*\*")
_SOURCE_CONTENT_RE = re.compile(
    r"\s*Source:\s*(?P<source>.*?)\nContent:\s*(?P<content>.*)", re.DOTALL
)

# Off-topic distractors used by the "inject" strategy. Deterministic content
# so noise is reproducible; deliberately unrelated to any likely query.
DEFAULT_IRRELEVANT_DOCS: Sequence[Tuple[str, str]] = (
    (
        "https://example.invalid/lorem-1",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
    ),
    (
        "https://example.invalid/lorem-2",
        "The quick brown fox jumps over the lazy dog. Pack my box with "
        "five dozen liquor jugs. Sphinx of black quartz, judge my vow.",
    ),
    (
        "https://example.invalid/lorem-3",
        "A panel of experts reviewed the annual report on regional "
        "weather patterns and submitted their findings to the committee.",
    ),
)


def _clip01(value: float) -> float:
    """Clamp a noise ratio into [0.0, 1.0]."""
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def parse_documents(doc_string: str) -> List[Tuple[str, str]]:
    """Parse a formatted document string into ``(source, content)`` tuples.

    Returns an empty list when the string has no ``**Document N.**`` markers
    (e.g. a provider's direct free-text answer), which lets callers decide
    on a fallback degradation strategy.
    """
    if not doc_string:
        return []
    docs: List[Tuple[str, str]] = []
    for part in _BLOCK_SPLIT_RE.split(doc_string)[1:]:
        match = _SOURCE_CONTENT_RE.match(part)
        if match:
            docs.append((match.group("source").strip(), match.group("content").strip()))
    return docs


def render_documents(docs: Iterable[Tuple[str, str]]) -> str:
    """Render ``(source, content)`` tuples back into the prompt format."""
    return "\n".join(
        f"\n**Document {i + 1}.** Source: {source}\nContent: {content}"
        for i, (source, content) in enumerate(docs)
    )


def _truncate(docs: List[Tuple[str, str]], noise_ratio: float) -> List[Tuple[str, str]]:
    keep_fraction = 1.0 - noise_ratio
    truncated: List[Tuple[str, str]] = []
    for source, content in docs:
        keep = max(1, int(len(content) * keep_fraction))
        truncated.append((source, content[:keep]))
    return truncated


def _drop(docs: List[Tuple[str, str]], noise_ratio: float, rng: random.Random) -> List[Tuple[str, str]]:
    n_drop = int(round(len(docs) * noise_ratio))
    if n_drop <= 0:
        return list(docs)
    drop_idx = set(rng.sample(range(len(docs)), min(n_drop, len(docs))))
    return [doc for i, doc in enumerate(docs) if i not in drop_idx]


def _inject(
    docs: List[Tuple[str, str]],
    noise_ratio: float,
    rng: random.Random,
    irrelevant_docs: Optional[Sequence[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    pool = list(irrelevant_docs) if irrelevant_docs else list(DEFAULT_IRRELEVANT_DOCS)
    if not pool:
        return list(docs)
    n_add = max(1, int(round(len(docs) * noise_ratio)))
    chosen = rng.sample(pool, min(n_add, len(pool)))
    return [*docs, *chosen]


_STRATEGIES = {"truncate": _truncate, "drop": _drop, "inject": _inject}


def apply_noise(
    doc_string: str,
    noise_ratio: float = 0.5,
    strategy: str = "truncate",
    seed: int = 0,
    irrelevant_docs: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    """Apply controlled noise to a formatted document string.

    Args:
        doc_string: The post-processed search context fed to answer
            extraction (the output of ``ProviderHandler.post_process``).
        noise_ratio: Fraction of degradation in ``[0.0, 1.0]``. ``0.0``
            returns the input unchanged.
        strategy: ``"truncate"`` (default), ``"inject"``, ``"drop"``, or a
            ``"+"``-joined combination such as ``"truncate+inject"``.
        seed: Seed for the deterministic RNG used by ``drop`` / ``inject``.
        irrelevant_docs: Optional pool of distractor documents for
            ``"inject"``; defaults to :data:`DEFAULT_IRRELEVANT_DOCS`.

    Returns:
        A degraded document string. When the input has no document
        structure (e.g. a provider's direct answer), it is degraded by
        character truncation instead.
    """
    if not doc_string:
        return doc_string
    noise_ratio = _clip01(noise_ratio)
    if noise_ratio <= 0.0:
        return doc_string

    docs = parse_documents(doc_string)
    if not docs:
        # Free-text tool output: degrade by truncation.
        keep = max(1, int(len(doc_string) * (1.0 - noise_ratio)))
        return doc_string[:keep]

    rng = random.Random(seed)
    out = list(docs)
    for name in (strategy or "truncate").split("+"):
        fn = _STRATEGIES.get(name.strip())
        if fn is None:
            continue
        if name.strip() in ("drop", "inject"):
            out = fn(out, noise_ratio, rng, irrelevant_docs)  # type: ignore[arg-type]
        else:
            out = fn(out, noise_ratio)  # type: ignore[arg-type]
        if not out:
            break
    return render_documents(out)


async def run_noisy_pass(
    query: str,
    search_ans: str,
    is_llm_response: bool,
    post_processor,
    evaluator,
    reference_answer: str,
    noise_ratio: float = 0.5,
    strategy: str = "truncate",
    seed: int = 0,
    irrelevant_docs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict:
    """Re-extract and re-grade a single example under controlled tool noise.

    Mirrors the clean pass in ``evaluate_provider_simple_qa``: perturb the
    search context, extract an answer from the degraded context, and grade
    it against the reference. Returns whether the noisy answer is correct.
    """
    noisy_context = apply_noise(
        search_ans,
        noise_ratio=noise_ratio,
        strategy=strategy,
        seed=seed,
        irrelevant_docs=irrelevant_docs,
    )
    noisy_answer = post_processor.extract_answer(
        query=query,
        is_llm_response=is_llm_response,
        search_result=noisy_context,
    )
    noisy_eval = await evaluator.evaluate(
        {"question": query},
        {"answer": noisy_answer},
        {"answer": reference_answer},
    )
    return {
        "noisy_answer": noisy_answer,
        "noisy_score": noisy_eval["score"],
        "noisy_is_correct": noisy_eval["score"] == 1.0,
    }


def compute_robustness(results: Sequence[Dict]) -> Dict:
    """Score noise-robustness from per-example clean + noisy correctness.

    Reads ``is_correct`` (clean) and ``noisy_is_correct`` (noisy) off each
    result. Examples where the noisy pass did not run (``noisy_is_correct``
    is ``None`` / absent) are ignored.

    ``relative_robustness`` is the repo analog of PredAct-Bench's Relative
    AI-Reliance (RAIR): among answers the agent got right with the clean
    tool, the fraction that *stay* right under controlled tool noise.
    """
    paired: List[Tuple[bool, bool]] = []
    for r in results:
        noisy = r.get("noisy_is_correct")
        if noisy is None:
            continue
        paired.append((bool(r.get("is_correct")), bool(noisy)))

    n = len(paired)
    if n == 0:
        return {
            "n": 0,
            "accuracy_clean": 0.0,
            "accuracy_noisy": 0.0,
            "robustness_drop": 0.0,
            "relative_robustness": 0.0,
            "flipped_to_incorrect": 0,
        }

    clean_correct = sum(c for c, _ in paired)
    noisy_correct = sum(ny for _, ny in paired)
    flipped = sum(1 for c, ny in paired if c and not ny)
    acc_clean = clean_correct / n
    acc_noisy = noisy_correct / n
    relative = (noisy_correct / clean_correct) if clean_correct else 0.0

    return {
        "n": n,
        "accuracy_clean": round(acc_clean, 3),
        "accuracy_noisy": round(acc_noisy, 3),
        "robustness_drop": round(acc_clean - acc_noisy, 3),
        "relative_robustness": round(relative, 3),
        "flipped_to_incorrect": flipped,
    }
