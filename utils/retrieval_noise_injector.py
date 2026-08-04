"""Controlled retrieval-noise injection and downstream-degradation measurement.

Adapted from PredAct-Bench (arXiv:2608.02372), "Benchmarking Tool-Augmented
Dialogue under Controlled Tool Noise". That paper's portable contribution is
its controlled tool-noise framework -- an "accuracy injector" that perturbs a
tool's output by a configurable amount -- together with the downstream-
degradation measurement that quantifies how much that noise hurts the agent.
Here the "tool output" is the post-processed search-result text (documents /
answer) that each handler produces in the SimpleQA pipeline, so a provider's
base quality can be stress-tested along a new "robustness under retrieval
noise" axis on top of the existing accuracy number.

Mode 2 (adapted port). The CORE mechanism (controlled noise injection at a
configurable ratio + the clean-vs-noisy degradation measurement) is kept at
full fidelity. AUXILIARY components are substituted with target-native
equivalents:

  * Grading is delegated to the repo's existing ``CorrectnessEvaluator``
    (see ``make_correctness_grader``) instead of being reimplemented.
  * PredAct-Bench's RAIR / RSR human-trust metrics and its educational
    dialogue / human-study scaffolding are intentionally NOT ported -- they
    are specific to the human-decision-making setting and have no analog in a
    search-API evaluation repo. The reported axis is accuracy degradation
    (downstream-degradation), the repo-native signal.
  * The noise target is the retrieved/answer text rather than a numeric
    predictor, matching this repo's documents-in / answer-out contract.
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

__all__ = [
    "NoiseMode",
    "NoiseConfig",
    "RetrievalNoiseInjector",
    "measure_noise_degradation",
    "make_correctness_grader",
    "evaluate_noise_robustness",
]


# Distractor prose used by the INJECT_IRRELEVANT mode. Stand-in for irrelevant
# documents mixed into a retrieval result.
_DEFAULT_IRRELEVANT_POOL: List[str] = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Annual rainfall in the region averages roughly six hundred millimetres.",
    "A nearby bakery opens at seven in the morning on most weekdays.",
    "Tickets for the local tour can be reserved online up to a week ahead.",
]


class NoiseMode(str, Enum):
    """Kind of controlled noise applied to the retrieved text."""

    TRUNCATE = "truncate"
    # Cut a fraction of characters from the end -- models lost key info.

    DROP = "drop"
    # Remove a fraction of words -- simulates missing documents.

    SHUFFLE = "shuffle"
    # Reorder a fraction of words -- simulates scrambled retrieval order.

    INJECT_IRRELEVANT = "inject_irrelevant"
    # Append irrelevant distractor text -- simulates off-topic documents.


@dataclass
class NoiseConfig:
    """Configuration for a single noise-injection operation.

    Args:
        noise_ratio: Fraction of the text to degrade, in ``[0.0, 1.0]``.
            ``0.0`` leaves the text untouched; ``1.0`` degrades it maximally.
        mode: Which :class:`NoiseMode` to apply.
        seed: Seed for the (otherwise stochastic) DROP / SHUFFLE /
            INJECT_IRRELEVANT modes. Fixed by default so a run is reproducible.
        irrelevant_pool: Source distractor sentences for INJECT_IRRELEVANT.
    """

    noise_ratio: float = 0.5
    mode: NoiseMode = NoiseMode.TRUNCATE
    seed: Optional[int] = 0
    irrelevant_pool: List[str] = field(
        default_factory=lambda: list(_DEFAULT_IRRELEVANT_POOL)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.noise_ratio <= 1.0:
            raise ValueError(
                f"noise_ratio must be in [0.0, 1.0], got {self.noise_ratio}"
            )
        # Accept a raw string from JSON configs / argparse.
        if isinstance(self.mode, str):
            self.mode = NoiseMode(self.mode)


class RetrievalNoiseInjector:
    """Apply a controlled amount of noise to a retrieved-text string.

    This is PredAct-Bench's "accuracy injector" adapted to the document/answer
    text surface of this repo's search handlers.
    """

    def __init__(self, config: Optional[NoiseConfig] = None) -> None:
        self.config = config or NoiseConfig()

    def inject(self, text: str, config: Optional[NoiseConfig] = None) -> str:
        """Return a noise-degraded copy of ``text``.

        ``config`` overrides the injector's default for this call only.
        """
        cfg = config or self.config
        if not text or cfg.noise_ratio <= 0.0:
            return text

        rng = random.Random(cfg.seed)
        mode = cfg.mode

        if mode == NoiseMode.TRUNCATE:
            keep = max(0, math.ceil(len(text) * (1.0 - cfg.noise_ratio)))
            return text[:keep]

        if mode == NoiseMode.DROP:
            words = text.split()
            if not words:
                return text
            n_drop = round(len(words) * cfg.noise_ratio)
            if n_drop <= 0:
                return text
            drop_idx = set(
                rng.sample(range(len(words)), min(n_drop, len(words)))
            )
            return " ".join(
                w for i, w in enumerate(words) if i not in drop_idx
            )

        if mode == NoiseMode.SHUFFLE:
            words = text.split()
            n_shuffle = round(len(words) * cfg.noise_ratio)
            if n_shuffle > 1 and len(words) > 1:
                idx = rng.sample(range(len(words)), min(n_shuffle, len(words)))
                subset = [words[i] for i in idx]
                rng.shuffle(subset)
                for i, replacement in zip(idx, subset):
                    words[i] = replacement
            return " ".join(words)

        if mode == NoiseMode.INJECT_IRRELEVANT:
            # Scale the volume of injected distractors with the noise ratio.
            n_inject = max(1, round(cfg.noise_ratio * 3))
            picks = [rng.choice(cfg.irrelevant_pool) for _ in range(n_inject)]
            return text + " " + " ".join(picks)

        raise ValueError(f"Unknown noise mode: {mode}")


# An async grader maps a (possibly noisy) answer text to a score in [0.0, 1.0].
AsyncGradeFn = Callable[[str], Awaitable[float]]


async def measure_noise_degradation(
    text: str,
    reference_answer: str,
    grade: AsyncGradeFn,
    config: Optional[NoiseConfig] = None,
) -> Dict[str, Any]:
    """Grade ``text`` clean and under noise; return the degradation.

    This is PredAct-Bench's downstream-degradation measurement: the agent is
    graded on the clean tool output and on a noise-degraded copy, and the
    drop in score characterises robustness to retrieval noise.

    Args:
        text: The clean tool output (retrieved documents / answer text).
        reference_answer: Gold answer, captured for the caller's convenience
            and typically closed over by ``grade``.
        grade: Async callable mapping an answer text to a score in [0.0, 1.0].
            Use :func:`make_correctness_grader` to build one from the repo's
            existing ``CorrectnessEvaluator``.
        config: Noise configuration. Defaults to ``NoiseConfig()``.

    Returns:
        Dict with ``clean_score``, ``noisy_score``, ``degradation``
        (``clean_score - noisy_score``), ``noisy_text``, plus the ``mode`` and
        ``noise_ratio`` that produced it.
    """
    cfg = config or NoiseConfig()
    injector = RetrievalNoiseInjector(cfg)

    clean_score = float(await grade(text))
    noisy_text = injector.inject(text)
    noisy_score = float(await grade(noisy_text))

    return {
        "reference_answer": reference_answer,
        "clean_score": clean_score,
        "noisy_score": noisy_score,
        "degradation": clean_score - noisy_score,
        "noisy_text": noisy_text,
        "mode": cfg.mode.value,
        "noise_ratio": cfg.noise_ratio,
    }


def make_correctness_grader(
    evaluator: Any, question: str, reference_answer: str
) -> AsyncGradeFn:
    """Adapt the repo's existing :class:`CorrectnessEvaluator` as a grader.

    The returned async callable grades an answer text against ``question`` /
    ``reference_answer`` using the evaluator's real SimpleQA grading prompt,
    returning its ``score`` (1.0 correct / 0.0 otherwise). This is the
    integration point between the noise framework and the existing pipeline.
    """
    async def grade(text: str) -> float:
        result = await evaluator.evaluate(
            {"question": question},
            {"answer": text},
            {"answer": reference_answer},
        )
        return float(result["score"])

    return grade


async def evaluate_noise_robustness(
    examples: List[Dict[str, str]],
    evaluator: Any,
    config: Optional[NoiseConfig] = None,
) -> Dict[str, Any]:
    """Aggregate clean-vs-noisy accuracy over a set of SimpleQA examples.

    Consumes the exact per-example shape the SimpleQA pipeline already emits
    (``question`` / ``predicted_answer`` / ``reference_answer``), so it can be
    run over a finished provider's results to add a robustness axis without
    re-issuing any search calls.

    Args:
        examples: List of dicts with ``question``, ``predicted_answer`` (the
            tool-output text to perturb) and ``reference_answer``.
        evaluator: Object exposing ``evaluate(inputs, outputs,
            reference_outputs)`` with the ``CorrectnessEvaluator`` contract.
        config: Noise configuration applied to every example.

    Returns:
        Dict with ``clean_accuracy``, ``noisy_accuracy``, ``accuracy_drop``
        and per-example ``results``.
    """
    cfg = config or NoiseConfig()
    total = len(examples)
    per_example: List[Dict[str, Any]] = []

    for example in examples:
        question = example["question"]
        text = example["predicted_answer"]
        reference_answer = example["reference_answer"]
        grade = make_correctness_grader(evaluator, question, reference_answer)
        per_example.append(
            await measure_noise_degradation(
                text, reference_answer, grade, cfg
            )
        )

    clean_correct = sum(r["clean_score"] >= 1.0 for r in per_example)
    noisy_correct = sum(r["noisy_score"] >= 1.0 for r in per_example)
    clean_accuracy = clean_correct / total if total else 0.0
    noisy_accuracy = noisy_correct / total if total else 0.0

    return {
        "clean_accuracy": round(clean_accuracy, 4),
        "noisy_accuracy": round(noisy_accuracy, 4),
        "accuracy_drop": round(clean_accuracy - noisy_accuracy, 4),
        "mode": cfg.mode.value,
        "noise_ratio": cfg.noise_ratio,
        "total_count": total,
        "results": per_example,
    }
