"""Tests for the DeepStress-derived evidence-stress capability.

These tests exercise the new ``utils.evidence_stress`` module *through* existing
repo utilities (``utils.EvaluationType``, ``utils.utils.load_document_relevance_eval_data``)
to prove the capability integrates with the repo's real data contract, not just
in isolation.
"""

import os

# Imported from NON-NEW repo modules -- this is the integration anchor.
from utils import EvaluationType
from utils.utils import load_document_relevance_eval_data

# The new capability module under test.
from utils.evidence_stress import (
    ALL_DIMENSIONS,
    DistractorDimension,
    inject_noise,
    make_distractors,
    relevance_proxy,
    stress_robustness,
    stress_sweep,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(REPO_ROOT, "datasets", "document_relevance_dynamic_test_set.json")

QUESTION = "What is the top emerging technology in 2025?"
ANSWER = "Generative AI is the top emerging technology in 2025."
CONTEXT = (
    "Generative AI is dominating as a key technology trend in 2025. "
    "It reshapes industries through highly sophisticated human-like content."
)


def test_relevance_proxy_distinguishes_relevant_from_offtopic():
    """The parameter-free proxy marks the relevant passage and rejects filler."""
    assert relevance_proxy(CONTEXT, QUESTION, ANSWER) is True
    off_topic = "A guide to growing tomatoes and basil in raised garden beds."
    assert relevance_proxy(off_topic, QUESTION, ANSWER) is False


def test_distractor_dimensions_are_distinct_signals():
    """Relevance distractors are caught; trust/factuality ones slip a topical filter.

    This is the DeepStress insight materialized: a relevance-only signal cannot
    see trustworthiness or factuality defects, so those distractors pass.
    """
    distractors = make_distractors(QUESTION, ANSWER, ALL_DIMENSIONS)
    assert len(distractors) == len(ALL_DIMENSIONS)
    by_dim = dict(zip(ALL_DIMENSIONS, distractors))
    assert relevance_proxy(by_dim[DistractorDimension.RELEVANCE], QUESTION, ANSWER) is False
    assert relevance_proxy(by_dim[DistractorDimension.TRUSTWORTHINESS], QUESTION, ANSWER) is True
    assert relevance_proxy(by_dim[DistractorDimension.FACTUALITY], QUESTION, ANSWER) is True


def test_inject_noise_preserves_baseline_and_is_deterministic():
    """Injection never mutates the input, always keeps baseline docs, and repeats."""
    baseline = [CONTEXT, "Generative AI content generation trends continue in 2025."]
    perturbed = inject_noise(baseline, QUESTION, ANSWER, noise_ratio=0.3, seed=7)

    # Baseline documents survive; perturbed set grew.
    assert all(d in perturbed for d in baseline)
    assert len(perturbed) > len(baseline)
    # Input was not mutated.
    assert baseline == [CONTEXT, "Generative AI content generation trends continue in 2025."]
    # Deterministic for a fixed seed.
    again = inject_noise(baseline, QUESTION, ANSWER, noise_ratio=0.3, seed=7)
    assert perturbed == again


def test_frequency_scales_with_noise_ratio():
    """Distractor frequency tracks the controlled-frequency axis on realistic sets."""
    baseline = ["relevant doc"] * 10
    seen = set()
    for ratio in [0.1, 0.2, 0.3, 0.5]:
        perturbed = inject_noise(baseline, QUESTION, ANSWER, noise_ratio=ratio, seed=0)
        frequency = (len(perturbed) - 10) / len(perturbed)
        # Frequency should be within ~0.12 of the requested ratio.
        assert abs(frequency - ratio) < 0.12
        seen.add(round(frequency, 2))
    # Different ratios produce different frequencies (the curve is not flat).
    assert len(seen) >= 3


def test_stress_robustness_degrades_under_noise():
    """Relevance degrades and the robustness retention ratio is well-formed."""
    result = stress_robustness([CONTEXT], QUESTION, ANSWER, noise_ratio=0.3, seed=0)
    assert result["baseline_relevant_pct"] == 100.0
    assert result["stress_relevant_pct"] < result["baseline_relevant_pct"]
    assert result["degradation_pp"] > 0.0
    assert 0.0 <= result["robustness"] <= 1.0
    assert result["injected_docs"] == result["stress_docs"] - result["baseline_docs"]


def test_stress_sweep_is_ordered_by_noise_ratio():
    sweep = stress_sweep([CONTEXT], QUESTION, ANSWER, noise_ratios=[0.0, 0.2, 0.5], seed=0)
    assert [row["noise_ratio"] for row in sweep] == [0.0, 0.2, 0.5]
    # Zero noise means no degradation.
    assert sweep[0]["degradation_pp"] == 0.0
    assert sweep[0]["robustness"] == 1.0


def test_integration_with_existing_data_loader():
    """The capability runs end-to-end on documents from the repo's data loader.

    This is the integration test: it pulls real examples through the existing
    ``load_document_relevance_eval_data`` utility and feeds them to the new
    stress capability, asserting a well-formed robustness result.
    """
    assert EvaluationType.DOCUMENT_RELEVANCE.value == "document_relevance"
    df = load_document_relevance_eval_data(DATASET_PATH, start_index=0, end_index=3)
    assert len(df) > 0

    row = df.iloc[0]
    question, answer = row["problem"], row["answer"]
    # Treat the ground-truth answer as a controlled relevant document.
    result = stress_robustness([str(answer)], question, answer, noise_ratio=0.3, seed=0)
    assert result["baseline_relevant_pct"] == 100.0
    assert 0.0 <= result["robustness"] <= 1.0
    assert result["stress_docs"] > result["baseline_docs"]
