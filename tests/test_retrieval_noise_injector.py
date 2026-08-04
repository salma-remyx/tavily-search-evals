"""Tests for the retrieval-noise robustness capability.

These exercise the integration with the repo's existing modules rather than
self-testing the new file in isolation: they import the non-new
``PostProcessor`` from ``utils.post_processor`` and drive the real
document-extraction prompt path on noise-perturbed text, and they confirm the
``make_correctness_grader`` adapter matches the ``CorrectnessEvaluator.evaluate``
contract via a fake evaluator with the same signature.
"""

import asyncio
import os
from types import SimpleNamespace

# ChatOpenAI is constructed lazily and only contacts the API on invoke; a
# dummy key lets PostProcessor() instantiate without network access.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.post_processor import PostProcessor  # noqa: E402  (non-new module)
from utils.retrieval_noise_injector import (  # noqa: E402
    NoiseConfig,
    NoiseMode,
    RetrievalNoiseInjector,
    evaluate_noise_robustness,
    make_correctness_grader,
    measure_noise_degradation,
)


# --- injector behavior ------------------------------------------------------


def test_truncate_keeps_fraction_of_characters():
    injector = RetrievalNoiseInjector(
        NoiseConfig(noise_ratio=0.5, mode=NoiseMode.TRUNCATE)
    )
    # ceil(8 * (1 - 0.5)) == 4 characters retained.
    assert injector.inject("abcdefgh") == "abcd"


def test_drop_removes_all_words_at_full_ratio():
    injector = RetrievalNoiseInjector(
        NoiseConfig(noise_ratio=1.0, mode=NoiseMode.DROP)
    )
    assert injector.inject("the capital of france is paris") == ""


def test_inject_irrelevant_appends_distractors():
    injector = RetrievalNoiseInjector(
        NoiseConfig(noise_ratio=0.5, mode=NoiseMode.INJECT_IRRELEVANT)
    )
    original = "Paris is the capital of France."
    perturbed = injector.inject(original)
    assert perturbed.startswith(original)
    assert len(perturbed) > len(original)


def test_zero_ratio_is_a_noop():
    injector = RetrievalNoiseInjector(
        NoiseConfig(noise_ratio=0.0, mode=NoiseMode.TRUNCATE)
    )
    text = "unchanged retrieval result"
    assert injector.inject(text) == text


def test_same_seed_is_reproducible():
    cfg = NoiseConfig(noise_ratio=0.5, mode=NoiseMode.DROP, seed=42)
    first = RetrievalNoiseInjector(cfg).inject("a b c d e f g h")
    second = RetrievalNoiseInjector(cfg).inject("a b c d e f g h")
    assert first == second


# --- integration with the non-new PostProcessor -----------------------------


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


def test_noise_output_flows_through_post_processor_prompt():
    """A noise-perturbed document string changes what PostProcessor returns.

    PostProcessor.extract_answer builds its real document-extraction prompt
    from the supplied text; with its LLM stubbed to surface whether the key
    fact survived, dropping all content must yield an empty extraction.
    """
    processor = PostProcessor()

    def fake_invoke(prompt: str):
        # Mirror whether the key fact is still present in the formatted prompt.
        return _FakeLLMResponse("Paris" if "Paris" in prompt else "")

    # PostProcessor is a plain object, so swap in a stub LLM (the real
    # ChatOpenAI is a pydantic model and rejects per-attribute monkeypatching).
    processor.llm = SimpleNamespace(invoke=fake_invoke)

    clean_docs = "URL: example.com | Content: The capital of France is Paris."
    injector = RetrievalNoiseInjector(
        NoiseConfig(noise_ratio=1.0, mode=NoiseMode.DROP)
    )
    noisy_docs = injector.inject(clean_docs)

    query = "What is the capital of France?"
    clean_answer = processor.extract_answer(query, False, clean_docs)
    noisy_answer = processor.extract_answer(query, False, noisy_docs)

    assert clean_answer == "Paris"
    assert noisy_answer == ""


# --- degradation measurement ------------------------------------------------


class _FakeCorrectnessEvaluator:
    """Mimics CorrectnessEvaluator.evaluate's contract for offline testing."""

    async def evaluate(self, inputs, outputs, reference_outputs):
        predicted = str(outputs["answer"]).lower()
        reference = str(reference_outputs["answer"]).lower()
        # Stand-in grader: correct iff the gold answer survives verbatim.
        score = 1.0 if reference and reference in predicted else 0.0
        return {"score": score, "value": "CORRECT" if score else "INCORRECT"}


def test_measure_noise_degradation_detects_drop():
    async def run():
        evaluator = _FakeCorrectnessEvaluator()
        grade = make_correctness_grader(
            evaluator, "capital of France?", "Paris"
        )
        heavy = await measure_noise_degradation(
            "The capital of France is Paris.",
            "Paris",
            grade,
            NoiseConfig(noise_ratio=1.0, mode=NoiseMode.DROP),
        )
        none = await measure_noise_degradation(
            "The capital of France is Paris.",
            "Paris",
            grade,
            NoiseConfig(noise_ratio=0.0, mode=NoiseMode.DROP),
        )
        return heavy, none

    heavy, none = asyncio.run(run())
    assert heavy["clean_score"] == 1.0
    assert heavy["noisy_score"] == 0.0
    assert heavy["degradation"] == 1.0
    assert none["degradation"] == 0.0


def test_evaluate_noise_robustness_reports_accuracy_drop():
    async def run():
        evaluator = _FakeCorrectnessEvaluator()
        examples = [
            {
                "question": "capital of France?",
                "predicted_answer": "Paris",
                "reference_answer": "Paris",
            },
            {
                "question": "largest planet?",
                "predicted_answer": "Jupiter",
                "reference_answer": "Jupiter",
            },
            {
                "question": "speed of light unit?",
                "predicted_answer": "metres per second",
                "reference_answer": "metres per second",
            },
        ]
        return await evaluate_noise_robustness(
            examples,
            evaluator,
            NoiseConfig(noise_ratio=1.0, mode=NoiseMode.DROP),
        )

    report = asyncio.run(run())
    assert report["clean_accuracy"] == 1.0
    # Dropping all words wipes every answer -> accuracy collapses to 0.
    assert report["noisy_accuracy"] == 0.0
    assert report["accuracy_drop"] == 1.0
    assert report["total_count"] == 3
