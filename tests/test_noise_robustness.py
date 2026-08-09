"""Tests for the noise-robustness evaluation (PredAct-Bench inspired).

The integration test drives the real call site
``run_evaluation.evaluate_provider_simple_qa`` with fakes for the search
handler and the LLM-backed extractor/evaluator, so it exercises the wiring
without network or API keys.
"""

import asyncio

import run_evaluation
from utils.noise_robustness import (
    apply_noise,
    compute_robustness,
    parse_documents,
    render_documents,
)
from utils.utils import EvaluationType

GOLD = "Michio Sugeno"


def _doc_string(answer_at_end=True):
    """A formatted retrieved-doc string with the gold answer buried at the end."""
    padding = "Background context about IEEE awards and prior recipients. " * 20
    tail = f" The 2010 IEEE Frank Rosenblatt Award was received by {GOLD}."
    content = (padding + tail) if answer_at_end else (tail + padding)
    return (
        "\n**Document 1.** Source: https://example.org/award\n"
        f"Content: {content}"
    )


class _FakePostProcessor:
    """Stand-in for PostProcessor.extract_answer.

    Returns the gold answer only when it is still present in the (possibly
    degraded) context, mirroring how truncation removes the answer-bearing
    span.
    """

    def extract_answer(self, query, is_llm_response, search_result):
        if GOLD in search_result:
            return GOLD
        return "unknown"


class _FakeEvaluator:
    """Stand-in for CorrectnessEvaluator: exact (case-insensitive) match."""

    def __init__(self, *args, **kwargs):
        pass

    async def evaluate(self, inputs, outputs, reference_outputs):
        predicted = outputs["answer"].strip().lower()
        gold = reference_outputs["answer"].strip().lower()
        correct = predicted == gold
        return {"score": 1.0 if correct else 0.0, "value": "CORRECT" if correct else "INCORRECT"}


class _FakeSearchHandler:
    is_llm_response = False

    def __init__(self, doc_string):
        self._doc_string = doc_string

    async def search(self, query):
        return {"answer": "", "search_response": {"results": []}}

    async def post_process(self, search_result, evaluation_type=None):
        return self._doc_string, 0, 0


def test_apply_noise_truncation_removes_answer_span():
    original = _doc_string(answer_at_end=True)
    degraded = apply_noise(original, noise_ratio=0.6, strategy="truncate", seed=0)
    assert GOLD in original
    # 60% truncation shears off the answer-bearing tail.
    assert GOLD not in degraded
    assert len(degraded) < len(original)


def test_apply_noise_inject_adds_distractors():
    original = _doc_string()
    before = len(parse_documents(original))
    degraded = apply_noise(original, noise_ratio=0.5, strategy="inject", seed=1)
    after = len(parse_documents(degraded))
    assert after == before + 1


def test_apply_noise_off_returns_input_unchanged():
    original = _doc_string()
    assert apply_noise(original, noise_ratio=0.0) == original
    # Free-text (no document structure) falls back to character truncation.
    assert apply_noise("a plain answer string", noise_ratio=0.5) == "a plain answer"[:10]


def test_parse_render_roundtrip():
    docs = parse_documents(_doc_string())
    assert len(docs) == 1
    assert docs[0][0] == "https://example.org/award"
    re_parsed = parse_documents(render_documents(docs))
    assert re_parsed == docs


def test_compute_robustness_metrics():
    results = [
        {"index": 0, "is_correct": True, "noisy_is_correct": True},
        {"index": 1, "is_correct": True, "noisy_is_correct": False},
        {"index": 2, "is_correct": False, "noisy_is_correct": False},
        {"index": 3, "is_correct": True, "noisy_is_correct": None},  # skipped
    ]
    metrics = compute_robustness(results)
    # 3 paired examples; 2 clean-correct, 1 stays correct under noise.
    assert metrics["n"] == 3
    assert metrics["accuracy_clean"] == 0.667
    assert metrics["accuracy_noisy"] == 0.333
    assert metrics["robustness_drop"] == 0.333
    assert metrics["relative_robustness"] == 0.5
    assert metrics["flipped_to_incorrect"] == 1


def test_integration_noise_pass_degrades_correctness(tmp_path, monkeypatch):
    """Drive the real call site end-to-end with noise on.

    Clean: the answer is present, extraction + grading succeed.
    Noisy (truncate 0.6): the answer span is removed, the noisy re-grade
    fails, so relative robustness collapses to 0.
    """
    monkeypatch.setattr(run_evaluation, "CorrectnessEvaluator", _FakeEvaluator)
    monkeypatch.setattr(run_evaluation.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(run_evaluation, "output_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(run_evaluation, "evaluation_type", EvaluationType.SIMPLEQA, raising=False)

    handler = _FakeSearchHandler(_doc_string(answer_at_end=True))
    examples = [{"question": "Who received the IEEE Frank Rosenblatt Award in 2010?",
                 "answer": GOLD, "index": 0}]

    summary = asyncio.run(run_evaluation.evaluate_provider_simple_qa(
        provider_name="fake",
        search_handler=handler,
        examples=examples,
        post_processor=_FakePostProcessor(),
        evaluator_model="ignored",
        batch_size=3,
        noise_ratio=0.6,
        noise_strategy="truncate",
    ))

    # Clean pass still correct.
    assert summary["accuracy"] == 1.0
    assert summary["results"][0]["is_correct"] is True
    # Noisy pass degraded this example.
    assert summary["results"][0]["noisy_is_correct"] is False
    assert summary["noisy_accuracy"] == 0.0
    assert summary["robustness_drop"] == 1.0
    assert summary["relative_robustness"] == 0.0
    assert summary["flipped_to_incorrect"] == 1


def test_integration_noise_off_leaves_clean_path_unchanged(tmp_path, monkeypatch):
    """With noise_ratio == 0 the existing path is unchanged."""
    monkeypatch.setattr(run_evaluation, "CorrectnessEvaluator", _FakeEvaluator)
    monkeypatch.setattr(run_evaluation.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(run_evaluation, "output_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(run_evaluation, "evaluation_type", EvaluationType.SIMPLEQA, raising=False)

    handler = _FakeSearchHandler(_doc_string(answer_at_end=False))
    examples = [{"question": "q?", "answer": GOLD, "index": 0}]

    summary = asyncio.run(run_evaluation.evaluate_provider_simple_qa(
        provider_name="fake",
        search_handler=handler,
        examples=examples,
        post_processor=_FakePostProcessor(),
        evaluator_model="ignored",
    ))

    assert summary["accuracy"] == 1.0
    assert summary["results"][0]["noisy_is_correct"] is None
    assert summary["noisy_accuracy"] is None
    assert summary["relative_robustness"] is None
