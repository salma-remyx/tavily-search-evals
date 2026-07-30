"""Tests for the required-element coverage evaluator and its wiring.

The aggregation tests inject a fake LLM so the coverage logic runs without a
live OpenAI call. The wiring tests drive the real per-provider SimpleQA loop
in ``run_evaluation`` with stubbed collaborators to prove the coverage
evaluator is invoked (and skipped when an example declares no
``required_elements``), and that its result is folded into the per-example
output and CSV.
"""

import asyncio

from evaluators.required_coverage_evaluator import (
    ElementVerdict,
    RequiredCoverageConfig,
    RequiredCoverageEvaluator,
    RequiredCoverageGrade,
)
from utils import EvaluationType


class _FakeStructured:
    """Behaves like a ``with_structured_output`` runnable."""

    def __init__(self, grade):
        self._grade = grade

    def invoke(self, messages):
        return self._grade


class _FakeChat:
    """Behaves like a ``ChatOpenAI`` that always returns ``grade``."""

    def __init__(self, grade):
        self._grade = grade

    def with_structured_output(self, schema):
        return _FakeStructured(self._grade)


def _evaluator_for(covered_mask):
    """Build (elements, evaluator) where coverage follows ``covered_mask``."""
    elements = [f"element-{i}" for i in range(len(covered_mask))]
    grade = RequiredCoverageGrade(
        verdicts=[
            ElementVerdict(index=i + 1, covered=covered)
            for i, covered in enumerate(covered_mask)
        ]
    )
    return elements, RequiredCoverageEvaluator(
        RequiredCoverageConfig(), llm=_FakeChat(grade)
    )


def _run(coverage_evaluator, elements):
    return asyncio.run(
        coverage_evaluator.evaluate(
            {
                "question": "What permit and documents do I need?",
                "required_elements": elements,
                "retrieved_context": "context text",
            },
            {"answer": "predicted answer"},
        )
    )


def test_coverage_partial():
    elements, evaluator = _evaluator_for([True, False, True])
    out = _run(evaluator, elements)
    assert out["score"] == round(2 / 3, 3)
    assert out["value"] == "PARTIALLY_COVERED"
    assert out["covered"] == ["element-0", "element-2"]
    assert out["missing"] == ["element-1"]
    assert out["total"] == 3


def test_coverage_full():
    elements, evaluator = _evaluator_for([True, True])
    out = _run(evaluator, elements)
    assert out["score"] == 1.0
    assert out["value"] == "FULLY_COVERED"
    assert out["missing"] == []


def test_coverage_none():
    elements, evaluator = _evaluator_for([False, False])
    out = _run(evaluator, elements)
    assert out["score"] == 0.0
    assert out["value"] == "NOT_COVERED"
    assert out["covered"] == []


def _wire_up(monkeypatch, tmp_path):
    """Shared stubbing for the run_evaluation SimpleQA loop.

    Returns nothing; sets the module-level globals that the pre-existing
    ``save_result(...)`` call references and neutralizes file I/O + the live
    correctness judge.
    """
    import run_evaluation

    monkeypatch.setattr(run_evaluation, "output_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(
        run_evaluation, "evaluation_type", EvaluationType.SIMPLEQA, raising=False
    )
    monkeypatch.setattr(run_evaluation, "save_result", lambda *a, **k: None)

    class _FakeCorrectness:
        def __init__(self, *args, **kwargs):
            pass

        async def evaluate(self, inputs, outputs, reference_outputs):
            return {"score": 1.0, "value": "CORRECT"}

    monkeypatch.setattr(run_evaluation, "CorrectnessEvaluator", _FakeCorrectness)
    return run_evaluation


class _FakePostProcessor:
    def extract_answer(self, query, is_llm_response, search_result):
        return "The 'Passeport Talent' permit applies."


class _FakeHandler:
    is_llm_response = True

    async def search(self, query):
        return {"answer": "The 'Passeport Talent' permit applies."}


def test_wiring_attaches_coverage_to_result(monkeypatch, tmp_path):
    """The SimpleQA loop must invoke the coverage judge and store its output."""
    run_evaluation = _wire_up(monkeypatch, tmp_path)

    # Real coverage evaluator with an injected fake LLM -> 1 of 2 covered.
    grade = RequiredCoverageGrade(
        verdicts=[
            ElementVerdict(index=1, covered=True),
            ElementVerdict(index=2, covered=False),
        ]
    )
    coverage_evaluator = RequiredCoverageEvaluator(
        RequiredCoverageConfig(), llm=_FakeChat(grade)
    )

    example = {
        "index": 1,
        "question": "Which permit and which documents do I need to hire abroad?",
        "answer": "Passeport Talent",
        "required_elements": ["permit type", "passport copy"],
    }

    result = asyncio.run(
        run_evaluation.evaluate_provider_simple_qa(
            "tavily",
            _FakeHandler(),
            [example],
            _FakePostProcessor(),
            evaluator_model="gpt-4.1",
            batch_size=1,
            coverage_evaluator=coverage_evaluator,
        )
    )

    row = result["results"][0]
    assert row["required_coverage"] == 0.5
    assert row["coverage_label"] == "PARTIALLY_COVERED"
    assert row["missing_elements"] == ["passport copy"]
    assert row["is_correct"] is True


def test_wiring_skips_coverage_without_required_elements(monkeypatch, tmp_path):
    """Examples without required_elements must behave exactly as before."""
    run_evaluation = _wire_up(monkeypatch, tmp_path)

    example = {"index": 1, "question": "q", "answer": "gold"}

    result = asyncio.run(
        run_evaluation.evaluate_provider_simple_qa(
            "tavily",
            _FakeHandler(),
            [example],
            _FakePostProcessor(),
            evaluator_model="gpt-4.1",
            batch_size=1,
        )
    )

    row = result["results"][0]
    assert "required_coverage" not in row
    assert "coverage_label" not in row
    assert "missing_elements" not in row
