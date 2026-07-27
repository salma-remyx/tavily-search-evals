"""Integration tests for the GroUSE-style judge meta-evaluation.

These tests exercise the audit through the *existing* repo surface -- the
:class:`evaluators.correctness_evaluator.CorrectnessEvaluator` contract and the
:data-loading path in :mod:`utils` -- using a deterministic in-process judge
stand-in so no network or API key is required.
"""

import inspect
import asyncio

from utils import EvaluationType, load_csv_data  # existing, non-new modules in utils/
from evaluators.correctness_evaluator import CorrectnessEvaluator  # the existing judge being audited

from evaluators.grouse_meta_eval import (
    DEFAULT_UNIT_TESTS,
    FailureMode,
    Judge,
    UnitTestCase,
    build_unit_tests_from_scenarios,
    corrupt_answer,
    evaluate_unit_tests,
    run_meta_evaluation,
    summarize,
)


class _RuleBasedJudge:
    """A network-free judge that quacks like ``CorrectnessEvaluator``.

    CORRECT iff the answer contains the reference's key tokens and carries no
    recognisable corruption marker. It is deliberately *blind* to
    ``EXTRA_INFORMATION`` (irrelevant but true padding) -- the canonical failure
    a correctness-only judge misses, and exactly what GroUSE-style probing
    surfaces.
    """

    _POISON = (
        "but in reality",
        "twelve planets",
        "need more information",
        "red and green",
        "yellow",
        "can't determine",
        "without more context",
        "because it floats",
        "more dense",
        "the moon is farther",
        "equally far",
        "eight sides",
    )

    async def evaluate(self, inputs, outputs, reference_outputs):
        # Honour the exact call contract of CorrectnessEvaluator.evaluate.
        assert set(inputs) == {"question"}
        assert set(outputs) == {"answer"}
        assert set(reference_outputs) == {"answer"}
        answer = outputs["answer"].lower()
        reference = reference_outputs["answer"].lower()
        correct = reference in answer and not any(marker in answer for marker in self._POISON)
        return {"score": 1.0 if correct else 0.0, "value": "CORRECT" if correct else "INCORRECT"}


class _LenientJudge:
    """Accepts any answer that contains the reference (no corruption markers)."""

    async def evaluate(self, inputs, outputs, reference_outputs):
        answer = outputs["answer"].lower()
        reference = reference_outputs["answer"].lower()
        correct = reference in answer
        return {"score": 1.0 if correct else 0.0, "value": "CORRECT" if correct else "INCORRECT"}


def test_audit_detects_planted_failures_and_exposes_extra_info_blind_spot():
    report = asyncio.run(run_meta_evaluation(_RuleBasedJudge(), DEFAULT_UNIT_TESTS))

    # A well-calibrated judge accepts every clean answer.
    assert report["calibration"] == 1.0

    # It catches every clearly-wrong failure mode...
    for mode in (
        FailureMode.MISINFORMATION,
        FailureMode.INCOMPLETE_INFORMATION,
        FailureMode.UNANSWERED_QUESTION,
        FailureMode.WRONG_CAUSALITY,
        FailureMode.WRONG_OBJECT_OF_COMPARISON,
        FailureMode.WRONG_QUANTITY,
    ):
        assert report["discrimination"][mode.value] == 1.0, mode

    # ...but a correctness-only judge does not penalise irrelevant extra info.
    assert report["discrimination"][FailureMode.EXTRA_INFORMATION.value] == 0.0


def test_wiring_matches_correctness_evaluator_interface():
    # The audit must call the same async evaluate(inputs, outputs, reference_outputs)
    # shape that CorrectnessEvaluator exposes.
    assert isinstance(_RuleBasedJudge(), Judge)
    assert hasattr(CorrectnessEvaluator, "evaluate")
    assert inspect.iscoroutinefunction(CorrectnessEvaluator.evaluate)

    # evaluate_unit_tests must forward the exact keys the real judge reads.
    captured = {}

    class _SnoopingJudge:
        async def evaluate(self, inputs, outputs, reference_outputs):
            captured.update(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)
            return {"score": 1.0, "value": "CORRECT"}

    asyncio.run(
        evaluate_unit_tests(
            _SnoopingJudge(),
            [UnitTestCase("s", "Q?", "gold", "gold", FailureMode.CLEAN, True)],
        )
    )
    assert captured["inputs"] == {"question": "Q?"}
    assert captured["outputs"] == {"answer": "gold"}
    assert captured["reference_outputs"] == {"answer": "gold"}


def test_corrupt_answer_generates_each_applicable_failure_mode():
    causal = "Ice floats on water because it is less dense."
    comparative = "The Sun is larger than the Moon."

    assert corrupt_answer("Paris", FailureMode.MISINFORMATION).endswith("opposite is in fact true.")
    assert corrupt_answer(causal, FailureMode.UNANSWERED_QUESTION) == (
        "I cannot answer this based on the information available."
    )
    assert corrupt_answer(causal, FailureMode.WRONG_CAUSALITY).startswith("it is less dense.")
    swapped = corrupt_answer(comparative, FailureMode.WRONG_OBJECT_OF_COMPARISON)
    assert swapped.lower().startswith("the moon.")
    # WRONG_QUANTITY perturbs a present number, and is N/A when there is none.
    assert corrupt_answer("There are 8 planets.", FailureMode.WRONG_QUANTITY) == (
        "There are 19 planets."
    )
    assert corrupt_answer("Paris", FailureMode.WRONG_QUANTITY) is None
    # CLEAN is a calibration probe, not a corruption.
    assert corrupt_answer("Paris", FailureMode.CLEAN) is None


def test_builds_unit_tests_from_real_repo_data():
    # Integration with the existing utils.load_csv_data SimpleQA loader.
    frame = load_csv_data("datasets/simple_qa_test_set.csv", start_index=0, end_index=3)
    scenarios = [
        {"question": row["problem"], "answer": row["answer"]} for _, row in frame.iterrows()
    ]
    assert len(scenarios) == 3

    tests = build_unit_tests_from_scenarios(scenarios)
    assert tests and all(isinstance(test, UnitTestCase) for test in tests)
    # Each scenario yields a CLEAN probe plus at least one corrupted probe.
    assert any(test.failure_mode == FailureMode.CLEAN for test in tests)
    assert any(test.failure_mode != FailureMode.CLEAN for test in tests)

    report = summarize(asyncio.run(evaluate_unit_tests(_LenientJudge(), tests)))
    # The lenient judge accepts every CLEAN (reference) answer -> perfect calibration.
    assert report["calibration"] == 1.0
    assert EvaluationType.SIMPLEQA.value == "simpleqa"
