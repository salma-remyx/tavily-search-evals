"""Grounded-QA judge meta-evaluation via planted failure modes.

An LLM-as-a-judge that agrees with humans in aggregate can still be blind to
specific, systematic mistakes. This module audits the repo's correctness
judge (:class:`evaluators.correctness_evaluator.CorrectnessEvaluator`) the way
a unit test audits code: it takes short grounded-QA situations whose answer
has been deliberately corrupted to exhibit one of seven generator failure
modes, asks the judge to grade each one, and reports whether the judge
**accepts the clean answer** (calibration) and **rejects each corrupted one**
(per-failure-mode discrimination). Systematic misses pin down exactly where
the judge is weak -- e.g. a correctness-only judge typically fails to penalise
answers padded with irrelevant but true ``extra information``.

Adapted from *GroUSE: A Benchmark to Evaluate Evaluators in Grounded Question
Answering* (Bavaresco et al., arXiv:2409.06595). GroUSE ships 144
hand-authored unit tests scored against a six-metric prompted judge. This is a
Mode 2 (adapted) port:

  * Core mechanism, kept faithful: the seven-failure-mode taxonomy and the
    planted-corruption audit ("does the judge detect each planted failure?").
  * Substituted auxiliaries (target-native equivalents):
      - GroUSE's 144-scenario dataset -> a smaller native unit-test suite
        (:data:`DEFAULT_UNIT_TESTS`), plus :func:`corrupt_answer` so the audit
        can also run over the repo's own SimpleQA examples via
        :func:`build_unit_tests_from_scenarios`.
      - GroUSE's six-metric prompted judge -> this repo's binary
        :class:`~evaluators.correctness_evaluator.CorrectnessEvaluator`
        (CORRECT / INCORRECT / NOT_ATTEMPTED).
      - Six-metric range matching -> binary detection (judge must accept the
        clean answer and reject the corrupted one), reported as calibration
        and per-failure-mode discrimination.
"""

import argparse
import asyncio
import csv
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Minimal contract of a grounded-QA correctness judge.

    Mirrors :meth:`CorrectnessEvaluator.evaluate` so any object exposing that
    method (the real evaluator or a test stub) can be audited.
    """

    async def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


class FailureMode(Enum):
    """The seven grounded-QA generator failure modes (GroUSE taxonomy).

    ``CLEAN`` is the calibration probe -- an uncorrupted, correct answer that a
    well-calibrated judge must accept.
    """

    CLEAN = "clean"
    MISINFORMATION = "misinformation"
    INCOMPLETE_INFORMATION = "incomplete_information"
    UNANSWERED_QUESTION = "unanswered_question"
    EXTRA_INFORMATION = "extra_information"
    WRONG_CAUSALITY = "wrong_causality"
    WRONG_OBJECT_OF_COMPARISON = "wrong_object_of_comparison"
    WRONG_QUANTITY = "wrong_quantity"


FAILURE_MODE_DESCRIPTIONS: Dict[FailureMode, str] = {
    FailureMode.CLEAN: "An uncorrupted, correct answer (calibration probe).",
    FailureMode.MISINFORMATION: (
        "The answer contains a claim contradicted by, or unsupported by, the "
        "grounding context (low faithfulness)."
    ),
    FailureMode.INCOMPLETE_INFORMATION: (
        "The answer omits relevant information the context provides and the "
        "question requires (low completeness)."
    ),
    FailureMode.UNANSWERED_QUESTION: (
        "The answer does not address the question being asked (low usefulness)."
    ),
    FailureMode.EXTRA_INFORMATION: (
        "The answer includes additional, irrelevant information the question did "
        "not ask for (low relevancy)."
    ),
    FailureMode.WRONG_CAUSALITY: (
        "The answer inverts a cause-and-effect relationship from the context."
    ),
    FailureMode.WRONG_OBJECT_OF_COMPARISON: (
        "The answer compares the wrong two entities, swapping the subject and "
        "object of the comparison."
    ),
    FailureMode.WRONG_QUANTITY: (
        "The answer reports an incorrect number or amount."
    ),
}


@dataclass
class UnitTestCase:
    """One grounded-QA answer placed in front of the judge."""

    situation: str
    question: str
    reference_answer: str
    answer: str
    failure_mode: FailureMode
    expected_correct: bool


# A small native analog of GroUSE's unit tests. Each situation contributes a
# CLEAN probe plus corrupted variants that clearly exhibit a failure mode.
# Together they cover all seven modes; EXTRA_INFORMATION is included because it
# is the canonical case a correctness-only judge misses.
_SITUATIONS: List[Dict[str, Any]] = [
    {
        "id": "planets",
        "question": "How many planets are in the Solar System?",
        "reference": "Eight planets",
        FailureMode.CLEAN: "There are eight planets in the Solar System.",
        FailureMode.WRONG_QUANTITY: "There are twelve planets in the Solar System.",
        FailureMode.MISINFORMATION: (
            "There are eight planets in the Solar System, but in reality there "
            "are only three."
        ),
        FailureMode.UNANSWERED_QUESTION: "I would need more information to answer that.",
        FailureMode.EXTRA_INFORMATION: (
            "There are eight planets in the Solar System. The chemical symbol "
            "for gold is Au."
        ),
    },
    {
        "id": "primary_colors",
        "question": "What are the three primary colors of light?",
        "reference": "Red, green, and blue",
        FailureMode.CLEAN: "The three primary colors of light are red, green, and blue.",
        FailureMode.INCOMPLETE_INFORMATION: "The primary colors of light are red and green.",
        FailureMode.MISINFORMATION: "The three primary colors of light are red, green, and yellow.",
        FailureMode.EXTRA_INFORMATION: (
            "The three primary colors of light are red, green, and blue. The "
            "Eiffel Tower is located in Paris."
        ),
        FailureMode.UNANSWERED_QUESTION: "I can't determine that without more context.",
    },
    {
        "id": "ice_density",
        "question": "Why does ice float on water?",
        "reference": "Ice is less dense than liquid water",
        FailureMode.CLEAN: (
            "Ice floats on water because ice is less dense than liquid water."
        ),
        FailureMode.WRONG_CAUSALITY: (
            "Ice is less dense than liquid water because it floats on water."
        ),
        FailureMode.MISINFORMATION: (
            "Ice floats on water because it is more dense than liquid water."
        ),
    },
    {
        "id": "sun_moon_distance",
        "question": "Which is farther from Earth, the Sun or the Moon?",
        "reference": "The Sun is farther from Earth than the Moon",
        FailureMode.CLEAN: "The Sun is farther from Earth than the Moon is.",
        FailureMode.WRONG_OBJECT_OF_COMPARISON: "The Moon is farther from Earth than the Sun is.",
        FailureMode.MISINFORMATION: "The Sun and the Moon are equally far from Earth.",
    },
    {
        "id": "hexagon",
        "question": "How many sides does a hexagon have?",
        "reference": "Six",
        FailureMode.CLEAN: "A hexagon has six sides.",
        FailureMode.WRONG_QUANTITY: "A hexagon has eight sides.",
    },
]


def _expand_situations(situations: Sequence[Dict[str, Any]]) -> List[UnitTestCase]:
    tests: List[UnitTestCase] = []
    for situation in situations:
        reference = situation["reference"]
        for mode, answer in situation.items():
            if mode in ("id", "question", "reference"):
                continue
            tests.append(
                UnitTestCase(
                    situation=situation["id"],
                    question=situation["question"],
                    reference_answer=reference,
                    answer=answer,
                    failure_mode=mode,
                    expected_correct=(mode == FailureMode.CLEAN),
                )
            )
    return tests


#: Default suite of grounded-QA unit tests covering all seven failure modes.
DEFAULT_UNIT_TESTS: List[UnitTestCase] = _expand_situations(_SITUATIONS)


def corrupt_answer(answer: str, failure_mode: FailureMode) -> Optional[str]:
    """Inject ``failure_mode`` into ``answer`` via a parameter-free transform.

    These deterministic transforms are a target-native substitute for GroUSE's
    hand-authored corruptions, letting the audit scale over the repo's own
    examples. They apply structurally and return ``None`` when a failure cannot
    be cleanly injected into the given answer (e.g. ``WRONG_QUANTITY`` on an
    answer with no number). ``CLEAN`` is not a corruption and returns ``None``.
    """
    text = answer.strip()
    if failure_mode == FailureMode.MISINFORMATION:
        return f"{text} However, this is incorrect; the opposite is in fact true."
    if failure_mode == FailureMode.INCOMPLETE_INFORMATION:
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])
    if failure_mode == FailureMode.UNANSWERED_QUESTION:
        return "I cannot answer this based on the information available."
    if failure_mode == FailureMode.EXTRA_INFORMATION:
        return f"{text} On an unrelated note, the chemical symbol for gold is Au."
    if failure_mode == FailureMode.WRONG_QUANTITY:
        match = re.search(r"\d+", text)
        if not match:
            return None
        perturbed = str(int(match.group()) + 11)
        return text[: match.start()] + perturbed + text[match.end():]
    if failure_mode == FailureMode.WRONG_CAUSALITY:
        return _swap_around(text, " because ")
    if failure_mode == FailureMode.WRONG_OBJECT_OF_COMPARISON:
        return _swap_around(text, " than ", max_side_words=6)
    return None


def _swap_around(text: str, marker: str, max_side_words: Optional[int] = None) -> Optional[str]:
    """Swap the two clauses around ``marker`` (case-insensitive)."""
    idx = text.lower().find(marker)
    if idx < 0:
        return None
    left, right = text[:idx], text[idx + len(marker):]
    if max_side_words is not None and (
        len(left.split()) > max_side_words or len(right.split()) > max_side_words
    ):
        return None
    return f"{right}{marker}{left}"


def build_unit_tests_from_scenarios(
    scenarios: Sequence[Dict[str, str]],
) -> List[UnitTestCase]:
    """Build unit tests from ``{"question", "answer"}`` scenarios.

    Each scenario becomes a CLEAN probe plus a corrupted probe for every mode
    :func:`corrupt_answer` can inject. Accepts the row shape produced by the
    repo's ``utils.load_csv_data`` / ``utils.prepare_examples`` loaders, so the
    audit can run over the project's own SimpleQA data.
    """
    tests: List[UnitTestCase] = []
    for index, scenario in enumerate(scenarios):
        question, reference = scenario["question"], scenario["answer"]
        tests.append(
            UnitTestCase(
                situation=f"scenario-{index}",
                question=question,
                reference_answer=reference,
                answer=reference,
                failure_mode=FailureMode.CLEAN,
                expected_correct=True,
            )
        )
        for mode in FailureMode:
            if mode == FailureMode.CLEAN:
                continue
            corrupted = corrupt_answer(reference, mode)
            if corrupted is not None:
                tests.append(
                    UnitTestCase(
                        situation=f"scenario-{index}",
                        question=question,
                        reference_answer=reference,
                        answer=corrupted,
                        failure_mode=mode,
                        expected_correct=False,
                    )
                )
    return tests


async def evaluate_unit_tests(judge: Judge, unit_tests: Sequence[UnitTestCase]) -> List[Dict[str, Any]]:
    """Grade every unit test with ``judge`` and return detailed rows.

    Calls ``judge.evaluate`` with the exact contract
    :meth:`CorrectnessEvaluator.evaluate` expects: ``inputs={"question": ...}``,
    ``outputs={"answer": ...}``, ``reference_outputs={"answer": ...}``.
    """
    rows: List[Dict[str, Any]] = []
    for test in unit_tests:
        result = await judge.evaluate(
            {"question": test.question},
            {"answer": test.answer},
            {"answer": test.reference_answer},
        )
        judged_correct = float(result.get("score", 0.0)) >= 1.0
        rows.append(
            {
                "situation": test.situation,
                "question": test.question,
                "reference_answer": test.reference_answer,
                "answer": test.answer,
                "failure_mode": test.failure_mode.value,
                "expected_correct": test.expected_correct,
                "judged_correct": judged_correct,
                "judge_value": result.get("value"),
            }
        )
    return rows


def _rate(values: Sequence[bool]) -> float:
    return round(sum(1 for value in values if value) / len(values), 3) if values else 0.0


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse per-test rows into calibration and per-mode discrimination.

    * ``calibration`` -- share of CLEAN answers the judge accepted.
    * ``discrimination[mode]`` -- share of that mode's corrupted answers the
      judge rejected (1.0 = catches every planted failure).
    """
    clean = [row for row in rows if row["failure_mode"] == FailureMode.CLEAN.value]
    corrupted = [row for row in rows if row["failure_mode"] != FailureMode.CLEAN.value]
    discrimination: Dict[str, float] = {}
    for mode in FailureMode:
        if mode == FailureMode.CLEAN:
            continue
        mode_rows = [row for row in corrupted if row["failure_mode"] == mode.value]
        if mode_rows:
            discrimination[mode.value] = _rate([not row["judged_correct"] for row in mode_rows])
    detected = sum(1 for row in corrupted if not row["judged_correct"])
    return {
        "n_tests": len(rows),
        "calibration": _rate([row["judged_correct"] for row in clean]),
        "discrimination": discrimination,
        "overall_discrimination": _rate([not row["judged_correct"] for row in corrupted]),
        "failures_detected": detected,
        "failures_total": len(corrupted),
        "details": list(rows),
    }


def write_report_csv(report: Dict[str, Any], output_path: str) -> None:
    """Write the per-test ``details`` of ``report`` to a CSV file."""
    fieldnames = [
        "situation",
        "question",
        "reference_answer",
        "answer",
        "failure_mode",
        "expected_correct",
        "judged_correct",
        "judge_value",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["details"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


async def run_meta_evaluation(
    judge: Judge,
    unit_tests: Optional[Sequence[UnitTestCase]] = None,
    output_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full GroUSE-style audit of ``judge`` and return the report."""
    if unit_tests is None:
        unit_tests = DEFAULT_UNIT_TESTS
    rows = await evaluate_unit_tests(judge, unit_tests)
    report = summarize(rows)
    if output_csv:
        write_report_csv(report, output_csv)
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("\n===== GROUSE-STYLE JUDGE META-EVALUATION =====")
    print(f"Calibration (accepts clean answers): {report['calibration']:.1%}")
    print(
        f"Overall discrimination (rejects planted failures): "
        f"{report['overall_discrimination']:.1%} "
        f"({report['failures_detected']}/{report['failures_total']})"
    )
    print("\nPer-failure-mode discrimination:")
    for mode in FailureMode:
        if mode == FailureMode.CLEAN:
            continue
        rate = report["discrimination"].get(mode.value)
        label = "  (not sampled)" if rate is None else f"{rate:.1%}"
        print(f"  {mode.value:<28} {label}")
    print("==============================================\n")


def _build_judge(model_name: str) -> Judge:
    # Imported lazily so the module (and its tests) can be imported without the
    # langchain/openai stack or an API key.
    from evaluators.correctness_evaluator import CorrectnessConfig, CorrectnessEvaluator

    return CorrectnessEvaluator(CorrectnessConfig(model_name=model_name))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Meta-evaluate the correctness judge with GroUSE-style unit tests."
    )
    parser.add_argument("--model", default="gpt-4.1", help="Judge model name (default: gpt-4.1).")
    parser.add_argument(
        "--from-csv",
        default=None,
        help=(
            "Path to a SimpleQA-style CSV (columns 'problem', 'answer') to build "
            "extra unit tests from via planted corruption."
        ),
    )
    parser.add_argument("--limit", type=int, default=10, help="Max CSV rows to turn into scenarios.")
    parser.add_argument("--output-csv", default=None, help="Optional path to write the per-test report.")
    args = parser.parse_args(argv)

    unit_tests: List[UnitTestCase] = list(DEFAULT_UNIT_TESTS)
    if args.from_csv:
        from utils import load_csv_data

        frame = load_csv_data(args.from_csv, start_index=0, end_index=args.limit)
        scenarios = [
            {"question": row["problem"], "answer": row["answer"]}
            for _, row in frame.iterrows()
        ]
        unit_tests.extend(build_unit_tests_from_scenarios(scenarios))

    judge = _build_judge(args.model)
    report = asyncio.run(run_meta_evaluation(judge, unit_tests, output_csv=args.output_csv))
    _print_report(report)


if __name__ == "__main__":
    main()
