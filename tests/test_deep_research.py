"""Tests for the deep-research (multi-hop) evaluation type.

These exercise the integration with *existing* repo modules (``utils.utils``
and ``run_evaluation``) plus the resolve-and-grade loop in the new
``evaluators.deep_research`` module. The loop is driven with fakes for the
search handler / post-processor / evaluator so it runs without network
access or API keys.
"""

import asyncio
import os
import sys

# Make the repo root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_evaluation  # existing (non-new) module
from evaluators.deep_research import (  # new module under test
    DeepResearchTask,
    ResearchStep,
    evaluate_provider_deep_research,
    load_deep_research_tasks,
    render_question,
)
from utils.utils import EvaluationType, save_result  # existing (non-new) module


# --- Integration with existing modules --------------------------------------


def test_deep_research_evaluation_type_registered():
    """The new type is wired into the existing utils enum."""
    assert EvaluationType.DEEP_RESEARCH.value == "deep_research"


def test_deep_research_dispatch_wired():
    """The existing CLI dispatch recognizes the new evaluation type."""
    assert "deep_research" in run_evaluation.get_dataset_path(
        EvaluationType.DEEP_RESEARCH
    )


def test_argparse_accepts_deep_research():
    """The --evaluation_type flag accepts the deep_research choice."""
    # Re-create the parser the same way the CLI does and assert the choice.
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--evaluation_type",
        choices=[
            EvaluationType.SIMPLEQA.value,
            EvaluationType.DOCUMENT_RELEVANCE.value,
            EvaluationType.DEEP_RESEARCH.value,
        ],
    )
    parsed, _ = p.parse_known_args(["--evaluation_type", "deep_research"])
    assert parsed.evaluation_type == "deep_research"


# --- Pure unit checks of the DAG mechanics ----------------------------------


def test_render_question_substitutes_resolved_answers():
    step = ResearchStep("s2", "When was {s1} born?", "1452", depends_on=["s1"])
    assert render_question(step, {"s1": "Leonardo da Vinci"}) == (
        "When was Leonardo da Vinci born?"
    )


def test_topo_order_respects_dependencies():
    task = DeepResearchTask(
        "t1",
        "seed?",
        "seed_ans",
        steps=[
            ResearchStep("s2", "q2", "a2", depends_on=["s1"]),
            ResearchStep("s1", "q1", "a1"),
            ResearchStep("s3", "q3", "a3", depends_on=["s2"]),
        ],
    )
    order = [step.step_id for step in task.topo_order()]
    assert order == ["s1", "s2", "s3"]


def test_catalog_loads_with_valid_dags():
    tasks = load_deep_research_tasks()
    assert tasks, "expected a non-empty evolved task catalog"
    for task in tasks:
        ids = [step.step_id for step in task.steps]
        assert len(ids) == len(set(ids)), f"duplicate step ids in {task.task_id}"
        for step in task.steps:
            assert all(dep in ids for dep in step.depends_on), (
                f"step {step.step_id} depends on unknown predecessor"
            )
        # Every dependency precedes its dependent in topo order.
        ordered = [step.step_id for step in task.topo_order()]
        for step in task.steps:
            for dep in step.depends_on:
                assert ordered.index(dep) < ordered.index(step.step_id)


# --- End-to-end resolve-and-grade loop with fakes ---------------------------


class _FakeHandler:
    """Returns a canned answer per query (LLM-response style)."""

    is_llm_response = True

    def __init__(self, answer_for_query=None, default="unknown"):
        self._answers = answer_for_query or {}
        self.default = default

    async def search(self, query):
        return {"answer": self._answers.get(query, self.default)}


class _FakePostProcessor:
    def extract_answer(self, query, is_llm_response, search_result):
        return search_result


class _FakeEvaluator:
    def __init__(self, correct):
        self._correct = correct

    async def evaluate(self, inputs, outputs, reference_outputs):
        if self._correct:
            return {"score": 1.0, "value": "CORRECT"}
        return {"score": 0.0, "value": "INCORRECT"}


def test_all_checkpoints_pass_yields_full_task_accuracy():
    tasks = load_deep_research_tasks()
    result = asyncio.run(
        evaluate_provider_deep_research(
            "tavily",
            _FakeHandler(),
            tasks,
            _FakePostProcessor(),
            evaluator=_FakeEvaluator(correct=True),
            output_dir=None,
        )
    )
    assert result["total_tasks"] == len(tasks)
    assert result["task_accuracy"] == 1.0
    assert result["step_accuracy"] == 1.0
    assert result["tasks_passed"] == len(tasks)


def test_wrong_answers_collapse_task_accuracy_but_dont_block():
    tasks = load_deep_research_tasks()
    result = asyncio.run(
        evaluate_provider_deep_research(
            "tavily",
            _FakeHandler(),
            tasks,
            _FakePostProcessor(),
            evaluator=_FakeEvaluator(correct=False),
            output_dir=None,
        )
    )
    # Every checkpoint graded wrong -> no task fully solved.
    assert result["task_accuracy"] == 0.0
    assert result["step_accuracy"] == 0.0
    # An answer was still extracted for every step, so nothing was blocked:
    # downstream queries ran with the (wrong) resolved predecessor answer.
    assert all(not row["blocked"] for row in result["results"])


class _EmptyAnswerHandler:
    is_llm_response = True

    async def search(self, query):
        return {"answer": ""}


def test_missing_dependency_blocks_downstream_steps():
    """Empty predecessor answer => dependent step is blocked (DAG semantics)."""
    task = DeepResearchTask(
        "mona_lisa",
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci",
        steps=[
            ResearchStep("s1", "Who painted the Mona Lisa?", "Leonardo da Vinci"),
            ResearchStep(
                "s2", "In which year was {s1} born?", "1452", depends_on=["s1"]
            ),
        ],
    )
    result = asyncio.run(
        evaluate_provider_deep_research(
            "tavily",
            _EmptyAnswerHandler(),
            [task],
            _FakePostProcessor(),
            evaluator=_FakeEvaluator(correct=True),
            output_dir=None,
        )
    )
    by_step = {row["step_id"]: row for row in result["results"]}
    assert by_step["s1"]["predicted_answer"] == ""
    assert by_step["s2"]["blocked"] is True
    assert by_step["s2"]["grade"] == "BLOCKED"
    assert by_step["s2"]["predicted_answer"] == ""
    assert result["task_accuracy"] == 0.0


def test_results_round_trip_through_existing_save_result(tmp_path):
    """Per-step rows save cleanly via the existing utils.save_result path."""
    output_dir = str(tmp_path)
    task = DeepResearchTask(
        "hamlet",
        "Who wrote the play Hamlet?",
        "William Shakespeare",
        steps=[
            ResearchStep("s1", "Who wrote the play Hamlet?", "William Shakespeare"),
            ResearchStep(
                "s2", "In which year was {s1} born?", "1564", depends_on=["s1"]
            ),
        ],
    )
    result = asyncio.run(
        evaluate_provider_deep_research(
            "tavily",
            _FakeHandler(default="William Shakespeare"),
            [task],
            _FakePostProcessor(),
            evaluator=_FakeEvaluator(correct=True),
            output_dir=output_dir,
        )
    )
    assert result["task_accuracy"] == 1.0
    csv_path = os.path.join(
        output_dir, f"tavily_{EvaluationType.DEEP_RESEARCH.value}_results.csv"
    )
    assert os.path.exists(csv_path)
    with open(csv_path) as fh:
        lines = [line for line in fh.read().splitlines() if line]
    # header + one row per step.
    assert len(lines) == 1 + len(task.steps)
    assert "checkpoint" in lines[0]
    assert "blocked" in lines[0]
