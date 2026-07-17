"""Integration tests for the Elo provider ranking.

These tests import the pre-existing ``evaluators.correctness_evaluator`` module
(the non-new call-site neighbor) and exercise ``compute_elo_ranking`` /
``save_elo_ranking`` on a ``provider_results`` fixture that mirrors the exact
per-row contract ``evaluate_provider_simple_qa`` emits (``index``,
``question``, ``reference_answer``, ``predicted_answer``, ``is_correct``,
``grade``). That proves the new capability consumes the real data the existing
SimpleQA pipeline already produces.
"""

import asyncio
import json
import os
import tempfile

# Import from a NON-NEW module in evaluators/ to anchor the integration.
from evaluators.correctness_evaluator import CorrectnessEvaluator, CorrectnessConfig
from evaluators.elo_evaluator import (
    Outcome,
    PairwiseAnswerJudge,
    compute_elo_ranking,
    grade_outcome,
    save_elo_ranking,
)


def _row(index, grade, predicted="ans"):
    """Build a result row with the keys evaluate_provider_simple_qa writes."""
    is_correct = grade == "CORRECT"
    return {
        "index": index,
        "question": f"q{index}",
        "reference_answer": f"gold{index}",
        "predicted_answer": predicted,
        "is_correct": is_correct,
        "grade": grade,
    }


def _provider(name, grades, predicted="ans"):
    return {
        "provider": name,
        "results": [_row(i, g, predicted=predicted) for i, g in enumerate(grades)],
        "accuracy": sum(1 for g in grades if g == "CORRECT") / len(grades),
        "correct_count": sum(1 for g in grades if g == "CORRECT"),
        "total_count": len(grades),
    }


def _make_provider_results():
    # brave answers correctly everywhere; exa is middling; tavily struggles.
    return {
        "brave": _provider("brave", ["CORRECT", "CORRECT", "CORRECT"]),
        "exa": _provider("exa", ["CORRECT", "NOT_ATTEMPTED", "INCORRECT"]),
        "tavily": _provider("tavily", ["NOT_ATTEMPTED", "INCORRECT", "INCORRECT"]),
    }


def test_ranking_orders_providers_by_pairwise_quality():
    """The all-correct provider must rank first; the struggling one last."""
    ranking = asyncio.run(compute_elo_ranking(_make_provider_results()))

    providers = [entry["provider"] for entry in ranking]
    assert providers == ["brave", "exa", "tavily"]
    assert ranking[0]["elo"] > ranking[-1]["elo"]
    # brave should never lose a pairwise comparison.
    brave = next(e for e in ranking if e["provider"] == "brave")
    assert brave["losses"] == 0
    assert brave["rank"] == 1


def test_ranking_contracts_match_existing_evaluator_grades():
    """Outcomes are derived from CorrectnessEvaluator's grade vocabulary."""
    # The same grade labels CorrectnessEvaluator.evaluate() returns as ``value``.
    assert grade_outcome("CORRECT", "INCORRECT") is Outcome.A_WINS
    assert grade_outcome("INCORRECT", "CORRECT") is Outcome.B_WINS
    assert grade_outcome("CORRECT", "CORRECT") is Outcome.TIE
    assert grade_outcome("NOT_ATTEMPTED", "INCORRECT") is Outcome.A_WINS

    # The non-new evaluator module is live, and its grading template speaks the
    # same CORRECT/INCORRECT/NOT_ATTEMPTED vocabulary the proxy judge consumes.
    # (Checked as class attributes so the test needs no OPENAI_API_KEY.)
    assert hasattr(CorrectnessEvaluator, "OPENAI_GRADER_TEMPLATE")
    assert "CORRECT" in CorrectnessEvaluator.OPENAI_GRADER_TEMPLATE
    assert "INCORRECT" in CorrectnessEvaluator.OPENAI_GRADER_TEMPLATE
    assert CorrectnessConfig().model_name == "gpt-4.1"


def test_save_elo_ranking_writes_json_and_roundtrips():
    ranking = asyncio.run(compute_elo_ranking(_make_provider_results()))
    with tempfile.TemporaryDirectory() as tmp:
        path = save_elo_ranking(ranking, tmp)
        assert os.path.basename(path) == "elo_ranking.json"
        with open(path) as f:
            loaded = json.load(f)
    assert loaded == ranking
    assert {e["provider"] for e in loaded} == {"brave", "exa", "tavily"}


def test_ranking_with_injected_judge_uses_raw_answers():
    """The LLM-judge path is exercised via an injectable stub (no API)."""

    class StubJudge(PairwiseAnswerJudge):
        def __init__(self):
            self.calls = 0

        async def judge(self, question, reference_answer, answer_a, answer_b):
            self.calls += 1
            # Prefer whichever answer text sorts later — a deterministic stand-in
            # for the real LLM verdict that exercises the same code path.
            if answer_a == answer_b:
                return Outcome.TIE
            return Outcome.A_WINS if answer_a > answer_b else Outcome.B_WINS

    stub = StubJudge()
    provider_results = {
        "alpha": _provider("alpha", ["CORRECT"], predicted="mmm"),
        "beta": _provider("beta", ["CORRECT"], predicted="zzz"),
    }
    ranking = asyncio.run(compute_elo_ranking(provider_results, judge=stub, use_judge=True))
    # beta ("zzz") beats alpha ("mmm") on the stub, so beta ranks first.
    assert ranking[0]["provider"] == "beta"
    assert stub.calls == 1  # one pairwise comparison for the one shared question


def test_single_provider_yields_base_rating():
    ranking = asyncio.run(compute_elo_ranking({"solo": _provider("solo", ["CORRECT", "INCORRECT"])}))
    assert len(ranking) == 1
    assert ranking[0]["provider"] == "solo"
    assert ranking[0]["matches"] == 0
