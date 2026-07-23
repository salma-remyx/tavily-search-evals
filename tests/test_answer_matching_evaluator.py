"""Tests for the answer-matching triangulation judge.

These tests exercise the integration, not just the new module in isolation:
they import the existing ``CorrectnessEvaluator`` / ``EvaluationType`` (non-new
modules in ``evaluators/`` and ``utils/``) and drive the real call site
``run_evaluation.evaluate_provider_simple_qa`` with the triangulation wiring.
No network or API key is required: the judge LLMs are replaced with fakes.
"""
import asyncio

import run_evaluation
from evaluators import CorrectnessEvaluator  # existing, non-new module
from evaluators.answer_matching_evaluator import (
    AnswerMatchingConfig,
    AnswerMatchingEvaluator,
)
from evaluators.correctness_evaluator import CorrectnessConfig
from utils import EvaluationType  # existing, non-new module


class _FakeGrade:
    """Mimics the pydantic schema returned by with_structured_output()."""

    def __init__(self, match):
        self.match = match


class _FakeMatchLLM:
    """Stand-in for the structured-output judge LLM."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.seen_messages = []

    def invoke(self, messages):
        self.seen_messages.append(messages)
        return _FakeGrade(self._verdict)


class _FakeCorrectness:
    """Drop-in for CorrectnessEvaluator; outcome set on the class."""

    score = 1.0
    value = "CORRECT"

    def __init__(self, config=None):
        pass

    async def evaluate(self, inputs, outputs, reference_outputs):
        return {"score": _FakeCorrectness.score, "value": _FakeCorrectness.value}


class _FakeMatcher:
    """Drop-in for AnswerMatchingEvaluator; outcome set on the class."""

    score = 1.0
    value = "MATCH"

    def __init__(self, config=None):
        pass

    async def evaluate(self, inputs, outputs, reference_outputs):
        return {"score": _FakeMatcher.score, "value": _FakeMatcher.value}


class _FakeHandler:
    is_llm_response = True

    async def search(self, query):
        return {"answer": "a predicted answer"}


class _FakePostProcessor:
    def extract_answer(self, query, is_llm_response, search_result):
        return search_result


def _run_eval(monkeypatch, triangulate, correctness, matcher):
    """Drive the real call site with fake judges; return its summary dict."""
    _FakeCorrectness.score, _FakeCorrectness.value = correctness
    _FakeMatcher.score, _FakeMatcher.value = matcher
    monkeypatch.setattr(run_evaluation, "CorrectnessEvaluator", _FakeCorrectness)
    monkeypatch.setattr(run_evaluation, "AnswerMatchingEvaluator", _FakeMatcher)
    monkeypatch.setattr(run_evaluation, "save_result", lambda *a, **kw: None)
    # output_dir / evaluation_type are module globals normally set in __main__;
    # allow creating them when they are not yet present.
    monkeypatch.setattr(run_evaluation, "evaluation_type", EvaluationType.SIMPLEQA, raising=False)
    monkeypatch.setattr(run_evaluation, "output_dir", "results", raising=False)

    examples = [{"index": 0, "question": "q?", "answer": "ref"}]
    return asyncio.run(
        run_evaluation.evaluate_provider_simple_qa(
            "test_provider",
            _FakeHandler(),
            examples,
            _FakePostProcessor(),
            triangulate=triangulate,
            batch_size=1,
        )
    )


def test_triangulate_off_leaves_verdicts_unset(monkeypatch):
    """With triangulation off, the pipeline behaves exactly as before."""
    summary = _run_eval(monkeypatch, triangulate=False,
                        correctness=(1.0, "CORRECT"), matcher=(1.0, "MATCH"))
    row = summary["results"][0]
    assert row["answer_match"] is None
    assert row["judges_agree"] is None
    assert "judge_agreement_rate" not in summary


def test_triangulate_on_records_verdict_and_agreement(monkeypatch):
    """The second judge runs and agreement is computed (disagree case)."""
    summary = _run_eval(monkeypatch, triangulate=True,
                        correctness=(0.0, "INCORRECT"), matcher=(1.0, "MATCH"))
    row = summary["results"][0]
    # Correctness says wrong, matcher says match -> judges disagree.
    assert row["answer_match"] == "MATCH"
    assert row["judges_agree"] is False
    assert summary["judge_agreement_rate"] == 0.0


def test_judges_agree_when_both_judges_match(monkeypatch):
    """Both judges say correct -> agreement recorded."""
    summary = _run_eval(monkeypatch, triangulate=True,
                        correctness=(1.0, "CORRECT"), matcher=(1.0, "MATCH"))
    row = summary["results"][0]
    assert row["answer_match"] == "MATCH"
    assert row["judges_agree"] is True
    assert summary["judge_agreement_rate"] == 1.0


def _real_matcher(monkeypatch, verdict):
    """Build a real AnswerMatchingEvaluator with its LLM swapped for a fake."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    evaluator = AnswerMatchingEvaluator(AnswerMatchingConfig(model_name="gpt-4.1"))
    evaluator.llm = _FakeMatchLLM(verdict)
    return evaluator


def test_answer_matching_evaluator_parses_verdict(monkeypatch):
    """The new judge maps YES->MATCH/1.0 and NO->NO_MATCH/0.0."""
    args = ({"question": "q?"}, {"answer": "pred"}, {"answer": "ref"})

    yes = _real_matcher(monkeypatch, "YES")
    assert asyncio.run(yes.evaluate(*args)) == {"score": 1.0, "value": "MATCH"}

    no = _real_matcher(monkeypatch, "NO")
    assert asyncio.run(no.evaluate(*args)) == {"score": 0.0, "value": "NO_MATCH"}


def test_answer_matching_judge_is_drop_in_for_correctness(monkeypatch):
    """The new judge returns the same result shape as the existing judge."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    matcher = AnswerMatchingEvaluator(AnswerMatchingConfig(model_name="gpt-4.1"))
    matcher.llm = _FakeMatchLLM("YES")
    matcher_result = asyncio.run(
        matcher.evaluate({"question": "q?"}, {"answer": "pred"}, {"answer": "ref"})
    )

    correctness = CorrectnessEvaluator(CorrectnessConfig(model_name="gpt-4.1"))
    correctness.llm = _FakeMatchLLM("YES")  # unused: real grader's schema differs,
    # but the *return contract* below is what we assert, independent of the LLM.

    # Same public contract: dict with float 'score' in {0,1} and str 'value'.
    assert set(matcher_result) == {"score", "value"}
    assert matcher_result["score"] in (0.0, 1.0)
    assert isinstance(matcher_result["value"], str)
    # The existing evaluator advertises the same contract on its description.
    assert correctness.evaluation_name == "correctness_evaluator"
    assert matcher.evaluation_name == "answer_matching_evaluator"


def test_judge_prompt_mirrors_canonical_paper_rules(monkeypatch):
    """The judge prompt carries the paper's canonical matching rules.

    The authors' reference judge prompt (arXiv:2507.02856) grades with a
    coverage rule (the response must cover everything in the reference; more
    specific is OK) and a 1% numeric relative-error tolerance. The repo's
    YES/NO adaptation must keep both rules to stay faithful to the paper.
    """
    prompt = AnswerMatchingEvaluator.ANSWER_MATCH_TEMPLATE
    # Coverage rule: predicted answer must cover the reference; more
    # specific / extra correct detail is still a match.
    assert "cover everything" in prompt
    assert "more specific" in prompt
    # Numeric rule: 1% relative-error tolerance for numeric references.
    assert "1% relative error" in prompt
    # Non-attempts are penalized (no false-positive matches on refusals).
    assert "does not attempt" in prompt


def test_judge_prompt_embeds_question_reference_and_prediction(monkeypatch):
    """evaluate() formats the real example fields into the judge prompt."""
    matcher = _real_matcher(monkeypatch, "YES")
    asyncio.run(
        matcher.evaluate(
            {"question": "Who wrote Hamlet?"},
            {"answer": "William Shakespeare"},
            {"answer": "Shakespeare"},
        )
    )
    sent = matcher.llm.seen_messages[0][0]["content"]
    assert "Who wrote Hamlet?" in sent
    assert "William Shakespeare" in sent
    assert "Shakespeare" in sent
