"""Tests for the span-level answer-grounding check.

These exercise both the deterministic scoring core of
``utils.span_grounding_check`` and the integration wiring through the
existing (non-new) ``utils.post_processor.PostProcessor`` and
``utils.utils.save_result`` modules.
"""
import csv
import os

import utils.post_processor as post_processor_module
from utils.span_grounding_check import SpanGroundingChecker, UngroundedSpans
from utils.utils import EvaluationType, save_result


# --- deterministic scoring core (no LLM, no network) ---


def test_locate_finds_verbatim_span_and_scores_it():
    answer = "Paris is the capital of France and has 50 million people."
    spans = SpanGroundingChecker._locate(answer, ["50 million people"])

    assert len(spans) == 1
    span = spans[0]
    assert span["text"] == "50 million people"
    assert answer[span["start"]:span["end"]] == "50 million people"
    assert span["grounded"] is False

    score = SpanGroundingChecker._coverage(answer, spans)
    # The unsupported span is a small fraction of a long answer.
    assert 0.0 < score < 1.0


def test_coverage_is_zero_when_nothing_ungrounded():
    answer = "The Eiffel Tower is in Paris."
    assert SpanGroundingChecker._coverage(answer, []) == 0.0


def test_coverage_merges_overlapping_spans_without_double_counting():
    answer = "abcdefghij"  # 10 chars
    # Two overlapping spans [0,4) and [1,5) merge to [0,5) = 5 distinct chars.
    spans = [
        {"text": "abcd", "start": 0, "end": 4, "grounded": False},
        {"text": "bcde", "start": 1, "end": 5, "grounded": False},
    ]
    assert SpanGroundingChecker._coverage(answer, spans) == 0.5


def test_locate_drops_non_verbatim_spans():
    answer = "The capital of France is Paris."
    # Judge paraphrased / mis-copied -> cannot be located -> dropped.
    spans = SpanGroundingChecker._locate(answer, ["paris city", "France"])
    assert [s["text"] for s in spans] == ["France"]


# --- check() end-to-end with an injected judge (no network) ---


class _FakeStructuredLLM:
    """Stand-in for ``ChatOpenAI(...).with_structured_output(...)``."""

    def __init__(self, ungrounded_spans):
        self._spans = ungrounded_spans

    def invoke(self, messages):
        return UngroundedSpans(ungrounded_spans=self._spans)


def test_check_flags_ungrounded_span_via_injected_judge():
    context = "The Eiffel Tower is located in Paris, France."
    answer = "The Eiffel Tower is in Berlin."
    checker = SpanGroundingChecker(structured_llm=_FakeStructuredLLM(["Berlin"]))

    result = checker.check(
        context=context, question="Where is the Eiffel Tower?", answer=answer
    )

    assert result["grounded"] is False
    assert result["hallucination_score"] > 0.0
    assert any(s["text"] == "Berlin" for s in result["ungrounded_spans"])


def test_check_returns_grounded_when_judge_finds_nothing():
    answer = "The Eiffel Tower is in Paris."
    checker = SpanGroundingChecker(structured_llm=_FakeStructuredLLM([]))

    result = checker.check(
        context="The Eiffel Tower is in Paris.",
        question="Where is the Eiffel Tower?",
        answer=answer,
    )

    assert result["grounded"] is True
    assert result["hallucination_score"] == 0.0
    assert result["ungrounded_spans"] == []


def test_check_handles_empty_answer_without_calling_judge():
    checker = SpanGroundingChecker(structured_llm=_FakeStructuredLLM(["unused"]))

    result = checker.check(context="some context", question="q?", answer="   ")

    assert result == {
        "hallucination_score": 0.0,
        "ungrounded_spans": [],
        "grounded": True,
    }


def test_check_is_resilient_to_judge_failure():
    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("API down")

    checker = SpanGroundingChecker(structured_llm=_BoomLLM())

    result = checker.check(context="ctx", question="q?", answer="an answer")

    # Infra failure must not break the run -> fail open to 0.0, flagged.
    assert result["hallucination_score"] == 0.0
    assert result["grounded"] is True
    assert result.get("error")


# --- integration wiring through existing (non-new) modules ---


def test_postprocessor_grounding_wiring(monkeypatch):
    """PostProcessor.check_answer_grounding must delegate to the checker.

    Patches ``SpanGroundingChecker`` in the ``post_processor`` namespace and
    asserts the wiring edit routes (context, question, answer) through it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # ChatOpenAI() construction

    captured = {}

    class _FakeChecker:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, context, question, answer):
            captured.update(context=context, question=question, answer=answer)
            return {
                "hallucination_score": 0.42,
                "ungrounded_spans": [{"text": "x", "start": 0, "end": 1, "grounded": False}],
                "grounded": False,
            }

    monkeypatch.setattr(post_processor_module, "SpanGroundingChecker", _FakeChecker)

    out = post_processor_module.PostProcessor().check_answer_grounding(
        query="Where is the Eiffel Tower?",
        answer="It is in Berlin.",
        search_result="The Eiffel Tower is in Paris.",
    )

    assert out["hallucination_score"] == 0.42
    assert out["grounded"] is False
    # (context, question, answer) contract preserved end-to-end.
    assert captured["question"] == "Where is the Eiffel Tower?"
    assert captured["answer"] == "It is in Berlin."
    assert captured["context"] == "The Eiffel Tower is in Paris."


def test_save_result_persists_hallucination_score(tmp_path):
    """The SimpleQA loop's result dict must surface hallucination_score in CSV.

    Exercises the ``utils.utils.save_result`` fieldname edit with the exact
    result-dict shape ``evaluate_provider_simple_qa`` now produces.
    """
    result = {
        "index": 0,
        "question": "Where is the Eiffel Tower?",
        "reference_answer": "Paris",
        "predicted_answer": "Berlin",
        "is_correct": False,
        "grade": "INCORRECT",
        "hallucination_score": 0.6,
        "answer_grounded": False,  # not in fieldnames -> intentionally dropped
        "token_count": 100,
        "token_avg": 50,
    }

    save_result(result, "tavily", str(tmp_path), EvaluationType.SIMPLEQA)

    csv_path = os.path.join(str(tmp_path), "tavily_simpleqa_results.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert "hallucination_score" in rows[0]
    assert float(rows[0]["hallucination_score"]) == 0.6
