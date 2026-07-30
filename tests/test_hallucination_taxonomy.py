"""Tests for the span-severity taxonomy.

Exercise the deterministic taxonomy core of ``utils.hallucination_taxonomy``
and its integration wiring through the existing (non-new)
``utils.post_processor.PostProcessor`` and ``utils.utils.save_result``
modules. No network / LLM calls -- the categorising judge is injected.
"""
import csv
import os

import utils.post_processor as post_processor_module
from utils.hallucination_taxonomy import (
    SpanCategories,
    SpanCategory,
    SpanLabel,
    SpanTaxonomyClassifier,
    summarize_taxonomy,
)
from utils.utils import EvaluationType, save_result


# --- deterministic taxonomy core (no LLM, no network) ---


def test_summarize_splits_contradicted_from_unverifiable():
    answer = "The tower is in Berlin and stands 500 metres tall."
    labelled = [
        {"text": "Berlin", "start": 16, "end": 22, "label": SpanLabel.CONTRADICTED},
        {"text": "500 metres", "start": 34, "end": 44, "label": SpanLabel.UNVERIFIABLE},
    ]
    summary = summarize_taxonomy(answer, labelled)

    assert summary["hallucinated"] is True
    assert summary["contradiction_score"] > 0.0
    assert summary["unverifiable_score"] > 0.0
    assert [s["text"] for s in summary["contradicted_spans"]] == ["Berlin"]
    assert [s["text"] for s in summary["unverifiable_spans"]] == ["500 metres"]


def test_summarize_is_all_clear_when_no_spans():
    summary = summarize_taxonomy("A fully grounded answer.", [])
    assert summary["hallucinated"] is False
    assert summary["contradiction_score"] == 0.0
    assert summary["unverifiable_score"] == 0.0
    assert summary["contradicted_spans"] == []


# --- classify() with an injected judge (no network) ---


class _FakeStructuredLLM:
    def __init__(self, categories):
        self._categories = categories

    def invoke(self, messages):
        return SpanCategories(
            spans=[SpanCategory(text=t, label=lbl) for t, lbl in self._categories]
        )


def test_classify_no_llm_call_when_nothing_ungrounded():
    class _BoomLLM:
        def invoke(self, messages):
            raise AssertionError("should not be called for a grounded answer")

    clf = SpanTaxonomyClassifier(structured_llm=_BoomLLM())
    summary = clf.classify(context="ctx", question="q?", answer="a", ungrounded_spans=[])
    assert summary["hallucinated"] is False


def test_classify_labels_contradiction_over_high_lexical_overlap():
    # "Paris is in Germany" overlaps heavily with a context about Paris, yet
    # it is CONTRADICTED, not merely unverifiable -- the taxonomy the paper
    # tracks and lexical overlap cannot capture.
    answer = "Paris is in Germany."
    ungrounded = [{"text": "Germany", "start": 12, "end": 19, "grounded": False}]
    clf = SpanTaxonomyClassifier(
        structured_llm=_FakeStructuredLLM([("Germany", SpanLabel.CONTRADICTED)])
    )

    summary = clf.classify(
        context="Paris is the capital of France.",
        question="Where is Paris?",
        answer=answer,
        ungrounded_spans=ungrounded,
    )

    assert summary["hallucinated"] is True
    assert [s["text"] for s in summary["contradicted_spans"]] == ["Germany"]
    assert summary["contradiction_score"] > 0.0


def test_classify_defaults_unlabelled_spans_to_unverifiable():
    ungrounded = [{"text": "42", "start": 0, "end": 2, "grounded": False}]
    # Judge returned nothing for this span -> must fail safe to unverifiable,
    # never silently drop or upgrade to contradicted.
    clf = SpanTaxonomyClassifier(structured_llm=_FakeStructuredLLM([]))

    summary = clf.classify(
        context="ctx", question="q?", answer="42", ungrounded_spans=ungrounded
    )

    assert summary["contradiction_score"] == 0.0
    assert [s["text"] for s in summary["unverifiable_spans"]] == ["42"]


def test_classify_is_resilient_to_judge_failure():
    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("API down")

    ungrounded = [{"text": "Berlin", "start": 0, "end": 6, "grounded": False}]
    clf = SpanTaxonomyClassifier(structured_llm=_BoomLLM())

    summary = clf.classify(
        context="ctx", question="q?", answer="Berlin", ungrounded_spans=ungrounded
    )

    # Fail open to the less-severe class: no false contradictions.
    assert summary["contradiction_score"] == 0.0
    assert [s["text"] for s in summary["unverifiable_spans"]] == ["Berlin"]


# --- integration wiring through existing (non-new) modules ---


def test_postprocessor_enriches_grounding_with_taxonomy(monkeypatch):
    """PostProcessor.check_answer_grounding must fold the taxonomy into its result."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # ChatOpenAI() construction

    class _FakeChecker:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, context, question, answer):
            return {
                "hallucination_score": 0.5,
                "ungrounded_spans": [
                    {"text": "Germany", "start": 12, "end": 19, "grounded": False}
                ],
                "grounded": False,
            }

    class _FakeTaxonomy:
        def __init__(self, *args, **kwargs):
            pass

        def classify(self, context, question, answer, ungrounded_spans):
            # Contract: receives the checker's located ungrounded spans.
            assert ungrounded_spans[0]["text"] == "Germany"
            return summarize_taxonomy(
                answer,
                [{"text": "Germany", "start": 12, "end": 19, "label": SpanLabel.CONTRADICTED}],
            )

    monkeypatch.setattr(post_processor_module, "SpanGroundingChecker", _FakeChecker)
    monkeypatch.setattr(post_processor_module, "SpanTaxonomyClassifier", _FakeTaxonomy)

    out = post_processor_module.PostProcessor().check_answer_grounding(
        query="Where is Paris?",
        answer="Paris is in Germany.",
        search_result="Paris is the capital of France.",
    )

    # Baseline grounding fields preserved, taxonomy fields merged in.
    assert out["hallucination_score"] == 0.5
    assert out["grounded"] is False
    assert out["contradiction_score"] > 0.0
    assert [s["text"] for s in out["contradicted_spans"]] == ["Germany"]


def test_save_result_persists_contradiction_score(tmp_path):
    """The SimpleQA result CSV must carry the new contradiction_score column."""
    result = {
        "index": 0,
        "question": "Where is Paris?",
        "reference_answer": "France",
        "predicted_answer": "Germany",
        "is_correct": False,
        "grade": "INCORRECT",
        "hallucination_score": 0.5,
        "contradiction_score": 0.5,
        "answer_grounded": False,  # not in fieldnames -> intentionally dropped
        "token_count": 100,
        "token_avg": 50,
    }

    save_result(result, "tavily", str(tmp_path), EvaluationType.SIMPLEQA)

    csv_path = os.path.join(str(tmp_path), "tavily_simpleqa_results.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert "contradiction_score" in rows[0]
    assert float(rows[0]["contradiction_score"]) == 0.5
