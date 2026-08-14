"""Tests for the schema-guided extraction evaluator.

These exercise the wiring through the public ``evaluators`` package (the
existing evaluator registry) rather than the new module directly.
"""

import asyncio

import pytest

from evaluators import CorrectnessEvaluator, SchemaExtractionEvaluator
from evaluators.schema_extraction_evaluator import (
    MATCH_CLASSES,
    SchemaAttribute,
    classify_locally,
    flatten_by_paths,
    match_by_path,
)


class _FakeRecord:
    """Stands in for the structured-output pydantic record."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _FakeGrade:
    def __init__(self, match_class):
        self.match_class = match_class


class ScriptedLLM:
    """Returns the extraction record first, then a grader verdict per call."""

    def __init__(self, extracted, verdict="USEFUL"):
        self.extracted = extracted
        self.verdict = verdict

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if "Extracting structured information" in messages[0]["content"] or "extracting" in messages[0]["content"].lower():
            return _FakeRecord(self.extracted)
        return _FakeGrade(self.verdict)


DOCUMENTS = [
    "Generative AI is dominating as a key technology trend in 2025, reshaping industries.",
    "Other trends include agentic AI and inference-time compute scaling.",
]
GOLD = {
    "answer": "Generative AI is the top emerging technology in 2025.",
    "supporting_snippets": [
        {"quote": "Generative AI is dominating as a key technology trend in 2025", "url": "https://example.com/trends"}
    ],
}


def _make_evaluator(extracted, verdict="USEFUL"):
    llm = ScriptedLLM(extracted, verdict)
    return SchemaExtractionEvaluator(llm=llm), llm


def test_registry_exports_evaluator():
    """The evaluator is importable from the existing evaluators package."""
    assert SchemaExtractionEvaluator is not None
    assert CorrectnessEvaluator is not None


def test_flatten_by_paths_drops_list_indices():
    record = {"a": "x", "snips": [{"q": "1", "u": "2"}, {"q": "3", "u": "4"}]}
    paths = [p for p, _ in flatten_by_paths(record)]
    assert paths == ["a", "snips[].q", "snips[].u", "snips[].q", "snips[].u"]


def test_match_by_path_aligns_variable_cardinality():
    extracted = [("s[].q", "alpha beta"), ("s[].q", "gamma delta")]
    gold = [("s[].q", "delta gamma"), ("s[].q", "beta alpha")]
    pairs = match_by_path(extracted, gold)
    assert [(p, g) for _, p, g in pairs] == [
        ("alpha beta", "beta alpha"),
        ("gamma delta", "delta gamma"),
    ]


def test_match_by_path_flags_unmatched_paths():
    pairs = match_by_path([("answer", "x")], [("other", "y")])
    assert pairs == [("answer", "x", "")]


def test_classify_locally_rubric_shortcuts():
    assert classify_locally("", "gold") == "NON_MATCH"
    assert classify_locally("N/A", "gold") == "NON_MATCH"
    assert classify_locally("Generative AI (2025)!", "generative ai 2025") == "EXACT"
    assert classify_locally("generative ai in 2025", "generative ai") == "SEMANTIC"
    assert classify_locally("completely different", "unrelated gold words") is None


def test_evaluate_scores_rubric_classes():
    extracted = {
        "answer": "Generative AI is the top emerging technology in 2025.",
        "supporting_snippets": [
            {"quote": "Generative AI is dominating", "url": "a different url entirely"}
        ],
    }
    evaluator, _ = _make_evaluator(extracted, verdict="USEFUL")
    result = asyncio.run(
        evaluator.evaluate(
            inputs={"query": "What is the top emerging technology in 2025?", "documents": DOCUMENTS},
            outputs={},
            reference_outputs={"answer": GOLD},
        )
    )
    # answer is an exact token-set match; the quote is a strict subset of the
    # gold quote (SEMANTIC); the url has no token overlap, so it goes to the
    # (scripted) grader.
    assert result["value"]["EXACT"] == 1
    assert result["value"]["SEMANTIC"] == 1
    assert result["value"]["USEFUL"] == 1
    assert result["value"]["NON_MATCH"] == 0
    assert result["score"] == round((1.0 + 1.0 + 0.5) / 3, 3)
    assert all(m["match_class"] in MATCH_CLASSES for m in result["matches"])


def test_evaluate_missing_extraction_is_non_match():
    evaluator, _ = _make_evaluator({"answer": "", "supporting_snippets": []})
    result = asyncio.run(
        evaluator.evaluate(
            inputs={"query": "top tech 2025?", "documents": DOCUMENTS},
            outputs={},
            reference_outputs={"answer": GOLD},
        )
    )
    assert result["value"]["NON_MATCH"] >= 1
    assert result["score"] == 0.0


def test_custom_schema_and_gold_record():
    schema = [
        SchemaAttribute("product_name", "Name of the product."),
        SchemaAttribute("price", "Price of the product."),
    ]
    extracted = {"product_name": "Acme Widget", "price": "$10"}
    gold = {"product_name": "Acme Widget", "price": "$10"}
    evaluator, _ = _make_evaluator(extracted)
    evaluator.schema = tuple(schema)
    result = asyncio.run(
        evaluator.evaluate(
            inputs={"query": "acme widget price", "documents": ["Acme Widget sells for $10."]},
            outputs={},
            reference_outputs={"answer": gold},
        )
    )
    assert result["value"]["EXACT"] == 2


def test_scalar_gold_answer_wrapped_into_record():
    evaluator, _ = _make_evaluator({"answer": "generative ai 2025"})
    result = asyncio.run(
        evaluator.evaluate(
            inputs={"query": "q", "documents": DOCUMENTS},
            outputs={},
            reference_outputs={"answer": "Generative AI is the top emerging technology in 2025."},
        )
    )
    assert any(m["path"] == "answer" for m in result["matches"])


@pytest.mark.parametrize("cls", list(MATCH_CLASSES))
def test_all_rubric_classes_are_scored(cls):
    """Every rubric class maps to a defined score contribution."""
    from evaluators.schema_extraction_evaluator import MATCH_WEIGHTS

    assert MATCH_WEIGHTS[cls] >= 0.0
