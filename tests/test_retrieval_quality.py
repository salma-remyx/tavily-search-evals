"""Tests for the eRAG reference-free retrieval-quality scoring.

Covers both the standalone metric aggregation (``utils.retrieval_quality``)
and the integration wiring on the existing ``utils.post_processor``
module, which is the production call site that invokes the scorer.

``langchain_openai`` (a heavy optional dependency of ``PostProcessor``) is
stubbed before import so the wiring can be exercised without network access
or the real SDK installed.
"""

import asyncio
import os
import sys
import types

# --- Stub the heavy optional dependencies pulled in by the utils package. -
# Inserted before importing anything from ``utils`` so the real SDKs are not
# required to test the integration wiring (utils/__init__ imports pandas and
# quotientai; PostProcessor imports langchain_openai).
def _install_stub(name, **attrs):
    if name not in sys.modules:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


class _FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt):
        # extract_answer returns result.content; echo the document so the
        # generator path runs end to end without a real model.
        return types.SimpleNamespace(content=str(prompt))


_install_stub("langchain_openai", ChatOpenAI=_FakeChatOpenAI)
_install_stub("pandas", DataFrame=object)
_install_stub("quotientai", QuotientAI=object, DetectionType=object)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Repo root on the path so `import utils...` resolves when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.retrieval_quality import (  # noqa: E402
    aggregate_labels,
    average_precision,
    mean_metrics,
    reciprocal_rank,
    split_formatted_documents,
)


def _format(docs):
    """Mimic ProviderHandler._format_search_results_for_prompt output."""
    return "\n".join(
        f"\n**Document {i + 1}.** Source: {url}\nContent: {content}"
        for i, (url, content) in enumerate(docs)
    )


def test_split_formatted_documents_roundtrip():
    docs = [("http://a", "alpha"), ("http://b", "beta"), ("http://c", "gamma")]
    parts = split_formatted_documents(_format(docs))
    assert len(parts) == 3
    assert "alpha" in parts[0] and "beta" in parts[1] and "gamma" in parts[2]
    assert split_formatted_documents("") == []


def test_aggregate_labels_metrics():
    # Relevant doc first, then two irrelevant.
    metrics = aggregate_labels([1, 0, 0])
    assert metrics["mrr"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["precision@3"] == round(1 / 3, 4)
    assert metrics["hit"] == 1.0

    # First relevant document at rank 2.
    assert reciprocal_rank([0, 1, 0]) == 0.5
    # No relevant documents -> zero signal.
    assert aggregate_labels([0, 0])["map"] == 0.0
    # AP with relevant docs at ranks 1 and 3: (1/1 + 2/3) / 2.
    assert abs(average_precision([1, 0, 1]) - (1.0 + 2 / 3) / 2) < 1e-9


def test_mean_metrics_skips_empty_queries():
    q1 = {**aggregate_labels([1, 0]), "num_documents": 2}
    q2 = {**aggregate_labels([0, 0]), "num_documents": 2}
    empty = {"num_documents": 0}
    summary = mean_metrics([q1, q2, empty, None])
    assert summary["queries_scored"] == 2
    assert summary["hit"] == 0.5  # one of the two scored queries had a hit


def test_post_processor_scores_retrieval_quality_via_wiring():
    """Exercise the real PostProcessor call site that invokes the scorer."""
    from utils.post_processor import PostProcessor

    processor = PostProcessor(llm_model="gpt-4.1")

    formatted = _format(
        [
            ("http://good", "GOOD supporting evidence"),
            ("http://bad", "unrelated filler"),
            ("http://good2", "another GOOD passage"),
        ]
    )

    async def fake_judge(query, predicted_answer, reference_answer):
        # The generator echoes the document text into the predicted answer,
        # so a "GOOD" document yields a relevant (score 1.0) downstream result.
        return 1.0 if "GOOD" in predicted_answer else 0.0

    metrics = asyncio.run(
        processor.score_retrieval_quality(
            query="who?",
            reference_answer="the gold answer",
            documents=formatted,
            judge_fn=fake_judge,
        )
    )

    assert metrics["num_documents"] == 3
    assert metrics["labels"] == [1, 0, 1]
    assert metrics["mrr"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["erag_precision"] == round(2 / 3, 4)
