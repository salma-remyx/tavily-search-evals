"""Tests for the source-recall evaluation wiring.

These tests exercise the integration with the *existing* harness, not just
the new module in isolation: they import from ``utils`` (the call-site
package) and assert the wiring edits (``EvaluationType.SOURCE_RECALL``,
``save_result`` fieldnames, ``prepare_examples`` column forwarding) plus the
core DeepScholar-Bench source-recall primitive.
"""
import asyncio
import csv
import os

# Imports from a NON-NEW module (utils) — exercises the wiring edits.
from utils import EvaluationType, prepare_examples, save_result

from evaluators.source_recall_evaluator import (
    SourceRecallEvaluator,
    evaluate_provider_source_recall,
    extract_result_urls,
    load_source_recall_eval_data,
    normalize_source,
)


DATASET_PATH = os.path.join("datasets", "source_recall_test_set.json")


# --------------------------------------------------------------------------- #
# Wiring: the new evaluation type is registered in the utils enum.
# --------------------------------------------------------------------------- #
def test_evaluation_type_source_recall_registered():
    assert EvaluationType.SOURCE_RECALL.value == "source_recall"
    # still side-by-side with the existing types
    values = {e.value for e in EvaluationType}
    assert {"simpleqa", "document_relevance", "source_recall"} <= values


# --------------------------------------------------------------------------- #
# Core metric: URL normalization makes near-duplicate URLs comparable.
# --------------------------------------------------------------------------- #
def test_normalize_source_strips_scheme_www_slash_query():
    a = normalize_source("https://www.en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)/")
    b = normalize_source("http://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)?oldid=1#sec")
    assert a == b
    assert a == "en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"


def test_normalize_source_handles_empty_and_garbage():
    assert normalize_source("") == ""
    assert normalize_source(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Core metric: recall@k / precision@k against a gold source set.
# --------------------------------------------------------------------------- #
def test_score_partial_recall():
    ev = SourceRecallEvaluator(k=10)
    gold = ["https://en.wikipedia.org/wiki/X", "https://arxiv.org/abs/1706.03762"]
    retrieved = ["https://www.en.wikipedia.org/wiki/X/", "https://example.com/unrelated"]
    s = ev.score(gold, retrieved)
    assert s["gold_count"] == 2
    assert s["matched_count"] == 1
    assert s["recall"] == 0.5
    assert s["precision"] == 0.5
    assert s["hit"] == 1.0


def test_score_full_recall():
    ev = SourceRecallEvaluator(k=10)
    gold = ["https://arxiv.org/abs/1810.04805"]
    retrieved = ["https://www.arxiv.org/abs/1810.04805/"]
    s = ev.score(gold, retrieved)
    assert s["recall"] == 1.0
    assert s["hit"] == 1.0


def test_score_no_match():
    ev = SourceRecallEvaluator(k=10)
    gold = ["https://arxiv.org/abs/1706.03762"]
    retrieved = ["https://example.com/something-else"]
    s = ev.score(gold, retrieved)
    assert s["recall"] == 0.0
    assert s["hit"] == 0.0


def test_recall_at_k_truncates_ranked_retrieval():
    # k caps how many retrieved results are considered, in rank order.
    ev = SourceRecallEvaluator(k=1)
    gold = ["https://a.example/x", "https://b.example/y"]
    retrieved = ["https://b.example/y", "https://a.example/x"]  # both present, but k=1
    s = ev.score(gold, retrieved)
    assert s["matched_count"] == 1
    assert s["recall"] == 0.5


def test_extract_result_urls_reads_provider_shape():
    response = {"answer": "", "search_response": {"results": [
        {"url": "https://a.example/1", "content": "..."},
        {"url": "https://b.example/2", "content": "..."},
    ]}}
    assert extract_result_urls(response) == ["https://a.example/1", "https://b.example/2"]
    assert extract_result_urls({}) == []


# --------------------------------------------------------------------------- #
# Wiring: save_result writes a source-recall row (utils integration).
# --------------------------------------------------------------------------- #
def test_save_result_writes_source_recall_row(tmp_path):
    row = {
        "index": 0,
        "question": "q",
        "gold_count": 2,
        "matched_count": 1,
        "recall": 0.5,
        "precision": 0.5,
        "hit": 1,
        "retrieved_count": 2,
        "grade": "completed",
        # keys outside the fieldnames get filtered, like the other types:
        "extra_ignored": "should_not_appear",
    }
    save_result(row, "tavily", str(tmp_path), EvaluationType.SOURCE_RECALL)

    out = tmp_path / "tavily_source_recall_results.csv"
    assert out.exists()
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["recall"] == "0.5"
    assert rows[0]["grade"] == "completed"
    assert "extra_ignored" not in rows[0]


# --------------------------------------------------------------------------- #
# Wiring: prepare_examples forwards the gold_sources column (utils integration).
# --------------------------------------------------------------------------- #
def test_dataset_loads_and_prepare_examples_forwards_gold_sources():
    df = load_source_recall_eval_data(DATASET_PATH)
    assert "gold_sources" in df.columns
    assert len(df) >= 1

    examples = prepare_examples(
        df, ["tavily"], rerun=False, results_dir="results",
        random_sample=None, evaluation_type=EvaluationType.SOURCE_RECALL,
    )
    first = examples["tavily"][0]
    # core fields still present
    assert first["question"] and "index" in first
    # the source-recall-specific column flows through prepare_examples
    assert isinstance(first["gold_sources"], list) and len(first["gold_sources"]) >= 1


# --------------------------------------------------------------------------- #
# Integration: the per-provider runner scores retrieval end-to-end with a
# mock handler (no network / API key) and persists via utils.save_result.
# --------------------------------------------------------------------------- #
class _FakeHandler:
    """Minimal stand-in for a search provider handler."""

    is_llm_response = False

    def __init__(self, urls_by_query):
        self.urls_by_query = urls_by_query

    async def search(self, query):
        urls = self.urls_by_query.get(query, [])
        return {"answer": "", "search_response": {"results": [{"url": u} for u in urls]}}


def test_evaluate_provider_source_recall_end_to_end(tmp_path):
    examples = [
        {"question": "q1", "gold_sources": ["https://en.wikipedia.org/wiki/X"], "index": 0},
        {"question": "q2", "gold_sources": ["https://arxiv.org/abs/1706.03762"], "index": 1},
    ]
    # q1: gold source retrieved (www + trailing slash variant). q2: not retrieved.
    fake = _FakeHandler({
        "q1": ["https://www.en.wikipedia.org/wiki/X/"],
        "q2": ["https://example.com/unrelated"],
    })

    result = asyncio.run(evaluate_provider_source_recall(
        "mock", fake, examples, str(tmp_path), k=10, batch_size=2,
    ))

    assert result["provider"] == "mock"
    assert result["total_count"] == 2
    assert result["mean_recall"] == 0.5  # 1.0 + 0.0 / 2
    assert result["hit_rate"] == 0.5
    assert result["matched_sources"] == 1
    assert result["total_gold_sources"] == 2

    # the runner persisted per-example rows through utils.save_result
    out = tmp_path / "mock_source_recall_results.csv"
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    recalls = sorted(float(r["recall"]) for r in rows)
    assert recalls == [0.0, 1.0]
