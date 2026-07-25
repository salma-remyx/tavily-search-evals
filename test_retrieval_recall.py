"""Integration tests for the LitSearch-style retrieval-recall evaluation.

These tests exercise the wiring edits in the existing ``utils/utils.py``
(``EvaluationType.LITSEARCH``, ``prepare_examples`` gold-papers passthrough,
and the ``save_result``/``save_summary`` LITSEARCH branches) together with the
new ``utils/retrieval_recall.py`` scorer and loader. They go through public
interfaces of existing modules rather than internal attributes.
"""

import json
import os

import pandas as pd

from utils.retrieval_recall import load_litsearch_data, recall_at_k
from utils.utils import EvaluationType, prepare_examples, save_result, save_summary

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# recall_at_k scoring (new module)
# ---------------------------------------------------------------------------

def test_recall_matches_by_arxiv_id():
    docs = [
        {"title": "Unrelated blog", "url": "https://example.com/blog", "content": ""},
        {
            "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            "url": "https://arxiv.org/abs/2201.11903",
            "content": "few-shot",
        },
    ]
    gold = ["arxiv:2201.11903 Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"]
    scored = recall_at_k(docs, gold)
    assert scored["recall"] == 1.0
    assert len(scored["matched"]) == 1


def test_recall_matches_by_title_without_arxiv_url():
    docs = [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "url": "https://www.semanticscholar.org/paper/xyz",
            "content": "abstract",
        }
    ]
    gold = ["arxiv:2005.11401 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"]
    assert recall_at_k(docs, gold)["recall"] == 1.0


def test_recall_zero_when_gold_absent():
    docs = [{"title": "Totally unrelated paper", "url": "https://example.com", "content": "blah"}]
    gold = ["arxiv:2106.09685 LoRA: Low-Rank Adaptation of Large Language Models"]
    scored = recall_at_k(docs, gold)
    assert scored["recall"] == 0.0
    assert scored["matched"] == []


def test_recall_partial_match_and_k_cutoff():
    gold = [
        "arxiv:2103.10360 Finetuned Language Models Are Zero-Shot Learners",
        "arxiv:2212.10560 Self-Instruct: Aligning Language Models with Self-Generated Instructions",
    ]
    docs = [
        {
            "title": "Finetuned Language Models Are Zero-Shot Learners",
            "url": "https://arxiv.org/abs/2103.10360",
            "content": "",
        }
    ]
    assert recall_at_k(docs, gold)["recall"] == 0.5
    # A k cutoff that excludes the only matching doc drops recall to 0.
    assert recall_at_k(docs, gold, k=0)["recall"] == 0.0


def test_recall_handles_stringified_documents():
    # Handlers' document-relevance path returns documents as stringified dicts.
    docs = ["{'title': 'LoRA: Low-Rank Adaptation of Large Language Models', 'url': 'https://arxiv.org/abs/2106.09685'}"]
    gold = ["arxiv:2106.09685 LoRA: Low-Rank Adaptation of Large Language Models"]
    assert recall_at_k(docs, gold)["recall"] == 1.0


# ---------------------------------------------------------------------------
# prepare_examples (NON-NEW utils.utils) — gold_papers passthrough
# ---------------------------------------------------------------------------

def test_prepare_examples_passes_gold_papers():
    df = pd.DataFrame(
        [
            {"problem": "q1", "answer": "a | b", "gold_papers": ["a", "b"], "index": 0},
            {"problem": "q2", "answer": "c", "gold_papers": ["c"], "index": 1},
        ]
    )
    prepared = prepare_examples(
        df, ["tavily"], rerun=False, results_dir="results", evaluation_type=EvaluationType.LITSEARCH
    )
    examples = prepared["tavily"]
    assert len(examples) == 2
    assert examples[0]["gold_papers"] == ["a", "b"]
    assert examples[1]["gold_papers"] == ["c"]
    assert examples[0]["question"] == "q1"


def test_prepare_examples_without_gold_column_is_unaffected():
    df = pd.DataFrame([{"problem": "q", "answer": "a", "index": 0}])
    prepared = prepare_examples(
        df, ["tavily"], rerun=False, results_dir="results", evaluation_type=EvaluationType.SIMPLEQA
    )
    assert "gold_papers" not in prepared["tavily"][0]
    assert prepared["tavily"][0]["answer"] == "a"


# ---------------------------------------------------------------------------
# save_result + save_summary (NON-NEW utils.utils) — LITSEARCH output schema
# ---------------------------------------------------------------------------

def test_save_result_and_summary_litsearch(tmp_path):
    provider = "tavily"
    # Three examples with recall 0.0, 0.5, 1.0 -> mean 0.5, summed matched 0+1+2.
    recalls = [0.0, 0.5, 1.0]
    for i, recall in enumerate(recalls):
        save_result(
            {
                "index": i,
                "question": f"q{i}",
                "gold_count": 2,
                "matched_count": i,
                "recall": recall,
                "matched": json.dumps(["x"] * i),
                "token_count": 0,
                "token_avg": 0,
            },
            provider,
            str(tmp_path),
            EvaluationType.LITSEARCH,
        )

    save_summary(
        {provider: {"provider": provider, "results": [], "mean_recall": 0.5, "total_count": 3}},
        str(tmp_path),
        EvaluationType.LITSEARCH,
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert "mean_recall" in summary.columns
    row = summary.iloc[0]
    assert row["provider"] == provider
    assert row["mean_recall"] == 0.5
    assert int(row["matched_count"]) == 3
    assert int(row["total_count"]) == 3

    per_provider = pd.read_csv(tmp_path / f"{provider}_litsearch_results.csv")
    assert {"index", "gold_count", "matched_count", "recall", "matched"} <= set(per_provider.columns)
    assert len(per_provider) == 3


# ---------------------------------------------------------------------------
# Shipped dataset round-trip
# ---------------------------------------------------------------------------

def test_load_shipped_litsearch_dataset():
    dataset_path = os.path.join(REPO_ROOT, "datasets", "litsearch_test_set.json")
    df = load_litsearch_data(dataset_path)
    assert len(df) == 6
    assert "gold_papers" in df.columns
    assert isinstance(df.iloc[0]["gold_papers"], list)
    # The instruction-tuning query has two gold papers.
    inst = df[df["problem"].str.contains("instruction tuning", case=False)].iloc[0]
    assert len(inst["gold_papers"]) == 2
