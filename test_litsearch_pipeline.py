"""End-to-end test of the LITSEARCH evaluation path through ``run_evaluation``.

``run_evaluation`` pulls in every search handler at import time, including the
``gptr`` handler which depends on the heavyweight ``gpt_researcher`` package.
That package is irrelevant to the LitSearch path, so we stub it (only if it is
not already importable) purely to let ``run_evaluation`` be imported here. The
handler under test is an in-memory fake, so no network or API keys are used.
"""

import asyncio
import importlib.util
import sys
import types

# Stub gpt_researcher only if the real package is absent; never override it.
if importlib.util.find_spec("gpt_researcher") is None:
    _stub = types.ModuleType("gpt_researcher")
    _stub.GPTResearcher = object
    sys.modules.setdefault("gpt_researcher", _stub)

import pandas as pd

import run_evaluation
from utils.utils import EvaluationType


class _FakeHandler:
    """Minimal stand-in for a ProviderHandler over a fixed document set."""

    is_llm_response = False

    def __init__(self, docs):
        self._docs = docs

    async def search(self, query):
        return {"search_response": {"results": self._docs}, "answer": ""}

    async def post_process(self, search_response, evaluation_type=EvaluationType.SIMPLEQA):
        # Mirror the document-relevance path: one stringified dict per result.
        docs = [str(doc) for doc in search_response["search_response"]["results"]]
        return docs, 0, 0


def test_evaluate_provider_litsearch_end_to_end(tmp_path):
    docs = [
        {
            "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            "url": "https://arxiv.org/abs/2201.11903",
            "content": "",
        }
    ]
    handler = _FakeHandler(docs)
    examples = [
        {
            "question": "chain-of-thought reasoning",
            "gold_papers": [
                "arxiv:2201.11903 Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
            ],
            "index": 0,
        },
        {
            "question": "a query whose gold paper is not retrieved",
            "gold_papers": ["arxiv:9999.99999 A Paper That Does Not Exist"],
            "index": 1,
        },
    ]

    result = asyncio.run(
        run_evaluation.evaluate_provider_litsearch(
            "tavily", handler, examples, str(tmp_path), EvaluationType.LITSEARCH, batch_size=2
        )
    )

    assert result["provider"] == "tavily"
    assert result["total_count"] == 2
    # q1 full recall (1.0), q2 no recall (0.0) -> mean 0.5
    assert result["mean_recall"] == 0.5

    written = pd.read_csv(tmp_path / "tavily_litsearch_results.csv")
    assert {"index", "gold_count", "matched_count", "recall", "matched"} <= set(written.columns)
    assert len(written) == 2
    assert sorted(written["recall"].tolist()) == [0.0, 1.0]
