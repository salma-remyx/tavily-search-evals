"""Tests for the delegated multi-provider search capability.

These tests import from NON-NEW modules in the repo to prove integration:

  * ``handlers.base_handler.ProviderHandler`` -- the real abstract handler the
    evaluation pipeline duck-types on. The fake providers below subclass it,
    so the swarm is exercised against the genuine handler contract.
  * ``utils.utils`` (``EvaluationType`` and ``save_result``) -- the swarm's
    aggregated output is fed through the framework's real result writer.

The headline assertion captures the WebSwarm insight this module delivers:
when each provider only knows one facet of a multi-faceted query, delegating
sub-queries across the swarm and aggregating recovers *every* facet, while any
single provider answering the whole query recovers only its own.
"""

import asyncio
import os

from handlers.base_handler import ProviderHandler  # non-new module
from utils.delegated_search import (  # the new module under test
    DelegatedSearch,
    default_decompose,
    select_answer,
)
from utils.utils import EvaluationType, save_result  # non-new module in utils/


class _FakeHandler(ProviderHandler):
    """A network-free provider that answers only when it recognizes a facet.

    Mirrors the real handler contract (``search`` coroutine returning a dict
    with an ``answer`` key, an ``is_llm_response`` flag, ``post_process``)
    without touching any API.
    """

    def __init__(self, known):
        super().__init__(api_key="dummy_key", api_url="dummy_url")
        self.known = known  # facet keyword -> answer
        self.is_llm_response = True

    async def search(self, query):
        for keyword, answer in self.known.items():
            if keyword.lower() in (query or "").lower():
                return {"answer": answer, "search_response": answer}
        return {"answer": "", "search_response": ""}

    async def post_process(self, search_response, evaluation_type=EvaluationType.SIMPLEQA):
        return search_response


def _build_swarm(strategy="fan_out"):
    return DelegatedSearch(
        handlers={
            "tavily": _FakeHandler({"france": "Paris"}),
            "brave": _FakeHandler({"japan": "Tokyo"}),
            "serper": _FakeHandler({"brazil": "Brasilia"}),
        },
        strategy=strategy,
    )


# --- decomposition (parameter-free proxy for the paper's task planner) -------


def test_default_decompose_splits_on_facet_boundaries():
    query = "What is the capital of France and the capital of Japan and the capital of Brazil"
    subqueries = default_decompose(query)
    assert len(subqueries) == 3
    assert "France" in subqueries[0]
    assert "Japan" in subqueries[1]
    assert "Brazil" in subqueries[2]


def test_default_decompose_falls_back_to_whole_query():
    assert default_decompose("single facet query") == ["single facet query"]
    assert default_decompose("") == []


def test_select_answer_votes_plurality_and_breaks_ties_by_length():
    assert select_answer(["Paris", "Paris", "Lyon"]) == "Paris"
    # one vote each -> longest wins
    assert select_answer(["NYC", "New York City"]) == "New York City"
    assert select_answer(["", "  ", None]) == ""


# --- the core WebSwarm insight: swarm beats single provider -----------------


def test_swarm_recovers_every_facet_that_any_single_provider_misses():
    swarm = _build_swarm()
    query = (
        "What is the capital of France and the capital of Japan "
        "and the capital of Brazil"
    )
    result = asyncio.run(swarm.search(query))

    combined = result["answer"]
    # The swarm aggregated evidence from all three specialized providers.
    assert "Paris" in combined
    assert "Tokyo" in combined
    assert "Brasilia" in combined
    assert result["coverage"] == 1.0
    assert result["answered_subqueries"] == 3
    assert result["total_subqueries"] == 3
    assert result["providers_used"] == ["brave", "serper", "tavily"]

    # Baseline: a single provider answering the WHOLE query only recovers its
    # own facet -- the gap the delegated swarm closes.
    single = asyncio.run(swarm.handlers["tavily"].search(query))
    assert "Paris" in single["answer"]
    assert "Tokyo" not in single["answer"]
    assert "Brasilia" not in single["answer"]


def test_round_robin_delegates_one_provider_per_subquery():
    swarm = _build_swarm(strategy="round_robin")
    delegation = asyncio.run(swarm.delegate("France and Japan and Brazil"))

    # Each sub-query was assigned to exactly one provider.
    providers_per_subquery = [
        len(entry["answers"]) for entry in delegation["assignments"]
    ]
    assert providers_per_subquery == [1, 1, 1]
    # Three distinct providers were used across the three sub-queries.
    used = {
        next(iter(entry["answers"].keys())) for entry in delegation["assignments"]
    }
    assert used == {"tavily", "brave", "serper"}


def test_failed_provider_does_not_sink_the_swarm():
    class _Boom(ProviderHandler):
        def __init__(self):
            super().__init__(api_key="dummy_key", api_url="dummy_url")
            self.is_llm_response = True

        async def search(self, query):
            raise RuntimeError("provider down")

        async def post_process(self, search_response, evaluation_type=EvaluationType.SIMPLEQA):
            return search_response

    swarm = DelegatedSearch(
        handlers={"tavily": _FakeHandler({"france": "Paris"}), "broken": _Boom()},
    )
    result = asyncio.run(swarm.search("capital of France"))
    # The healthy node still answered despite its sibling throwing.
    assert "Paris" in result["answer"]


# --- integration with the framework's real result writer --------------------


def test_swarm_output_flows_through_real_save_result(tmp_path):
    swarm = _build_swarm()
    query = (
        "What is the capital of France and the capital of Japan "
        "and the capital of Brazil"
    )
    result = asyncio.run(swarm.search(query))

    # Shape the aggregated answer exactly like a SIMPLEQA result row and write
    # it with the framework's existing save_result (non-new utils function).
    row = {
        "index": 0,
        "question": query,
        "reference_answer": "Paris | Tokyo | Brasilia",
        "predicted_answer": result["answer"],
        "is_correct": True,
        "grade": "correct",
        "token_count": 0,
        "token_avg": 0,
    }
    save_result(row, "delegated_swarm", str(tmp_path), EvaluationType.SIMPLEQA)

    written = tmp_path / "delegated_swarm_simpleqa_results.csv"
    assert os.path.exists(written)
    content = written.read_text()
    assert "Paris" in content and "Tokyo" in content and "Brasilia" in content
