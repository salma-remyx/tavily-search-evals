"""Delegated multi-provider web search.

Inspired by WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide
Web Search. The paper's central observation is that a single search agent is
bounded by one trajectory, while delegating sub-queries across a *swarm* of
cooperating search nodes and aggregating their evidence upward yields better
coverage on multi-faceted, research-style queries.

This module applies that insight to this repo's actual surface: the diverse
set of search providers the framework already integrates (Tavily, Brave,
Serper, Perplexity, ...) are treated as the swarm. A complex query is
decomposed into sub-queries, each sub-query is delegated across the providers,
and the returned evidence is aggregated into a single answer.

Implementation mode: Mode 3 (inspired experiment). The paper's full method
shape (an LLM runtime that recursively instantiates agentic nodes, probes the
web's information structure, and reuses process-level experience) is not
something this benchmark harness hosts. The core *insight* -- decompose,
delegate across many searchers, aggregate -- maps cleanly onto the existing
handler infrastructure and is implemented here. The paper's auxiliary LLM
components are substituted with parameter-free proxies (Mode 2-style):

  * LLM task decomposition      -> deterministic facet splitting
                                   (inject ``decomposer`` to restore an
                                   LLM planner).
  * Recursive delegation of     -> bounded one-level delegation across the
    unbounded depth                available providers (the providers ARE
                                   the swarm nodes); the aggregation result
                                   is returned upward exactly as the paper
                                   describes.
  * Learned answer aggregation  -> plurality vote + evidence union
                                   (inject ``combiner`` to restore an
                                   LLM aggregator).

``DelegatedSearch`` exposes the same ``search`` / ``post_process`` /
``is_llm_response`` surface the evaluation pipeline duck-types on every
provider, so it can be passed to ``evaluate_provider_simple_qa`` as if it were
another provider -- the swarm then competes head-to-head with the individual
providers the team already benchmarks.
"""

import asyncio
import logging
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Conjunction / facet cues used to split a multi-faceted query into
# independent sub-queries -- a parameter-free proxy for the paper's LLM-driven
# task decomposition. Non-capturing groups keep re.split from injecting the
# matched separators back into the result.
_FACET_SPLIT_RE = re.compile(
    r"\s+(?:and|vs\.?|versus|compared\s+to|compared\s+with)\s+"
    r"|;+"
    r"|\n+",
    flags=re.IGNORECASE,
)


def default_decompose(query: str, max_subqueries: int = 8) -> List[str]:
    """Split a multi-faceted query into sub-queries without an LLM.

    Falls back to the whole query when no facet boundary is found, so a
    single-faceted query still produces one sub-query (and the swarm degrades
    gracefully into a plain multi-provider ensemble on that query).
    """
    if not query or not query.strip():
        return []
    parts = [p.strip(" \t.,;-") for p in _FACET_SPLIT_RE.split(query)]

    seen = set()
    subqueries: List[str] = []
    for part in parts:
        key = part.lower()
        if part and key not in seen:
            seen.add(key)
            subqueries.append(part)

    return subqueries[:max_subqueries] or [query.strip()]


def select_answer(answers: List[str]) -> str:
    """Plurality vote over a group of answers to the *same* sub-query.

    Ties break toward the longest answer, then the earliest -- a
    parameter-free proxy for the paper's evidence-grounded aggregation. Empty
    answers are ignored; if every answer is empty, "" is returned.
    """
    non_empty = [a for a in answers if a and str(a).strip()]
    if not non_empty:
        return ""

    groups: Dict[str, List[str]] = {}
    for answer in non_empty:
        norm = re.sub(r"\s+", " ", answer.strip()).lower()
        groups.setdefault(norm, []).append(answer)

    # Most votes first, then longest representative, preserving first-seen.
    ranked = sorted(
        groups.values(),
        key=lambda group: (-len(group), -len(group[0])),
    )
    return ranked[0][0]


async def _gather(task_map: Dict[str, Coroutine]) -> Dict[str, str]:
    """Await a name->coroutine mapping, never letting one failure sink others."""
    keys = list(task_map.keys())
    values = await asyncio.gather(*[task_map[key] for key in keys])
    return dict(zip(keys, values))


class DelegatedSearch:
    """Decompose a query, delegate sub-queries across providers, aggregate."""

    def __init__(
        self,
        handlers: Dict[str, Any],
        decomposer: Optional[Callable[[str], List[str]]] = None,
        combiner: Optional[Callable[[List[str]], str]] = None,
        strategy: str = "fan_out",
        max_subqueries: int = 6,
    ):
        """Initialize the delegated search swarm.

        Args:
            handlers: Mapping of provider name -> provider handler. Each
                handler must expose the ``search`` coroutine used throughout
                the evaluation pipeline.
            decomposer: Optional callable turning a query into sub-queries.
                Defaults to :func:`default_decompose`.
            combiner: Optional callable merging per-sub-query answers into the
                final answer. Defaults to a deterministic evidence union.
            strategy: ``"fan_out"`` sends every sub-query to every provider
                (widest coverage); ``"round_robin"`` assigns one provider per
                sub-query (cheapest, maximizes provider diversity).
            max_subqueries: Cap on the number of sub-queries delegated, to
                bound the wide-search fan-out cost.
        """
        if not handlers:
            raise ValueError("DelegatedSearch requires at least one handler")
        if strategy not in ("fan_out", "round_robin"):
            raise ValueError(f"unknown strategy: {strategy!r}")

        self.handlers = dict(handlers)
        self.decomposer = decomposer or default_decompose
        self.combiner = combiner
        self.strategy = strategy
        self.max_subqueries = max_subqueries
        # The swarm synthesizes its own answer, so it behaves like an
        # LLM-response provider (e.g. GPTR) in the existing pipeline.
        self.is_llm_response = True

    def decompose(self, query: str) -> List[str]:
        subqueries = list(self.decomposer(query) or [])[: self.max_subqueries]
        if subqueries:
            return subqueries
        return [query.strip()] if query and query.strip() else []

    async def _query_handler(self, handler: Any, subquery: str) -> str:
        try:
            result = await handler.search(subquery)
            return str((result or {}).get("answer", "") or "")
        except Exception as exc:  # a failed node must not sink the whole swarm
            logger.warning("delegated sub-query failed: %s", exc)
            return ""

    async def delegate(self, query: str) -> Dict[str, Any]:
        """Decompose the query and fan the sub-queries out across providers."""
        subqueries = self.decompose(query)
        names = list(self.handlers.keys())
        assignments: List[Dict[str, Any]] = []

        if self.strategy == "fan_out":
            for subquery in subqueries:
                answers = await _gather(
                    {
                        name: self._query_handler(handler, subquery)
                        for name, handler in self.handlers.items()
                    }
                )
                assignments.append({"subquery": subquery, "answers": answers})
        else:  # round_robin
            for index, subquery in enumerate(subqueries):
                name = names[index % len(names)]
                answer = await self._query_handler(self.handlers[name], subquery)
                assignments.append({"subquery": subquery, "answers": {name: answer}})

        return {"query": query, "subqueries": subqueries, "assignments": assignments}

    def aggregate(self, delegation: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate delegated evidence upward into a single combined answer."""
        per_subquery: List[Dict[str, Any]] = []
        winners: List[str] = []
        providers_used = set()

        for entry in delegation.get("assignments", []):
            answers = entry.get("answers", {}) or {}
            winner = select_answer(list(answers.values()))
            contributors = [name for name, ans in answers.items() if ans and str(ans).strip()]
            providers_used.update(contributors)
            per_subquery.append(
                {"subquery": entry.get("subquery"), "answer": winner, "providers": contributors}
            )
            if winner:
                winners.append(winner)

        answered = sum(1 for item in per_subquery if item["answer"])
        total = len(per_subquery)
        coverage = (answered / total) if total else 0.0

        if self.combiner is not None:
            combined = self.combiner(winners)
        else:
            # Distinct sub-queries are distinct facets -- union their evidence
            # rather than voting across them.
            combined = " | ".join(winners)

        return {
            "answer": combined,
            "search_response": {
                "query": delegation.get("query"),
                "subqueries": delegation.get("subqueries", []),
                "per_subquery": per_subquery,
            },
            "coverage": round(coverage, 3),
            "answered_subqueries": answered,
            "total_subqueries": total,
            "providers_used": sorted(providers_used),
        }

    async def search(self, query: str) -> Dict[str, Any]:
        """Run the full decompose -> delegate -> aggregate pipeline.

        Returns a dict shaped like a provider handler result (with ``answer``
        and ``search_response``) plus swarm-level coverage stats.
        """
        delegation = await self.delegate(query)
        return self.aggregate(delegation)

    async def post_process(self, search_response: Any, evaluation_type: Any = None) -> str:
        """Identity post-process -- the answer is already synthesized."""
        if isinstance(search_response, dict):
            return search_response.get("answer", "")
        return search_response


async def run_delegated_evaluation(
    query: str,
    search_provider_params: Dict[str, Dict[str, Any]],
    strategy: str = "fan_out",
    token_model: str = "gpt-4.1",
) -> Dict[str, Any]:
    """Run the swarm over a single query using the framework's real handlers.

    Builds providers through the existing ``get_search_handlers`` factory so
    the delegation targets the same live providers the benchmark uses, then
    delegates and aggregates.
    """
    # Lazy import keeps this module importable without the full pipeline's
    # transitive dependencies loaded at import time.
    from run_evaluation import get_search_handlers

    handler_objects = await get_search_handlers(search_provider_params, token_model)
    names = [name.lower() for name in search_provider_params.keys()]
    handlers = {
        name: handler
        for name, handler in zip(names, handler_objects)
        if handler is not None
    }
    if not handlers:
        raise ValueError("no search handlers could be initialized from the given params")

    swarm = DelegatedSearch(handlers, strategy=strategy)
    return await swarm.search(query)


__all__ = [
    "DelegatedSearch",
    "default_decompose",
    "select_answer",
    "run_delegated_evaluation",
]
