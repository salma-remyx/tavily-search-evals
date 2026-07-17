"""Provider Elo ranking from pairwise answer comparisons.

Adapted from RAGElo (Carrasco et al., 2024, "Evaluating RAG-Fusion with
RAGElo: an Automated Elo-based Framework", arXiv:2406.14783). RAGElo's core
idea is to stop ranking retrieval systems by raw accuracy and instead rank
them with an Elo rating built from per-question *pairwise* comparisons: on
each question two systems are played against each other, an outcome
(win / loss / tie) is decided, and the outcomes are aggregated into Elo
ratings. The pairwise view is more informative than accuracy because a
system can outrank another even when neither lands the gold answer, and Elo
softens the noise of any single question.

This module delivers that core mechanism (pairwise comparison -> Elo) at full
fidelity while substituting RAGElo's auxiliary components for target-native
equivalents (Mode 2 / adapted port):

  * RAGElo's bespoke LLM pairwise judge is replaced, by default, with a
    parameter-free proxy that derives win/loss/tie from the per-question
    correctness grades the repo's existing ``CorrectnessEvaluator`` already
    produces. This reuses the existing data contract and adds no extra API
    calls. The original LLM pairwise judge is still available
    (``PairwiseAnswerJudge``, structured-output pattern mirroring
    ``CorrectnessEvaluator``) for callers who want RAGElo's full-fidelity
    behavior on the raw predicted answers.
  * RAGElo's separate RAG-QA harness / domain benchmark is cut: ranking runs
    on the ``provider_results`` the existing SimpleQA pipeline already
    produces, so evaluation stays in the existing pipeline.

The deterministic Elo aggregation (``EloRanker``) is kept verbatim.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated

load_dotenv()

logger = logging.getLogger(__name__)


# Ordering used by the parameter-free proxy judge. A CORRECT answer beats a
# NOT_ATTEMPTED answer, which beats an INCORRECT / ERROR one. This matches the
# semantics CorrectnessEvaluator already assigns to its grades.
GRADE_STRENGTH: Dict[str, int] = {
    "CORRECT": 2,
    "NOT_ATTEMPTED": 1,
    "INCORRECT": 0,
    "ERROR": 0,
}


class Outcome(str, Enum):
    """Result of a single pairwise comparison between two providers."""

    A_WINS = "A"
    B_WINS = "B"
    TIE = "TIE"


@dataclass
class EloRanker:
    """Deterministic Elo rating aggregator over pairwise matches.

    Standard Elo expected-score update: for players ``a`` (rating ``Ra``) and
    ``b`` (rating ``Rb``), ``Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))`` and each
    player's rating moves by ``k * (actual - expected)``. A win scores 1, a
    loss 0, a tie 0.5 for both.
    """

    k: float = 32.0
    base_rating: float = 1000.0
    _ratings: Dict[str, float] = field(default_factory=dict)
    _wins: Dict[str, int] = field(default_factory=dict)
    _losses: Dict[str, int] = field(default_factory=dict)
    _ties: Dict[str, int] = field(default_factory=dict)

    def register(self, player: str) -> None:
        """Seed a player at the base rating with zeroed match counts."""
        self._ensure(player)

    def _ensure(self, player: str) -> None:
        for table in (self._ratings, self._wins, self._losses, self._ties):
            table.setdefault(player, self.base_rating if table is self._ratings else 0)

    def record_match(self, a: str, b: str, outcome: Outcome) -> None:
        """Apply one pairwise outcome to both players' ratings."""
        self._ensure(a)
        self._ensure(b)
        ra, rb = self._ratings[a], self._ratings[b]
        expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        expected_b = 1.0 - expected_a

        if outcome is Outcome.A_WINS:
            score_a, score_b = 1.0, 0.0
            self._wins[a] += 1
            self._losses[b] += 1
        elif outcome is Outcome.B_WINS:
            score_a, score_b = 0.0, 1.0
            self._losses[a] += 1
            self._wins[b] += 1
        else:  # Outcome.TIE
            score_a, score_b = 0.5, 0.5
            self._ties[a] += 1
            self._ties[b] += 1

        self._ratings[a] += self.k * (score_a - expected_a)
        self._ratings[b] += self.k * (score_b - expected_b)

    @property
    def ratings(self) -> Dict[str, float]:
        return dict(self._ratings)

    def standings(self) -> List[Tuple[str, float]]:
        """Players sorted by rating, highest first."""
        return sorted(self._ratings.items(), key=lambda kv: kv[1], reverse=True)


class PairwiseGrade(BaseModel):
    """Schema for the LLM pairwise judge's structured output."""

    verdict: Annotated[str, "A, B, or TIE"]


class PairwiseAnswerJudge:
    """LLM pairwise judge mirroring ``CorrectnessEvaluator``'s style.

    Compares two predicted answers for the same question against the reference
    (gold) answer and returns which is better (or a tie). This is RAGElo's
    original auxiliary judge; it is optional — the default ranking path uses
    the parameter-free correctness-grade proxy and needs no API calls.
    """

    TEMPLATE = """
You are judging two search providers' answers to the same question.
Compare them against the reference (gold) answer and decide which is better.

Only semantic meaning matters; capitalization, punctuation, grammar, and order do not.
An answer that fully contains the important information in the reference, with no
contradictions, is better than one that is missing information or contradicts it.
If both are equally correct (or equally wrong), reply TIE.

```
Question: {question}
Reference answer: {reference_answer}
Provider A answer: {answer_a}
Provider B answer: {answer_b}
```

Reply with exactly one of:
A: Provider A is better
B: Provider B is better
TIE: They are equally good

Just return "A", "B", or "TIE", with no text around it.
""".strip()

    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0):
        from langchain_openai import ChatOpenAI  # imported lazily so the core
        # ranking path never requires an LLM client or API key.

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(PairwiseGrade)

    async def judge(
        self,
        question: str,
        reference_answer: str,
        answer_a: str,
        answer_b: str,
    ) -> Outcome:
        prompt = self.TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        verdict = response.verdict.strip().upper()
        if verdict in ("A", Outcome.A_WINS.value):
            return Outcome.A_WINS
        if verdict in ("B", Outcome.B_WINS.value):
            return Outcome.B_WINS
        return Outcome.TIE


def grade_outcome(grade_a: str, grade_b: str) -> Outcome:
    """Parameter-free proxy judge: derive win/loss/tie from correctness grades.

    This substitutes for RAGElo's LLM pairwise judge. It reuses the
    ``CorrectnessEvaluator`` grades the pipeline already computes, so it adds
    no extra API calls and no new data contract.
    """
    strength_a = GRADE_STRENGTH.get(str(grade_a).strip().upper(), 0)
    strength_b = GRADE_STRENGTH.get(str(grade_b).strip().upper(), 0)
    if strength_a > strength_b:
        return Outcome.A_WINS
    if strength_b > strength_a:
        return Outcome.B_WINS
    return Outcome.TIE


def _index_by_query(provider_results: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Group per-provider per-question results by question index.

    Mirrors the per-row contract that ``evaluate_provider_simple_qa`` writes:
    each result row carries ``index``, ``question``, ``reference_answer``,
    ``predicted_answer`` and ``grade``. Rows whose grade cannot be ranked
    (e.g. unparsed errors) are skipped so they neither help nor hurt a rating.
    """
    by_query: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for provider, payload in provider_results.items():
        for row in payload.get("results", []):
            grade = str(row.get("grade", "")).strip().upper()
            if grade not in GRADE_STRENGTH:
                continue
            try:
                idx = int(row["index"])
            except (KeyError, TypeError, ValueError):
                continue
            by_query.setdefault(idx, {})[provider] = row
    return by_query


async def compute_elo_ranking(
    provider_results: Dict[str, Dict[str, Any]],
    judge: Optional[PairwiseAnswerJudge] = None,
    use_judge: bool = False,
    evaluator_model: str = "gpt-4.1",
    k: float = 32.0,
    base_rating: float = 1000.0,
) -> List[Dict[str, Any]]:
    """Rank search providers with Elo over pairwise per-question comparisons.

    Args:
        provider_results: Output of the SimpleQA pipeline (one entry per
            provider, each with a ``results`` list of per-question rows).
        judge: Optional injected pairwise judge (e.g. a stub in tests). When
            ``use_judge`` is true and this is None, a ``PairwiseAnswerJudge``
            is constructed.
        use_judge: Use the LLM pairwise judge on the raw answers instead of the
            default parameter-free correctness-grade proxy.
        evaluator_model: Model for the LLM judge when one must be constructed.
        k, base_rating: Elo parameters.

    Returns:
        Providers sorted by Elo rating (highest first). Each entry is::

            {"provider", "elo", "rank", "wins", "losses", "ties", "matches"}
    """
    ranker = EloRanker(k=k, base_rating=base_rating)
    # Every evaluated provider appears in the ranking, even when pairwise data
    # is thin (no shared questions / a single provider) — at the base rating.
    for provider in provider_results:
        ranker.register(provider)

    # Lazily construct the LLM judge only if it is actually needed, so the
    # default path stays API-free.
    active_judge: Optional[PairwiseAnswerJudge]
    if use_judge and judge is None:
        active_judge = PairwiseAnswerJudge(model_name=evaluator_model)
    else:
        active_judge = judge if use_judge else None

    by_query = _index_by_query(provider_results)

    # Deterministic iteration order: question index ascending, then providers
    # in sorted order within each question.
    for idx in sorted(by_query):
        rows = by_query[idx]
        for a, b in combinations(sorted(rows), 2):
            row_a, row_b = rows[a], rows[b]
            if active_judge is not None:
                outcome = await active_judge.judge(
                    question=row_a.get("question", ""),
                    reference_answer=row_a.get("reference_answer", ""),
                    answer_a=str(row_a.get("predicted_answer", "")),
                    answer_b=str(row_b.get("predicted_answer", "")),
                )
            else:
                outcome = grade_outcome(row_a.get("grade", ""), row_b.get("grade", ""))
            ranker.record_match(a, b, outcome)

    standings = ranker.standings()
    ranking: List[Dict[str, Any]] = []
    for position, (provider, elo) in enumerate(standings, start=1):
        wins = ranker._wins.get(provider, 0)
        losses = ranker._losses.get(provider, 0)
        ties = ranker._ties.get(provider, 0)
        ranking.append(
            {
                "provider": provider,
                "elo": round(elo, 1),
                "rank": position,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "matches": wins + losses + ties,
            }
        )
    return ranking


def save_elo_ranking(ranking: List[Dict[str, Any]], output_dir: str) -> str:
    """Persist the Elo ranking to ``<output_dir>/elo_ranking.json``.

    Returns the path written. Kept here (rather than in utils/) so the
    run_evaluation wiring edit stays a one-line call.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "elo_ranking.json")
    with open(path, "w") as f:
        json.dump(ranking, f, indent=2)
    logger.info(f"Saved Elo ranking to {path}")
    return path
