"""Interpretable binary-question evaluation for SimpleQA answers.

Adapted from BINEVAL ("Ask, Don't Judge: Binary Questions for
Interpretable LLM Evaluation and Self-Improvement", arXiv:2606.27226).

The core mechanism is kept at full fidelity: an evaluation is decomposed
into atomic *binary* (yes/no) questions, an LLM answers each one
independently for a given output, and the verdicts are aggregated into
per-dimension and overall scores. Because every verdict is preserved, the
resulting scores are inspectable and debuggable rather than a single
opaque grade -- the same critique BINEVAL levels at holistic LLM judges
such as the repo's ``CorrectnessEvaluator`` (one CORRECT / INCORRECT /
NOT_ATTEMPTED label with no signal about *which* facet failed).

Adaptation (Mode 2): BINEVAL generates its question set at runtime via a
meta-prompt. That question *generation* step is replaced here by a
curated, task-specific question bank for factual question answering
(SimpleQA). The question bank is just an input to the evaluator, so a
runtime meta-prompt could be dropped in later without touching the
scoring path. The paper's separate prompt-optimization / self-improvement
loop is intentionally out of scope -- optimization is a downstream concern.
"""

import logging
from dataclasses import dataclass
from typing import Annotated, Dict, List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class BinaryQuestion:
    """A single atomic yes/no question probing one facet of an answer.

    Attributes:
        id: Stable identifier persisted with the verdict.
        dimension: Evaluation dimension this question rolls up into
            (e.g. ``"factual_consistency"``, ``"relevance"``).
        text: The question the LLM answers with yes / no.
        polarity: ``"yes_good"`` if a "yes" answer is favourable,
            ``"yes_bad"`` if a "yes" answer is unfavourable (e.g. a
            question like "does the answer contradict the reference?").
    """

    id: str
    dimension: str
    text: str
    polarity: str = "yes_good"


# Curated SimpleQA question bank. These mirror the facets SimpleQA's
# holistic grader bundles into a single label, decomposed so each one can
# be inspected and scored on its own. This stands in for BINEVAL's
# runtime meta-prompt question-generation step (see module docstring).
DEFAULT_SIMPLEQA_QUESTIONS: List[BinaryQuestion] = [
    BinaryQuestion(
        id="contains_key_info",
        dimension="factual_consistency",
        text=(
            "Does the predicted answer contain the key information from "
            "the reference answer?"
        ),
        polarity="yes_good",
    ),
    BinaryQuestion(
        id="contradicts_reference",
        dimension="factual_consistency",
        text=(
            "Does the predicted answer contain any information that "
            "contradicts the reference answer?"
        ),
        polarity="yes_bad",
    ),
    BinaryQuestion(
        id="addresses_question",
        dimension="relevance",
        text="Does the predicted answer directly address the question being asked?",
        polarity="yes_good",
    ),
    BinaryQuestion(
        id="fully_answers",
        dimension="completeness",
        text=(
            "Does the predicted answer fully answer the question without "
            "omitting key requested information?"
        ),
        polarity="yes_good",
    ),
    BinaryQuestion(
        id="is_concise",
        dimension="conciseness",
        text="Is the predicted answer concise and free of irrelevant information?",
        polarity="yes_good",
    ),
]


class _QuestionVerdict(BaseModel):
    """The model's yes / no answer for a single question."""

    question_id: Annotated[str, "The id of the question being answered"]
    answer: Annotated[bool, "True for yes, False for no"]


class _VerdictSet(BaseModel):
    """Structured container holding the model's answer to every question."""

    verdicts: List[_QuestionVerdict] = Field(default_factory=list)


@dataclass
class BinaryQuestionConfig:
    """Configuration for the binary-question evaluator."""

    model_name: str = "gpt-4.1"
    temperature: float = 0.0


def aggregate_verdicts(
    questions: List[BinaryQuestion], verdict_map: Dict[str, bool]
) -> Dict[str, object]:
    """Aggregate raw yes/no verdicts into per-dimension and overall scores.

    This is the pure scoring core of BINEVAL, kept separate from the LLM
    call so it can be unit-tested without network access. Each verdict is
    normalised to a "good" boolean via the question's polarity, then
    averaged within each dimension and across all questions.

    Args:
        questions: The ordered question bank that was judged.
        verdict_map: Mapping of question id -> raw yes/no answer. A missing
            id is treated as "no".

    Returns:
        ``{"score": float, "dimensions": {dim: float}, "verdicts": {id: bool}}``.
    """
    by_dimension: Dict[str, List[bool]] = {}
    goods: List[bool] = []
    recorded: Dict[str, bool] = {}

    for question in questions:
        said_yes = bool(verdict_map.get(question.id, False))
        recorded[question.id] = said_yes
        good = said_yes if question.polarity == "yes_good" else (not said_yes)
        goods.append(good)
        by_dimension.setdefault(question.dimension, []).append(good)

    dimensions = {
        dim: round(sum(vals) / len(vals), 3) for dim, vals in by_dimension.items()
    }
    score = round(sum(goods) / len(goods), 3) if goods else 0.0

    return {"score": score, "dimensions": dimensions, "verdicts": recorded}


class BinaryQuestionEvaluator:
    """Decompose answer evaluation into binary questions and aggregate verdicts."""

    PROMPT_TEMPLATE = """
You are evaluating a predicted answer to a factual question. Judge each
binary question below on its OWN merit -- do not collapse them into a single
overall verdict.

Question: {question}
Reference answer: {reference_answer}
Predicted answer: {predicted_answer}

Binary questions (answer EVERY one with yes or no):
{question_list}

Return exactly one verdict per question id. Set "answer" to true for "yes"
and false for "no".
""".strip()

    def __init__(
        self,
        config: BinaryQuestionConfig = BinaryQuestionConfig(),
        questions: List[BinaryQuestion] = None,
    ):
        """Initialize the evaluator.

        Args:
            config: Model / temperature configuration.
            questions: Question bank to judge. Defaults to the curated
                SimpleQA bank.
        """
        self.config = config
        self.questions = (
            questions if questions is not None else DEFAULT_SIMPLEQA_QUESTIONS
        )
        self.llm = ChatOpenAI(
            model=config.model_name, temperature=config.temperature
        ).with_structured_output(_VerdictSet)

    def _build_prompt(
        self, question: str, reference_answer: str, predicted_answer: str
    ) -> str:
        lines = [
            f"{i}. [{q.id}] {q.text}" for i, q in enumerate(self.questions, start=1)
        ]
        return self.PROMPT_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
            question_list="\n".join(lines),
        )

    async def evaluate(
        self, question: str, predicted_answer: str, reference_answer: str
    ) -> Dict[str, object]:
        """Judge a predicted answer against every binary question.

        All questions are answered in a single structured-output call; the
        prompt explicitly instructs independent per-question judgment, which
        preserves BINEVAL's per-question decomposition while keeping cost
        comparable to the existing holistic correctness judge.

        Args:
            question: The original query.
            predicted_answer: The answer produced by the search provider.
            reference_answer: The gold answer.

        Returns:
            The ``aggregate_verdicts`` result dict (overall score, per
            dimension, and the raw per-question verdicts).
        """
        prompt = self._build_prompt(question, reference_answer, predicted_answer)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        verdict_map: Dict[str, bool] = {}
        for verdict in response.verdicts:
            verdict_map[verdict.question_id] = bool(verdict.answer)

        answered = len(verdict_map)
        if answered != len(self.questions):
            logger.warning(
                "Binary-question judge returned %d verdicts for %d questions; "
                "missing questions default to 'no'",
                answered,
                len(self.questions),
            )

        return aggregate_verdicts(self.questions, verdict_map)

    @property
    def evaluation_name(self) -> str:
        """Name of this evaluator."""
        return "binary_question_evaluator"

    @property
    def evaluation_description(self) -> str:
        """Description of what this evaluator does."""
        return (
            "Decomposes answer evaluation into atomic binary questions and "
            "aggregates the verdicts into interpretable per-dimension scores."
        )
