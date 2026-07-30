"""Required-element coverage evaluator.

A multi-faceted relevance judge that decomposes whether a predicted answer
(together with its retrieved context) covers the set of *required elements*
declared for a question, instead of collapsing relevance into a single
binary verdict.

This is a target-native (Mode 2) adaptation of the multi-facet RAG
evaluation introduced in "Evaluating RAG for French immigration law: a
benchmark and baseline study" (arXiv:2607.24449v1). That paper's
"required-document retrieval" and "legal citation coverage" facets both
reduce to the same underlying question this evaluator answers: *did the
retrieval surface everything this task demands?*

What is kept at full fidelity (the paper's core mechanism):
  - Relevance is decomposed into per-element coverage and aggregated into a
    single coverage score. The team can read *which* required entities /
    document types / citations a provider misses, per question, rather than
    only whether an answer was correct overall.

What is substituted with target-native equivalents (Mode 2):
  - The paper's French-immigration legal corpus and 52 synthetic profiles
    are replaced by the repo's existing dataset shape. Coverage is driven by
    generic per-question ``required_elements`` metadata (key entities /
    required document types / citations) that the team can populate for any
    domain, not just immigration law.
  - The paper's bespoke retrieval/judgment stack is replaced by this repo's
    existing LLM-as-judge pattern (``langchain_openai.ChatOpenAI`` with
    structured output), mirroring ``CorrectnessEvaluator``.

What is intentionally out of scope:
  - The paper's full benchmark framework and its parametric-vs-retrieval
    comparison across model scales belong in a downstream evaluation PR;
    this module contributes only the coverage judge and its aggregation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ElementVerdict(BaseModel):
    """Judge verdict for a single required element."""

    index: int = Field(
        description="1-based index of the element in the required-elements list"
    )
    covered: bool


class RequiredCoverageGrade(BaseModel):
    """Structured-output schema returned by the coverage judge."""

    verdicts: List[ElementVerdict]


@dataclass
class RequiredCoverageConfig:
    """Configuration for required-element coverage evaluation."""

    model_name: str = "gpt-4.1"
    temperature: float = 0.0


class RequiredCoverageEvaluator:
    """Judge whether an answer + retrieved context covers required elements.

    Mirrors the ``CorrectnessEvaluator`` surface (an async ``evaluate`` plus
    ``evaluation_name`` / ``evaluation_description`` properties) so it drops
    into the same per-example judge loop without introducing a new pipeline
    shape. An optional ``llm`` can be injected (any object exposing
    ``with_structured_output``) to keep the aggregation logic testable
    without a live OpenAI call.
    """

    GRADER_TEMPLATE = """
You are evaluating a retrieval-augmented answer for completeness.

For each numbered REQUIRED ELEMENT below, decide whether the specific
information it asks for is present and correct in the PREDICTED ANSWER. You
may use the RETRIEVED CONTEXT as supporting evidence. Mark an element
"covered" only when the exact information it names is actually stated - not
merely implied by a related but different fact, and not when the answer is
vague or hedged about it.

QUESTION:
{question}

REQUIRED ELEMENTS (the answer must address each one):
{required_elements}

PREDICTED ANSWER:
{predicted_answer}

RETRIEVED CONTEXT:
{retrieved_context}

Return one verdict per element, using its 1-based index from the list above
and a boolean "covered". Do not skip elements and do not add commentary.
""".strip()

    def __init__(
        self,
        config: RequiredCoverageConfig = RequiredCoverageConfig(),
        llm: Optional[Any] = None,
    ):
        """Initialize the evaluator.

        Args:
            config: Model / temperature configuration for the default judge.
            llm: Optional ChatOpenAI-like client. When omitted a default
                ``ChatOpenAI`` is constructed. Either way it is wrapped with
                ``with_structured_output(RequiredCoverageGrade)``.
        """
        self.config = config
        if llm is None:
            llm = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
            )
        self.llm = llm.with_structured_output(RequiredCoverageGrade)

    async def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Score how completely the answer covers the required elements.

        Args:
            inputs: ``question``, ``required_elements`` (list[str]) and an
                optional ``retrieved_context`` string.
            outputs: ``answer`` - the predicted answer to check.
            reference_outputs: Unused; accepted for interface parity with
                ``CorrectnessEvaluator``.

        Returns:
            A dict with ``score`` (fraction of elements covered, 0..1),
            ``value`` (``FULLY_COVERED`` / ``PARTIALLY_COVERED`` /
            ``NOT_COVERED``), ``covered`` and ``missing`` element lists, and
            ``total``.
        """
        question = inputs["question"]
        required_elements: List[str] = inputs["required_elements"]
        predicted_answer = outputs["answer"]
        retrieved_context = inputs.get("retrieved_context", "")

        elements_block = "\n".join(
            f"{i + 1}. {element}" for i, element in enumerate(required_elements)
        )
        grader_prompt = self.GRADER_TEMPLATE.format(
            question=question,
            required_elements=elements_block,
            predicted_answer=predicted_answer,
            retrieved_context=retrieved_context or "(none provided)",
        )

        grade_response = self.llm.invoke([{"role": "user", "content": grader_prompt}])

        # Index by the element's 1-based position so a judge that rephrases an
        # element's wording still maps back to the right item.
        verdict_by_index = {
            verdict.index: verdict.covered for verdict in grade_response.verdicts
        }
        covered: List[str] = []
        missing: List[str] = []
        for i, element in enumerate(required_elements):
            if verdict_by_index.get(i + 1, False):
                covered.append(element)
            else:
                missing.append(element)

        total = len(required_elements)
        score = len(covered) / total if total else 0.0

        if score >= 1.0:
            label = "FULLY_COVERED"
        elif score > 0.0:
            label = "PARTIALLY_COVERED"
        else:
            label = "NOT_COVERED"

        return {
            "score": round(score, 3),
            "value": label,
            "covered": covered,
            "missing": missing,
            "total": total,
        }

    @property
    def evaluation_name(self) -> str:
        """Name of this evaluator."""
        return "required_coverage_evaluator"

    @property
    def evaluation_description(self) -> str:
        """Description of what this evaluator does."""
        return (
            "Scores how completely a predicted answer and its retrieved "
            "context cover the per-question required elements "
            "(key entities / required document types / citations)."
        )
