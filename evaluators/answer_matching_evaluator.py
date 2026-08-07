import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Annotated
from dotenv import load_dotenv

# Answer matching: grade a free-form predicted answer by asking a judge LLM
# whether it *means the same thing* as the reference answer (binary YES/NO),
# rather than forcing a discriminative multiple-choice pick (A/B/C).
#
# Adapted from: Rein (2025), "Answer Matching Outperforms Multiple Choice for
# Language Model Evaluation" (arXiv:2507.02856). The paper shows that a
# generative answer graded by a binary match judge is more reliable than
# multiple-choice grading, which is shortcut-prone. Here it serves as a second,
# methodologically distinct judge that triangulates the repo's lone A/B/C
# CorrectnessEvaluator on the SimpleQA per-example loop. The prompt rules
# (coverage rule, 1% numeric tolerance) mirror the canonical judge prompt from
# the authors' reference implementation, rephrased for this repo.

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class AnswerMatchGrade(BaseModel):
    """Schema for answer-matching verdict."""
    match: Annotated[str, "YES if the predicted answer means the same thing as the reference, NO otherwise"]


@dataclass
class AnswerMatchingConfig:
    """Configuration for answer-matching evaluation."""
    model_name: str = "gpt-4.1"
    temperature: float = 0.0


class AnswerMatchingEvaluator:
    """Grades a predicted answer by binary semantic match to the reference.

    Unlike CorrectnessEvaluator (which picks a discriminative CORRECT /
    INCORRECT / NOT_ATTEMPTED label), this judge answers a single yes/no
    question: does the predicted answer mean the same thing as the reference?
    The shared judge model and temperature keep it comparable to the existing
    evaluator while the prompt framing is methodologically independent, which
    is what makes it useful for triangulation.
    """
    ANSWER_MATCH_TEMPLATE = """
Your job is to decide whether a predicted answer means the same thing as a reference answer.

Question: {question}
Reference answer: {reference_answer}
Predicted answer: {predicted_answer}

Does the predicted answer mean the same thing as the reference answer?
- Answer YES when the predicted answer conveys the same meaning as the reference answer. Only semantic meaning matters: capitalization, punctuation, grammar, and word order do not matter. The predicted answer is YES even if it hedges, as long as the reference answer's information is fully present and nothing in the predicted answer contradicts it. A name with an obvious typo (e.g. "Hyung Won Chung" vs "Hyungwon Chung") still counts as YES.
- Coverage rule: the predicted answer must cover everything stated in the reference answer. Being more specific or adding extra correct details (e.g. a full name when the reference gives a surname, or a paraphrase with more context) is still YES — extra information only makes it NO when it contradicts the reference answer.
- Numeric rule: if the reference answer is a number, a predicted number within 1% relative error of the reference counts as YES (e.g. reference "1000" matches "1005" but not "1020").
- Answer NO when the predicted answer is missing the reference answer's information, adds information that contradicts the reference answer, or does not attempt the question.

Respond with only YES or NO.
""".strip()

    def __init__(self, config: AnswerMatchingConfig = AnswerMatchingConfig()):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature
        ).with_structured_output(
            AnswerMatchGrade
        )

    async def evaluate(self, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> dict:
        """Evaluate whether a predicted answer matches the reference answer.

        Args:
            inputs: Dictionary containing 'question'.
            outputs: Dictionary containing 'predicted_answer' (keyed as 'answer').
            reference_outputs: Dictionary containing 'reference_answer' (keyed as 'answer').

        Returns:
            dict with 'score' (1.0 for a match, 0.0 otherwise) and 'value'
            ('MATCH' or 'NO_MATCH'). This is the same shape as
            CorrectnessEvaluator.evaluate so the two judges are
            interchangeable in the per-example loop.
        """
        grader_prompt = self.ANSWER_MATCH_TEMPLATE.format(
            question=inputs["question"],
            reference_answer=reference_outputs["answer"],
            predicted_answer=outputs["answer"]
        )

        grade_response = self.llm.invoke([
            {"role": "user", "content": grader_prompt}
        ])

        verdict = grade_response.match.strip().upper()
        # Defensive parsing: the structured output should already be YES/NO,
        # but tolerate stray words (e.g. "yes.") the same way the A/B/C
        # judge tolerates full words.
        is_match = verdict.startswith("Y")

        return {
            "score": 1.0 if is_match else 0.0,
            "value": "MATCH" if is_match else "NO_MATCH",
        }

    @property
    def evaluation_name(self) -> str:
        """Name of this evaluator."""
        return "answer_matching_evaluator"

    @property
    def evaluation_description(self) -> str:
        """Description of what this evaluator does."""
        return "Grades a predicted answer by binary semantic match to the reference answer."
