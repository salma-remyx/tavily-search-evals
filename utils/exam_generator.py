"""Task-specific exam generation for retrieval evaluation.

Generates a synthetic short-answer exam from a knowledge corpus and emits it in
the repo's SimpleQA CSV schema (``metadata,problem,answer``), so the generated
exam is a drop-in for ``datasets/simple_qa_test_set.csv`` and is scored by the
existing ``CorrectnessEvaluator`` + ``evaluate_provider_simple_qa`` pipeline
without any further plumbing.

This is an adapted port (Mode 2) of:

    Es, Jia, et al. "Automated Evaluation of Retrieval-Augmented Language
    Models with Task-Specific Exam Generation." arXiv:2405.13622 (2024).

What is ported at fidelity
    The paper's core contribution -- an LLM that turns a corpus of documents
    into a synthetic multiple-choice exam whose items are grounded in that
    corpus. Each item carries a concise reference (gold) answer plus plausible
    distractors, mirroring the SimpleQA gold-target style this repo already
    grades against.

What is substituted (target-native)
    * The paper's separate MCQ scoring harness (lm-harness / Bedrock) is
      replaced by this repo's own handlers + ``CorrectnessEvaluator``. We do
      that by emitting the exam as a SimpleQA CSV rather than an MCQ
      answer-sheet, so ``evaluate_provider_simple_qa`` grades it unchanged.
    * The paper's fitted Item Response Theory (IRT) item-calibration -- a
      learned estimator that needs correctness vectors across many examinees
      -- is replaced by a parameter-free item-quality gate
      (``filter_items``): distractor sanity plus corpus-grounding overlap.
      Full fitted IRT is intentionally downstream scope: it only becomes
      meaningful once the repo has per-item correctness across providers.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)


class ExamQuestion(BaseModel):
    """A single generated exam item."""

    question: str = Field(description="A self-contained factual short-answer question.")
    answer: str = Field(description="A concise reference (gold) answer, as an entity or short phrase.")
    distractors: List[str] = Field(
        description="Three plausible but incorrect answer options for the multiple-choice form of the item.",
    )


class Exam(BaseModel):
    """Structured output schema for the generated exam."""

    questions: List[ExamQuestion]


@dataclass
class ExamGeneratorConfig:
    """Configuration for exam generation. Mirrors CorrectnessConfig's shape."""

    model_name: str = "gpt-4.1"
    temperature: float = 0.0
    n_distractors: int = 3


# The generation prompt asks for corpus-grounded, SimpleQA-style short answers.
# It mirrors the discipline the repo's CorrectnessEvaluator expects of gold
# targets: concise, entity-style, fully determined by the question.
_EXAM_GENERATION_TEMPLATE = """
You are an exam author building a task-specific test for a retrieval system.

Using ONLY the documents below, write {n_questions} factual short-answer questions.
Each question must:
- Be self-contained (no "according to the passage" phrasing).
- Have an answer that is explicitly supported by the documents.
- Have a concise reference answer: a single entity, name, number, date, or short
  phrase (mirroring SimpleQA gold targets). Do not write full sentences as answers.
- Include {n_distractors} plausible but clearly INCORRECT distractor options that
  share the answer's type (e.g. if the answer is a year, the distractors are years).

Prefer questions whose answers are unambiguous and verifiable from the documents.
Do not invent facts that are not in the documents.

Task / topic: {topic}

Documents:
{corpus}
""".strip()


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


class ExamGenerator:
    """Generate a corpus-grounded short-answer exam.

    The generated items are emitted through :meth:`to_simple_qa_records` /
    :meth:`write_simple_qa_csv` in the repo's SimpleQA schema so they are graded
    by the existing pipeline without modification.
    """

    def __init__(self, config: ExamGeneratorConfig = ExamGeneratorConfig()):
        self.config = config
        # Constructor mirrors CorrectnessEvaluator: build the LLM client lazily;
        # no network call happens until generate_exam() is invoked.
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
        ).with_structured_output(Exam)

    # -- corpus loading ---------------------------------------------------

    @staticmethod
    def load_corpus(corpus_path: str) -> str:
        """Load a corpus from a file or a directory of text/markdown files.

        Returns the concatenated corpus text. Skips files larger than
        ``max_chars`` per file to keep the prompt bounded.
        """
        max_chars_per_file = 8000
        chunks: List[str] = []

        def read_one(path: str) -> Optional[str]:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Skipping unreadable corpus file %s: %s", path, exc)
                return None
            return text[:max_chars_per_file]

        if os.path.isdir(corpus_path):
            for root, _dirs, files in os.walk(corpus_path):
                for name in sorted(files):
                    if name.startswith("."):
                        continue
                    piece = read_one(os.path.join(root, name))
                    if piece:
                        chunks.append(piece)
        elif os.path.isfile(corpus_path):
            piece = read_one(corpus_path)
            if piece is not None:
                chunks.append(piece)
        else:
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")

        corpus = "\n\n".join(chunks).strip()
        if not corpus:
            raise ValueError(f"Corpus at {corpus_path} is empty or unreadable")
        return corpus

    # -- generation -------------------------------------------------------

    def _build_prompt(self, corpus: str, n_questions: int, topic: str) -> str:
        return _EXAM_GENERATION_TEMPLATE.format(
            n_questions=n_questions,
            n_distractors=self.config.n_distractors,
            topic=topic or "(unspecified)",
            corpus=corpus,
        )

    def _llm_generate(self, corpus: str, n_questions: int, topic: str) -> Exam:
        """Call the LLM to produce a structured Exam. (Network call.)"""
        prompt = self._build_prompt(corpus, n_questions, topic)
        logger.info("Generating exam: %d questions on topic '%s'", n_questions, topic)
        return self.llm.invoke([{"role": "user", "content": prompt}])

    def generate_exam(
        self,
        documents: str,
        n_questions: int = 10,
        topic: str = "",
        filter_items: bool = True,
    ) -> List[ExamQuestion]:
        """Generate a synthetic exam from a corpus string.

        Args:
            documents: The corpus text to ground questions in.
            n_questions: Number of questions to request from the LLM.
            topic: Optional task/topic label for the exam.
            filter_items: Apply the parameter-free quality gate (distractor
                sanity + corpus grounding). Defaults to True.

        Returns:
            A list of :class:`ExamQuestion` items, ready to be written as a
            SimpleQA CSV via :meth:`write_simple_qa_csv`.
        """
        exam = self._llm_generate(documents, n_questions, topic)
        items = list(exam.questions)
        logger.info("LLM returned %d raw items", len(items))
        if filter_items:
            before = len(items)
            items = self.filter_items(items, documents)
            logger.info("Quality gate kept %d/%d items", len(items), before)
        return items

    # -- parameter-free item-quality gate (IRT-inspired) -----------------

    @staticmethod
    def filter_items(items: List[ExamQuestion], documents: str) -> List[ExamQuestion]:
        """Parameter-free proxy for the paper's IRT item-quality estimation.

        The paper fits an IRT model over examinee correctness vectors to drop
        low-quality items; that needs per-item results across many examinees
        and is downstream scope here. As a parameter-free substitute we drop
        items that are structurally unsound regardless of examinees:

        * empty or sentence-length reference answers (ambiguous gold targets),
        * distractors that are missing, too few, non-distinct, or that collide
          with the reference answer (a non-discriminating MCQ item),
        * reference answers with no token overlap with the corpus (not
          corpus-grounded).
        """
        corpus_tokens = set(_normalize(documents).split())
        kept: List[ExamQuestion] = []
        for item in items:
            answer = _normalize(item.answer)
            if not answer or len(answer.split()) > 8:
                continue
            distractors = [_normalize(d) for d in (item.distractors or [])]
            if len(distractors) < 2:
                continue
            if answer in distractors or len(set(distractors)) != len(distractors):
                continue
            if not (corpus_tokens & set(answer.split())):
                continue
            kept.append(item)
        return kept

    # -- SimpleQA drop-in emission ---------------------------------------

    @staticmethod
    def to_simple_qa_records(
        items: List[ExamQuestion],
        topic: str = "",
    ) -> List[Dict[str, Any]]:
        """Map generated items to the repo's SimpleQA CSV row schema.

        The output rows have ``metadata``, ``problem``, and ``answer`` columns,
        matching ``datasets/simple_qa_test_set.csv`` so the existing
        ``load_csv_data`` / ``prepare_examples`` / ``evaluate_provider_simple_qa``
        chain consumes them unchanged. ``index`` is assigned by ``load_csv_data``.
        """
        records: List[Dict[str, Any]] = []
        for item in items:
            metadata = {
                "topic": topic,
                "answer_type": "Exam",
                "source": "auto-rag-eval",
                "distractors": list(item.distractors),
            }
            records.append(
                {
                    "metadata": json.dumps(metadata),
                    "problem": item.question,
                    "answer": item.answer,
                }
            )
        return records

    @staticmethod
    def write_simple_qa_csv(
        items: List[ExamQuestion],
        output_path: str,
        topic: str = "",
    ) -> str:
        """Write generated items as a SimpleQA-format CSV (drop-in dataset).

        Returns the path written. The header is ``metadata,problem,answer`` --
        the columns ``load_csv_data`` requires.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        records = ExamGenerator.to_simple_qa_records(items, topic=topic)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["metadata", "problem", "answer"])
            writer.writeheader()
            writer.writerows(records)
        logger.info("Wrote %d-item exam CSV to %s", len(records), output_path)
        return output_path
