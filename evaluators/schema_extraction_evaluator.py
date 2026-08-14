"""Schema-guided extraction and rubric-based semantic evaluation of retrieved documents.

Adapted from "Schema-Guided Hierarchical Information Extraction and Semantic
Evaluation Using Generative AI" (arXiv:2608.06167). The paper's contract is
kept: a schema acts as the information model, extraction happens in a single
zero-shot model call, extracted values are aligned to the gold standard by
flattened attribute path (so nested, variable-cardinality attributes compare
correctly), and each aligned pair is graded by a generative model against a
four-way rubric: EXACT / SEMANTIC / USEFUL / NON_MATCH.

Target-native substitutions (the paper's NICE/HTA domain does not exist here):
the schema defaults to one suited to the document-relevance benchmark (an
extracted answer plus its supporting snippets), the gold standard is the
dataset's reference answer, and the paper's alignment similarity is replaced
by a parameter-free normalized token-overlap proxy.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, create_model

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Rubric classes, ordered from strongest to weakest agreement.
MATCH_CLASSES = ("EXACT", "SEMANTIC", "USEFUL", "NON_MATCH")

# Partial credit per rubric class: exact and semantic both count as matches
# (the paper computes F1 over attributes found), useful is worth half.
MATCH_WEIGHTS = {"EXACT": 1.0, "SEMANTIC": 1.0, "USEFUL": 0.5, "NON_MATCH": 0.0}


@dataclass(frozen=True)
class SchemaAttribute:
    """One node of the hierarchical schema (the paper's information model)."""

    name: str
    description: str
    kind: str = "scalar"  # "scalar" or "list"
    attributes: Tuple["SchemaAttribute", ...] = ()


def default_answer_schema() -> List[SchemaAttribute]:
    """Schema for the document-relevance benchmark: an answer plus its evidence.

    ``supporting_snippets`` is a variable-cardinality list of nested records,
    which exercises the hierarchical / nested part of the paper's algorithm.
    """
    return [
        SchemaAttribute(
            "answer",
            "Direct answer to the query, extracted from the retrieved documents.",
        ),
        SchemaAttribute(
            "supporting_snippets",
            "Passages from the retrieved documents that support the answer.",
            kind="list",
            attributes=(
                SchemaAttribute("quote", "Verbatim quote from one document."),
                SchemaAttribute("url", "URL of the document the quote came from."),
            ),
        ),
    ]


def _build_record_model(schema: Tuple[SchemaAttribute, ...], name: str = "Record") -> type:
    """Build a pydantic model from the schema so extraction is schema-guided."""
    fields: Dict[str, Any] = {}
    for attr in schema:
        if attr.kind == "list":
            fields[attr.name] = (List[_build_record_model(attr.attributes, attr.name.title())], ...)
        else:
            fields[attr.name] = (str, ...)
    return create_model(name, **fields)


def flatten_by_paths(value: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten a nested record into ``(path, value)`` pairs.

    List indices are dropped (``supporting_snippets[]`` rather than
    ``supporting_snippets[0]``) so attributes with different cardinalities in
    the extraction and the gold standard still land on the same path template.
    """
    paths: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(flatten_by_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for item in value:
            paths.extend(flatten_by_paths(item, f"{prefix}[]"))
    elif value is None:
        pass
    else:
        paths.append((prefix, str(value)))
    return paths


def _normalize(text: str) -> str:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _token_overlap(a: str, b: str) -> float:
    """Parameter-free similarity proxy used to propose value alignments."""
    sa, sb = _normalize(a), _normalize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def match_by_path(
    extracted: List[Tuple[str, str]],
    gold: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """Path-based semantic matching: align extracted values to gold values.

    Within a shared path template (cardinality-agnostic), values are paired
    greedily by token overlap, which handles the variable-cardinality case the
    paper targets. Returns ``(path, extracted_value, gold_value)`` triples plus
    unmatched extracted values paired against an empty gold.
    """
    pairs: List[Tuple[str, str, str]] = []
    gold_by_path: Dict[str, List[str]] = {}
    for path, value in gold:
        gold_by_path.setdefault(path, []).append(value)

    for path, value in extracted:
        candidates = gold_by_path.get(path, [])
        if not candidates:
            pairs.append((path, value, ""))
            continue
        best_idx = max(range(len(candidates)), key=lambda i: _token_overlap(value, candidates[i]))
        pairs.append((path, value, candidates.pop(best_idx)))
    return pairs


class MatchGrade(BaseModel):
    """Rubric outcome for one aligned attribute pair."""

    match_class: str
    rationale: str = ""


MATCH_TEMPLATE = """
You are comparing a value extracted from retrieved web documents against a
gold-standard value for the attribute "{path}", in the context of the query
"{query}".

Attribute description: {description}
Extracted value: {extracted}
Gold value: {gold}

Classify the extracted value as exactly one of:
- EXACT: same value, ignoring case, formatting, units and ordering.
- SEMANTIC: same meaning, different wording or level of detail.
- USEFUL: partially correct or a relevant subset, but incomplete or mixed with
  unrelated information.
- NON_MATCH: missing, wrong, or contradicting the gold value.

Reply with just the class name.
""".strip()

EXTRACTION_TEMPLATE = """
You are extracting structured information from documents retrieved by a web
search. Follow the schema exactly: fill every scalar attribute with a string,
and every list attribute with as many records as the documents support
(zero or more).

Query: {query}

Schema:
{schema}

Documents:
{documents}

Return the extracted record. Use only information present in the documents;
use an empty string for a scalar attribute the documents do not answer.
""".strip()


def render_schema(schema: Tuple[SchemaAttribute, ...], indent: int = 0) -> str:
    """Render the schema as the indented attribute list used in the prompt."""
    lines = []
    for attr in schema:
        kind = "list of records" if attr.kind == "list" else "string"
        lines.append(f"{'  ' * indent}- {attr.name} ({kind}): {attr.description}")
        if attr.attributes:
            lines.append(render_schema(attr.attributes, indent + 1))
    return "\n".join(lines)


@dataclass
class SchemaExtractionConfig:
    """Configuration for schema-guided extraction and evaluation."""

    model_name: str = "gpt-4.1"
    temperature: float = 0.0
    max_documents: int = 5


@dataclass
class SchemaEvaluationReport:
    """Per-query report: one rubric class per aligned attribute path."""

    query: str
    matches: List[Dict[str, str]] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return {cls: sum(1 for m in self.matches if m["match_class"] == cls) for cls in MATCH_CLASSES}

    @property
    def score(self) -> float:
        total = len(self.matches)
        if total == 0:
            return 0.0
        weighted = sum(MATCH_WEIGHTS.get(m["match_class"], 0.0) for m in self.matches)
        return round(weighted / total, 3)


def classify_locally(extracted_value: str, gold_value: str) -> Optional[str]:
    """Deterministic rubric shortcut for trivially decidable pairs.

    Returns a MATCH_CLASS when the pair can be graded without a model call,
    otherwise ``None`` so the caller falls back to the grader.
    """
    if not extracted_value.strip() or extracted_value.strip().lower() in ("n/a", "none", "unknown"):
        return "NON_MATCH"
    if _normalize(extracted_value) == _normalize(gold_value):
        return "EXACT"
    if not _normalize(extracted_value) & _normalize(gold_value):
        return None
    if _normalize(extracted_value) >= _normalize(gold_value) or _normalize(gold_value) >= _normalize(extracted_value):
        return "SEMANTIC"
    return None


class SchemaExtractionEvaluator:
    """Extract schema-shaped records from retrieved documents and grade them.

    Mirrors :class:`evaluators.correctness_evaluator.CorrectnessEvaluator`
    (same constructor shape, async ``evaluate`` entry point, structured-output
    grader) so it can be wired into a provider evaluation loop the same way.
    """

    EXTRACTION_TEMPLATE = EXTRACTION_TEMPLATE
    MATCH_TEMPLATE = MATCH_TEMPLATE

    def __init__(
        self,
        config: SchemaExtractionConfig = SchemaExtractionConfig(),
        schema: Optional[List[SchemaAttribute]] = None,
        llm: Optional[Any] = None,
    ):
        self.config = config
        self.schema = tuple(schema) if schema else tuple(default_answer_schema())
        self.record_model = _build_record_model(self.schema)
        self.llm = llm or ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
        ).with_structured_output(self.record_model)
        self.grader = llm if llm is not None else ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
        ).with_structured_output(MatchGrade)

    async def extract(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """Single zero-shot call: documents + schema -> structured record."""
        docs = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents[: self.config.max_documents]))
        prompt = self.EXTRACTION_TEMPLATE.format(
            query=query,
            schema=render_schema(self.schema),
            documents=docs,
        )
        record = self.llm.invoke([{"role": "user", "content": prompt}])
        return record.model_dump()

    def _descriptions(self) -> Dict[str, str]:
        descriptions: Dict[str, str] = {}

        def walk(schema: Tuple[SchemaAttribute, ...], prefix: str) -> None:
            for attr in schema:
                path = f"{prefix}.{attr.name}" if prefix else attr.name
                descriptions[path] = attr.description
                walk(attr.attributes, f"{path}[]" if attr.kind == "list" else path)

        walk(self.schema, "")
        return descriptions

    def grade_pair(self, query: str, path: str, extracted_value: str, gold_value: str) -> str:
        """Grade one aligned pair against the four-way rubric."""
        local = classify_locally(extracted_value, gold_value)
        if local is not None:
            return local
        prompt = self.MATCH_TEMPLATE.format(
            query=query,
            path=path,
            description=self._descriptions().get(path, path),
            extracted=extracted_value,
            gold=gold_value,
        )
        response = self.grader.invoke([{"role": "user", "content": prompt}])
        match_class = response.match_class.strip().upper().replace(" ", "_")
        return match_class if match_class in MATCH_CLASSES else "NON_MATCH"

    async def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract from documents and grade against the gold standard.

        Args:
            inputs: Dictionary containing 'query' and 'documents'.
            outputs: Unused; extraction happens here (kept for signature parity
                with CorrectnessEvaluator).
            reference_outputs: Dictionary containing 'answer' (gold record or
                scalar gold answer).

        Returns:
            Report dict with per-path rubric classes, class counts and a score.
        """
        query = inputs["query"]
        documents = inputs["documents"]
        gold = reference_outputs["answer"]
        if isinstance(gold, str):
            gold = {"answer": gold}

        extracted_record = await self.extract(query, documents)
        pairs = match_by_path(flatten_by_paths(extracted_record), flatten_by_paths(gold))

        report = SchemaEvaluationReport(query=query)
        for path, extracted_value, gold_value in pairs:
            report.matches.append(
                {
                    "path": path,
                    "extracted": extracted_value,
                    "gold": gold_value,
                    "match_class": self.grade_pair(query, path, extracted_value, gold_value),
                }
            )
        return {
            "score": report.score,
            "value": report.counts(),
            "matches": report.matches,
        }

    @property
    def evaluation_name(self) -> str:
        return "schema_extraction_evaluator"

    @property
    def evaluation_description(self) -> str:
        return (
            "Extracts schema-shaped records from retrieved documents in one zero-shot "
            "call and grades each attribute against the gold standard on the "
            "EXACT/SEMANTIC/USEFUL/NON_MATCH rubric."
        )
