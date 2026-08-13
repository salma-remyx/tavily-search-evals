"""Integration tests for the auto-rag-eval exam generator.

These tests prove the generated exam is a drop-in for the existing SimpleQA
pipeline: they flow generated items through the EXISTING ``load_csv_data`` and
``prepare_examples`` loaders (non-new modules in ``utils/``) and assert the
record shape that ``evaluate_provider_simple_qa`` consumes.

No network calls are made: the LLM step is monkeypatched.
"""

import csv

import pytest

from utils import EvaluationType, load_csv_data, prepare_examples
from utils.exam_generator import Exam, ExamGenerator, ExamGeneratorConfig, ExamQuestion


CORPUS = (
    "The IEEE Frank Rosenblatt Award for 2010 was given to Michio Sugeno. "
    "Radcliffe College was a women's liberal arts college in Cambridge, Massachusetts. "
    "The Leipzig 1877 chess tournament was organized in honor of Adolf Anderssen."
)


def _make_item(question, answer, distractors):
    return ExamQuestion(question=question, answer=answer, distractors=distractors)


def test_filter_items_drops_unsound_and_ungrounded_items():
    items = [
        # Kept: grounded, clean distractors.
        _make_item("Who received the IEEE Frank Rosenblatt Award in 2010?",
                   "Michio Sugeno", ["Adolf Anderssen", "Annick Bricaud", "Marie Curie"]),
        # Dropped: distractor collides with the answer.
        _make_item("Where was Radcliffe College?",
                   "Cambridge", ["Cambridge", "Boston", "New Haven"]),
        # Dropped: too few distractors.
        _make_item("Who was the Leipzig 1877 tournament for?",
                   "Adolf Anderssen", ["Bobby Fischer"]),
        # Dropped: reference answer not grounded in the corpus.
        _make_item("Who won the 1992 Olympics?",
                   "Carl Lewis", ["Ben Johnson", "Mike Powell", "Linford Christie"]),
    ]

    kept = ExamGenerator.filter_items(items, CORPUS)

    assert len(kept) == 1
    assert kept[0].answer == "Michio Sugeno"


def test_to_simple_qa_records_matches_simpleqa_schema():
    item = _make_item("Who received the IEEE Frank Rosenblatt Award in 2010?",
                      "Michio Sugeno", ["Adolf Anderssen", "Annick Bricaud", "Marie Curie"])

    records = ExamGenerator.to_simple_qa_records([item], topic="Awards")

    assert list(records[0].keys()) == ["metadata", "problem", "answer"]
    assert records[0]["problem"] == item.question
    assert records[0]["answer"] == "Michio Sugeno"
    assert "Awards" in records[0]["metadata"]


def test_generated_exam_is_dropin_for_simpleqa_pipeline(monkeypatch, tmp_path):
    """End-to-end (LLM mocked): generated CSV loads through the existing pipeline."""
    # ChatOpenAI is constructed lazily; set a placeholder key so construction is
    # version-safe. The LLM is never invoked -- _llm_generate is stubbed.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    generated = Exam(questions=[
        _make_item("Who received the IEEE Frank Rosenblatt Award in 2010?",
                   "Michio Sugeno", ["Adolf Anderssen", "Annick Bricaud", "Marie Curie"]),
        _make_item("What women's liberal arts college was in Cambridge, Massachusetts?",
                   "Radcliffe College", ["Wellesley", "Smith", "Barnard"]),
    ])

    generator = ExamGenerator(ExamGeneratorConfig(model_name="gpt-4.1"))
    monkeypatch.setattr(generator, "_llm_generate", lambda *a, **kw: generated)

    items = generator.generate_exam(documents=CORPUS, n_questions=2, topic="Awards")
    assert len(items) == 2

    csv_path = str(tmp_path / "awards_exam.csv")
    written = ExamGenerator.write_simple_qa_csv(items, csv_path, topic="Awards")
    assert written == csv_path

    # Existing utils loader (non-new module) consumes the generated CSV.
    with open(csv_path) as fh:
        header = next(csv.reader(fh))
    assert header == ["metadata", "problem", "answer"], "header must match simple_qa_test_set.csv"

    df = load_csv_data(csv_path)
    examples = prepare_examples(df, ["tavily"], evaluation_type=EvaluationType.SIMPLEQA)

    # This is the exact record shape evaluate_provider_simple_qa reads per example.
    for example in examples["tavily"]:
        assert set(example.keys()) == {"question", "answer", "index"}
    assert examples["tavily"][0]["answer"] == "Michio Sugeno"
    assert examples["tavily"][1]["answer"] == "Radcliffe College"


def test_load_corpus_reads_directory(monkeypatch, tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("Ada Lovelace wrote notes on the Analytical Engine.", encoding="utf-8")
    (d / "b.md").write_text("Alan Turing proposed the Turing test in 1950.", encoding="utf-8")

    corpus = ExamGenerator.load_corpus(str(d))
    assert "Ada Lovelace" in corpus
    assert "Alan Turing" in corpus


def test_load_corpus_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExamGenerator.load_corpus(str(tmp_path / "does_not_exist"))
