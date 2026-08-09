"""Integration test for the reference-free benchmark-quality audit.

Exercises the wiring added to ``run_evaluation.run_benchmark_audit`` (a
NON-NEW module -- the call site) with the LLM judge mocked out, so no
network or API key is required.
"""

import json

import pytest

import run_evaluation  # non-new module: the call site
import evaluators.benchmark_quality_evaluator as bqe


def _make_examples():
    """Build the per-provider examples shape produced by prepare_examples."""
    items = [
        {"question": "Who wrote the novel 1984?", "answer": "George Orwell", "index": 0},
        {"question": "What is the capital of Australia?", "answer": "Canberra", "index": 1},
        {"question": "In what year did the Titanic sink?", "answer": "1912", "index": 2},
        {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci", "index": 3},
        {"question": "What is the chemical symbol for gold?", "answer": "Au", "index": 4},
        {"question": "How many continents are there?", "answer": "Seven", "index": 5},
    ]
    # Both providers share the same questions, so dedup must collapse them.
    return {"tavily": items, "exa": items}


# Canned grades the fake judge cycles through.
_GRADES = [
    bqe.BenchmarkQualityGrade(consistency_score=5, complexity_score=4, issue="none"),
    bqe.BenchmarkQualityGrade(consistency_score=2, complexity_score=1, issue="trivial"),
    bqe.BenchmarkQualityGrade(consistency_score=3, complexity_score=3, issue="ambiguous"),
]


class _FakeStructured:
    def __init__(self):
        self._i = 0

    def invoke(self, messages):
        grade = _GRADES[self._i % len(_GRADES)]
        self._i += 1
        return grade


class _FakeChat:
    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, schema):
        return _FakeStructured()


def test_run_benchmark_audit_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(bqe, "ChatOpenAI", _FakeChat)

    report = run_evaluation.run_benchmark_audit(
        _make_examples(),
        model_name="gpt-4.1",
        output_dir=str(tmp_path),
        sample_size=6,
    )

    # The audit file was written and round-trips as JSON.
    audit_file = tmp_path / "benchmark_audit.json"
    assert audit_file.exists()
    on_disk = json.loads(audit_file.read_text())
    assert on_disk == report

    # Dedup across two providers -> 6 unique items, all judged, no failures.
    assert report["n_total"] == 6
    assert report["n_judged"] == 6
    assert report["judge_failures"] == 0

    # Grades cycle 5,2,3,5,2,3 -> mean consistency 20/6.
    assert report["mean_consistency"] == pytest.approx(3.333, abs=0.001)
    # 4,1,3,4,1,3 -> mean complexity 16/6.
    assert report["mean_complexity"] == pytest.approx(2.667, abs=0.001)
    # 4 of 6 flagged (trivial x2, ambiguous x2).
    assert report["flagged_ratio"] == pytest.approx(4 / 6, abs=0.001)
    assert report["issue_counts"] == {"none": 2, "trivial": 2, "ambiguous": 2}

    # Coverage proxy is the parameter-free substitute for policy coverage.
    coverage = report["coverage"]
    assert 0.0 <= coverage["coverage_score"] <= 1.0
    assert coverage["unique_question_ratio"] == 1.0  # all questions distinct


def test_audit_empty_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(bqe, "ChatOpenAI", _FakeChat)
    report = run_evaluation.run_benchmark_audit(
        {"tavily": []}, model_name="gpt-4.1", output_dir=str(tmp_path), sample_size=5
    )
    assert report["n_total"] == 0
    assert report["n_judged"] == 0
    assert (tmp_path / "benchmark_audit.json").exists()


def test_coverage_proxy_is_parameter_free():
    # _coverage_proxy needs no LLM; bypass __init__ (which builds the judge).
    auditor = bqe.BenchmarkQualityEvaluator.__new__(bqe.BenchmarkQualityEvaluator)
    auditor.config = bqe.BenchmarkQualityConfig()
    coverage = auditor._coverage_proxy([
        {"question": "a a b", "answer": "x"},
        {"question": "a a b", "answer": "y"},
    ])
    assert coverage["unique_question_ratio"] == 0.5  # two identical questions
    assert coverage["unique_answer_ratio"] == 1.0
    assert 0.0 <= coverage["coverage_score"] <= 1.0
