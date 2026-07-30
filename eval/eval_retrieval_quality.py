"""Evaluation harness for eRAG reference-free retrieval-quality scoring.

Runs the same fixture on baseline (pre-PR) and PR-head code. On PR head the
real `utils.retrieval_quality` / `utils.post_processor.PostProcessor` wiring
scores each retrieved document via a stand-in downstream judge and produces
per-document relevance labels; on baseline (module absent) we fall back to
degraded zeroed labels so the script still emits comparable metrics.

Metrics:
  - erag_correctness_rate: fraction of per-document labels that match the
    known ground-truth relevance of the fixture documents (target).
  - wiring_labels_match: 1.0 iff the full label vector exactly matches the
    expected ground truth, else 0.0 (guardrail on correct wiring).
"""
import argparse
import asyncio
import json
import os
import sys
import types


def _install_stub(name, **attrs):
    if name not in sys.modules:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


class _FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt):
        return types.SimpleNamespace(content=str(prompt))


# Mirror tests/test_retrieval_quality.py: stub heavy optional deps before
# importing anything from the `utils` package so no real SDKs are required.
_install_stub("langchain_openai", ChatOpenAI=_FakeChatOpenAI)
_install_stub("pandas", DataFrame=object)
_install_stub("quotientai", QuotientAI=object, DetectionType=object)
os.environ.setdefault("OPENAI_API_KEY", "test-key")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FEATURE_AVAILABLE = True
try:
    from utils.post_processor import PostProcessor
except Exception:
    FEATURE_AVAILABLE = False
    PostProcessor = None


def _format(docs):
    """Mimic ProviderHandler._format_search_results_for_prompt output."""
    return "\n".join(
        f"\n**Document {i + 1}.** Source: {url}\nContent: {content}"
        for i, (url, content) in enumerate(docs)
    )


# Fixture: two documents contain "GOOD" (truly relevant), one does not.
DOCS = [
    ("http://good", "GOOD supporting evidence"),
    ("http://bad", "unrelated filler"),
    ("http://good2", "another GOOD passage"),
]
EXPECTED_LABELS = [1, 0, 1]


async def fake_judge(query, predicted_answer, reference_answer):
    # The extractor echoes the document text into the predicted answer, so a
    # "GOOD" document should yield a relevant (score 1.0) downstream result.
    return 1.0 if "GOOD" in predicted_answer else 0.0


def _fallback_labels():
    # Degraded pre-change behaviour: no per-document eRAG scoring exists,
    # so every document is treated as unscored/irrelevant.
    return [0] * len(DOCS)


def _compute_labels():
    if not FEATURE_AVAILABLE:
        return _fallback_labels()
    try:
        processor = PostProcessor(llm_model="gpt-4.1")
        formatted = _format(DOCS)
        metrics = asyncio.run(
            processor.score_retrieval_quality(
                query="who?",
                reference_answer="the gold answer",
                documents=formatted,
                judge_fn=fake_judge,
            )
        )
        labels = list(metrics.get("labels", []))
        if len(labels) != len(EXPECTED_LABELS):
            labels = (labels + [0] * len(EXPECTED_LABELS))[: len(EXPECTED_LABELS)]
        return labels
    except Exception:
        return _fallback_labels()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--seed", default=None)
    parser.parse_known_args()

    try:
        labels = _compute_labels()
    except Exception:
        labels = _fallback_labels()

    matches = sum(1 for a, b in zip(labels, EXPECTED_LABELS) if a == b)
    erag_correctness_rate = matches / len(EXPECTED_LABELS)
    wiring_labels_match = 1.0 if labels == EXPECTED_LABELS else 0.0

    print(json.dumps({
        "erag_correctness_rate": erag_correctness_rate,
        "wiring_labels_match": wiring_labels_match,
    }))


if __name__ == "__main__":
    main()