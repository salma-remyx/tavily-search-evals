"""Tests for the document condenser and its wiring into answer extraction.

The integration tests import the *existing* call-site module
``utils.post_processor`` and exercise the opt-in Extract-then-Evaluate wiring
added there, proving the condenser is actually invoked before the LLM prompt
is built (not just a self-test of the new module).
"""

from utils.document_condenser import (
    condense_document_content,
    condense_search_results,
    split_sentences,
)
from utils.post_processor import PostProcessor


def _build_results(docs):
    """Reproduce base_handler._format_search_results_for_prompt output."""
    return "\n".join(
        f"\n**Document {i + 1}.** Source: {url}\nContent: {content}"
        for i, (url, content) in enumerate(docs)
    )


class _RecordingLLM:
    """Stand-in for ChatOpenAI that records the prompt it was invoked with."""

    def __init__(self, *args, **kwargs):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt

        class _Result:
            content = "extracted-answer"

        return _Result()


def test_split_sentences_basic():
    assert split_sentences("Hello world. This is a test! Really? Yes.") == [
        "Hello world.",
        "This is a test!",
        "Really?",
        "Yes.",
    ]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_condense_document_content_keeps_query_relevant_sentence():
    query = "What is the capital of France?"
    content = (
        "The weather today is sunny and warm across most regions. "
        "Paris is the capital of France and a major European city. "
        "Many people enjoy eating croissants for breakfast each morning. "
        "The train arrives at the station around noon every day. "
        "Rivers flow downhill toward the sea in most landscapes. "
        "Birds migrate south during the colder months of winter."
    )
    condensed = condense_document_content(query, content, max_sentences=2)

    assert "Paris is the capital of France" in condensed
    assert "train arrives" not in condensed          # filler dropped
    assert len(condensed) < len(content)             # strictly shorter


def test_condense_document_content_short_docs_passthrough():
    content = "Short doc. Only two sentences. Nothing more."
    # Fewer than min_sentences_to_condense -> returned verbatim.
    assert condense_document_content("query", content, max_sentences=1) == content


def test_condense_document_content_no_query_keeps_leading():
    content = "First sentence here. Second one now. Third as well. Fourth line. Fifth bit. Sixth item."
    condensed = condense_document_content("", content, max_sentences=2)
    # With no query signal, the earliest sentences are kept, in order.
    assert condensed == "First sentence here. Second one now."


def test_condense_search_results_preserves_structure():
    query = "capital of France"
    docs = [
        (
            "https://en.wikipedia.org/wiki/Paris",
            "Paris is the capital of France. "
            "Trains run on time today. "
            "Birds fly south in winter. "
            "The river is wide and deep. "
            "Mountains are cold this season. "
            "Coffee is a popular morning drink.",
        )
    ]
    formatted = _build_results(docs)
    condensed = condense_search_results(query, formatted, max_sentences_per_doc=2)

    assert "**Document 1.**" in condensed                      # numbering kept
    assert "en.wikipedia.org/wiki/Paris" in condensed          # URL kept
    assert "Paris is the capital of France" in condensed        # key sentence kept
    assert "Coffee is a popular morning drink" not in condensed  # filler dropped
    assert len(condensed) < len(formatted)


def test_condense_search_results_unrecognized_format_unchanged():
    raw = "Just some text with no document markers at all."
    assert condense_search_results("query", raw, max_sentences_per_doc=2) == raw


def test_condense_search_results_disabled_is_noop():
    formatted = _build_results([("https://example.com/x", "One. Two. Three. Four. Five. Six.")])
    assert condense_search_results("query", formatted, max_sentences_per_doc=0) == formatted


def test_extract_answer_condenses_when_enabled(monkeypatch):
    monkeypatch.setattr("utils.post_processor.ChatOpenAI", _RecordingLLM)
    pp = PostProcessor(llm_model="gpt-4.1", extract_sentences_per_doc=1)

    content = (
        "Paris is the capital of France. "
        "Trains run on time today. "
        "Birds fly south in winter. "
        "The river is wide and deep. "
        "Mountains are cold this season. "
        "Coffee is a popular morning drink."
    )
    search_result = _build_results([("https://example.com/paris", content)])

    answer = pp.extract_answer(query="capital of France", is_llm_response=False, search_result=search_result)

    prompt = pp.llm.last_prompt
    assert answer == "extracted-answer"                        # extraction still returns
    assert "Paris is the capital of France" in prompt          # key sentence reaches LLM
    assert "Coffee is a popular morning drink" not in prompt   # filler was condensed out


def test_extract_answer_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr("utils.post_processor.ChatOpenAI", _RecordingLLM)
    pp = PostProcessor(llm_model="gpt-4.1")  # default: condensing off

    filler = "Coffee is a popular morning drink. " * 6
    search_result = _build_results([("https://example.com/x", filler.strip())])

    pp.extract_answer(query="capital of France", is_llm_response=False, search_result=search_result)

    assert "Coffee is a popular morning drink" in pp.llm.last_prompt
