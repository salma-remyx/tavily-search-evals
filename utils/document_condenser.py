"""Document condenser — extract query-relevant key sentences from retrieved docs.

Adapted from "Less is More for Long Document Summary Evaluation by LLMs"
(arXiv:2309.07382), whose *Extract-then-Evaluate* method extracts key
sentences from a long source document before an LLM evaluates the summary.
That curbs the Lost-in-the-Middle effect (long documents' middles get
overlooked) and trims prompt cost.

This is a Mode 2 (adapted) port: the paper scores candidate sentences with
learned / similarity estimators (LEAD, ROUGE, BERTScore, NLI). Those
auxiliaries are substituted here with a parameter-free query-token-overlap
scorer -- no model, no extra dependency -- and the extract step plugs into
this repo's existing answer-extraction path rather than the paper's
standalone summary-evaluation benchmark.

The public entry point :func:`condense_search_results` consumes the document
string that ``_format_search_results_for_prompt`` emits (one block per
retrieved doc) and returns a shorter string in the *same* format, with each
document's content reduced to its most query-relevant sentences. Short
documents and inputs that do not match the expected format are returned
unchanged, so condensing can never break answer extraction.
"""

import re
from typing import List

# Per-document block format produced by
# base_handler._format_search_results_for_prompt:
#   "\n**Document {n}.** Source: {url}\nContent: {content}"  (joined by "\n")
_DOC_SPLIT_RE = re.compile(r"(?=\n\*\*Document \d+\.\*\* Source:)")
_DOC_PARSE_RE = re.compile(
    r"\n\*\*Document (?P<num>\d+)\.\*\* Source: (?P<url>[^\n]*)\nContent: (?P<content>.*)",
    re.DOTALL,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# A small, dependency-free stopword set so filler words do not dominate the
# overlap score. This is a proxy signal, not an NLP pipeline.
_STOPWORDS = frozenset(
    """
    a an the and or but if while of to in on at by for with from into over
    is are was were be been being this that these those it its as not no
    what which who whom whose when where why how many much more most some
    any all each every did do does done has have had will would can could
    should shall may might must about""".split()
)


def _content_tokens(text: str) -> set:
    """Lowercase content-word tokens (>2 chars, no stopwords) from ``text``."""
    tokens = set()
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        if len(lowered) > 2 and lowered not in _STOPWORDS:
            tokens.add(lowered)
    return tokens


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences on sentence-final punctuation."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def condense_document_content(
    query: str,
    content: str,
    max_sentences: int,
    min_sentences_to_condense: int = 5,
) -> str:
    """Reduce one document's ``content`` to its most query-relevant sentences.

    Sentences are ranked by the count of their content tokens that also appear
    in the query (the parameter-free proxy for the paper's relevance
    estimator). The top ``max_sentences`` are kept and returned in their
    original order so the condensed text still reads naturally. Documents with
    fewer than ``min_sentences_to_condense`` sentences are returned unchanged --
    the paper's "less is more" claim targets *long* documents, and shorter
    snippets have nothing to gain from extraction.
    """
    sentences = split_sentences(content)
    if max_sentences <= 0 or len(sentences) <= min_sentences_to_condense:
        return content
    if max_sentences >= len(sentences):
        return content

    query_tokens = _content_tokens(query or "")
    # Highest query-overlap first; original index breaks ties deterministically.
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: (-len(query_tokens & _content_tokens(sentences[i])), i),
    )
    keep = sorted(ranked[:max_sentences])
    return " ".join(sentences[i] for i in keep)


def condense_search_results(
    query: str,
    search_results: str,
    max_sentences_per_doc: int = 3,
    min_sentences_to_condense: int = 5,
) -> str:
    """Condense the formatted search-result string fed to answer extraction.

    Parses the per-document blocks emitted by
    ``_format_search_results_for_prompt`` and reduces each document's content
    via :func:`condense_document_content`. URLs and document numbering are
    preserved so downstream prompts that reference "Source" / "Document N"
    keep working. Any input that does not match the expected format is
    returned unchanged.
    """
    if not search_results or not max_sentences_per_doc or max_sentences_per_doc <= 0:
        return search_results

    matches = list(filter(None, (_DOC_PARSE_RE.match(b) for b in _DOC_SPLIT_RE.split(search_results))))
    if not matches:
        return search_results

    condensed_blocks = [
        f"\n**Document {m.group('num')}.** Source: {m.group('url')}"
        f"\nContent: {condense_document_content(query, m.group('content'), max_sentences_per_doc, min_sentences_to_condense)}"
        for m in matches
    ]
    return "\n".join(condensed_blocks)
