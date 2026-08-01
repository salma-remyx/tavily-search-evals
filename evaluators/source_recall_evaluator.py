"""Source-recall evaluation for live-web retrieval providers.

Adapted from the core evaluation primitive of DeepScholar-Bench
(arXiv:2508.20033) — *A Live Benchmark and Automated Evaluation for
Generative Research Synthesis*. DeepScholar-Bench argues that QA-style
benchmarks miss whether a research-synthesis system actually surfaces the
right *sources*: a retriever can return a plausible answer while never
retrieving the ground-truth citations. Its central metric is, per research
question, what fraction of the gold-truth cited sources the system
retrieved.

This module ports that retrieval-coverage primitive into this repo's
search-API evaluation harness. The other benchmarks here grade answer
correctness (SimpleQA) or document-level relevance via an external scorer
(document_relevance). Source recall adds the missing axis: against a
human-curated gold source set per query, how much of it does each provider
retrieve?

Adaptation scope (Mode 2 — core mechanism preserved, auxiliaries
substituted):
  * Preserved at full fidelity: per-query source recall@k and precision@k
    against a human-curated gold source set, the paper's central metric.
  * DeepScholar's full research-synthesis benchmark suite + its live
    human-curated dataset -> a small target-native JSON dataset
    (question + gold source URLs), loaded through the same harness as the
    other benchmarks (`load_source_recall_eval_data`).
  * DeepScholar's automated citation-faithfulness / synthesis-quality LLM
    judges -> intentionally cut. This measures retrieval source coverage
    only; synthesis-quality evaluation belongs in a downstream PR.
  * Source matching is parameter-free URL normalization (no learned
    estimator), so the metric runs with no extra model calls.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from utils.utils import EvaluationType, save_result

logger = logging.getLogger(__name__)


def normalize_source(url: str) -> str:
    """Normalize a source URL for gold-vs-retrieved matching.

    Strips scheme, leading ``www.``, trailing slash, and query/fragment so
    that ``https://www.en.wikipedia.org/wiki/X/`` and
    ``http://en.wikipedia.org/wiki/X?oldid=1#sec`` compare equal. Returns
    an empty string for empty / unparseable input so callers can drop it.
    """
    if not url or not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    try:
        parsed = urlparse(cleaned)
    except (ValueError, TypeError):
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").rstrip("/")
    return f"{host}{path}"


def extract_result_urls(search_response: Dict) -> List[str]:
    """Pull the ordered list of result URLs out of a provider search response.

    Providers return ``{"search_response": {"results": [{"url": ...}, ...]}}``
    (Tavily/Serper/Brave/Perplexity-Search). Exa nests under ``results`` too.
    We read defensively so the evaluator works across handler shapes.
    """
    if not isinstance(search_response, dict):
        return []
    inner = search_response.get("search_response", search_response)
    if isinstance(inner, dict):
        results = inner.get("results", [])
    elif isinstance(inner, list):
        results = inner
    else:
        results = []
    urls: List[str] = []
    for res in results:
        if isinstance(res, dict):
            url = res.get("url") or res.get("link") or res.get("href")
            if url:
                urls.append(url)
    return urls


class SourceRecallEvaluator:
    """Score a single query's retrieval against its gold source set.

    This is the DeepScholar-Bench evaluation primitive: given the gold-truth
    cited sources for a research question and the sources a provider
    retrieved, how much of the gold set was covered (recall@k) and how much
    of what was retrieved was on-target (precision@k)?
    """

    def __init__(self, k: int = 10):
        self.k = k

    def score(
        self,
        gold_sources: List[str],
        retrieved_sources: List[str],
    ) -> Dict[str, float]:
        """Return recall@k / precision@k plus raw match counts for one query.

        Args:
            gold_sources: Ground-truth cited source URLs for the query.
            retrieved_sources: URLs the provider returned, in rank order.

        Returns:
            Dict with ``recall`` (matched/gold), ``precision`` (matched/k),
            ``matched_count``, ``gold_count`` and ``hit`` (1.0 if any gold
            source was retrieved, else 0.0).
        """
        gold = {
            norm
            for norm in (normalize_source(s) for s in (gold_sources or []))
            if norm
        }
        retrieved_topk = [
            norm
            for norm in (normalize_source(s) for s in (retrieved_sources or []))
            if norm
        ][: self.k]
        topk_set = set(retrieved_topk)
        matched = gold & topk_set
        recall = len(matched) / len(gold) if gold else 0.0
        precision = len(matched) / len(retrieved_topk) if retrieved_topk else 0.0
        return {
            "recall": recall,
            "precision": precision,
            "matched_count": float(len(matched)),
            "gold_count": float(len(gold)),
            "hit": 1.0 if matched else 0.0,
        }


async def evaluate_provider_source_recall(
    provider_name: str,
    search_handler,
    examples: List[Dict],
    output_dir: str,
    k: int = 10,
    batch_size: int = 3,
) -> Dict:
    """Evaluate one provider's source recall across the whole dataset.

    Mirrors the shape of ``evaluate_provider_document_relevance``: runs the
    provider's search per query, scores retrieved URLs against that query's
    gold sources, appends each row through ``save_result``, and returns
    aggregate metrics the harness writes to the summary.
    """
    evaluator = SourceRecallEvaluator(k=k)
    results: List[Dict] = []
    recall_sum = 0.0
    precision_sum = 0.0
    hit_count = 0
    matched_total = 0
    gold_total = 0

    async def process_example(example: Dict):
        nonlocal recall_sum, precision_sum, hit_count, matched_total, gold_total
        query = example["question"]
        gold_sources = example.get("gold_sources", [])
        index = example["index"]
        try:
            search_result = await search_handler.search(query)
            retrieved_urls = extract_result_urls(search_result)
            score = evaluator.score(gold_sources, retrieved_urls)

            recall_sum += score["recall"]
            precision_sum += score["precision"]
            hit_count += int(score["hit"])
            matched_total += int(score["matched_count"])
            gold_total += int(score["gold_count"])

            result = {
                "index": index,
                "question": query,
                "gold_count": int(score["gold_count"]),
                "matched_count": int(score["matched_count"]),
                "recall": round(score["recall"], 4),
                "precision": round(score["precision"], 4),
                "hit": int(score["hit"]),
                "retrieved_count": len(retrieved_urls),
                "grade": "completed",
            }
            results.append(result)
            save_result(result, provider_name, output_dir, EvaluationType.SOURCE_RECALL)
            logger.info(
                f"[{provider_name}] Q{index}: recall={score['recall']:.2f} "
                f"({result['matched_count']}/{result['gold_count']})"
            )
            return result
        except Exception as e:
            logger.error(f"[{provider_name}] Error evaluating example {index}: {str(e)}")
            err = {
                "index": index,
                "question": query,
                "gold_count": len(gold_sources),
                "matched_count": 0,
                "recall": 0.0,
                "precision": 0.0,
                "hit": 0,
                "retrieved_count": 0,
                "grade": "ERROR",
            }
            results.append(err)
            return None

    n = len(examples)
    for i in range(0, n, batch_size):
        batch = examples[i : i + batch_size]
        await asyncio.gather(*[process_example(ex) for ex in batch])
        time.sleep(3.0)  # avoid rate limiting, matching the SimpleQA loop

    mean_recall = recall_sum / n if n else 0.0
    mean_precision = precision_sum / n if n else 0.0
    hit_rate = hit_count / n if n else 0.0
    return {
        "provider": provider_name,
        "results": results,
        "mean_recall": round(mean_recall, 3),
        "mean_precision": round(mean_precision, 3),
        "hit_rate": round(hit_rate, 3),
        "matched_sources": matched_total,
        "total_gold_sources": gold_total,
        "total_count": n,
    }


def load_source_recall_eval_data(
    json_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    random_sample: Optional[int] = None,
) -> pd.DataFrame:
    """Load a source-recall dataset (question + gold source URLs).

    JSON shape: ``{"dataset": [{"question": str, "gold_sources": [url, ...]}, ...]}``.
    Returns a DataFrame with ``problem``, ``answer`` (unused, kept for
    parity with the shared loader), ``gold_sources`` and ``index`` columns
    so it flows through ``prepare_examples`` unchanged.
    """
    logger.info(f"Loading source-recall data from JSON file: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset = data["dataset"] if isinstance(data, dict) and "dataset" in data else data
    total_rows = len(dataset)

    if random_sample is not None and random_sample > 0:
        sample_size = min(random_sample, total_rows)
        logger.info(f"Randomly sampling {sample_size} examples from {total_rows} total")
        import random

        dataset = random.sample(dataset, sample_size)
        start = 0
    else:
        if end_index is None:
            end_index = total_rows
        start = max(0, min(start_index, total_rows - 1))
        end_index = max(start + 1, min(end_index, total_rows))
        logger.info(
            f"Using examples from index {start} to {end_index - 1} (total: {end_index - start})"
        )
        dataset = dataset[start:end_index]

    examples = []
    for i, item in enumerate(dataset):
        examples.append(
            {
                "problem": item["question"],
                "answer": "",  # source recall keys off gold_sources, not a gold answer
                "gold_sources": item.get("gold_sources", []),
                "index": start + i,
            }
        )
    return pd.DataFrame(examples)
