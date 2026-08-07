"""
Cost-of-pass economics for search-provider evaluation.

Adapted from "Cost-of-Pass: An Economic Framework for Evaluating Language
Models" (arXiv:2504.13359), which grounds LLM evaluation in production
theory: rather than ranking models on accuracy alone, rank them on the
expected monetary cost of producing a *correct* answer.

The paper's core metric is

    cost_of_pass = per-attempt inference cost / accuracy

and the "frontier cost-of-pass" is the minimum cost_of_pass across the
field of models being compared -- the cheapest way to buy one correct
answer.

Mode-2 adaptation.  The paper's full framework prices an end-to-end model
serving stack.  In search-augmented QA the dominant inference-cost driver
is the retrieved-content tokens fed to the extraction LLM, and this repo
already logs per-example ``token_count``.  We therefore substitute the
paper's pricing model with a small, parameter-free price table keyed by the
repo's ``token_model`` (default ``gpt-4.1``) and convert recorded tokens
into a USD cost-of-pass per provider.  Everything else (the per-attempt
cost / accuracy definition, the frontier min) is kept at full fidelity.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Approximate USD price per token for the models this repo uses as its
# token_model, expressed as input-token pricing per 1M tokens.  These are a
# fixed, swappable constant standing in for the paper's pricing model -- not a
# live price feed.  Update this table when adding a new token_model.
PRICE_TABLE_USD_PER_TOKEN: Dict[str, float] = {
    'gpt-4.1': 2.0e-6,        # $2.00 / 1M tokens
    'gpt-4.1-mini': 0.4e-6,   # $0.40 / 1M tokens
    'gpt-4.1-nano': 0.1e-6,   # $0.10 / 1M tokens
    'gpt-4o': 2.5e-6,         # $2.50 / 1M tokens
    'gpt-4o-mini': 0.15e-6,   # $0.15 / 1M tokens
}

# Falls back to gpt-4.1 (the repo's default token_model) for unknown models.
DEFAULT_PRICE_USD_PER_TOKEN = PRICE_TABLE_USD_PER_TOKEN['gpt-4.1']


def price_per_token(model: Optional[str]) -> float:
    """USD price per token for a token_model name.

    Case-insensitive match against ``PRICE_TABLE_USD_PER_TOKEN``; unknown
    models fall back to the gpt-4.1 default so cost-of-pass is always defined.
    """
    if not model:
        return DEFAULT_PRICE_USD_PER_TOKEN
    key = str(model).strip().lower()
    for name, price in PRICE_TABLE_USD_PER_TOKEN.items():
        if name.lower() == key:
            return price
    logger.info(
        "No price entry for token_model '%s'; using gpt-4.1 default", model
    )
    return DEFAULT_PRICE_USD_PER_TOKEN


def cost_of_pass(
    total_tokens: float,
    total_attempts: int,
    correct_count: int,
    model: Optional[str] = None,
) -> float:
    """Expected USD cost of producing one correct answer.

    Implements ``cost_of_pass = per_attempt_inference_cost / accuracy`` where
    ``per_attempt_inference_cost = (total_tokens * price) / total_attempts``
    and ``accuracy = correct_count / total_attempts``.  The ``total_attempts``
    terms cancel, reducing the result to ``(total_tokens * price) /
    correct_count``; we keep the per-attempt form to mirror the paper's
    definition directly.

    A provider that never answers correctly has unbounded cost-of-pass
    (``float('inf')``) -- it cannot buy a correct answer at any finite price.
    """
    if correct_count <= 0 or total_attempts <= 0:
        return float('inf')
    accuracy = correct_count / total_attempts
    inference_cost = total_tokens * price_per_token(model)
    per_attempt_cost = inference_cost / total_attempts
    return per_attempt_cost / accuracy


def frontier_cost_of_pass(
    costs: Dict[str, float],
) -> Tuple[Optional[str], float]:
    """Return ``(provider, cost)`` for the cheapest correct answer.

    The frontier is the minimum cost-of-pass across the field; providers with
    infinite cost-of-pass (no correct answers) are excluded.  Returns
    ``(None, inf)`` when every provider is off the frontier.
    """
    finite = {p: c for p, c in costs.items() if c != float('inf')}
    if not finite:
        return None, float('inf')
    provider = min(finite, key=finite.get)
    return provider, finite[provider]
