# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

_INSTRUCTIONS = """ROLE: Explanation summary module
INSTRUCTIONS: Your goal is to extract meaningful insight from structured explanations of a prediction. You will be given
- a predictive query which defines a predictive problem
- the prediction for that particular query
- column analysis
- subgraph explanation
- documentation on how to understand explanations (column analysis and subgraph explanation)

Provide insight to the user, make reference to specific quantitative details which led the model to it's prediction."""

_EXPLAINABILITY = """# Explainability

Kumo Relational explanations provide two complementary views of model predictions:

1. **Global View (Cohorts):** Column-level patterns across in-context examples that reveal what data characteristics drive predictions
1. **Local View (Subgraph):** Cell-level attribution scores showing which specific values in this entity's subgraph influenced the prediction

Together, these views answer: "What patterns does the model see globally?" and "Which specific data points matter for this prediction?"

## Understanding the Global View: Cohorts

Cohorts reveal how different value ranges or categories in columns correlate with prediction outcomes across all in-context examples.

- `table_name`: Which table this analysis covers
- `column_name`: Which column or statistic (e.g., `COUNT(*)`) this analysis covers
- `hop`: Distance from the entity table (0 = entity attributes, 1 = direct neighbors, 2 = second-degree neighbors, ...)
- `stype`: Semantic type (numerical, categorical, timestamp, etc)
- `cohorts`: List of value ranges/categories (e.g., `["[0-5]", "(5-10]", "(10-20+]"]`)
- `populations`: Proportion of in-context examples in each cohort
- `targets`: Average prediction score within each cohort

High-impact columns usually have large variance in `targets` across different cohorts.

**Example for a churn predictive query:**

```
table_name: "orders"
column_name: "COUNT(*)"
hop: 1
cohorts: ["[0-0]", "(0-1]", "(1-2]", "(2-4]", "(4-6+]"]
populations: [0.20, 0.08, 0.07, 0.11, 0.54]
targets: [0.0, 0.78, 0.74, 0.64, 0.35]
```

**What this means:**

- Users with 0 orders have 0% churn risk (they already churned)
- Users with 1-2 orders have ~75% churn risk (early stage, not sticky)
- Users with 6+ orders have 35% churn risk (established, but not immune)
- Key insight: Order count is strongly predictive; more orders = lower churn

## Understanding the Local View: Subgraph

Subgraphs show the actual data neighborhood around the specific entity being predicted, with attribution scores indicating importance.
Node indices are different from primary keys and are mapped to a contiguous range from 0 to N.
The entity being predicted is guaranteed to have ID 0.
Some cells may have a `null` value with non-zero scores, indicating missingness itself is informative.

Each node represents a row from a table, containing:

- `cells`: Dictionary of column values with attribution scores
  - `value`: Actual data value
  - `score`: Gradient-based importance between 0 and 1 (higher = more influential)
- `links`: Connections to other nodes via foreign keys

Scores reflect how much changing this value would change the prediction.
High scores on specific cells explain "why this prediction, not another".

**Score Magnitude Interpretation:**

- 0.00 - 0.05: Negligible influence
- 0.05 - 0.15: Moderate influence
- 0.15 - 0.30: Strong influence
- 0.30+: Critical influence

**Example:**

```
cells: {
  "account_status": {value: "ACTIVE", score: 1.0},
  "age": {value: 49, score: 0.089},
  "email_subscription": {value: "Weekly", score: 0.411}
}
links: {
  "user_id->orders": [1,2,3,...,32]
}
```

**What this means:**

Account status is the most important attribute (score=1.0)
Email subscription is moderately important (score=0.411).
Age contributes but is less critical (score=0.089).
User has 32 orders linked (indicates high activity).

You can follow paths in the subgraph to understand data connectivity and how tables/cells far away may contribute to the prediction.

## Connecting Global and Local Views

Often times, you can understand high subgraph attribution scores by relating their cell values to the average prediction of the cohort.

1. **Find influential cells for the prediction in the local view:**
   Which cells have scores > 0.15?
1. **Locate entity in global context:**
   Find which cohorts the specific entity falls into and compare entity's values to high/low risk cohorts.
   Focus on highest-scoring cells and most divergent cohorts.
1. **Relate attribution score and cohort prediction:**
   Check if entity exhibits typical or atypical patterns.
1. **Find general global trends** in the data that might explain the prediction.
   Additionally, look for missing expected signals (why ISN'T something important?)

Tell a coherent story connecting global patterns to local evidence.
Use concrete numbers from the subgraph.
Avoid jargon; explain in business terms.

## Common Interpretation Pitfalls

- **Don't assume correlation = causation:**
  High scores show model importance, not real-world causality.
  For example, "black clothing" might correlate with churn, but color isn't the cause.
- **Consider data distribution:**
  Rare cohorts may show extreme `targets` with small `populations`.
  Focus on cohorts with both significant population AND divergent targets.
- **Missing cohort analysis:**
  Not all columns have a cohort analysis since some semantic types are unsupported.
  For example, text and ID columns typically only appear in local view."""

SYSTEM_PROMPT = _INSTRUCTIONS + '\n\n' + _EXPLAINABILITY + '\n\n'

_DEFAULT_MODEL = 'gpt-4.1-mini-2025-04-14'
_DEFAULT_TIMEOUT = 20.0

_STRUCTURED_NOTE = (
    'The structured explanation is still available on .cohorts and .subgraphs.'
)

# Names the extra on the distribution users install (`sdm-client`), not on this
# engine package, which the client documents as an implementation detail.
SUMMARY_NEEDS_EXTRA_MESSAGE = (
    "Natural-language explanation summary needs the 'explain' extra: "
    "pip install 'sdm-relational-engine[explain]'. " + _STRUCTURED_NOTE
)

SUMMARY_UNAVAILABLE_MESSAGE = (
    'Natural-language explanation summary needs an API key. Set '
    'KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY, which sends the rows behind the '
    'prediction to the summary endpoint (the default is OpenAI gpt-4.1-mini, a '
    'non-NVIDIA service). For another OpenAI-compatible endpoint also set '
    'KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL and KUMO_RELATIONAL_EXPLAIN_LLM_MODEL. '
    + _STRUCTURED_NOTE
)

SUMMARY_NEEDS_MODEL_MESSAGE = (
    'Natural-language explanation summary: a custom endpoint is set '
    '(KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL) but no model. Set KUMO_RELATIONAL_EXPLAIN_LLM_MODEL to a '
    'model that endpoint serves. ' + _STRUCTURED_NOTE
)

SUMMARY_ERROR_MESSAGE = (
    'Natural-language explanation summary could not be generated. Check the '
    'endpoint URL, API key, and model.'
)

SUMMARY_TIMEOUT_MESSAGE = (
    'Natural-language explanation summary timed out after {timeout:g}s. Raise '
    'KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT (or point KUMO_RELATIONAL_EXPLAIN_LLM_MODEL at a faster model) '
    'and retry. ' + _STRUCTURED_NOTE
)


def _build_input_message(
    query: str,
    prediction: Any,
    cohorts: Sequence[Any],
    subgraphs: Sequence[Any],
) -> str:
    r"""Render the model prediction and its structured explanation into the
    prompt input expected by the summary model, mirroring the fields described
    in the embedded explainability instructions.
    """
    subgraph = subgraphs[0] if subgraphs else ''
    return (
        f'USER QUERY: {query}\n\n'
        f'MODEL PREDICTION: {prediction}\n\n'
        f'COLUMN ANALYSIS: {cohorts}\n\n'
        f'SUBGRAPH EXPLANATION: {subgraph}'
    )


def _make_client(
    base_url: str | None, api_key: str | None, timeout: float
) -> Any:
    r"""Build an OpenAI-compatible client for the given endpoint, or return
    ``None`` when the optional dependency or API key is missing so summary
    generation can degrade gracefully instead of raising.

    ``base_url`` targets any OpenAI-compatible chat-completions deployment
    (OpenAI, NVIDIA inference, vLLM, a self-hosted gateway, ...); when ``None``
    the client uses the default OpenAI host.
    """
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    kwargs: dict[str, Any] = {'api_key': api_key, 'timeout': timeout}
    if base_url:
        kwargs['base_url'] = base_url
    return OpenAI(**kwargs)


def _env(*names: str) -> str | None:
    r"""Return the first non-empty value among the given environment vars."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_float(*names_and_default: object) -> float:
    r"""Return the first of ``names`` parsed as a float, or the default.

    The default is the final argument; every argument before it is an
    environment variable name, tried in order.
    """
    *names, default = names_and_default
    assert isinstance(default, float)
    value = _env(*[str(name) for name in names])
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def generate_summary(
    query: str,
    prediction: Any,
    cohorts: Sequence[Any],
    subgraphs: Sequence[Any],
    *,
    client: Any = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> str:
    r"""Generate a human-readable summary of a Kumo Relational explanation.

    .. warning::

        This sends ``query``, ``prediction``, ``cohorts`` and ``subgraphs`` to
        the configured chat-completions endpoint. The last carries the **raw
        cell values** of the explained entity's subgraph.
        Unless ``base_url`` or ``KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL`` says
        otherwise, that endpoint is OpenAI's ``https://api.openai.com/v1/``, a
        non-NVIDIA service. Callers that must not egress row-level data should
        not reach this function. Set ``skip_summary=True`` on
        :class:`~sdm_relational_engine.rfm.ExplainConfig`.

    Built for OpenAI (default model ``gpt-4.1-mini``), it works with any
    OpenAI-compatible chat-completions endpoint. Configure once via the
    environment: the API key from ``KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY``,
    ``KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL`` for another endpoint,
    ``KUMO_RELATIONAL_EXPLAIN_LLM_MODEL``, and
    ``KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT`` in seconds. Passing ``base_url`` /
    ``api_key`` / ``model`` / ``timeout`` or an already-built ``client`` takes
    precedence. A slow model only needs a larger timeout. Instead of raising,
    the summary degrades to a short message naming what to set, and the
    structured explanation (``.cohorts`` / ``.subgraphs``) remains available.
    """
    if timeout is None:
        timeout = _env_float(
            'KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT', _DEFAULT_TIMEOUT
        )
    base_url = base_url or _env('KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL')
    # No ambient-key fallback: a key exported for an unrelated tool must not be
    # enough to start sending row values to a third party.
    api_key = api_key or _env('KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY')
    model_set = model or _env('KUMO_RELATIONAL_EXPLAIN_LLM_MODEL')
    model = model_set or _DEFAULT_MODEL

    if client is None:
        try:
            import openai  # noqa: F401
        except ImportError:
            return SUMMARY_NEEDS_EXTRA_MESSAGE
        if not api_key:
            return SUMMARY_UNAVAILABLE_MESSAGE
        if base_url and not model_set:
            return SUMMARY_NEEDS_MODEL_MESSAGE
        client = _make_client(base_url, api_key, timeout)
    if client is None:
        return SUMMARY_UNAVAILABLE_MESSAGE

    message = _build_input_message(query, prediction, cohorts, subgraphs)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': message},
            ],
        )
        content = response.choices[0].message.content
    except Exception as exc:
        if type(exc).__name__ == 'APITimeoutError':
            return SUMMARY_TIMEOUT_MESSAGE.format(timeout=timeout)
        return f'{SUMMARY_ERROR_MESSAGE} (reason: {type(exc).__name__})'
    if not content:
        return SUMMARY_ERROR_MESSAGE
    return content.strip()
