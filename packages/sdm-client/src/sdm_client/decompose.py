# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Asking a question the model does not take, in questions it does.

Two kinds of question have no task on the endpoint. A ranking asks which rows
of one table an entity will reach in a window; a forecast asks what a window
holds at each of several steps ahead. The models serve classification and
regression, and nothing else, so a request for either comes back refused.

Both are the same question asked repeatedly, though, and the repetition is
what the endpoint does serve:

* a ranking over candidates is, per candidate, *will this entity reach this
  one?* -- a binary question whose probability orders the candidates;
* a forecast over steps is, per step, *what does this window hold?* -- the
  question already being asked, with its window moved along.

Each part needs its own context, because the labels differ: what was true in
the second window is not what was true in the first, and whether an entity
reached *this* candidate is not whether it reached any. So the decomposition
happens where contexts are gathered rather than where envelopes are built,
and what comes back is merged into the one answer that was asked for.

The cost is a call per part, which for a ranking is a call per candidate.
That is the whole reason candidates must be named rather than assumed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

#: More parts than this and the fan-out is the cost of the call, not a detail
#: of it. A caller who means it can raise it; nobody discovers it by accident.
MOST_PARTS = 200


def parts_of(
    query: ValidatedPredictiveQuery,
    text: str,
    candidates: list[Any] | None,
) -> list[tuple[str, Any]] | None:
    r"""The queries this one is asked as, or nothing if it is asked as itself.

    Returns query text paired with what the part is about -- the candidate it
    asks after, or the step it covers. The pair matters: reading the candidate
    back out of the text it was written into loses its type and trusts a
    string the caller chose.

    Each part is gathered like any other query, which is what gives it the
    context its own labels need.
    """
    kind = str(getattr(query, 'problem_type', ''))
    if kind.endswith('RANK'):
        return _per_candidate(query, candidates)
    if kind.endswith('FORECAST'):
        return _per_step(query)
    if candidates is not None:
        raise StructuredDataError(
            f'candidates are the rows a ranking is over, and {text!r} does '
            f'not rank anything; it is a '
            f'{kind.rsplit(".", 1)[-1].lower()} query',
            code='INVALID_REQUEST',
        )
    return None


def _window(query: ValidatedPredictiveQuery) -> tuple[int, int, str]:
    span = query.get_final_target_aggregation().date_offset_range
    return int(span.start), int(span.end), str(span.unit)


def _per_candidate(
    query: ValidatedPredictiveQuery,
    candidates: list[Any] | None,
) -> list[tuple[str, Any]]:
    r"""One binary question per candidate, ordered by how sure the answer is.

    The endpoint has no ranking task, and a ranking it could serve would still
    be over whatever the sampler reached. Asking per candidate makes the set
    the caller's, which is the only version of this question with a knowable
    answer.
    """
    aggregation = query.get_final_target_aggregation()
    ranked = str(aggregation.target)
    if ' WHERE ' in ranked:
        raise StructuredDataError(
            f'the rows to rank are chosen by a filter ({ranked!r}), and each '
            f'candidate is asked about with a filter of its own, so the two '
            f'would have to be combined into one question this cannot write. '
            f'Rank without the filter, or narrow the rows before building the '
            f'schema.',
            code='INVALID_REQUEST',
        )
    table, _, column = ranked.partition('.')
    if candidates is None:
        raise StructuredDataError(
            f'a ranking is over a set of candidates, and this model has no '
            f'ranking task to choose them with, so they have to be named: '
            f'materialize(..., candidates=frame[{column!r}]). '
            f'One call is made per candidate.',
            code='INVALID_REQUEST',
        )
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise StructuredDataError(
            'candidates is empty, so there is nothing to rank',
            code='INVALID_REQUEST',
        )
    _refuse_too_many(len(unique), 'candidates')
    start, end, unit = _window(query)
    return [
        (
            f'PREDICT COUNT({table}.* WHERE {ranked}={_as_literal(value)}, '
            f'{start}, {end}, {unit}) > 0 FOR EACH {query.entity_column}',
            value,
        )
        for value in unique
    ]


def _per_step(query: ValidatedPredictiveQuery) -> list[tuple[str, Any]]:
    r"""One question per step, each about the window that step covers.

    The aggregation is rebuilt from what it aggregates, which carries any
    filter it had: `COUNT(buys.* WHERE channel="web", ...)` keeps its channel.
    Only the window moves.
    """
    steps = int(getattr(query, 'num_forecasts', 1) or 1)
    _refuse_too_many(steps, 'forecast steps')
    aggregation = query.get_final_target_aggregation()
    start, end, unit = _window(query)
    span = end - start
    return [
        (
            f'PREDICT {aggregation.aggr}({aggregation.target}, '
            f'{start + step * span}, {end + step * span}, {unit}) '
            f'FOR EACH {query.entity_column}',
            # The contract numbers steps from one.
            step + 1,
        )
        for step in range(steps)
    ]


def _refuse_too_many(many: int, what: str) -> None:
    if many <= MOST_PARTS:
        return
    raise StructuredDataError(
        f'{many} {what} means {many} calls, one for each, because the model '
        f'has no task that answers them together. That is past the {MOST_PARTS} '
        f'this will do without being told to; ask about fewer, or raise '
        f'sdm_client.decompose.MOST_PARTS if the wait is what you want.',
        code='INVALID_REQUEST',
    )


def _as_literal(value: Any) -> str:
    r"""``value`` as the query language writes it.

    A candidate is a key, and a key that cannot be written as one is refused
    rather than escaped into something that parses as a different question.
    """
    # Candidates usually come out of a frame, where a whole number is a numpy
    # integer and not an ``int``, and the two have to be written alike.
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if value is None:
        raise StructuredDataError(
            'a candidate names a row, and a missing value names none',
            code='INVALID_REQUEST',
        )
    if isinstance(value, bool):
        raise StructuredDataError(
            'a candidate names a row, and True and False do not',
            code='INVALID_REQUEST',
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise StructuredDataError(
                f'{value!r} cannot name a row to rank',
                code='INVALID_REQUEST',
            )
        return str(int(value))
    text = str(value)
    if any(character in text for character in '"\'\\,()') or not text.strip():
        raise StructuredDataError(
            f'the candidate {value!r} cannot be written into a query: a key '
            f'holding a quote, a comma, a bracket or nothing at all would '
            f'read as part of the question rather than as the row it names',
            code='INVALID_REQUEST',
        )
    return f'"{text}"'
