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

import contextlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from sdm_client.errors import StructuredDataError
from sdm_client.wire.dtypes import _decimal_dtype, _is_missing

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client.schema import Schema

#: More parts than this and the fan-out is the cost of the call, not a detail
#: of it. A caller who means it can raise it; nobody discovers it by accident.
MOST_PARTS = 200

RANKING = 'ranking'
FORECAST = 'forecast'


@dataclass(frozen=True)
class Decomposition:
    r"""The questions one question was rewritten as, and what each is about.

    Carried on the request rather than re-derived from its text: a query
    naming a table ``rank_events`` is not a ranking, and a candidate read back
    out of the text it was written into is a string where a key was given.
    """

    kind: str
    parts: tuple[tuple[str, Any], ...]
    top: int | None = None


def parts_of(
    query: ValidatedPredictiveQuery,
    text: str,
    candidates: list[Any] | None,
    options: Mapping[str, Any],
    schema: Schema | None = None,
) -> Decomposition | None:
    r"""The queries this one is asked as, or nothing if it is asked as itself.

    Returns query text paired with what the part is about -- the candidate it
    asks after, or the step it covers. The pair matters: reading the candidate
    back out of the text it was written into loses its type and trusts a
    string the caller chose.

    Each part is gathered like any other query, which is what gives it the
    context its own labels need.
    """
    # Imported here because how many candidates to rank is settled by the
    # relational materializer, which is built on the request this returns.
    from sdm_client.materializers.relational import _candidates_to_rank

    kind = str(getattr(query, 'problem_type', ''))
    if kind.endswith('RANK'):
        if 'prediction_statistic' in options:
            # Said here: each part is a classification, and refusing there
            # would name a binary_classification the caller never wrote.
            raise StructuredDataError(
                'prediction_statistic applies only to a regression, and this '
                'question is a ranking, ordered by how likely each candidate '
                'is; ask it without prediction_statistic',
                code='INVALID_REQUEST',
            )
        return Decomposition(
            RANKING,
            tuple(_per_candidate(query, candidates, schema)),
            _candidates_to_rank(query, options),
        )
    if candidates is not None:
        raise StructuredDataError(
            f'candidates are the rows a ranking is over, and {text!r} does '
            f'not rank anything. Ask it without candidates, or rank with '
            f'"RANK TOP k", which is the question they are for',
            code='INVALID_REQUEST',
        )
    if kind.endswith('FORECAST'):
        if 'top_k' in options:
            raise StructuredDataError(
                'top_k is how many candidates a ranking keeps, and a forecast '
                'ranks nothing; it gives back one value per step',
                code='INVALID_REQUEST',
            )
        return Decomposition(FORECAST, tuple(_per_step(query)))
    return None


def _ended(message: Any) -> str:
    """``message`` with a full stop, so what is appended reads as a sentence.

    Not every message this wraps ends in one, and two of them run together
    without it.
    """
    text = str(message).rstrip()
    return text if not text or text[-1] in '.!?' else f'{text}.'


#: How the engine says a window reaches back past the data's earliest row.
_OUTRUNS_THE_DATA = 'we would need data back to'


def about_the_part(
    error: StructuredDataError, rewritten: Decomposition, about: Any
) -> StructuredDataError:
    r"""``error``, said of the part it happened in, and still its own type.

    A ranking's part is about one candidate, which can be dropped; a forecast's
    covers a window further out than the one written, which is asked for by
    asking for fewer steps. One wording served both and was false of the first.

    The step named is the one that failed: anything at or above it fails alike.
    The type is kept, because it is what `predict` documents.
    """
    how_many = len(rewritten.parts)
    first = _ended(error.args[0])
    if rewritten.kind == RANKING:
        said = (
            f'This is the question asked for candidate {about!r}, one of '
            f'{how_many} this ranking is asked as.'
        )
    elif _OUTRUNS_THE_DATA in first:
        said = (
            f'This is forecast step {about} of {how_many}, and the window '
            f"named above is that step's rather than the one written; ask "
            f'for fewer than {about} steps if the data does not reach that '
            f'far.'
        )
    else:
        # Fewer steps only helps when this step's window outran the data; an
        # unknown option fails at step 1 alike, and "fewer than 1" is no remedy.
        said = f'This is forecast step {about} of {how_many}.'
    said = f'{first} {said}'
    carried = error.__class__.__new__(error.__class__)
    carried.__dict__.update(error.__dict__)
    carried.args = (said, *error.args[1:])
    carried.message = said
    return carried


def _for_each(query: ValidatedPredictiveQuery) -> str:
    r"""The entity clause as it was written, filter and all.

    An entity filter chooses which rows become labelled examples rather than
    which rows are scored, so a part that drops it is answered from a
    different population than the one asked about.
    """
    return f'{query.for_each} {query.entity_ast}'


def _window(query: ValidatedPredictiveQuery) -> tuple[int, int, str]:
    span = query.get_final_target_aggregation().date_offset_range
    return int(span.start), int(span.end), str(span.unit)


def _per_candidate(
    query: ValidatedPredictiveQuery,
    candidates: list[Any] | None,
    schema: Schema | None = None,
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
    dtype = _how_the_column_travels(schema, table, column)
    start, end, unit = _window(query)
    return [
        (
            f'PREDICT COUNT({table}.* WHERE {ranked}='
            f'{_as_literal(value, dtype)}, '
            f'{start}, {end}, {unit}) > 0 {_for_each(query)}',
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
            f'{_for_each(query)}',
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


def _how_the_column_travels(
    schema: Schema | None, table: str, column: str
) -> str | None:
    r"""The dtype the schema computed for the column being ranked.

    How a candidate is written is the column's question, not the value's: a key
    column straddling int64 travels as text, and so must a small key from it.
    """
    if schema is None or table not in schema:
        return None
    described = schema[table]
    if column not in described:
        return None
    return str(described[column].dtype)


def _as_literal(value: Any, dtype: str | None = None) -> str:
    r"""``value`` as the query language writes it.

    A candidate is a key, and a key that cannot be written as one is refused
    rather than escaped into something that parses as a different question.
    """
    # Candidates usually come out of a frame, where a whole number is a numpy
    # integer and not an ``int``, and the two have to be written alike.
    if hasattr(value, 'item'):
        with contextlib.suppress(AttributeError, ValueError):
            value = value.item()
    if _is_missing(value):
        raise StructuredDataError(
            'a candidate names a row, and a missing value names none',
            code='INVALID_REQUEST',
        )
    if isinstance(value, bool):
        raise StructuredDataError(
            'a candidate names a row, and True and False do not',
            code='INVALID_REQUEST',
        )
    if isinstance(value, Decimal):
        written = _decimal_as_typed(value, dtype)
        if isinstance(written, (int, float)) and Decimal(written) != value:
            raise StructuredDataError(
                f'the candidate {value} names no row of a column of '
                f'{dtype or "numbers"}: written as one it becomes {written}, '
                f'which is a different row',
                code='INVALID_REQUEST',
            )
        value = written
    if dtype == 'string' and isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise StructuredDataError(
                f'{value!r} cannot name a row to rank',
                code='INVALID_REQUEST',
            )
        value = format(Decimal(str(value)), 'f')
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


def _decimal_as_typed(value: Decimal, dtype: str | None = None) -> Any:
    r"""``value`` as the Python type its column is typed as on the wire.

    A warehouse ``DECIMAL`` reaches a caller as a :class:`~decimal.Decimal`,
    and written as a string it asks after the row named ``"1"`` on a column of
    integers -- a question the validator refuses and nobody wrote. Which type
    that is does not follow from the Python type: :func:`_decimal_dtype` is
    what decides whether such a column travels as an integer, a float, or its
    digits. ``dtype`` carries the column's; the value alone is used only where
    the column is not known.

    The digits are formatted rather than ``str``-ed, since ``str`` turns to
    scientific notation past int64's width, which is the width that makes such
    a column text in the first place.
    """
    if dtype is None:
        dtype = _decimal_dtype(pd.Series([value], dtype=object))
    if dtype == 'int64':
        return int(value)
    if dtype == 'float64':
        return float(value)
    return format(value, 'f')
