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
  one?* This is a binary question whose probability orders the candidates.
* a forecast over steps is, per step, *what does this window hold?* This is the
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

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from sdm_client.errors import StructuredDataError
from sdm_client.wire.dtypes import _decimal_dtype

if TYPE_CHECKING:
    pass

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


def _ended(message: Any) -> str:
    """``message`` with a full stop, so what is appended reads as a sentence.

    Not every message this wraps ends in one, and two of them run together
    without it.
    """
    text = str(message).rstrip()
    return text if not text or text[-1] in '.!?' else f'{text}.'


def refuse_a_part_that_would_take_several_calls(
    rewritten: Decomposition, asked: int, per_call: int
) -> None:
    r"""Refuse a question asked in parts over more rows than one call scores.

    A part is one call. The guard above counts parts on that basis, so a
    batched part would make the parts times the batches: a ranking over 200
    candidates and 50,000 entities passes as 200 calls and sends 1,000.

    A ranking could not be batched in any case. Each batch samples its own
    context, and a batch whose context holds one class leaves the candidate
    out of every row rather than out of its own, because a candidate is
    reported unscored for the whole ranking.

    Checked before anything is gathered, from the count the caller gave, so a
    refusal costs nothing and arrives before the ceilings that are about what
    gathering holds.
    """
    if per_call <= 0 or asked <= per_call:
        return
    each = 'candidate' if rewritten.kind == RANKING else 'step'
    why = (
        'and a candidate is reported unscored for the whole ranking, so one '
        'cold batch would drop it from every row rather than from its own'
        if rewritten.kind == RANKING
        else 'and the calls would be the batches times the steps rather than '
        'the steps that were counted'
    )
    raise StructuredDataError(
        f'{asked:,} entities are more than the {per_call:,} one call scores, '
        f'and this question is asked as {len(rewritten.parts):,} of them, one '
        f'per {each}. A part is not batched, {why}. Ask about fewer entities '
        f'at a time',
        code='UNSUPPORTED_REQUEST',
        details={'entities': asked, 'rows_per_call': per_call},
    )


def about_the_part(
    error: StructuredDataError, rewritten: Decomposition, about: Any
) -> StructuredDataError:
    r"""``error``, said of the part it happened in, and still its own type.

    A ranking's part is about one candidate, which can be dropped; a forecast's
    covers a window further out than the one written, which is asked for by
    asking for fewer steps. One wording served both and was false of the first.

    The step named is the one that failed: anything at or above it fails alike.
    The type is kept, because it is what `predict` documents.

    Which part failed is always said, because it is always true. A remedy is
    added only where the failure is one it addresses. An unknown model is not
    answered by asking for fewer steps, and at step 1 there are no fewer to ask
    for.
    """
    how_many = len(rewritten.parts)
    first = _ended(error.args[0])
    if rewritten.kind == RANKING:
        said = (
            f'This is the question asked for candidate {about!r}, one of '
            f'{how_many} this ranking is asked as.'
        )
    else:
        said = f'This is forecast step {about} of {how_many}.'
        if (getattr(error, 'details', None) or {}).get(
            'window_outran_the_data'
        ):
            said += (
                f" The window named above is that step's rather than the one "
                f'written; ask for fewer than {about} steps if the data does '
                f'not reach that far.'
            )
    said = f'{first} {said}'
    carried = error.__class__.__new__(error.__class__)
    carried.__dict__.update(error.__dict__)
    carried.args = (said, *error.args[1:])
    carried.message = said
    return carried


def said_as_parts_are(
    error: StructuredDataError,
    candidates: list[Any] | None,
) -> StructuredDataError:
    r"""A refusal the core gave of a question it asks in parts, on the core
    route (T1.36 fix 21, T2.8), with the details the old route's has: the
    core's sentences already name the part it happened in (T3.17 fix 2), so
    the part's fields (``ranking_candidate``, ``ranking_parts``,
    ``forecast_step``, ``forecast_steps``) leave its details, and a
    candidate named by its position among ``candidates`` is named as
    given."""
    details = getattr(error, 'details', None)
    if not isinstance(details, dict):
        return error
    for name in (
        'ranking_candidate',
        'ranking_parts',
        'forecast_step',
        'forecast_steps',
    ):
        details.pop(name, None)
    if candidates is not None and isinstance(details.get('candidate'), int):
        details['candidate'] = candidates[details['candidate']]
    return error


def _decimal_as_typed(value: Decimal, dtype: str | None = None) -> Any:
    r"""``value`` as the Python type its column is typed as on the wire.

    A warehouse ``DECIMAL`` reaches a caller as a :class:`~decimal.Decimal`,
    and written as a string it asks after the row named ``"1"`` on a column of
    integers, a question the validator refuses and nobody wrote. Which type
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
