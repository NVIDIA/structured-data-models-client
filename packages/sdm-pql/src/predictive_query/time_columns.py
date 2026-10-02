# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Which timestamp columns may be inferred as a table's time column.

A time column says when a row became true, and a query counts or windows rows
by it. Two kinds of timestamp say nothing of the sort: one that holds a single
value in every row, and one that records when a row was last edited or
loaded. Either, taken as the time column, puts every row at a moment unrelated
to the event it records, and a window over it is empty or refused.

This module is the one rule both inference paths apply: the client's, which
reads a frame, and the relational engine's, which reads a source. It sits here
because both already import this distribution and neither imports the other.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

#: Name tokens that mark a column recording an edit or a load.
EDIT_STAMP_TOKENS = frozenset(
    {
        'last',
        'edit',
        'edited',
        'changed',
        'synced',
        'sync',
        'loaded',
        'load',
        'reload',
        'reloaded',
        'ingested',
        'batched',
        'extracted',
        'etl',
        'refresh',
        'refreshdate',
        'refreshed',
        'deleted',
        'deletion',
    }
)

# A name in one case with no separators, as Snowflake folds `lastLogin` into
# `LASTLOGIN`, has no token boundary, so these are matched as its prefix.
_JOINED_PREFIXES = (
    'last',
    'load',
    'sync',
    'etl',
    'refresh',
    'ingest',
    'extract',
)

# Endings that make one of those prefixes a different word (LASTING, LASTLY,
# SYNCHRONOUS, REFRESHMENT). A short list of English suffixes, rather than a
# dictionary, keeps every joined stamp refused and only these words kept.
_WORD_CONTINUATIONS = ('ing', 'ly', 'hron', 'ment')

#: Name tokens that mark a column recording when a row was created.
CREATION_TOKENS = frozenset({'create', 'created', 'creation'})

# An acronym run, a capitalised or lower word, or digits: `ETLLoadedAt` reads
# as ETL, Loaded, At, and `LASTMODIFIEDDATE` as one token.
_TOKEN = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+')


def name_tokens(name: str) -> tuple[str, ...]:
    r"""The lowercased words of a column name, split on snake_case and
    camelCase.
    """
    return tuple(token.lower() for token in _TOKEN.findall(name))


def is_edit_stamp(name: str) -> bool:
    r"""Whether ``name`` reads as when a row was last changed or loaded,
    rather than when it began.

    ``created_at`` and other ``*_at`` names are kept: only a name that says
    modification, update, or one of :data:`EDIT_STAMP_TOKENS` is refused.
    """
    lowered = name.lower()
    if 'modif' in lowered or 'updat' in lowered:
        return True
    if name.isalnum() and (name.isupper() or name.islower()):
        for prefix in _JOINED_PREFIXES:
            rest = lowered.removeprefix(prefix)
            if rest != lowered and not rest.startswith(_WORD_CONTINUATIONS):
                return True
    # By token, so `last_seen` is refused and `lastingDate` is kept.
    return not EDIT_STAMP_TOKENS.isdisjoint(name_tokens(name))


def is_creation_stamp(name: str) -> bool:
    r"""Whether ``name`` reads as when a row was created."""
    # The prefix keeps an all caps `CREATEDDATE`, which splits into one token.
    return name.lower().startswith('create') or not CREATION_TOKENS.isdisjoint(
        name_tokens(name)
    )


@dataclass(frozen=True)
class RejectedTimeColumn:
    r"""A timestamp column inference declined to take as the time column.

    Attributes:
        column: The column's name.
        reason: ``'constant'`` when every row holds the same value or none,
            ``'edit_stamp'`` when its name says when a row last changed.
        value: The one value every row holds, or ``None`` when no row holds
            one or the reason is the name.
    """

    column: str
    reason: Literal['constant', 'edit_stamp']
    value: object = None

    @property
    def why(self) -> str:
        r"""The reason, said as a clause about the column."""
        if self.reason == 'edit_stamp':
            return (
                'it looks like when a row was last changed, not when it began'
            )
        if self.value is None:
            return 'no row holds a value'
        return f'every row holds {_as_timestamp(self.value)}'


@dataclass(frozen=True)
class TimeColumnScreen:
    r"""What screening a table's timestamp columns left.

    Attributes:
        eligible: The columns that may be the time column, in the order they
            were offered. Narrowed to the creation stamps when there are any.
        rejected: The columns refused, each with its reason.
    """

    eligible: tuple[str, ...]
    rejected: tuple[RejectedTimeColumn, ...]

    @property
    def chosen(self) -> str | None:
        r"""The time column, when exactly one column is eligible."""
        return self.eligible[0] if len(self.eligible) == 1 else None


_VARIES = object()

#: Reads a column's MIN and MAX over every row of its table, or ``None`` when
#: it cannot.
FullRange = Callable[[str], tuple[object, object] | None]


def screen_time_columns(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    full_range: FullRange | None = None,
) -> TimeColumnScreen:
    r"""Refuse the candidates that cannot be a time column.

    Args:
        frame: The rows read, holding every candidate.
        candidates: The timestamp columns, in table order.
        full_range: ``None`` when ``frame`` holds every row. Otherwise asked
            for each column that looks constant in the rows read, so that a
            column constant only in its first rows is kept.

    Returns:
        The eligible columns and the rejected ones.
    """
    eligible: list[str] = []
    rejected: list[RejectedTimeColumn] = []
    for name in candidates:
        if is_edit_stamp(name):
            rejected.append(RejectedTimeColumn(name, 'edit_stamp'))
            continue
        held = _the_one_value(frame[name])
        if held is not _VARIES and full_range is not None:
            seen = full_range(name)
            # A table that cannot be asked keeps its column: only evidence
            # rejects one. The value read keeps the type the rows were read as.
            whole = _VARIES if seen is None else _the_one_value(pd.Series(seen))
            held = whole if whole is _VARIES or held is None else held
        if held is _VARIES:
            eligible.append(name)
        else:
            rejected.append(RejectedTimeColumn(name, 'constant', held))
    created = [name for name in eligible if is_creation_stamp(name)]
    return TimeColumnScreen(tuple(created or eligible), tuple(rejected))


def _as_timestamp(value: object) -> object:
    r"""``value`` as a timestamp when it spells one, so that a date a source
    keeps as text reads the way a frame's does.
    """
    if isinstance(value, (str, datetime.date)):
        try:
            return pd.Timestamp(value)
        except ValueError:
            pass
    return value


def _the_one_value(values: pd.Series) -> object:
    r"""The only value ``values`` holds, ``None`` when it holds none, or
    ``_VARIES``.
    """
    present = values.dropna()
    if len(present) == 0:
        return None
    first = present.iloc[0]
    return first if bool((present == first).all()) else _VARIES
