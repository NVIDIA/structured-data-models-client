# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Which timestamp column inference takes as a table's time column.

A time column says when a row became true, and a query counts or windows rows
by it. Some timestamps say nothing of the sort, and are never taken: one that
holds a single value in every row, or none, and one whose name says when a row
was loaded or synced, such as `_fivetran_synced`.
A stamp of the last edit, such as `ModifiedDate` or `updated_at`, is taken
only when it is a table's one timestamp left, since in an append-only log of
changes it is when the row began. Of several left, a `created` stamp is taken,
and otherwise the one whose values start earliest.

:func:`choose_time_column` is the one rule both inference paths apply: the
client's, which reads a frame, and the relational engine's, which reads a
source. It sits here because both already import this distribution and
neither imports the other.
"""

from __future__ import annotations

import datetime
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

#: Name tokens that mark a column recording when a row was loaded or synced.
LOAD_STAMP_TOKENS = frozenset(
    {
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
    }
)

#: Name tokens that mark a column recording when a row was last changed.
EDIT_STAMP_TOKENS = frozenset(
    {'last', 'edit', 'edited', 'changed', 'deleted', 'deletion'}
)

#: Name tokens that mark a column recording when a row was created.
CREATION_TOKENS = frozenset({'create', 'created', 'creation'})

# A name in one case with no separators, as Snowflake folds `lastLogin` into
# `LASTLOGIN`, has no token boundary, so these are matched as its prefix. Each
# keeps the endings that make it a different word: LASTING, LASTLY,
# SYNCHRONOUS, REFRESHMENT. A short list of endings, rather than a dictionary,
# keeps every joined stamp refused and only these words kept.
_JOINED_LOAD_PREFIXES: Mapping[str, tuple[str, ...]] = {
    'load': ('ing',),
    'sync': ('hron',),
    'etl': (),
    'refresh': ('ing', 'ment'),
    'ingest': (),
    'extract': (),
}
_JOINED_EDIT_PREFIXES: Mapping[str, tuple[str, ...]] = {
    'last': ('ing', 'ly'),
}

# An acronym run, a capitalised or lower word, or digits: `ETLLoadedAt` reads
# as ETL, Loaded, At, and `LASTMODIFIEDDATE` as one token.
_TOKEN = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+')


def name_tokens(name: str) -> tuple[str, ...]:
    r"""The lowercased words of a column name, split on snake_case and
    camelCase.
    """
    return tuple(token.lower() for token in _TOKEN.findall(name))


def _is_joined(name: str) -> bool:
    return name.isalnum() and (name.isupper() or name.islower())


def _reads_as(
    name: str,
    tokens: frozenset[str],
    prefixes: Mapping[str, tuple[str, ...]],
) -> bool:
    if _is_joined(name):
        lowered = name.lower()
        for prefix, other_words in prefixes.items():
            rest = lowered.removeprefix(prefix)
            if rest != lowered and not rest.startswith(other_words):
                return True
    # By token, so `last_seen` is matched and `lastingDate` is not.
    return not tokens.isdisjoint(name_tokens(name))


def is_load_stamp(name: str) -> bool:
    r"""Whether ``name`` reads as when a row was loaded, synced or
    refreshed, which says nothing of when it began.
    """
    return _reads_as(name, LOAD_STAMP_TOKENS, _JOINED_LOAD_PREFIXES)


def is_edit_stamp(name: str) -> bool:
    r"""Whether ``name`` reads as when a row was last changed, rather than
    when it began.

    ``created_at`` and other ``*_at`` names are not: only a name that says
    modification, update, or one of :data:`EDIT_STAMP_TOKENS` is.
    """
    lowered = name.lower()
    if 'modif' in lowered or 'updat' in lowered:
        return True
    return _reads_as(name, EDIT_STAMP_TOKENS, _JOINED_EDIT_PREFIXES)


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
        reason: Why it was refused. ``'constant'`` when every row holds
            the same value, or none.
            ``'load_stamp'`` when its name says when a row was loaded.
            ``'edit_stamp'`` when it says when a row was last changed.
        value: The one value every row holds, or ``None`` when no row holds
            one or the reason is the name.
    """

    column: str
    reason: Literal['constant', 'load_stamp', 'edit_stamp']
    value: object = None

    @property
    def why(self) -> str:
        r"""The reason, said as a clause about the column."""
        if self.reason == 'edit_stamp':
            return (
                'it looks like when a row was last changed, not when it began'
            )
        if self.reason == 'load_stamp':
            return 'it looks like when a row was loaded, not when it began'
        if self.value is None:
            return 'no row holds a value'
        return f'every row holds {_as_timestamp(self.value)}'


@dataclass(frozen=True)
class TimeColumnChoice:
    r"""The time column inference took, and how.

    Attributes:
        column: The time column, or ``None`` when no timestamp may be it.
        how: ``'only'`` when it was the one timestamp left, ``'created'``
            when it was the one creation stamp among several, ``'earliest'``
            when its values start earliest of :attr:`among`, and
            ``'edit_stamp'`` when it looks like a stamp of the last edit but
            no other timestamp was left. ``None`` when no column was taken.
        among: The columns it was chosen from, itself included.
        rejected: The columns refused, each with its reason, in table order.
    """

    column: str | None
    how: Literal['only', 'created', 'earliest', 'edit_stamp'] | None
    among: tuple[str, ...]
    rejected: tuple[RejectedTimeColumn, ...]


_VARIES = object()

#: Reads a column's MIN and MAX over every row of its table, or ``None`` when
#: it cannot.
FullRange = Callable[[str], tuple[object, object] | None]


def choose_time_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    full_range: FullRange | None = None,
) -> TimeColumnChoice:
    r"""Choose a table's time column among its timestamps.

    A column holding one value in every row, or none, is
    refused, as is one named as a load stamp.
    Of those left, a stamp of the last edit is taken only when it is the one
    column left. Otherwise a creation stamp is preferred, and of several the
    one whose values start earliest is taken, the first in table order on a
    tie.

    Args:
        frame: The rows read, holding every candidate.
        candidates: The timestamp columns, in table order.
        full_range: ``None`` when ``frame`` holds every row. Otherwise asked
            for each column that looks constant in the rows read, and for
            each column a choice by earliest value is made among, so that
            the answer is the one every row gives.

    Returns:
        The column taken, how, and the columns refused.
    """
    whole = _WholeTable(full_range)
    rejected: dict[str, RejectedTimeColumn] = {}
    plain: list[str] = []
    edited: list[str] = []
    for name in candidates:
        if is_load_stamp(name):
            rejected[name] = RejectedTimeColumn(name, 'load_stamp')
        elif (held := _constant(frame[name], name, whole)) is not _VARIES:
            rejected[name] = RejectedTimeColumn(name, 'constant', held)
        elif is_edit_stamp(name):
            rejected[name] = RejectedTimeColumn(name, 'edit_stamp')
            edited.append(name)
        else:
            plain.append(name)

    def chosen(
        column: str | None,
        how: Literal['only', 'created', 'earliest', 'edit_stamp'] | None,
        among: Sequence[str],
    ) -> TimeColumnChoice:
        return TimeColumnChoice(
            column,
            how,
            tuple(among),
            tuple(r for name, r in rejected.items() if name != column),
        )

    if not plain:
        if len(edited) == 1:
            return chosen(edited[0], 'edit_stamp', edited)
        return chosen(None, None, ())
    created = [name for name in plain if is_creation_stamp(name)]
    if len(created) == 1:
        return chosen(
            created[0], 'created' if len(plain) > 1 else 'only', plain
        )
    among = created or plain
    if len(among) == 1:
        return chosen(among[0], 'only', among)
    return chosen(_earliest(frame, among, whole), 'earliest', among)


class _WholeTable:
    r"""Asks the table for a column's range once."""

    def __init__(self, full_range: FullRange | None) -> None:
        self._full_range = full_range
        self._seen: dict[str, tuple[object, object] | None] = {}

    @property
    def can_ask(self) -> bool:
        return self._full_range is not None

    def __call__(self, name: str) -> tuple[object, object] | None:
        if self._full_range is None:
            return None
        if name not in self._seen:
            self._seen[name] = self._full_range(name)
        return self._seen[name]


def _constant(values: pd.Series, name: str, whole: _WholeTable) -> object:
    r"""The one value every row of a column holds, ``None`` when it holds
    none, or ``_VARIES``.
    """
    present = values.dropna()
    first = present.iloc[0] if len(present) else None
    if len(present) > 1 and not bool((present == first).all()):
        return _VARIES
    if whole.can_ask:
        seen = whole(name)
        if seen is None:
            # A table that cannot be asked keeps its column: only evidence
            # rejects one.
            return _VARIES
        low, high = seen
        if not bool(low == high):
            return _VARIES
        # The value read keeps the type the rows were read as.
        first = low if first is None else first
    return first


def _earliest(
    frame: pd.DataFrame, among: Sequence[str], whole: _WholeTable
) -> str:
    r"""The column of ``among`` whose values start earliest, the first in
    table order on a tie or when none can be read as a time.
    """
    starts: dict[str, pd.Timestamp] = {}
    for name in among:
        seen = whole(name)
        start = (
            _as_instant(seen[0])
            if seen is not None
            else _first_instant(frame[name])
        )
        if start is not None:
            starts[name] = start
    if not starts:
        return among[0]
    return min(starts, key=starts.__getitem__)


def _first_instant(values: pd.Series) -> pd.Timestamp | None:
    r"""The earliest of ``values`` as a UTC timestamp, or ``None``."""
    if isinstance(values.dtype, pd.ArrowDtype):
        values = pd.Series(values.to_numpy(), index=values.index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parsed = pd.to_datetime(values, errors='coerce', utc=True)
    except (TypeError, ValueError, OverflowError):
        return None
    return _as_instant(parsed.min())


def _as_instant(value: object) -> pd.Timestamp | None:
    r"""``value`` as a UTC timestamp, or ``None`` when it is not a time."""
    if value is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize('UTC')
    return stamp.tz_convert('UTC')


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
