# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Which timestamp column inference takes as a table's time column.

A time column says when a row became true, and a query counts or windows rows
by it. Some timestamps say nothing of the sort, and are never taken: one that
is empty in more than half the rows, or holds a single value in every row that
holds one, and one whose name says when a row was loaded or synced, such as
`_fivetran_synced`.
Nor is a date of birth.
A stamp of the last edit, such as `ModifiedDate` or `updated_at`, is never
taken on a table another table links to, an entity whose rows are edited in
place. On a table nothing links to it is taken only when it is the one
timestamp left, since in an append-only log of changes it is when the row
began. Of several left, a `created` stamp is taken, and otherwise the one
whose values start earliest, placeholders such as 1900-01-01 or 1970-01-01
left out.

Whether a column is empty or holds one value is read over every row of the
table. When the query that reads it fails, a column the rows read would
refuse is kept behind any they do not, and what they say of it is returned
for the caller to warn about: the first rows of a table kept in time order
often share one day, so they are no proof.

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
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

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
        'ingest',
        'ingested',
        'ingestion',
        'batched',
        'extract',
        'extracted',
        'extraction',
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

#: Name tokens that mark a date of birth.
BIRTH_DATE_TOKENS = frozenset({'birth', 'birthday', 'birthdate', 'dob', 'born'})

#: Name tokens that mark a column recording when a row was created.
CREATION_TOKENS = frozenset({'create', 'created', 'creation'})

# A name in one case with no separators, as Snowflake folds `lastLogin` into
# `LASTLOGIN`, has no token boundary, so these are matched as its prefix. Each
# keeps the endings that make it a different word: LASTING, EDITION,
# SYNCHRONOUS, and CHANGEDATE, which reads as change date. A short list of
# endings, rather than a dictionary, keeps every joined stamp refused and only
# these words kept.
_JOINED_LOAD_PREFIXES: Mapping[str, tuple[str, ...]] = {
    'load': ('ing',),
    'reload': ('ing',),
    'sync': ('hron',),
    'etl': (),
    'refresh': ('ing', 'ment'),
    'ingest': (),
    'extract': (),
    'batched': (),
}
_JOINED_EDIT_PREFIXES: Mapping[str, tuple[str, ...]] = {
    'last': ('ing', 'ly'),
    'edit': ('ion', 'or'),
    'changed': ('ate',),
    'deleted': ('ate',),
    'deletion': (),
}
_JOINED_BIRTH_PREFIXES: Mapping[str, tuple[str, ...]] = {'dob': (), 'born': ()}

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


def is_birth_date(name: str) -> bool:
    r"""Whether ``name`` reads as a date of birth, which says when someone
    was born and not when a row became true.
    """
    # A name folded into one word, such as `DATEOFBIRTH`, by its letters.
    if _is_joined(name) and 'birth' in name.lower():
        return True
    return _reads_as(name, BIRTH_DATE_TOKENS, _JOINED_BIRTH_PREFIXES)


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
        reason: Why it was refused. ``'constant'`` when no row holds a
            value, or the table has more than one row and every row that
            holds a value holds the same one. ``'sparse'`` when more than
            half the rows hold no value.
            ``'load_stamp'`` when its name says when a row was loaded.
            ``'edit_stamp'`` when it says when a row was last changed.
            ``'birth_date'`` when it says when someone was born.
        value: The one value every row holds, or ``None`` when no row holds
            one or the reason is not ``'constant'``.
        missing: How many rows hold no value, when the reason is the values.
        rows: How many rows the table has, when the reason is the values.
        whole: Whether ``missing`` and ``rows`` count every row of the
            table. ``False`` when they count only the rows read, because
            the query over every row failed.
    """

    column: str
    reason: Literal[
        'constant', 'sparse', 'load_stamp', 'edit_stamp', 'birth_date'
    ]
    value: object = None
    missing: int = field(default=0, compare=False)
    rows: int = field(default=0, compare=False)
    whole: bool = field(default=True, compare=False)

    @property
    def why(self) -> str:
        r"""The reason, said as a clause about the column."""
        if self.reason == 'edit_stamp':
            return (
                'it looks like when a row was last changed, not when it began'
            )
        if self.reason == 'load_stamp':
            return 'it looks like when a row was loaded, not when it began'
        if self.reason == 'birth_date':
            return 'it looks like a date of birth, not when a row began'
        if not self.whole:
            return self._why_in_rows_read
        if self.reason == 'sparse':
            return f'{self.missing:,} of its {self.rows:,} rows hold no value'
        if self.value is None:
            return 'no row holds a value'
        if self.missing:
            return (
                f'every row that holds a value holds '
                f'{_as_timestamp(self.value)}'
            )
        return f'every row holds {_as_timestamp(self.value)}'

    @property
    def _why_in_rows_read(self) -> str:
        if self.reason == 'sparse':
            return (
                f'{self.missing:,} of the {self.rows:,} rows read hold no value'
            )
        if self.value is None:
            return f'none of the {self.rows:,} rows read holds a value'
        if self.missing:
            return (
                f'every one of the {self.rows:,} rows read that holds a value '
                f'holds {_as_timestamp(self.value)}'
            )
        return (
            f'every one of the {self.rows:,} rows read holds '
            f'{_as_timestamp(self.value)}'
        )


@dataclass(frozen=True)
class TimeColumnChoice:
    r"""The time column inference took, and how.

    Attributes:
        column: The time column, or ``None`` when no timestamp may be it.
        how: ``'only'`` when it was the one timestamp left, ``'created'``
            when it was the one creation stamp among several, ``'earliest'``
            when its values start earliest of :attr:`among`, and
            ``'edit_stamp'`` when it looks like a stamp of the last edit but
            no other timestamp was left and no table links to this one.
            ``None`` when no column was taken.
        among: The columns it was chosen from, itself included.
        refused: Every column screened out, each with its reason, in table
            order, a lone edit stamp taken as the last resort included.
        fallback: The lone edit stamp left, when no other timestamp was.
        unconfirmed: Every column the rows read would refuse for its values
            but that was kept, because the query over every row failed,
            each with what the rows read say.
    """

    column: str | None
    how: Literal['only', 'created', 'earliest', 'edit_stamp'] | None
    among: tuple[str, ...]
    refused: tuple[RejectedTimeColumn, ...]
    fallback: str | None = None
    unconfirmed: tuple[RejectedTimeColumn, ...] = ()

    @property
    def rejected(self) -> tuple[RejectedTimeColumn, ...]:
        r"""The columns refused, each with its reason, in table order."""
        return tuple(r for r in self.refused if r.column != self.column)

    def for_links(self, linked: bool) -> TimeColumnChoice:
        r"""This choice for a table another table links to, or none does.

        A table something links to is an entity, one row per thing, edited
        in place, so a stamp of its last edit says when the row last changed
        and is refused. A table nothing links to is read as an append-only
        log, where the stamp says when the row was written, and it is taken
        when nothing else is left. The links are known only once every table
        is described, after the time column was chosen, so the choice is
        kept both ways.
        """
        if self.fallback is None:
            return self
        if linked:
            return replace(self, column=None, how=None, among=())
        return replace(
            self, column=self.fallback, how='edit_stamp', among=(self.fallback,)
        )


def taken_unchecked(table: str, doubt: RejectedTimeColumn, clear: str) -> str:
    r"""Say that ``doubt``, a column of :attr:`TimeColumnChoice.unconfirmed`,
    was taken as ``table``'s time column, and that ``clear`` undoes it.

    The client and the engine both say it, each with its own override.
    """
    return (
        f'{table}.{doubt.column} was taken as the time column although '
        f'{doubt.why}, because the query over the whole table failed and the '
        f'rows past them were not checked. If the whole table is like that, '
        f'clear it: {clear}.'
    )


class ColumnRange(NamedTuple):
    r"""A column's ``MIN``, ``MAX`` and ``COUNT`` over every row of its
    table.
    """

    low: object
    high: object
    #: How many rows hold a value.
    held: int


class TableRange(NamedTuple):
    r"""What one query over every row of a table read."""

    rows: int
    columns: Mapping[str, ColumnRange]


#: Reads the row count of a table and the range of each named column over
#: every row, in one query, or ``None`` when it cannot.
FullRange = Callable[[Sequence[str]], TableRange | None]


def choose_time_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    full_range: FullRange | None = None,
    linked: bool = False,
) -> TimeColumnChoice:
    r"""Choose a table's time column among its timestamps.

    A column empty in more than half the rows is refused, as is one holding
    one value in every row that holds one, unless the table has a single
    row, and one named as a load stamp.
    So is one named as a date of birth.
    Of those left, a stamp of the last edit is taken only when it is the one
    column left and no table links to this one. Otherwise a creation stamp
    is preferred, and of several the one whose values start earliest is
    taken, the first in table order on a tie.

    When the query over every row fails, a column the rows read would
    refuse for its values is kept, behind every column they do not but
    ahead of a lone stamp of the last edit, and named in
    :attr:`TimeColumnChoice.unconfirmed`.

    Args:
        frame: The rows read, holding every candidate.
        candidates: The timestamp columns, in table order.
        full_range: ``None`` when ``frame`` holds every row. Otherwise asked
            once, for every column not refused by its name, so that whether
            a column is constant and where it starts are what every row
            says.
        linked: Whether another table links to this one. Usually not known
            yet; see :meth:`TimeColumnChoice.for_links`.

    Returns:
        The column taken, how, and the columns refused.
    """
    whole = _WholeTable(
        full_range,
        [
            name
            for name in candidates
            if not is_load_stamp(name) and not is_birth_date(name)
        ],
    )
    rejected: dict[str, RejectedTimeColumn] = {}
    unconfirmed: list[RejectedTimeColumn] = []
    plain: list[str] = []
    doubted: list[str] = []
    edited: list[str] = []
    for name in candidates:
        if is_load_stamp(name):
            rejected[name] = RejectedTimeColumn(name, 'load_stamp')
            continue
        if is_birth_date(name):
            rejected[name] = RejectedTimeColumn(name, 'birth_date')
            continue
        refused = _refused_for_values(name, frame, whole)
        if refused is not None and refused.whole:
            rejected[name] = refused
            continue
        if refused is not None:
            unconfirmed.append(refused)
        if is_edit_stamp(name):
            rejected[name] = RejectedTimeColumn(name, 'edit_stamp')
            edited.append(name)
        elif refused is not None:
            doubted.append(name)
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
            tuple(rejected.values()),
            unconfirmed=tuple(unconfirmed),
        )

    # A column the rows read cast doubt on is taken only when no other is.
    plain = plain or doubted
    if not plain:
        if len(edited) == 1:
            last_resort = replace(chosen(None, None, ()), fallback=edited[0])
            return last_resort.for_links(linked)
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
    r"""What one query over every row of the table said, if it was asked."""

    def __init__(
        self, full_range: FullRange | None, names: Sequence[str]
    ) -> None:
        self.can_ask = full_range is not None
        self.read = (
            full_range(names) if full_range is not None and names else None
        )

    def __call__(self, name: str) -> ColumnRange | None:
        return None if self.read is None else self.read.columns.get(name)


def _refused_for_values(
    name: str, frame: pd.DataFrame, whole: _WholeTable
) -> RejectedTimeColumn | None:
    r"""Refuse a column that is mostly empty or holds one value, by what
    every row of the table holds.

    A single value is no evidence that a column never changes, so a one-row
    table keeps its timestamp. When the table was asked and did not answer,
    what the rows read say is returned with ``whole`` false, for the caller
    to keep the column and say so.
    """
    present = frame[name].dropna()
    rows, held = len(frame), len(present)
    first = present.iloc[0] if held else None
    same = held > 0 and bool((present == first).all())
    seen = whole(name)
    if seen is not None and whole.read is not None:
        rows, held = whole.read.rows, seen.held
        same = held > 0 and bool(seen.low == seen.high)
        # The value read keeps the type the rows were read as.
        first = seen.low if first is None else first
    every_row = seen is not None or not whole.can_ask
    missing = rows - held
    if held == 0:
        return RejectedTimeColumn(
            name, 'constant', None, missing, rows, every_row
        )
    if 2 * missing > rows:
        return RejectedTimeColumn(
            name, 'sparse', None, missing, rows, every_row
        )
    if rows > 1 and same:
        return RejectedTimeColumn(
            name, 'constant', first, missing, rows, every_row
        )
    return None


# Values that stand for no date: anything before 1901, such as 0001-01-01 or
# 1900-01-01, and the Unix epoch at midnight, which a zero or a missing value
# becomes on its way into a timestamp.
_FIRST_REAL_DAY = pd.Timestamp('1901-01-01', tz='UTC')
_EPOCH = pd.Timestamp('1970-01-01', tz='UTC')


def _is_placeholder(stamp: pd.Timestamp) -> bool:
    return stamp < _FIRST_REAL_DAY or stamp == _EPOCH


def _earliest(
    frame: pd.DataFrame, among: Sequence[str], whole: _WholeTable
) -> str:
    r"""The column of ``among`` whose values start earliest, the first in
    table order on a tie or when none can be read as a time.

    A placeholder is no start: a `cancelled_at` filled with 1970-01-01 does
    not begin before the orders it cancels. Where a column starts is read
    over the whole table when it was asked; when its whole-table minimum is
    a placeholder, the earliest real value in the rows read is used, since
    one query cannot skip the placeholders in every SQL dialect.
    """
    starts: dict[str, pd.Timestamp] = {}
    for name in among:
        seen = whole(name)
        start = None if seen is None else _as_instant(seen.low)
        if start is None or _is_placeholder(start):
            start = _first_instant(frame[name])
        if start is not None:
            starts[name] = start
    if not starts:
        return among[0]
    return min(starts, key=starts.__getitem__)


def _first_instant(values: pd.Series) -> pd.Timestamp | None:
    r"""The earliest of ``values`` that is not a placeholder, as a UTC
    timestamp, or ``None``.
    """
    if isinstance(values.dtype, pd.ArrowDtype):
        values = pd.Series(values.to_numpy(), index=values.index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parsed = pd.to_datetime(values, errors='coerce', utc=True)
    except (TypeError, ValueError, OverflowError):
        return None
    real = parsed[(parsed >= _FIRST_REAL_DAY) & (parsed != _EPOCH)]
    return _as_instant(real.min())


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
