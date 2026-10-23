# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The tables a request is about, and what the model should make of them.

A :class:`Schema` holds your tables, what each column means, and how the tables
relate. One table or many is the same object. A single-table model is the
degenerate case of the same shape, not a second system, which is why this is
not called a graph.

Everything is inferred. You override only where inference is wrong:

>>> schema = Schema.from_data({'users': users_df})
>>> schema['users'].primary_key = 'user_id'
>>> schema['users']['churn'].stype = 'categorical'

**Inference reads values, so a code stored as a number reads as a number.** A
postcode, a store number or a status code is ``numerical`` here, and the model
will treat its distances as meaningful. No amount of inference settles that.
Only you know the column is a label. Correct it:

>>> schema['users']['zip_code'].stype = 'categorical'

Relationship inference is deliberately conservative in the same spirit: it
links a column only to a primary key of exactly the same name, so
``orders.user_id`` finds ``users.user_id`` but not ``users.id``. A wrong link
changes what the model is shown and is harder to notice than a missing one, so
pass ``relationships=`` when the names do not line up.

Nothing here contacts a model. Inference reads your frames locally.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import pandas as pd
from predictive_query.time_columns import (
    TimeColumnChoice,
    choose_time_column,
    taken_unchecked,
)

from sdm_client.core import dates
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.wire import (
    UNTYPED,
    infer_tfm_dtype,
    require_scalar_cells,
)

if TYPE_CHECKING:
    from sdm_connectors import Source

# Semantic types this client can declare. These are the contract's, and match
# ``predictive_query.Stype`` one for one.
SEMANTIC_TYPES = frozenset(
    {
        'ID',
        'categorical',
        'multicategorical',
        'numerical',
        'sequence',
        'text',
        'timestamp',
    }
)

# Above this many distinct values, a string column is prose rather than a set
# of labels. The exact number matters less than having one: a model encodes
# the two differently, and 'text' on a 3-value column wastes the signal.
_MAX_CATEGORICAL_CARDINALITY = 64

# A column is only a candidate identifier if nearly every value is distinct.
# Exactly 1.0 would reject a key with a single duplicated row, which is common
# in an extract and still the column the user means.
_KEY_UNIQUENESS = 0.99

# And below this, the repeats are the ordinary many-to-one shape rather than a
# fault in a key: a fact table's foreign key repeats on the order of its row
# count, while a key with a bad extract behind it repeats a handful of times.
# Between the two is where a column was meant to be the key and missed.
_NEARLY_A_KEY = 0.9

_KEY_SUFFIXES = ('_id', '_key', 'id', 'key')

_T = TypeVar('_T', bound='_TableShape')


@dataclass
class Column:
    r"""Describe one column and how the model should treat it.

    Attributes:
        name: The column name, as it appears in the frame.
        dtype: The wire dtype, e.g. ``'int64'`` or ``'timestamp[us]'``.
            Inferred from the values; correcting it is rarely necessary.
        stype: What the column *means*: ``'numerical'``, ``'categorical'``,
            ``'ID'``, ``'text'``, ``'timestamp'``. This is the one worth
            correcting: it decides how the model encodes the column.
        stype_declared: Whether the meaning was assigned rather than inferred.
            An all-null column is typed ``categorical`` because a schema has to
            say something, and that placeholder is not a claim about the
            values; a meaning the caller assigned is. Recorded so the two can
            be told apart, since only the second is worth refusing a
            contradiction against.
        untyped: Whether the values carried no type at all, as in an all-null
            object column. ``dtype`` is ``'string'`` for those, because a
            schema has to name something and a string round-trips whatever
            arrives, but the frame being scored may hold real values and
            decide the dtype instead. Recorded so the adapter can read it
            here rather than re-deriving it from the frame, which is how the
            schema and the request came to disagree.
    """

    name: str
    dtype: str
    stype: str
    untyped: bool = field(init=False, default=False)
    stype_declared: bool = field(init=False, default=False)
    _stype_before_key: str | None = field(init=False, default=None, repr=False)
    #: What the source said of the column, where the core described it
    #: (``sdm_client._rust.results.SourceFacts``, T3.16): kept with the
    #: column, copied and pickled with it, and handed back whole to the
    #: core, which reads a column with none from the source. Out of repr
    #: and equality, so a column is what it says, however it was made.
    _source: Any = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._check_stype(self.stype)

    def _mark_as_key(self) -> None:
        r"""Type this column as an identifier because the schema joins on it.

        A key typed from its dtype alone becomes ``numerical``, the model
        stores it as a float, and the join against an integer key fails
        downstream. That is true of a column the caller called ``numerical``
        too: what a key means is settled by its being a key, not by a
        preference, so a declared meaning is overridden here rather than
        honoured. Like an inferred meaning, it is recorded so it returns if
        the column stops being a key.

        This goes around ``__setattr__``: routed through it the schema's own
        marking set ``stype_declared``, becoming indistinguishable from a
        meaning the caller chose, and what had been decided was gone with
        nothing to restore.
        """
        if self.stype == 'ID':
            return
        object.__setattr__(self, '_stype_before_key', self.stype)
        object.__setattr__(self, 'stype', 'ID')

    def _release_as_key(self) -> None:
        r"""Restore a column's meaning after the schema stops joining on it.

        Through the same adoption an assignment goes through: a column that
        means `timestamp` orders the table it belongs to, and restoring the
        meaning around that left a table saying it held a timestamp while
        naming no time column. `check_consistent()` accepted that state, and
        every temporal query then failed on.
        """
        if self._stype_before_key is None:
            return
        restored = self._stype_before_key
        object.__setattr__(self, 'stype', restored)
        object.__setattr__(self, '_stype_before_key', None)
        if restored == 'timestamp':
            self._adopt_as_time_column()

    def __setattr__(self, key: str, value: Any) -> None:
        r"""Assign a field, keeping the three that describe each other true.

        `untyped` is derived rather than declared. It records that inference
        found no type, and the adapter reads it as authority over `dtype`, so
        letting it be set is letting a column claim a dtype and hold none.
        Naming the dtype is the information it was missing, which is why doing
        so clears it and the adapter stops substituting.

        `stype_declared` separates a meaning that was chosen from one that was
        inferred, and a second assignment is what distinguishes them.
        """
        if key == 'untyped':
            raise StructuredDataError(
                "'untyped' is inferred from the values and cannot be set; "
                'assign the dtype you want instead',
                code='INVALID_REQUEST',
            )
        if key == 'stype' and _described_on_the_core(
            getattr(getattr(self, '_table', None), '_schema', None)
        ):
            # A core-built schema's column: the core checks and makes the
            # edit, and its answer is stored (T1.24b, decision E66(g)).
            from sdm_client._rust import edits

            if edits.column(self, value):
                return
        if key == 'stype':
            self._check_stype(value)
            self._check_supports_dtype(value, getattr(self, 'dtype', None))
            if hasattr(self, 'stype'):
                object.__setattr__(self, 'stype_declared', True)
        if key == 'dtype' and getattr(self, 'untyped', False):
            object.__setattr__(self, 'untyped', False)
        object.__setattr__(self, key, value)
        if key == 'stype' and value == 'timestamp':
            self._adopt_as_time_column()
        if key == 'stype':
            self._stay_a_key_if_the_schema_joins_on_this(value)

    def _stay_a_key_if_the_schema_joins_on_this(self, declared: str) -> None:
        r"""Put ``ID`` back if this column is one the schema joins on.

        A key's meaning is settled by its being a key, and the schema says so
        wherever it is asked. The answer was only re-derived when the
        keys changed, so between declaring a meaning on a live key and the
        next such moment the column read as whatever was declared. Nothing was
        sent that way, because a request re-derives as it takes its copy; the
        object simply described itself wrongly in the meantime.

        The declaration replaces what there was to give back, rather than
        being discarded beside it: declaring ``ID`` on a column that is
        already a key left the meaning inference had found still recorded, and
        moving the key elsewhere handed that back over the caller's own word.
        """
        if declared == 'ID':
            object.__setattr__(self, '_stype_before_key', None)
            return
        table = getattr(self, '_table', None)
        schema = getattr(table, '_schema', None) if table is not None else None
        if schema is None:
            if table is not None and self.name in _as_key_tuple(
                getattr(table, 'primary_key', None)
            ):
                self._mark_as_key()
            return
        schema._mark_key_columns()

    def _adopt_as_time_column(self) -> None:
        r"""Let a column named as a timestamp order the table it belongs to.

        `Table._check_time_column` already sets a column's stype when the
        column is named as the time column. Only that direction was wired, so
        a caller correcting an inferred schema the way the SDK's own error
        messages teach, ``schema[table][column].stype = 'timestamp'``, got
        a table reporting `stype='timestamp'` and `time_column=None` at the
        same time, and every temporal query on it was refused for want of a
        time column the schema said was there.

        Only an absent time column is filled: naming a second timestamp does
        not displace the first. Nothing is adopted when inference already
        chose, so this resolves the contradiction rather than overriding a
        decision.

        A column that is also part of the key is left alone rather than
        refused: the two meanings contradict, and raising from an stype
        assignment would report it from the wrong place. `check_consistent`
        is where that is said.

        The time column is set directly because the checked route assigns this
        column's stype, which would re-enter here before the attribute it
        guards on exists.
        """
        table = getattr(self, '_table', None)
        if table is None or getattr(table, 'time_column', None) is not None:
            return
        if self.name in _as_key_tuple(getattr(table, 'primary_key', None)):
            return
        object.__setattr__(table, 'time_column', self.name)

    def check_consistent(self) -> None:
        r"""Refuse a column whose storage and meaning contradict each other.

        Assigning a meaning is checked against the storage as it happens, but
        the reverse is not, because a caller re-declaring storage often has to
        set the two one at a time and either order passes through a pair that
        does not hold. The pair is asked about here instead, once the caller
        has finished and before anything is sent.
        """
        if self.dtype is None or self.stype is None:
            return
        self._check_supports_dtype(self.stype, self.dtype)

    @staticmethod
    def _check_stype(value: Any) -> None:
        if value not in SEMANTIC_TYPES:
            raise StructuredDataError(
                f'{value!r} is not a semantic type; use one of '
                f'{sorted(SEMANTIC_TYPES)}',
                code='INVALID_REQUEST',
            )

    def _check_supports_dtype(self, stype: str, dtype: str | None) -> None:
        r"""Refuse a meaning the values cannot carry.

        The language already decides this, and used to decide it at
        ``materialize()``, several cells after the assignment that caused it,
        as a bare ``ValueError``. Asking the same question here changes only
        when the answer arrives. Nothing is refused that was going to be
        accepted.
        """
        if dtype is None:
            return
        if not supports_dtype(stype, dtype):
            raise StructuredDataError(
                f'column {self.name!r} holds {dtype!r}, which cannot mean '
                f'{stype!r}',
                code='INVALID_REQUEST',
            )

    def __repr__(self) -> str:
        untyped = ', untyped=True' if self.untyped else ''
        return (
            f'Column({self.name!r}, dtype={self.dtype!r}, '
            f'stype={self.stype!r}{untyped})'
        )


class _TableShape:
    r"""What a table says about itself, whether or not it holds the rows.

    Keys, the time column and what each column means are declared the same
    way on a table of rows and on one that stays in its source; only the
    checks that read rows differ, and those ask for the rows rather than
    assume them.
    """

    name: str
    columns: dict[str, Column]
    primary_key: str | Sequence[str] | None
    time_column: str | None
    duplicate_keys: int | None

    def _own_columns(self) -> None:
        r"""Point every column at this table, and take a copy of any that is
        already pointing somewhere else.

        The back-reference is what lets a column make itself the time column
        when it is told it is a timestamp. One `Column` handed to two tables
        could only point at one of them, so setting the stype through the
        first would have ordered the second. A column belongs to a table;
        handing the same one to two is asking for two columns.

        A borrowed column is copied into a new dict, so that a caller holding
        the one they passed does not find entries in it they did not put
        there. The back-reference itself is set directly: it is not a field,
        and must stay out of equality and `repr`.
        """
        # By identity. A `Table` is a dataclass, so `in (None, self)` asks
        # `==`, which compares the frames inside and raises on any table that
        # holds one.
        borrowed = [
            name
            for name, column in self.columns.items()
            if (owner := getattr(column, '_table', None)) is not None
            and owner is not self
        ]
        if borrowed:
            object.__setattr__(self, 'columns', dict(self.columns))
            for name in borrowed:
                self.columns[name] = _copy_column(self.columns[name])
        for column in self.columns.values():
            object.__setattr__(column, '_table', self)

    def __eq__(self, other: object) -> bool:
        r"""Whether two tables describe the same thing.

        Written out because the generated one compares the frames with `==`,
        which for a DataFrame is a frame of booleans. Asking whether two
        tables are equal raised `ValueError: The truth value of a DataFrame is
        ambiguous` from inside pandas, naming nothing the caller wrote. The
        rows are compared as rows, and everything else by value, which is what
        the question means.
        """
        if not isinstance(other, _TableShape) or type(other) is not type(self):
            return NotImplemented
        described = (
            self.name == other.name
            and self.columns == other.columns
            and _as_key_tuple(self.primary_key)
            == _as_key_tuple(other.primary_key)
            and self.time_column == other.time_column
            and getattr(self, 'source_name', None)
            == getattr(other, 'source_name', None)
        )
        if not described:
            return False
        mine = getattr(self, 'data', None)
        theirs = getattr(other, 'data', None)
        if mine is None or theirs is None:
            return mine is theirs
        return bool(mine.equals(theirs))

    __hash__ = None  # type: ignore[assignment]

    def __copy__(self: _T) -> _T:
        r"""A shallow copy is still a table of its own columns.

        Sharing them left the copy unable to answer for itself. Its columns
        pointed at the table they came from, so changing the copy's key typed
        nothing. It could also retype the original once the copy was put in a
        schema.
        """
        return _copy_table(self)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in ('primary_key', 'time_column') and _described_on_the_core(
            getattr(self, '_schema', None)
        ):
            # A core-built schema's table: the core checks and makes the
            # edit, and its answer is stored (T1.24b, decision E66(g)).
            from sdm_client._rust import edits

            if edits.table(self, key, value):
                return
        if key in ('primary_key', 'time_column') and 'columns' in self.__dict__:
            # Before anything is marked, not after. Naming a key or a time
            # column types the columns it names, and ownership was taken in
            # `__post_init__` (which runs after the fields are assigned), so
            # a table built on another's columns typed *those* first and left
            # the table it borrowed them from changed.
            self._own_columns()
        if key == 'primary_key':
            # Read once, here, and what was read is what is kept. A key is
            # consulted many times afterwards: to check it, to type its
            # columns, and to copy the table. An iterable that empties on
            # the first read arrived as "was given an empty primary key"
            # about a key that named two columns. Reading it before anything
            # is assigned also stops a key this cannot read from being taken
            # and leaving a table that can be neither copied nor sent.
            names = _as_key_tuple(value)
            if value is not None and not isinstance(value, str):
                value = names
            if value is None:
                object.__setattr__(self, 'duplicate_keys', 0)
                object.__setattr__(self, key, None)
                self._re_derive_key_columns()
                return
            self._check_key(value)
            object.__setattr__(self, key, value)
            self._adopt_key(value)
            self._re_derive_key_columns()
            return
        if key == 'time_column' and value is not None:
            self._check_time_column(value)
        object.__setattr__(self, key, value)

    def _re_derive_key_columns(self) -> None:
        r"""Ask this table's schema which of its columns are still keys.

        A key that has just been replaced is no longer a key, and nothing said
        so: the column it was kept the ``ID`` the old key gave it. Only the
        schema knows whether some relationship still joins on that column, so
        the question goes there; a table held on its own answers it for its own
        key alone.
        """
        schema = getattr(self, '_schema', None)
        if schema is not None:
            schema._mark_key_columns()
            return
        named = set(_as_key_tuple(getattr(self, 'primary_key', None)))
        for name, column in getattr(self, 'columns', {}).items():
            if name not in named:
                column._release_as_key()

    def _adopt_key(self, value: str | Sequence[str]) -> None:
        r"""Make the rest of the table agree with the key it was just given.

        A key identifies a row, so its columns mean ``ID``. The engine's own
        setter does the same thing. Any imperfection in the uniqueness
        check tolerated is recorded and said out loud. Inference does all three
        after nominating a key; assignment has to as well, or the two routes
        describe the same table differently.

        A repeat is the one visible sign that the column identifies something
        coarser than a row, most often a parent on a table whose real key is
        that column together with another. Naming that possibility is what
        makes the warning actionable; saying only that the rows are
        indistinguishable describes the symptom. A column that cannot be a key
        at all has already been refused by `_check_key`.

        Which is why only a column `_check_key` would take is offered. The
        search filtered nothing, and the column that distinguishes two rows of
        the same entity is very often the time column, as in the ordinary "two
        snapshots of one account" shape. The composite key the warning
        recommended was the one the next line refuses. Where the only
        disambiguating column is one a key cannot hold, that is said instead
        of recommending it.
        """
        names = _as_key_tuple(value)
        for name in names:
            self.columns[name]._mark_as_key()
        frame = getattr(self, 'data', None)
        if frame is None:
            # A table in its source is counted by `Schema.from_source`, and a
            # key named afterwards has not been.
            object.__setattr__(self, 'duplicate_keys', None)
            return
        if frame.empty:
            return
        quality = _key_quality(frame, names)
        object.__setattr__(self, 'duplicate_keys', quality.duplicates)
        if quality.duplicates:
            distinguishing = [
                other
                for other in self.columns
                if other not in names
                and _key_quality(frame, (*names, other)).duplicates == 0
            ]
            usable = [
                other
                for other in distinguishing
                if self._could_hold_a_key(other)
            ]
            if usable:
                remedy = (
                    f' {".".join(names)!r} is unique together with '
                    f'{", ".join(repr(other) for other in usable)}, which is '
                    f'what a composite key looks like; pass that as the '
                    f'primary key if so.'
                )
            elif distinguishing:
                remedy = (
                    f' They differ only by '
                    f'{_and_list(distinguishing)}, which a primary key '
                    f'cannot be made of, so there is no composite key to name '
                    f'here: keep one row per {".".join(names)!r} if the model '
                    f'should see one, or name a different primary key.'
                )
            else:
                remedy = _NAME_ANOTHER_KEY
            _warn_a_key_repeats(
                self.name, names, quality.duplicates, quality.example, remedy
            )

    def _could_hold_a_key(self, name: str) -> bool:
        r"""Whether `_check_key` would take this column as part of a key.

        The conditions it refuses on that a column carries on its own: a dtype
        no identifier can be made of, being the table's time column, and
        holding a null. Asked here so that what is recommended is what is
        accepted. A column with one null was offered as half of a composite
        key and refused by `_check_key` on the very next call.

        Uniqueness is not asked here. It is a property of the combination, and
        the caller of this is choosing what to combine.
        """
        column = self.columns.get(name)
        if column is None or name == getattr(self, 'time_column', None):
            return False
        if not supports_dtype('ID', column.dtype):
            return False
        frame = getattr(self, 'data', None)
        if frame is None or name not in frame.columns:
            return True
        return not bool(frame[name].isna().any())

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise StructuredDataError(
                f'The name {name!r} must be a non-empty string.',
                code='INVALID_REQUEST',
            )
        name_invalid_chars = _invalid_name_chars(name)
        if len(name_invalid_chars) > 0:
            raise StructuredDataError(
                f'The name {name!r} contains '
                f'one or more invalid characters: {name_invalid_chars}.',
                code='INVALID_REQUEST',
            )

    def _check_column(self, name: str) -> None:
        if name not in getattr(self, 'columns', {}):
            raise StructuredDataError(
                f'table {self.name!r} has no column {name!r}; it has '
                f'{sorted(getattr(self, "columns", {}))}',
                code='INVALID_REQUEST',
            )

    def _check_key(self, value: str | Sequence[str]) -> None:
        r"""A key has to exist, and has to identify a row.

        Inference deliberately tolerates a nearly-unique column, because a
        single duplicated row is common in an extract and still the column the
        caller means. An explicit key is an assertion rather than a guess, so
        it is held to the same standard the inference applies. A caller
        who states something false should hear so.

        A null is refused however few there are. It is not a near miss on
        uniqueness: the engine drops those rows rather than scoring them, and
        a single one passed silently because `duplicated` does not flag one
        null.
        """
        names = (value,) if isinstance(value, str) else tuple(value)
        if not names:
            raise StructuredDataError(
                f'table {self.name!r} was given an empty primary key',
                code='INVALID_REQUEST',
            )
        for name in names:
            self._check_column(name)
            column = self.columns[name]
            if not supports_dtype('ID', column.dtype):
                raise StructuredDataError(
                    f'column {name!r} holds {column.dtype!r}, which cannot '
                    f'identify a row, so it cannot be a primary key',
                    code='INVALID_REQUEST',
                )
            if name == getattr(self, 'time_column', None):
                raise StructuredDataError(
                    f'column {name!r} is already the time column of table '
                    f'{self.name!r}, so it cannot also be its primary key',
                    code='INVALID_REQUEST',
                )
        frame = getattr(self, 'data', None)
        if frame is None or frame.empty:
            return
        quality = _key_quality(frame, names)
        if quality.nulls:
            empty = [
                name
                for name in names
                if name in frame.columns and frame[name].isna().any()
            ]
            raise StructuredDataError(
                _a_key_missing_values(self.name, names, quality.nulls, empty),
                code='INVALID_REQUEST',
            )
        if _too_many_repeats(quality.duplicates, len(frame)):
            raise StructuredDataError(
                _not_unique_enough(
                    self.name,
                    names,
                    quality.duplicates,
                    len(frame),
                    quality.example,
                ),
                code='INVALID_REQUEST',
            )

    def _check_time_column(self, name: str) -> None:
        self._check_column(name)
        if name in _as_key_tuple(getattr(self, 'primary_key', None)):
            raise StructuredDataError(
                f'column {name!r} is already the primary key of table '
                f'{self.name!r}, so it cannot also be its time column',
                code='INVALID_REQUEST',
            )
        if name in getattr(self, 'columns', {}):
            self.columns[name].stype = 'timestamp'

    def __getitem__(self, column: str) -> Column:
        try:
            return self.columns[column]
        except KeyError:
            raise StructuredDataError(
                f'table {self.name!r} has no column {column!r}; it has '
                f'{sorted(self.columns)}',
                code='INVALID_REQUEST',
            ) from None

    def __contains__(self, column: object) -> bool:
        return column in self.columns

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)


# `eq=False` on both, so the comparison above is the one they use: a
# generated one would compare the frames with `==`.
@dataclass(eq=False)
class Table(_TableShape):
    r"""One table: its rows, its columns, and which column identifies a row.

    Attributes:
        name: The name the schema knows this table by. A query names columns
            through it, so it is part of the interface rather than a label.
        data: The rows.
        columns: Column metadata, keyed by column name.
        primary_key: The column identifying a row, or ``None``. A single-table
            request needs none; a relational one needs one per table, and it is
            inferred, so you rarely set it.
        time_column: The column ordering rows in time, or ``None``. Only a
            temporal question needs one.
    """

    name: str
    data: pd.DataFrame
    columns: dict[str, Column]
    primary_key: str | Sequence[str] | None = None
    time_column: str | None = None
    #: How many rows repeat a primary key value another row already used.
    #: Inference tolerates a few, deliberately; this is what it tolerated.
    #: Not an init argument: the generated ``__init__`` assigns fields in
    #: order, so a settable one would overwrite what assigning the key just
    #: computed, and every explicitly-built table would report zero.
    duplicate_keys: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        r"""Check the names and the cells, and take ownership of the columns.

        `Schema.from_data` checks the cells while inferring, but a hand-built
        `Table` reached the adapters unchecked, where a list cell surfaced
        much later as `TypeError: unhashable type: 'list'` raised from label
        collection, naming nothing the caller had written.
        """
        self._check_name(self.name)
        frame = getattr(self, 'data', None)
        if frame is not None:
            for column in frame.columns:
                self._check_name(column)
                require_scalar_cells(frame[column], f'{self.name}.{column}')
        self._own_columns()

    def __repr__(self) -> str:
        key = f', primary_key={self.primary_key!r}' if self.primary_key else ''
        time = f', time_column={self.time_column!r}' if self.time_column else ''
        repeats = (
            f', {self.duplicate_keys} repeated keys'
            if self.duplicate_keys
            else ''
        )
        return (
            f'Table({self.name!r}, {len(self.data)} rows, '
            f'{len(self.columns)} columns{key}{time}{repeats})'
        )


@dataclass(eq=False)
class SourceTable(_TableShape):
    r"""One table that stays in its source: its columns, and its keys.

    Built by :meth:`Schema.from_source`. It holds no rows: sampling reads them
    from the source, a hop at a time, when a query is materialized. Keys, time
    columns and semantic types are corrected exactly as on a :class:`Table`.

    Attributes:
        name: The name the schema knows this table by. A query names columns
            through it.
        source_name: The table's name in the source.
        columns: Column metadata, keyed by column name.
        primary_key: The column identifying a row, or ``None``.
        time_column: The column ordering rows in time, or ``None``.
    """

    name: str
    source_name: str
    columns: dict[str, Column]
    primary_key: str | Sequence[str] | None = None
    time_column: str | None = None
    #: How many rows repeat a primary key value another row already used, as
    #: :meth:`Schema.from_source` counted them in the source. Zero for a key
    #: the source enforces, and for a table with no key. ``None`` where
    #: nothing was counted: ``check_keys=False``, a semantic model's table,
    #: or a key named after the schema was built.
    duplicate_keys: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._check_name(self.name)
        if not isinstance(self.source_name, str) or not self.source_name:
            raise StructuredDataError(
                f'table {self.name!r} needs the name it has in its source, '
                f'got {self.source_name!r}',
                code='INVALID_REQUEST',
            )
        for column in self.columns:
            self._check_name(column)
        self._own_columns()

    def __delitem__(self, column: str) -> None:
        r"""Leave a column out: it is neither read from the source nor sent.

        >>> del schema['orders']['rowguid']

        A table in its source cannot be edited to suit a request, and some
        columns only mislead a model: a per-row UUID identifies a row and
        says nothing about it. A table of frames drops a column from its frame
        before :meth:`Schema.from_data` instead.
        """
        if _described_on_the_core(getattr(self, '_schema', None)):
            # A core-built schema's table: the core checks and makes the
            # edit, and its answer is stored (T1.24b).
            from sdm_client._rust import edits

            if edits.leave_out(self, column):
                return
        self._check_column(column)
        role = (
            'primary key'
            if column in _as_key_tuple(self.primary_key)
            else 'time column'
            if column == self.time_column
            else None
        )
        if role is not None:
            raise StructuredDataError(
                f"{self.name}.{column} is the table's {role}; set that to "
                f'something else, or None, before leaving the column out',
                code='INVALID_REQUEST',
            )
        del self.columns[column]

    def __repr__(self) -> str:
        key = f', primary_key={self.primary_key!r}' if self.primary_key else ''
        time = f', time_column={self.time_column!r}' if self.time_column else ''
        return (
            f'SourceTable({self.name!r}, from {self.source_name!r}, '
            f'{len(self.columns)} columns{key}{time})'
        )


@dataclass(frozen=True)
class Relationship:
    r"""A foreign key: ``source_table.source_column`` points at a table's key.

    Directed from the many side to the one side, which is the direction a join
    reads and the direction the wire envelope carries.
    """

    source_table: str
    source_column: str
    target_table: str
    target_column: str

    def __repr__(self) -> str:
        return (
            f'{self.source_table}.{self.source_column} -> '
            f'{self.target_table}.{self.target_column}'
        )


class Schema:
    r"""The tables a request is about.

    Build one with :meth:`from_data`. Index it by table name, and index a table
    by column name:

    >>> schema = Schema.from_data({'users': users, 'orders': orders})
    >>> schema['orders']['price'].stype
    'numerical'

    One table and many tables are the same object. A single-table model simply
    rejects a schema that has relationships, and says so before anything is
    sent.
    """

    def check_consistent(self) -> None:
        r"""Refuse any column whose storage and meaning contradict, and any
        relationship that cannot join.

        Asked once, as the schema becomes a request, because a caller
        re-declaring a column sets its storage and its meaning one at a time
        and either order passes through a pair that does not hold.

        The relationships are asked about here as well as on assignment.
        `relationships` is an ordinary list, so `schema.relationships.append(
        link)` reaches it without passing the setter, and this is the question
        a caller runs to find out whether their schema holds together. It
        should not be the one place that does not look.

        Which columns are keys is re-derived first. `tables` and a table's
        `columns` are ordinary dictionaries, so a table or a column replaced
        through one arrives never having been asked, and a relationship key
        left typed ``numerical`` would otherwise pass this and be sent as a
        float.
        """
        self._mark_key_columns()
        for table in self.tables.values():
            for column in table.columns.values():
                column.check_consistent()
        self._check_relationships(self.relationships)

    def __init__(
        self,
        tables: Mapping[str, Table | SourceTable],
        relationships: Sequence[Relationship] = (),
        *,
        source: Source | None = None,
    ) -> None:
        _require_one_kind_of_table(tables, source)
        self._source = source
        # Copied, because a schema types the key columns of its own tables
        # and two schemas handed the same ones fought over them: building an
        # unlinked schema retyped a linked schema's join key, and checking the
        # linked one retyped the unlinked one's back.
        self._tables = {
            name: _copy_table(table) for name, table in tables.items()
        }
        self.relationships = relationships
        # The engine's own reading of the source's tables, kept so that each
        # question does not read them again. Set by the first read.
        self._engine_tables: Any = None
        self._conversion_notes: tuple[str, ...] = ()

    @property
    def source(self) -> Source | None:
        r"""Where the tables are, for a schema built with :meth:`from_source`.

        ``None`` for a schema of frames. A schema has one source or none,
        because sampling runs against one: see :meth:`from_source`.
        """
        return self._source

    @property
    def conversion_notes(self) -> tuple[str, ...]:
        r"""How :meth:`from_source` read a semantic model where nothing was
        lost, for a person to read.

        For a Databricks metric view, one note each: the predictive query
        aggregation that matches a simple measure, such as
        ``'revenue' is SUM(order_lines.LineTotal, 0, 30, days)``, the time
        column taken in place of a date the view truncates, unless
        ``time_columns`` named another, and a parameter read as its default.
        Tables are named as the schema names them. What a model loses is
        said in an :class:`SDMWarning` instead, and a time column taken in
        place of a truncated date is said in the warning on how time columns
        were inferred as well, with the override that names another. Empty
        for every other schema.

        The wording of a note is not part of the contract, so do not parse
        it.
        """
        return self._conversion_notes

    def __setattr__(self, key: str, value: Any) -> None:
        r"""Hold an assigned relationship to the same standard as a passed one.

        `Schema.from_data(relationships=[link])` checks that both columns exist
        and that their dtypes can join; `schema.relationships = [link]` is the
        same statement and checked nothing, so a link that one route refuses
        the other accepted. `check_consistent()`, which a caller runs to
        ask exactly this, passed it.
        """
        if (
            key == 'relationships'
            # An edit of a schema that stands: building one is no edit.
            and 'relationships' in vars(self)
            and _described_on_the_core(self)
        ):
            # A core-built schema: the core checks and makes the edit, and
            # its answer is stored (T1.24b, decision E66(g)).
            from sdm_client._rust import edits

            if edits.relationships(self, value):
                return
        if key == 'relationships' and hasattr(self, '_tables'):
            value = _Links(self._settle(value), self)
            object.__setattr__(self, key, value)
            self._mark_key_columns()
            return
        object.__setattr__(self, key, value)

    def __copy__(self) -> Schema:
        r"""A shallow copy is still a schema of its own tables.

        `copy.copy` shared the tables and their columns, so editing the copy
        edited the original. This was always a trap, and became a
        destructive one once key columns were re-derived: clearing the copy's
        relationships retyped a column of a schema that still had them.

        The rows are still shared, as they are in `snapshot()`: they are the
        data, and copying a warehouse extract is not what a caller asking for
        a copy of the schema is asking for.
        """
        return self.snapshot()

    def __setstate__(self, state: dict[str, Any]) -> None:
        r"""Restore a copied or unpickled schema with its links still checked.

        Pickle and `deepcopy` put the attributes back without going through
        `__setattr__`, so the copy would hold a plain list and quietly stop
        checking what is added to it. Said nothing while it does: a copy is
        not a new declaration, and the notices were given when the schema it
        was copied from was built.

        A schema pickled before it could be bound to a source has neither
        that source nor the engine's reading of it, which is what it is: a
        schema of frames.
        """
        object.__setattr__(
            self,
            '__dict__',
            {
                '_source': None,
                '_engine_tables': None,
                '_conversion_notes': (),
                # The core's description of the source an earlier client
                # kept beside the schema (to T3.16): its columns now carry
                # what the source said of them, and a column with none is
                # read from the source again.
                **{
                    key: value
                    for key, value in state.items()
                    if key != '_core_description'
                },
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            self.relationships = list(state.get('relationships') or ())

    def _settle(self, links: Sequence[Relationship]) -> list[Relationship]:
        r"""The relationships as this schema will hold them, having said so.

        One place, so that what is checked, de-duplicated and warned about does
        not depend on how the link was added.
        """
        given = list(links)
        self._check_relationships(given)
        settled = _each_link_once(given)
        _warn_two_ways_to_reach_one_table(self._tables, settled)
        return settled

    def _check_relationships(self, links: Sequence[Relationship]) -> None:
        for link in links:
            if not isinstance(link, Relationship):
                raise StructuredDataError(
                    f'a relationship must be a Relationship, got '
                    f'{type(link).__name__}: {link!r}. Build one with '
                    f'Relationship(source_table, source_column, '
                    f'target_table, target_column)',
                    code='INVALID_REQUEST',
                )
            _require_column(self._tables, link.source_table, link.source_column)
            _require_column(self._tables, link.target_table, link.target_column)
            _require_joinable(self._tables, link)

    def snapshot(self) -> Schema:
        r"""This schema as it stands, so a later edit cannot reach back.

        `materialize()` is the gathering step and what it returns is read as
        finished, but the request kept the caller's schema by reference and the
        adapter reads it again at send time. Editing a column's meaning
        after materializing changed what was sent, silently, and an agent
        reusing one schema across several questions had no way to see it.

        The rows are shared rather than copied: they are the data, they are
        what the request is about, and duplicating a warehouse extract per
        question to guard against an edit nobody makes to a frame would cost
        more than the defect. What is copied is everything a schema *says*.
        """
        copied = Schema(
            {name: _copy_table(table) for name, table in self._tables.items()},
            list(self.relationships),
            source=self._source,
        )
        # What the engine read of the source is shared, not copied: it is what
        # the tables are, and every gather is given its own copy of it.
        copied._engine_tables = self._engine_tables
        copied._conversion_notes = self._conversion_notes
        return copied

    @classmethod
    def from_data(
        cls,
        tables: Mapping[str, pd.DataFrame],
        *,
        primary_keys: Mapping[str, str | Sequence[str]] | None = None,
        time_columns: Mapping[str, str | None] | None = None,
        relationships: Sequence[Relationship] | None = None,
    ) -> Schema:
        r"""Infer a schema from frames.

        Args:
            tables: Frames by table name. The name matters: a predictive query
                names columns through it.
            primary_keys: Overrides for the inferred key of a table. A
                sequence of column names declares a composite key, which the
                setter has always taken; the annotation said otherwise and so
                pushed anyone with one onto the assignment route, which until
                recently validated nothing.
            time_columns: Overrides for the inferred time column of a table.
                ``None`` says the table has none.
            relationships: Overrides for the inferred relationships. Pass an
                empty sequence to assert there are none.

        Returns:
            A schema with every dtype, semantic type, key, time column and
            relationship filled in.

        Raises:
            StructuredDataError: If ``tables`` is empty, a value is not a
                DataFrame, a name is not a string, or an override names
                something that does not exist.

        An inferred relationship is held to the same contract as an explicit
        one, and dropped rather than refused when it fails. Inference matches
        on name alone, so it can pair columns whose dtypes cannot join, and
        `query.py` would send that edge unchanged; a caller who wrote the
        relationship is told it cannot join, while a caller who never asked
        for it is not refused for a guess this made on their behalf.
        """
        if not tables:
            raise StructuredDataError(
                'a schema needs at least one table', code='INVALID_REQUEST'
            )

        built: dict[str, Table] = {}
        chose: dict[str, TimeColumnChoice | None] = {}
        for name, frame in tables.items():
            if not isinstance(name, str) or not name:
                raise StructuredDataError(
                    f'table names must be non-empty strings; got {name!r}',
                    code='INVALID_REQUEST',
                )
            if not isinstance(frame, pd.DataFrame):
                raise StructuredDataError(
                    f'table {name!r} must be a pandas DataFrame, got '
                    f'{type(frame).__name__}',
                    code='INVALID_REQUEST',
                )
            built[name], chose[name] = _infer_table_and_time_column(name, frame)

        _apply_keys_and_times(built, primary_keys, time_columns)
        if relationships is None:
            inferred = _joinable_inferred_links(
                built, _infer_relationships(built)
            )
        else:
            inferred = _checked_links(built, relationships)

        _settle_edit_stamps(built, chose, time_columns, inferred)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            schema = cls(built, inferred)
        schema._mark_key_columns()
        dates = {
            name: named
            for name, table in built.items()
            if (named := _warn_dates_stored_as_text(table)) is not None
        }
        if relationships is None:
            spoken_for = _warn_repeats_cost_the_table_its_key(built)
            _warn_table_cannot_be_linked_to(built, dates, spoken_for)
            _warn_one_key_name_keys_several_tables(built)
        _warn_stype_was_decided_by_the_rows_read(built)
        _warn_how_time_columns_were_inferred(schema, chose, time_columns)
        _warn_two_ways_to_reach_one_table(built, schema.relationships)
        return schema

    @classmethod
    def from_source(
        cls,
        source: Source,
        tables: Mapping[str, str] | None = None,
        *,
        primary_keys: Mapping[str, str | Sequence[str]] | None = None,
        time_columns: Mapping[str, str | None] | None = None,
        relationships: Sequence[Relationship] | None = None,
        check_keys: bool = True,
    ) -> Schema:
        r"""Describe tables that stay in their source.

        >>> source = DatabricksSource(spark, catalog='main', schema='sales')
        >>> schema = Schema.from_source(
        ...     source,
        ...     {'customers': 'adventureworks_customers',
        ...      'orders': 'adventureworks_orders'},
        ...     time_columns={'orders': 'order_date'},
        ... )

        Nothing is read into memory. Column types come from the source's
        catalog and semantic types from a preview of at most 1,000 rows. Keys
        and relationships come from the constraints the catalog declares, and
        where it declares none, from names, by the same rule as
        :meth:`from_data`. When a query is materialized, the engine samples
        the tables with SQL, a hop at a time, and only the rows it samples
        leave the source.

        Each key is then checked against every row, by the rules
        :meth:`from_data` applies to a frame. Databricks and Snowflake do not
        enforce a declared key, and a key inferred from a preview has only
        been seen to hold there, so a repeated value would otherwise surface
        as rows that change between seeds and context examples that
        contradict each other. A declared key, or one named in
        ``primary_keys``, warns when up to 1% of rows repeat a value and is
        refused beyond that; an inferred key is declined beyond it. The count
        is one statement for all the tables, a full scan of each, and it is
        skipped for a key the source enforces, such as a PostgreSQL or DuckDB
        primary key. :attr:`SourceTable.duplicate_keys` holds what it found.

        A semantic model, such as a Databricks metric view or a Snowflake
        semantic view, is a source of its own, and already says how its
        tables join:

        >>> schema = Schema.from_source(source.semantic_model('sales_metrics'))

        Its keys and relationships come from the model, and its tables are
        named as the model names them. No relationship is inferred from
        column names beside the model's own joins, which would give a table
        a second path the model never declared. ``relationships=`` replaces
        the model's own, so to add one, pass them with it:
        ``relationships=[*Schema.from_source(model).relationships, extra]``.
        What the model has that a schema cannot hold is said in one
        :class:`SDMWarning`, and how it was read where nothing was lost, such
        as the aggregation each measure is, is in :attr:`conversion_notes`.

        Everything that differs between warehouses belongs to the source, so
        this reads the same whichever one the tables are in. A schema has
        exactly one source: sampling sends the ids each hop found to the next
        query, so it runs against one engine.

        Args:
            source: Where the tables are: an ``sdm_connectors.Source``, or a
                ``SemanticModel`` got from one.
            tables: For each name a query will use, the table's name in the
                source. Left out, every table the source describes is taken,
                under its own name: every table in the namespace, or every
                table of the semantic model.
            primary_keys: Overrides for a table's key.
            time_columns: Overrides for a table's time column. ``None``
                says the table has none.
            relationships: Replaces the relationships, a semantic model's
                own included. Pass an empty sequence to assert there are
                none.
            check_keys: Count each key's repeats and missing values in the
                source. ``False`` skips the scan, and the checks with it, and
                leaves :attr:`SourceTable.duplicate_keys` ``None``. The tables
                of a semantic model are never counted: the model derives them.

        Returns:
            A schema of :class:`SourceTable`, bound to ``source``.

        Raises:
            StructuredDataError: If ``tables`` is empty, a name is not a
                string, a table is not in the source, the source describes no
                table, an override names something that does not exist, or a
                declared or named key repeats a value in more than 1% of rows
                or a named key has no value in some row.
            MissingExtraError: If the relational engine, which reads the
                source, is not installed.
            GraphConstructionError: The relational engine's, if the
                warehouse fails while a semantic model is read, with the
                driver's error as its ``__cause__``.
            EmptyTableError: The relational engine's, a ``RuntimeError``, if
                a semantic model's source table holds no rows.
        """
        from sdm_client.materializers.sources import (
            EngineTables,
            engine_graph,
            is_semantic_model,
            semantic_model_graph,
        )

        _require_a_source(source)
        if tables is not None:
            _require_table_names(tables)

        from sdm_client import _routing

        if _routing.on_the_core(
            source,
            _routing.FROM_SOURCE
            if tables is None
            else _routing.FROM_SOURCE_TABLES,
        ):
            from sdm_client._rust import schema as on_the_core

            return on_the_core.from_source(
                cls,
                source,
                tables,
                primary_keys=primary_keys,
                time_columns=time_columns,
                relationships=relationships,
                check_keys=check_keys,
            )

        read = EngineTables()
        engine_tables: dict[str, Any] | None = None
        chose: dict[str, TimeColumnChoice | None] = {}
        notes: tuple[str, ...] = ()
        if is_semantic_model(source):
            graph = semantic_model_graph(source)
            read.keep({name: graph[name] for name in graph.tables})
            offered = list(graph.tables)
            chosen = (
                dict(tables)
                if tables is not None
                else {name: name for name in offered}
            )
            absent = sorted(set(chosen.values()) - set(offered))
            if absent:
                raise StructuredDataError(
                    f'{absent} are not tables of {source!r}; it has '
                    f'{sorted(offered)}',
                    code='INVALID_REQUEST',
                )
            as_named = {model: name for name, model in chosen.items()}
            built = {
                name: _describe_source_table(graph[model], model, name=name)
                for name, model in chosen.items()
            }
            chose = {
                name: graph[model].time_column_choice
                for name, model in chosen.items()
            }
            declared = _model_links(graph, built, as_named)
        else:
            graph = engine_graph(source, tables, infer_metadata=True)
            chosen = (
                dict(tables)
                if tables is not None
                else {name: name for name in graph.tables}
            )
            read.keep({named: graph[name] for name, named in chosen.items()})
            built = {
                name: _describe_source_table(graph[name], named)
                for name, named in chosen.items()
            }
            chose = {name: graph[name].time_column_choice for name in chosen}
            declared = _declared_links(graph, chosen, built)
            if relationships is None:
                _key_by_the_referenced_column(built, declared)
            engine_tables = {name: graph[name] for name in chosen}

        found = {
            name: _as_key_tuple(table.primary_key)
            for name, table in built.items()
        }
        _apply_keys_and_times(built, primary_keys, time_columns)
        if is_semantic_model(source):
            notes = _schema_notes(graph.conversion_notes, built, as_named)
        if check_keys and engine_tables is not None:
            # Before relationships are inferred: a declined key is one no
            # relationship can point at.
            _check_source_keys(
                source,
                built,
                engine_tables,
                found,
                set(primary_keys or ()),
                infer_links=relationships is None,
            )
        else:
            for table in built.values():
                object.__setattr__(table, 'duplicate_keys', None)
        if relationships is None:
            covered = {
                (link.source_table, link.source_column) for link in declared
            }
            # A semantic model's joins are all the links it means.
            by_name = (
                []
                if is_semantic_model(source)
                else [
                    link
                    for link in _infer_relationships(built)
                    if (link.source_table, link.source_column) not in covered
                ]
            )
            links = _joinable_inferred_links(built, declared + by_name)
        else:
            links = _checked_links(built, relationships)

        _settle_edit_stamps(built, chose, time_columns, links)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            schema = cls(built, links, source=source)
        schema._engine_tables = read
        schema._conversion_notes = notes
        schema._mark_key_columns()
        _warn_how_time_columns_were_inferred(schema, chose, time_columns)
        _warn_two_ways_to_reach_one_table(built, schema.relationships)
        return schema

    @property
    def tables(self) -> dict[str, Table | SourceTable]:
        r"""The tables, by name."""
        return self._tables

    def __getitem__(self, table: str) -> Table | SourceTable:
        try:
            return self._tables[table]
        except KeyError:
            raise StructuredDataError(
                f'no table {table!r} in this schema; it has '
                f'{sorted(self._tables)}',
                code='INVALID_REQUEST',
            ) from None

    def __contains__(self, table: object) -> bool:
        return table in self._tables

    def __iter__(self) -> Iterator[str]:
        return iter(self._tables)

    def __len__(self) -> int:
        return len(self._tables)

    def _own_tables(self) -> None:
        r"""Take a copy of any table that still belongs to another schema.

        `tables` hands back the live dictionary, so a table can arrive from a
        schema that is still using it. Typing its key columns for this
        schema would retype them for that one. The same rule a table applies
        to the columns it is handed.
        """
        borrowed = [
            name
            for name, table in self._tables.items()
            if (owner := getattr(table, '_schema', None)) is not None
            and owner is not self
        ]
        for name in borrowed:
            self._tables[name] = _copy_table(self._tables[name])
        for table in self._tables.values():
            object.__setattr__(table, '_schema', self)
            # And the columns inside them, for the same reason: a column
            # inserted through `table.columns` arrives still belonging to the
            # table it came from, and typing it here typed it there.
            table._own_columns()

    def _mark_key_columns(self) -> None:
        r"""Mark columns this schema joins on or no longer joins on.

        Derived from the schema as it stands rather than applied once when it
        was built. Marking only on the way in meant a relationship dropped
        afterwards left its source column typed ``ID``, so two schemas with
        the same tables and no relationships asked different questions: one
        a regression and the other a multiclass classification, according to a
        link only one of them had ever had.
        """
        self._own_tables()
        keys: dict[str, set[str]] = {}
        for name, table in self._tables.items():
            keys[name] = {
                column
                for column in _as_key_tuple(table.primary_key)
                if column in table.columns
            }
        for link in self.relationships:
            source = self._tables.get(link.source_table)
            if source is not None and link.source_column in source.columns:
                keys[link.source_table].add(link.source_column)

        for name, table in self._tables.items():
            for column_name, column in table.columns.items():
                # A column replaced through the ordinary `columns` dictionary
                # arrives with no idea which table it is in, and could not be
                # asked about again.
                object.__setattr__(column, '_table', table)
                if column_name in keys[name]:
                    column._mark_as_key()
                else:
                    column._release_as_key()

    def __repr__(self) -> str:
        return (
            f'Schema({len(self._tables)} tables: {sorted(self._tables)}, '
            f'{len(self.relationships)} relationships)'
        )


def _require_table(tables: Mapping[str, _T], name: str) -> _T:
    if name not in tables:
        raise StructuredDataError(
            f'no table {name!r} in this schema; it has {sorted(tables)}',
            code='INVALID_REQUEST',
        )
    return tables[name]


def _schema_notes(
    notes: Sequence[Any],
    built: Mapping[str, _TableShape],
    as_named: Mapping[str, str],
) -> tuple[str, ...]:
    r"""A semantic model's conversion notes as text, naming its tables as
    the schema names them.

    A note about a table the schema leaves out, or about a time column its
    table no longer has, says nothing the schema holds, so it is left out.

    Args:
        notes: The engine's notes on how it read the model.
        built: The schema's tables, by name.
        as_named: The schema's name for each of the model's tables it holds.
    """
    from sdm_relational_engine.rfm.view_conversion import (
        MeasureNote,
        TimeColumnNote,
    )

    said: list[str] = []
    for note in notes:
        if isinstance(note, MeasureNote | TimeColumnNote):
            name = as_named.get(note.table)
            if name is None or (
                isinstance(note, TimeColumnNote)
                and built[name].time_column != note.column
            ):
                continue
            note = replace(note, table=name)
        said.append(str(note))
    return tuple(said)


def _apply_keys_and_times(
    built: Mapping[str, _TableShape],
    primary_keys: Mapping[str, str | Sequence[str]] | None,
    time_columns: Mapping[str, str | None] | None,
) -> None:
    for name, key in (primary_keys or {}).items():
        table = _require_table(built, name)
        # Read once, and the names are what is passed on. Reading it here
        # to check the columns exist and then handing the original along
        # emptied a key that can only be read once, exactly as the setter
        # used to before it read once itself.
        names = _as_key_tuple(key)
        for column in names:
            _require_column(built, name, column)
        table.primary_key = key if isinstance(key, str) else names
    for name, named in (time_columns or {}).items():
        _require_table(built, name).time_column = (
            None if named is None else _require_column(built, name, named)
        )


def _joinable_inferred_links(
    built: Mapping[str, _TableShape], inferred: Sequence[Relationship]
) -> list[Relationship]:
    r"""The inferred links that can join, having said which could not.

    An inferred relationship is held to the same contract as an explicit one,
    and dropped rather than refused when it fails: a caller who never asked
    for it is not refused for a guess made on their behalf.
    """
    unjoinable = [link for link in inferred if not _is_joinable(built, link)]
    _warn_of_a_link_that_cannot_join(built, unjoinable)
    return [link for link in inferred if _is_joinable(built, link)]


def _key_by_the_referenced_column(
    built: Mapping[str, SourceTable], declared: Sequence[Relationship]
) -> None:
    r"""Key a keyless table by the column declared foreign keys refer to.

    A catalog lets a foreign key refer only to a primary key or a unique
    column, so that column identifies a row, as a lone unique column does when
    the engine infers a key. Before the keys are checked, so this one is too.
    Foreign keys naming different columns of one table leave it keyless.
    """
    referenced: dict[str, set[str]] = {}
    for link in declared:
        referenced.setdefault(link.target_table, set()).add(link.target_column)
    for name, columns in referenced.items():
        if built[name].primary_key is None and len(columns) == 1:
            built[name].primary_key = next(iter(columns))


def _checked_links(
    built: Mapping[str, _TableShape], relationships: Sequence[Relationship]
) -> list[Relationship]:
    links = list(relationships)
    for link in links:
        if not isinstance(link, Relationship):
            raise StructuredDataError(
                f'a relationship must be a Relationship, got '
                f'{type(link).__name__}: {link!r}. Build one with '
                f'Relationship(source_table, source_column, '
                f'target_table, target_column)',
                code='INVALID_REQUEST',
            )
        _require_column(built, link.source_table, link.source_column)
        _require_column(built, link.target_table, link.target_column)
        _require_joinable(built, link)
    return links


#: How a message names each source's engine.
_BACKEND_NAMES = {
    'databricks': 'Databricks',
    'snowflake': 'Snowflake',
    'postgres': 'PostgreSQL',
    'duckdb': 'DuckDB',
    'sqlite': 'SQLite',
}

_SKIP_THE_CHECK = (
    'Name another key with primary_keys=, or pass check_keys=False to skip '
    '(one full scan per table).'
)


def _check_source_keys(
    source: Any,
    built: Mapping[str, SourceTable],
    engine_tables: Mapping[str, Any],
    found: Mapping[str, tuple[str, ...]],
    stated: AbstractSet[str],
    *,
    infer_links: bool,
) -> None:
    r"""Hold the keys of tables in a source to the rules a key of frames
    meets, counting the rows where they are.

    A stated key, from ``primary_keys=`` or the catalog, warns at up to 1% of
    rows repeating a value and is refused above that. A key the engine
    inferred is kept up to the same 1% and declined beyond it, as inference
    over frames declines it. A key the source enforces is not counted.

    Args:
        source: The source the tables are in.
        built: The tables, keyed as they will be.
        engine_tables: The engine's reading of each table, by schema name.
        found: The key each table had before ``primary_keys=`` was applied.
        stated: The tables whose key ``primary_keys=`` named.
        infer_links: Whether relationships will be inferred, which is when a
            declined key is worth a word.
    """
    from sdm_relational_engine.rfm.base import count_keys

    for table in built.values():
        object.__setattr__(table, 'duplicate_keys', 0)
    keyed = {
        name: _as_key_tuple(table.primary_key)
        for name, table in built.items()
        if table.primary_key is not None
    }
    if not keyed:
        return
    counts = dict(
        zip(
            keyed,
            count_keys([(engine_tables[name], keyed[name]) for name in keyed]),
            strict=True,
        )
    )
    backend = _BACKEND_NAMES.get(source.backend, source.backend)
    declined: dict[str, Any] = {}
    for name, names in keyed.items():
        count = counts[name]
        rows, distinct = count.rows, count.distinct
        if rows is None or distinct is None:
            continue
        table = built[name]
        declared = (
            name not in stated
            and names == found[name]
            and engine_tables[name]._declares_key(names)
        )
        repeats = count.repeats
        if name not in stated and not declared:
            # Inferred, so declined as inference over a frame declines it.
            if distinct < rows * _KEY_UNIQUENESS:
                declined[name] = count
                table.primary_key = None
                continue
        elif count.nulls and name in stated:
            empty = names if len(names) == 1 else None
            missing = _a_key_missing_values(name, names, count.nulls, empty)
            raise StructuredDataError(
                f'{missing}. {_SKIP_THE_CHECK}', code='INVALID_REQUEST'
            )
        elif _too_many_repeats(repeats, rows):
            found_out = _not_unique_enough(
                name, names, repeats, rows, count.example
            )
            unenforced = (
                f'{backend} does not enforce declared keys. '
                if declared and not count.enforced
                else ''
            )
            raise StructuredDataError(
                f'{found_out}. {unenforced}{_SKIP_THE_CHECK}',
                code='INVALID_REQUEST',
            )
        if count.nulls:
            _warn_a_key_has_no_value(name, '.'.join(names), count.nulls)
        if repeats:
            _warn_a_key_repeats(
                name, names, repeats, count.example, _NAME_ANOTHER_KEY
            )
        object.__setattr__(table, 'duplicate_keys', repeats)

    if not infer_links or not declined:
        return
    named_elsewhere = _tables_naming_each_column(built)
    keyed_by = {
        table.primary_key
        for table in built.values()
        if isinstance(table.primary_key, str)
    }
    for name, count in declined.items():
        column = keyed[name][0]
        rows, distinct = count.rows or 0, count.distinct or 0
        if (
            len(keyed[name]) > 1
            or column in keyed_by
            or not named_elsewhere.get(column, set()) - {name}
            or distinct < rows * _NEARLY_A_KEY
            or not count.repeats
        ):
            continue
        _warn_a_key_was_declined(
            name,
            column,
            count.repeats,
            rows,
            count.example,
            f'Keep one row per {column!r} in the source, or name a column '
            f'that does identify a row with primary_keys={{{name!r}: ...}}.',
        )


def _require_table_names(tables: Mapping[str, str]) -> None:
    if not tables:
        raise StructuredDataError(
            'a schema needs at least one table; leave tables out to take every '
            'table the source describes',
            code='INVALID_REQUEST',
        )
    for name, source_name in tables.items():
        if not isinstance(name, str) or not name:
            raise StructuredDataError(
                f'table names must be non-empty strings; got {name!r}',
                code='INVALID_REQUEST',
            )
        if not isinstance(source_name, str) or not source_name:
            raise StructuredDataError(
                f'table {name!r} must map to its name in the source, a '
                f'non-empty string; got {source_name!r}',
                code='INVALID_REQUEST',
            )


def _require_a_source(source: object) -> None:
    try:
        from sdm_connectors import Source
    except ImportError:  # pragma: no cover - nothing could have built one
        Source = None  # type: ignore[assignment,misc]
    if Source is None or not isinstance(source, Source):
        raise StructuredDataError(
            f'a schema over tables in place needs an sdm_connectors.Source, '
            f'got {type(source).__name__}. Build one from a connection: '
            f'DatabricksSource(databricks.sql.connect(...), catalog=..., '
            f'schema=...), or DatabricksSource(spark, ...) in a notebook',
            code='INVALID_REQUEST',
        )


def _require_one_kind_of_table(
    tables: Mapping[str, Table | SourceTable], source: object
) -> None:
    r"""Refuse a schema that mixes frames with tables in place, or tables in
    place without the source they are in.

    The engine samples one graph with one sampler: an in-memory one over
    frames, or SQL against one source. A mixture has no sampler to run it.
    """
    in_place = sorted(
        name for name, table in tables.items() if isinstance(table, SourceTable)
    )
    in_memory = sorted(
        name
        for name, table in tables.items()
        if not isinstance(table, SourceTable)
    )
    if in_place and in_memory:
        raise StructuredDataError(
            f'{in_place} stay in their source and {in_memory} are frames; a '
            f'schema samples one or the other, so read the first into frames '
            f'or put the second in the source',
            code='INVALID_REQUEST',
        )
    if in_place and source is None:
        raise StructuredDataError(
            f'{in_place} stay in their source, so the schema needs that '
            f'source; build it with Schema.from_source',
            code='INVALID_REQUEST',
        )
    if in_memory and source is not None:
        raise StructuredDataError(
            f'{in_memory} are frames, which a source does not hold; drop '
            f'source= or build the schema with Schema.from_source',
            code='INVALID_REQUEST',
        )
    if source is not None:
        _require_a_source(source)


# What a name in a predictive query cannot hold: `table.column` splits at the
# dot, and a backtick or line break ends the name.
_INVALID_NAME_CHARS = ('`', '\n', '\r', '.')


def _invalid_name_chars(name: str) -> list[str]:
    return [char for char in _INVALID_NAME_CHARS if char in name]


def names_a_query_cannot_say(schema: Schema) -> bool:
    r"""Whether a table of ``schema`` holds a column whose name the old
    validator's language cannot say: one holding a dot, a backtick or a
    line break. Only the core route keeps such a column, and only the core
    resolves a query naming it (KO2-36, T2.18).
    """
    return any(
        _invalid_name_chars(name)
        for table in schema.tables.values()
        for name in table.columns
    )


def _describe_source_table(
    engine_table: Any, source_name: str, *, name: str | None = None
) -> SourceTable:
    r"""A table in place, described from what the engine read of it.

    A column whose values have no scalar form on the wire, such as a list,
    binary or a duration, is left out and named, rather than refusing the
    whole table:
    the caller cannot change a warehouse table's columns to suit a request, and
    the engine then leaves it unread as well. So is a column whose name a
    predictive query cannot say, such as a metric view's dimension `a.b`.
    """
    from sdm_client.materializers.sources import wire_dtype

    columns: dict[str, Column] = {}
    unreadable: list[str] = []
    unnamable: list[str] = []
    for engine_column in engine_table.columns:
        if _invalid_name_chars(engine_column.name):
            unnamable.append(repr(engine_column.name))
            continue
        dtype = wire_dtype(engine_column.dtype)
        stype = str(getattr(engine_column.stype, 'value', engine_column.stype))
        if dtype is None or stype not in SEMANTIC_TYPES:
            unreadable.append(f'{engine_column.name} ({engine_column.dtype})')
            continue
        columns[engine_column.name] = Column(engine_column.name, dtype, stype)
    if unreadable:
        warnings.warn(
            f'{name or engine_table.name} leaves out {", ".join(unreadable)}: a '
            f'request carries one scalar value per cell, and these hold none',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    if unnamable:
        warnings.warn(
            f'{name or engine_table.name} leaves out {", ".join(unnamable)}: '
            f'a predictive query cannot name a column holding a dot, a '
            f'backtick or a line break',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    table = SourceTable(
        name=name or engine_table.name,
        source_name=source_name,
        columns=columns,
    )
    keys = tuple(engine_table.primary_key_columns or ())
    if keys and all(key in columns for key in keys):
        table.primary_key = keys[0] if len(keys) == 1 else keys
    time = engine_table.time_column
    if time is not None and time.name in columns and time.name not in keys:
        table.time_column = time.name
    return table


def _model_links(
    graph: Any, built: Mapping[str, SourceTable], as_named: Mapping[str, str]
) -> list[Relationship]:
    r"""The relationships a semantic model's joins make, as the schema says
    them.

    Each join links the many side's key to the one side's, which the model
    makes that table's primary key. A join to a table the schema did not take
    links nothing.
    """
    links = []
    for edge in graph.edges:
        source_name = as_named.get(edge.src_table)
        target_name = as_named.get(edge.dst_table)
        if source_name is None or target_name is None:
            continue
        target = built[target_name]
        source = built[source_name]
        key = target.primary_key
        if not isinstance(key, str) or edge.fkey not in source.columns:
            continue
        links.append(
            Relationship(
                source_table=source_name,
                source_column=edge.fkey,
                target_table=target_name,
                target_column=key,
            )
        )
    return links


def _declared_links(
    graph: Any, tables: Mapping[str, str], built: Mapping[str, SourceTable]
) -> list[Relationship]:
    r"""The foreign keys the source's catalog declares between these tables.

    A constraint naming a table outside the schema links nothing here, and
    one naming a column the schema left out is skipped with it.
    """
    # `dst_table` is qualified the way `source_name` is, not the way the caller
    # named the table.
    by_source_name = {graph[name].source_name: name for name in tables}
    links = []
    for name in tables:
        for key in graph[name].source_foreign_keys:
            target = _table_named(by_source_name, key.dst_table)
            if target is None:
                continue
            if key.name not in built[name].columns:
                continue
            if key.primary_key not in built[target].columns:
                continue
            links.append(
                Relationship(
                    source_table=name,
                    source_column=key.name,
                    target_table=target,
                    target_column=key.primary_key,
                )
            )
    return links


def _table_named(
    by_source_name: Mapping[str, str], dst_table: str
) -> str | None:
    r"""The table a foreign key's target names, matched exactly if it can be.

    SQLite and Databricks fold names to lower case, so a case-insensitive match
    still counts, but only when one table answers to it. PostgreSQL and
    Snowflake can hold both `"Parent"` and `parent`, and guessing between them
    would attach the key to the wrong one.
    """
    if dst_table in by_source_name:
        return by_source_name[dst_table]
    folded = [
        name
        for source_name, name in by_source_name.items()
        if source_name.lower() == dst_table.lower()
    ]
    return folded[0] if len(folded) == 1 else None


def _require_column(
    tables: Mapping[str, _TableShape], table: str, column: str
) -> str:
    if column not in _require_table(tables, table).columns:
        raise StructuredDataError(
            f'table {table!r} has no column {column!r}; it has '
            f'{sorted(tables[table].columns)}',
            code='INVALID_REQUEST',
        )
    return column


def _described_on_the_core(schema: Any) -> bool:
    r"""Whether ``schema`` was described on the core: a column of it holds
    what the source said of it (``Column._source``, T3.16), so its edits are
    the core's ``edit_schema``'s (T1.24b, decision E66(g)); ``None`` and a
    schema of frames are not."""
    tables = vars(schema).get('_tables') if schema is not None else None
    return any(
        column._source is not None
        for table in (tables or {}).values()
        for column in table.columns.values()
    )


def _as_key_tuple(key: Any) -> tuple[str, ...]:
    r"""A primary key as the tuple of column names it is.

    A single name, or any ordered iterable of them, including a `pd.Index`,
    which is what `frame.columns` is and so the most natural way to name a
    composite key from the data it came from. `pd.Index` is not a
    `collections.abc.Sequence`, so asking that question rejected it.

    A set is not one of them. A composite key's columns are in an order, and
    a set has none, so the key it named would differ between runs.
    """
    if key is None:
        return ()
    if isinstance(key, str):
        return (key,)
    if isinstance(key, (bytes, Mapping, AbstractSet)) or not isinstance(
        key, Iterable
    ):
        raise StructuredDataError(
            f'a primary key names a column or a sequence of them, got {key!r}',
            code='INVALID_REQUEST',
        )
    names = list(key)
    if not all(isinstance(name, str) for name in names):
        raise StructuredDataError(
            f'a primary key names a column or a sequence of them, got {key!r}',
            code='INVALID_REQUEST',
        )
    return tuple(names)


def _infer_table(name: str, frame: pd.DataFrame) -> Table:
    r"""Describe a frame as a table; see
    :func:`_infer_table_and_time_column`.
    """
    return _infer_table_and_time_column(name, frame)[0]


def _infer_table_and_time_column(
    name: str, frame: pd.DataFrame
) -> tuple[Table, TimeColumnChoice]:
    r"""Describe a frame as a table, inferring the types, key and time column,
    and return with it how the time column was chosen.

    The cells are checked before any column is described. Typing a column asks
    for its distinct values, and a list cell makes that raise `unhashable
    type: 'list'` from inside pandas, naming nothing the caller wrote.
    `Table.__post_init__` asks the same question again for a table built by
    hand, which is the route that has not passed through here.
    """
    if frame.columns.has_duplicates:
        duplicates = sorted(
            {str(c) for c in frame.columns[frame.columns.duplicated()]}
        )
        raise StructuredDataError(
            f'table {name!r} has duplicate column name(s) {duplicates}',
            code='INVALID_REQUEST',
        )
    non_strings = [c for c in frame.columns if not isinstance(c, str)]
    if non_strings:
        raise StructuredDataError(
            f'table {name!r} column names must be strings; got {non_strings}',
            code='INVALID_REQUEST',
        )
    for column in frame.columns:
        require_scalar_cells(frame[column], f'{name}.{column}')

    columns = {
        column: _column(frame[column], column) for column in frame.columns
    }
    table = Table(name=name, data=frame, columns=columns)
    inferred = _infer_primary_key(frame, columns)
    if inferred is not None and frame[inferred].isna().any():
        # Tolerated here and refused on assignment, the same way a nearly
        # unique column is: inference is a guess and a caller's key is an
        # assertion. Refusing would break `Schema.from_data` on frames that
        # worked before, including any relationship pointing at this table,
        # but the engine drops these rows without a word, so the guess is
        # said out loud.
        _warn_a_key_has_no_value(
            name, inferred, int(frame[inferred].isna().sum())
        )
        object.__setattr__(table, 'primary_key', inferred)
        table._adopt_key(inferred)
        return table, _infer_time_column(table, frame)
    # Assigning the key is what marks its column, counts what the uniqueness
    # threshold tolerated and says so; doing any of it a second time here would
    # report the same duplicate twice, from inside this module.
    table.primary_key = inferred
    return table, _infer_time_column(table, frame)


def _copy_table(table: _T) -> _T:
    r"""The same table again, field for field, judging nothing.

    Built without the constructor and without the checked setters, for two
    reasons. Inference deliberately accepts a key with a null in it. It warns
    and continues because refusing would break frames that worked. However,
    assignment refuses one, so copying through the setter turned a schema
    `from_data` had accepted into a request that could not be materialized.
    And `Table.__post_init__` walks every cell of every column, which on a
    warehouse extract is a second full scan added to every single request.
    """
    copied = type(table).__new__(type(table))
    object.__setattr__(copied, 'name', table.name)
    for held in ('data', 'source_name'):
        if hasattr(table, held):
            object.__setattr__(copied, held, getattr(table, held))
    object.__setattr__(
        copied,
        'columns',
        {name: _copy_column(column) for name, column in table.columns.items()},
    )
    for carried in ('primary_key', 'time_column', 'duplicate_keys'):
        object.__setattr__(copied, carried, getattr(table, carried))
    for column in copied.columns.values():
        object.__setattr__(column, '_table', copied)
    return copied


def _copy_column(column: Column) -> Column:
    r"""The same column again, without re-judging it.

    Built field by field rather than through the constructor: a schema is
    allowed to be inconsistent between two assignments. That is what
    `check_consistent` checks later, so copying one must not be
    the moment its storage and meaning are held against each other.
    """
    made = Column.__new__(Column)
    for field_name in (
        'name',
        'dtype',
        'stype',
        'untyped',
        'stype_declared',
        '_stype_before_key',
    ):
        object.__setattr__(made, field_name, getattr(column, field_name))
    # What the source said of it, frozen, so shared (T3.16); a column
    # unpickled from before it had any holds none.
    object.__setattr__(made, '_source', getattr(column, '_source', None))
    return made


def _column(series: pd.Series, name: str) -> Column:
    r"""One column, with the record of whether inference had anything to go on.

    `untyped` is set after construction because it is not an init field: a
    caller must not be able to build a column that names a dtype and claims to
    have none, which is the state the adapter resolves by discarding the dtype.

    The values are read once and the answer passed down. `infer_tfm_dtype` on
    an object column walks every cell. Asking it three times for the dtype,
    again inside the stype, and again for `untyped` is what made
    describing a million rows of warehouse `Decimal` take five seconds before
    anything was sent. The same data already typed takes three hundredths.
    """
    inferred = infer_tfm_dtype(series)
    built = Column(
        name=name,
        dtype=_dtype_of(inferred),
        stype=_stype_of(series, inferred),
    )
    if inferred == UNTYPED:
        object.__setattr__(built, 'untyped', True)
    return built


def _dtype_of(inferred: str) -> str:
    # An all-null object column carries no type. Calling it a string is the
    # safe fallback: it round-trips whatever arrives without claiming a width.
    return 'string' if inferred == UNTYPED else inferred


def _stype_of(series: pd.Series, inferred: str | None = None) -> str:
    dtype = _dtype_of(infer_tfm_dtype(series) if inferred is None else inferred)
    if dtype.startswith('timestamp'):
        return 'timestamp'
    if dtype in {'int32', 'int64', 'float32', 'float64'}:
        return 'numerical'
    if dtype == 'bool':
        return 'categorical'
    distinct = series.nunique(dropna=True)
    return 'categorical' if distinct <= _MAX_CATEGORICAL_CARDINALITY else 'text'


def _infer_primary_key(
    frame: pd.DataFrame, columns: Mapping[str, Column]
) -> str | None:
    r"""The column that identifies a row, if one is obvious.

    Requires a key-shaped name, values that can identify a row, and near-total
    uniqueness. Uniqueness alone would nominate any high-cardinality
    measurement, and a name alone would nominate a foreign key.

    **Uniqueness here is a property of the frame, not of the table it came
    from.** A foreign key drawn sparsely enough looks exactly like a key: take
    200 rows of a fact table whose real key is `(order, line)` and `order`
    will very likely hold 200 distinct values, because two rows of the same
    order are unlikely to be drawn together. Nothing in the frame distinguishes
    that from a genuine key, so a frame that is a sample can nominate a foreign
    key and no amount of checking here will notice. Declare the key when the
    frame is a sample; inference is for the case where the frame is the table.
    """
    if len(frame) == 0:
        return None
    for name, column in columns.items():
        lowered = name.lower()
        if not lowered.endswith(_KEY_SUFFIXES):
            continue
        if not supports_dtype('ID', column.dtype):
            # A key-shaped name on values that cannot identify a row, such as a
            # bool
            # column called `id`, say. Nominating it would refuse the frame at
            # construction for a key nobody asked for.
            continue
        distinct = frame[name].nunique(dropna=True)
        if distinct >= len(frame) * _KEY_UNIQUENESS:
            return name
    return None


#: Above this share of rows being distinct, a string column's value count is
#: bounded by the frame rather than by the column, so it says nothing about
#: how many values the column really holds.
_UNSATURATED = 0.5

#: And below this share of the categorical limit, growth has too far to go for
#: the frame to be what decides: two values in four rows are unsaturated by
#: the measure above and are still nowhere near becoming sixty-five.
_WITHIN_REACH = 0.5


def _warn_stype_was_decided_by_the_rows_read(
    tables: Mapping[str, Table],
) -> None:
    r"""Say when `categorical` was decided by the size of the frame.

    A string column is called `categorical` when it holds few enough distinct
    values and `text` otherwise, and the count is taken from the frame handed
    in. For a column whose values repeat that is the same answer at any size:
    `state` holds 27 values in ten thousand rows and in a hundred thousand.
    For one that does not, the count is not a property of the column at all.
    It is a property of how many rows were read, and reading more rows moves
    it. So a request written against `df.head(100)` and then pointed at the
    table describes a column two different ways, with nothing in between to
    say the description changed.

    Which is worth saying only where it can happen, and that takes both of two
    measures. The first is whether the values have run out: a column whose
    distinct count tracks its row count has not shown how many values it has,
    only how many rows were looked at. The second is whether there is anything
    within reach. Two values in four rows have not saturated either, and are
    nowhere near becoming sixty-five, so saying so is noise on every small
    frame. Nearness to the threshold alone is not the signal, because 53
    distinct in 100 rows is not near 64 by any measure and still flips.

    This is the same property the primary-key inference documents about
    uniqueness, and it has the same remedy, which is to say what the column is
    when the frame is a sample.

    The count decides the meaning for strings only, which is what this is
    about: a bool is categorical by its type rather than by how many values it
    holds. That makes the dtype check a precondition rather than a filter
    today, since the only other categorical dtype is bool and two values never
    come within reach of the limit. It is stated so that a dtype added to the
    categorical side later does not inherit a message about a count that did
    not decide it.

    The share is measured against the rows that hold a value, because that is
    what the distinct count was taken from. A column of forty values among a
    hundred and sixty nulls has saturated nothing, and dividing by the height
    of the frame said it had.
    """
    for name, table in tables.items():
        frame = table.data
        if frame is None or len(frame) == 0:
            continue
        for column, meta in table.columns.items():
            if meta.stype_declared or meta.stype != 'categorical':
                continue
            if meta.dtype != 'string' or column not in frame:
                continue
            distinct = frame[column].nunique(dropna=True)
            present = int(frame[column].count())
            if distinct < present * _UNSATURATED:
                continue
            if distinct < _MAX_CATEGORICAL_CARDINALITY * _WITHIN_REACH:
                continue
            warnings.warn(
                f'{name}.{column} holds {distinct} distinct values in the '
                f'{present} rows that have one, so the rows ran out before '
                f'the values did and the count is a measure of the frame '
                f'rather than of the column. It is read as categorical here '
                f'because {distinct} is at most '
                f'{_MAX_CATEGORICAL_CARDINALITY}; on more rows the same '
                f'column could pass that and be read as text, which the '
                f'models treat differently. If this frame is a sample, say '
                f'what the column is with '
                f"schema[{name!r}][{column!r}].stype = 'categorical' (or "
                f"'text'), so that it does not change under a larger one.",
                category=SDMWarning,
                stacklevel=caller_stacklevel(),
            )


def _warn_dates_stored_as_text(table: Table) -> str | None:
    r"""Say when the only candidate timestamp is a string, and was not used.

    Inference types a string by how many distinct values it holds and never
    reads it as a date, which is deliberate: a parser loose enough to read a
    warehouse `DATE` column also reads part codes as dates and dies later, on
    a value it should not have touched. The cost is that a wholly temporal
    dataset, such as a warehouse that ordinarily returns `DATE` as `STRING`,
    infers to a schema with no time in it. The caller learns
    only when a temporal query is refused, or never, if the query they wrote
    happened to be static.

    So the column is still not parsed, and the declining is no longer silent.
    Only a table that ended with no time column is worth a word; one that
    found its timestamp does not need to hear about a date-shaped string.

    The shape test is not a correctness check: the conversion below rejects
    everything it would, and more. It is there so that the ordinary case, a
    table whose string columns are names and descriptions, does not pay to
    have every one of them parsed as a date.

    Every remaining question is answered by running that conversion over the
    whole column rather than sampling it or reasoning about it: whether the
    values past the sample hold their shape, whether they are real dates, and
    which form of the call the caller should be told to make.
    """
    frame = table.data
    if table.time_column is not None or frame is None or frame.empty:
        return None
    for name, column in table.columns.items():
        if column.dtype != 'string' or column.stype == 'ID':
            continue
        present = frame[name].dropna()
        if not dates.looks_unambiguous(present):
            continue
        how = dates.conversion_that_works(present)
        if how is None:
            continue
        warnings.warn(
            f'{table.name}.{name} holds dates but is stored as text, so it is '
            f'not the time column and no temporal query can use it. Convert '
            f'it before inferring: '
            f'frame[{name!r}] = pd.to_datetime(frame[{name!r}]{how}), or '
            f'set schema[{table.name!r}][{name!r}].stype = {"timestamp"!r} '
            f'afterwards.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
        return name
    return None


def _infer_time_column(table: Table, frame: pd.DataFrame) -> TimeColumnChoice:
    r"""Set the time column, and return how it was chosen.

    The rule is the one the engine applies to a source, so a frame and the
    same rows in a warehouse infer the same column.
    """
    stamps = [
        name
        for name, column in table.columns.items()
        if column.stype == 'timestamp' and name != table.primary_key
    ]
    choice = choose_time_column(frame, stamps)
    table.time_column = choice.column
    return choice


def _settle_edit_stamps(
    built: Mapping[str, _TableShape],
    chose: dict[str, TimeColumnChoice | None],
    time_columns: Mapping[str, str | None] | None,
    links: Sequence[Relationship],
) -> None:
    r"""Take or refuse a lone edit stamp by whether another table links to
    the table it is in.

    A table something links to is an entity and refuses the stamp of its
    last edit; one nothing links to is a log and takes it. Decided on the
    schema's final links, the one fact both paths share by now: the two
    guess keys differently, so a log's `update_id` is a key from a frame
    and none from a source, and a rule on the key made them disagree.
    """
    linked = {link.target_table for link in links}
    for name, choice in chose.items():
        if choice is None or name in (time_columns or {}):
            continue
        table = built[name]
        settled = choice.for_links(name in linked)
        if settled != choice:
            chose[name] = settled
            table.time_column = settled.column


def _warn_how_time_columns_were_inferred(
    schema: Schema,
    chose: Mapping[str, TimeColumnChoice | None],
    given: Mapping[str, str | None] | None,
) -> None:
    """Say, in one warning, which time columns inference took or refused on
    a guess, and the override that settles each.

    A guess about a column's name or values decides what the model may know
    at the moment it answers, so it is said out loud: a column taken as the
    earliest of several, a stamp of the last edit taken as the only timestamp
    left, one taken although the rows read hold one value or mostly none,
    because the query over the whole table failed, and, for a table left
    without a time column, each timestamp refused and why. That table fails its first windowed query, and the refusal
    cannot say which of its columns inference passed over.

    Only where something could be windowed: a lone table is asked
    single-table questions, which have no window. A table whose time column
    the caller named is not mentioned.
    """
    tables = schema.tables
    if len(tables) < 2 and not schema.relationships:
        return
    lines: list[str] = []
    for name, table in sorted(tables.items()):
        choice = chose.get(name)
        if choice is None or name in (given or {}):
            continue
        column = table.time_column
        if column is not None and column == choice.column:
            if choice.instead_of:
                one = len(choice.instead_of) == 1
                lines.append(
                    f'{name}.{column} was taken as the time column rather '
                    f'than {_and_list(choice.instead_of)}, which '
                    f'{"truncates or casts" if one else "truncate or cast"} '
                    f'it, since it says when a row happened more precisely. '
                    f'If a question should count by '
                    f'{"it" if one else "one of those"}, name it: '
                    f'time_columns={{{name!r}: {choice.instead_of[0]!r}}}.'
                )
            if choice.how == 'earliest':
                others = [other for other in choice.among if other != column]
                lines.append(
                    f'{name}.{column} was taken as the time column because '
                    f'its values start earliest of {_and_list(choice.among)}. '
                    f'If another says when a row became true, name it: '
                    f'time_columns={{{name!r}: {others[0]!r}}}. If none does, '
                    f'clear it: time_columns={{{name!r}: None}}.'
                )
            elif choice.how == 'edit_stamp':
                lines.append(
                    f'{name}.{column} was taken as the time column although '
                    f'it looks like when a row was last changed, because the '
                    f'table has no other timestamp and no table links to it, '
                    f'so it reads as a log of changes. If it does not say when a row '
                    f'became true, clear it: time_columns={{{name!r}: None}}.'
                )
            lines.extend(
                taken_unchecked(name, doubt, f'time_columns={{{name!r}: None}}')
                for doubt in choice.unconfirmed
                if doubt.column == column
            )
        elif column is None:
            lines.extend(
                f'{name}.{rejected.column} was not taken as the time column '
                f'because {rejected.why}. If it does say when a row became '
                f'true, name it: '
                f'time_columns={{{name!r}: {rejected.column!r}}}.'
                for rejected in choice.rejected
                if rejected.column in table.columns
            )
    if lines:
        warnings.warn(
            'Some time columns were inferred, or refused, on a guess about '
            'what a column holds:\n' + '\n'.join(f'- {line}' for line in lines),
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _warn_table_cannot_be_linked_to(
    tables: Mapping[str, Table],
    dates: Mapping[str, str],
    already_said: AbstractSet[str] = frozenset(),
) -> None:
    r"""Say when a table has no key and something wanted to point at it.

    `_infer_primary_key` requires a key-shaped name, and rightly: uniqueness
    alone would nominate any high-cardinality measurement. But a great many
    real keys are domain codes such as `IATA_CODE`, `SKU`, `ISBN`, and `VIN`.
    No suffix rule will cover them.

    That is not merely a missing key. `_infer_relationships` can only point an
    edge at a *declared* key, so a table that has none is not a candidate
    target for anything: every relationship into it is lost, the tables are
    dropped from the request, and the model answers from less data than it was
    given with nothing on screen to say so.

    Nominating a key here was tried and withdrawn. Uniqueness within one frame
    does not distinguish a key from a foreign key that happens not to repeat in
    the rows to hand. A two-row extract of `line_items` has a unique
    `ORDER_NO`, so nominating turns a child into a parent and invents an edge
    between two facts. Worse, a second table keyed on a name makes that name
    ambiguous, and `_infer_relationships` then declines *every* edge using it:
    the guess does not merely risk a wrong link, it can delete the right ones
    from a schema whose keys the caller declared correctly.

    So the caller is asked instead, and only where the question is real. A
    keyless table no other table names loses no edges. Nor does one whose
    candidate column is already another table's key. That column is a
    foreign key, naming it here would make the name ambiguous, and ambiguity
    deletes the edges that were working. A warning that talks a caller into
    the failure it was written to prevent is worse than silence.

    `already_said` names the tables the repeated-key warning has spoken about.
    Said of one table the two contradict, with remedies that cannot both be
    followed.

    `dates` names, per table, a column already reported as a date stored as
    text. A column cannot be both the time column and the primary key, so
    asking for it here would contradict the other warning, and a caller who
    followed both in order would end with neither.

    How many columns could be the key and how many tables they are shared with
    are two counts, and they do not move together: several columns can be
    shared with one table, and one column with several. Agreeing either by the
    other made the sentence disagree with the graph it describes.
    """
    named_elsewhere = _tables_naming_each_column(tables)
    keyed_by = {
        table.primary_key
        for table in tables.values()
        if isinstance(table.primary_key, str)
    }

    for name, table in tables.items():
        if table.primary_key is not None or name in already_said:
            continue
        frame = table.data
        if frame is None or frame.empty:
            continue
        candidates = [
            column
            for column, meta in table.columns.items()
            if _could_identify_a_row(table, frame, column, meta)
            and len(named_elsewhere.get(column, ())) > 1
            and column not in keyed_by
            and column != dates.get(name)
        ]
        if not candidates:
            continue
        elsewhere = {
            other
            for column in candidates
            for other in named_elsewhere.get(column, ())
            if other != name
        }
        warnings.warn(
            _unlinkable_table_reads(name, candidates, len(elsewhere)),
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _tables_naming_each_column(
    tables: Mapping[str, _TableShape],
) -> dict[str, set[str]]:
    r"""Which tables hold a column of each name.

    A name shared by two tables is what an inferred edge is made of, so it is
    also what every question about a missing edge is asked against.
    """
    named: dict[str, set[str]] = {}
    for name, table in tables.items():
        for column in table.columns:
            named.setdefault(column, set()).add(name)
    return named


def _warn_repeats_cost_the_table_its_key(
    tables: Mapping[str, Table],
) -> set[str]:
    r"""Say when a key-shaped column was declined for its repeats.

    `_infer_primary_key` wants `_KEY_UNIQUENESS` of the values distinct, and
    tolerating a little short of total uniqueness is right: one duplicated row
    is common in an extract and still the column the caller means. But the
    tolerance has an edge, and the behaviour either side of it is the wrong way
    round. One duplicated row nominates the key and warns; two nominate
    nothing, and a table with no key is not a candidate target for any edge, so
    every relationship pointing at it disappears, with nothing said at
    inference or at materialization. The louder outcome is the smaller
    problem.

    `_warn_table_cannot_be_linked_to` is the warning for a table nothing can
    point at, and it cannot see this one: it asks for columns that identify a
    row perfectly, and the column that caused the loss is by definition not
    one of them. So it is asked about here instead, where the repeats are the
    subject rather than the disqualification.

    Only a column another table names is worth a word, on the same reasoning:
    a key nothing points at loses no edge by being absent.

    `_NEARLY_A_KEY` is what separates a key with a fault in it from an
    ordinary foreign key, which repeats on the order of the row count rather
    than a handful of times. A sampled extract makes an ordinary foreign key
    nearly unique too, so a column another table is already keyed by is left
    alone: the edge through it works, and the remedy printed here deletes it.

    Returns the tables it spoke about, so the keyless-table warning does not
    follow it with a contradicting account of the same one.
    """
    named_elsewhere = _tables_naming_each_column(tables)
    keyed_by = {
        table.primary_key
        for table in tables.values()
        if isinstance(table.primary_key, str)
    }
    spoken_for = set()
    for name, table in tables.items():
        if table.primary_key is not None:
            continue
        frame = table.data
        if frame is None or frame.empty:
            continue
        for column, meta in table.columns.items():
            if not column.lower().endswith(_KEY_SUFFIXES):
                continue
            if not supports_dtype('ID', meta.dtype):
                continue
            if column in keyed_by:
                continue
            if not named_elsewhere.get(column, set()) - {name}:
                continue
            if frame[column].nunique(dropna=True) < len(frame) * _NEARLY_A_KEY:
                continue
            quality = _key_quality(frame, (column,))
            if not quality.duplicates:
                continue
            _warn_a_key_was_declined(
                name,
                column,
                quality.duplicates,
                len(frame),
                quality.example,
                f'Keep one row per {column!r}: '
                f'frame = frame.drop_duplicates({column!r}), and infer '
                f'again, or name a column that does identify a row with '
                f'primary_keys={{{name!r}: ...}}.',
            )
            spoken_for.add(name)
            break
    return spoken_for


def _warn_one_key_name_keys_several_tables(
    tables: Mapping[str, Table],
) -> None:
    r"""Say when a key name is shared, so no edge can be inferred through it.

    `_infer_relationships` declines every edge naming a key that more than one
    table is keyed by, and rightly: `id` on four tables identifies a row in
    each of them and names a parent in none, so choosing one would be a guess.
    What was missing is that the declining was silent. Four frames infer to
    four tables and no relationships at all, and the only thing on screen is
    whatever unrelated notice some column happened to earn, which then reads
    as the whole account of what inference did.

    Said per shared name rather than per lost edge. A table keyed by an
    ambiguous name can never be the target of an inferred relationship, so the
    loss is the name's, and listing candidate edges that were never built
    would be inventing a graph to describe.
    """
    keyed_by: dict[str, list[str]] = {}
    for name, table in tables.items():
        if isinstance(table.primary_key, str):
            keyed_by.setdefault(table.primary_key, []).append(name)

    for key, names in sorted(keyed_by.items()):
        if len(names) < 2:
            continue
        shared = sorted(names)
        warnings.warn(
            f'{key!r} is the primary key of {_and_list(shared)}, so a column '
            f'named {key!r} identifies a row in each of them and names a '
            f'parent in none. No relationship is inferred through an '
            f'ambiguous name, so nothing can point at any of those tables: '
            f"none of them is gathered as another table's related rows, and "
            f'a question about one of them is answered from that table alone. '
            f'Say what the graph is: '
            f'relationships=[Relationship(child, column, {shared[0]!r}, '
            f'{key!r}), ...], or give each table a key of its own name, so '
            f'that the column naming it can be matched to it.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _unlinkable_table_reads(
    name: str, candidates: Sequence[str], tables_sharing: int
) -> str:
    r"""What to say about a table nothing can link to.

    Kept apart from the search that found it, because the two are different
    kinds of difficult. Deciding which columns could be the key is about the
    data; saying so is about English, and the sentence has to agree with the
    graph on two counts that move independently: how many columns could be
    the key, and how many tables they are shared with. Several columns can be
    shared with one table, and one column with several.
    """
    many, shared_by_many = len(candidates) > 1, tables_sharing > 1
    subject, verb = ('they', 'are') if many else ('it', 'is')
    tables = 'tables' if shared_by_many else 'table'
    return (
        f'{name!r} has no primary key, and nothing can link to a table '
        f'without one: every relationship pointing at {name!r} is dropped, so '
        f'nothing is gathered through it, and a question about another table '
        f'reaches it only where {name!r} names that table itself. '
        f'{_and_list(candidates)} {"identify" if many else "identifies"} '
        f'every row of {name!r} and {verb} named by '
        f'{"other tables" if shared_by_many else "another table"} too, which '
        f'makes {"each of them" if many else "it"} either a key or a foreign '
        f'key that happens not to repeat in the rows read. '
        f'If {"one of them identifies" if many else "it identifies"} '
        f'{name!r}, say so with primary_keys={{{name!r}: '
        f'{repr(candidates[0]) if not many else "..."}}}; '
        f'if {subject} instead '
        f'{"name" if many else "names"} the rows of the {tables} {subject} '
        f'{verb} shared with, '
        f'{"those tables are the ones that need" if shared_by_many else "that table is the one that needs"} '
        f'the key.'
    )


def _each_link_once(
    links: Sequence[Relationship],
) -> list[Relationship]:
    """Deduplicate relationships while preserving their order.

    The materializer already sends one edge for it, so keeping two here made
    the two halves of the client disagree about what the graph is, and the
    ambiguity warning read a repeat as a second way across the pair.
    """
    seen: set[tuple[str, str, str, str]] = set()
    once = []
    for link in links:
        named = (
            link.source_table,
            link.source_column,
            link.target_table,
            link.target_column,
        )
        if named in seen:
            continue
        seen.add(named)
        once.append(link)
    return once


def _warn_of_a_pair_joined_both_ways(links: Sequence[Relationship]) -> None:
    """Say when two tables name each other, which reads as an ambiguous pair.

    A parent naming one representative child is an ordinary shape and the
    schema takes it. The sampler walks an edge in both directions, so a
    declared reverse edge is a second way across the same pair, and every
    aggregation between them is then refused for a foreign key it says it
    cannot find uniquely when there is exactly one. The reverse edge is
    what made it two.

    A table pointing at itself is one link, not two: the test for a reverse
    edge is otherwise satisfied by the edge itself.
    """
    pairs = {(link.source_table, link.target_table) for link in links}
    said = set()
    for source, target in sorted(pairs):
        if source == target:
            continue
        if (target, source) not in pairs or (target, source) in said:
            continue
        said.add((source, target))
        warnings.warn(
            f'{source!r} and {target!r} name each other, and a relationship is '
            f'crossed in both directions, so this is two ways over one pair and '
            f'an aggregation between them is refused for not having a single '
            f'one. Declare whichever of the two a question is about and leave '
            f'the other out.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


class _Links(list):
    _schema: Schema

    r"""The relationships of one schema, checked however they are added.

    `append` is the route this client's own advice prints, and a check on
    assignment alone left it unchecked.
    """

    def __init__(self, links: Sequence[Relationship], schema: Schema) -> None:
        super().__init__(links)
        object.__setattr__(self, '_schema', schema)

    def _resettle(self, links: list[Relationship]) -> None:
        if _described_on_the_core(self._schema):
            # A core-built schema: the core checks and makes the edit, and
            # its answer is stored in this list (T1.24b).
            from sdm_client._rust import edits

            if edits.relationships(self._schema, links, into=self):
                return
        settled = self._schema._settle(links)
        super().clear()
        super().extend(settled)
        self._schema._mark_key_columns()

    def remove(self, link: Any) -> None:
        kept = list(self)
        kept.remove(link)
        self._resettle(kept)

    def pop(self, index: Any = -1) -> Relationship:
        kept = list(self)
        gone = kept.pop(index)
        self._resettle(kept)
        return gone

    def clear(self) -> None:
        self._resettle([])

    def __delitem__(self, index: Any) -> None:
        kept = list(self)
        del kept[index]
        self._resettle(kept)

    def append(self, link: Relationship) -> None:
        self._resettle([*self, link])

    def extend(self, links: Any) -> None:
        self._resettle([*self, *links])

    def insert(self, index: Any, link: Any) -> None:
        kept = list(self)
        kept.insert(index, link)
        self._resettle(kept)

    def __iadd__(self, links: Any) -> _Links:  # type: ignore[misc]
        self._resettle([*self, *links])
        return self

    def __imul__(self, count: Any) -> _Links:  # type: ignore[misc]
        self._resettle(list(self) * count)
        return self

    def __setitem__(self, index: Any, link: Any) -> None:
        kept = list(self)
        kept[index] = link
        self._resettle(kept)

    def __reduce__(self) -> tuple[Any, ...]:
        r"""Copied and pickled as a plain list, and re-checked on arrival.

        The schema this belongs to is held here, so copying the list the
        ordinary way copies the schema, whose state is restored after its
        own `__setattr__` would have run. `Schema.__setstate__` puts the list
        back under this type once the copy has its tables.
        """
        return (list, (list(self),))


def _warn_two_ways_to_reach_one_table(
    tables: Mapping[str, _TableShape], links: Sequence[Relationship]
) -> None:
    r"""Say when one table reaches another by more than one column.

    Both are real relationships and keeping both is right: a sales line names
    the customer it was sold to and the one it ships to, and they are
    different facts. But an aggregation across a pair of tables has to be
    resolved to one column, and a query that crosses an ambiguous pair is
    refused when it is asked, which reads as a problem with the
    query rather than with the graph it was written against.

    Said here so that it is read where the cause is. Nothing is dropped: which
    of them a question means is the caller's to decide, and choosing for them
    is choosing what their model sees.
    """
    ways: dict[tuple[str, str], list[str]] = {}
    for link in links:
        pair = (link.source_table, link.target_table)
        columns = ways.setdefault(pair, [])
        if link.source_column not in columns:
            columns.append(link.source_column)

    _warn_of_a_pair_joined_both_ways(links)
    for (source, target), columns in sorted(ways.items()):
        if len(columns) < 2:
            continue
        kept, *rest = sorted(columns)
        warnings.warn(
            f'{source!r} reaches {target!r} through {_and_list(sorted(columns))}'
            f', and an aggregation across the two tables has to be resolved to '
            f'one of them, so a query that crosses this pair is refused until '
            f'it is. They are all kept because they are all real, and which '
            f'one a question means is not something this can decide. To keep '
            f'{kept!r}, drop the others: schema.relationships = [link for '
            f'link in schema.relationships if not (link.source_table == '
            f'{source!r} and link.target_table == {target!r} and '
            f'link.source_column in {sorted(rest)!r})].',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _could_identify_a_row(
    table: Table, frame: pd.DataFrame, column: str, meta: Column
) -> bool:
    if column == table.time_column:
        return False
    if meta.dtype != 'string' and not meta.dtype.startswith('int'):
        return False
    if frame[column].isna().any():
        return False
    return _key_quality(frame, (column,)).duplicates == 0


def _and_list(names: Sequence[str]) -> str:
    quoted = [repr(name) for name in names]
    if len(quoted) < 3:
        return ' and '.join(quoted)
    return f'{", ".join(quoted[:-1])} and {quoted[-1]}'


def _infer_relationships(
    tables: Mapping[str, _TableShape],
) -> list[Relationship]:
    r"""Link a column to another table's primary key of the same name.

    Deliberately conservative: only an exact name match against a declared
    primary key counts. A wrong link changes what the model is shown, and is
    harder to notice than a missing one, which simply narrows the question.

    Which is why a name more than one table is keyed by links nothing. `ID` on
    four tables identifies a row in each of them and names a parent in none, so
    choosing one would be a guess in the direction this is supposed to avoid.
    Value overlap cannot break the tie either: a
    fabricated edge has been measured joining 97% of rows and a real one 91%,
    so no threshold separates them.
    """
    ambiguous: set[str] = set()
    keys: dict[str, str] = {}
    for name, table in tables.items():
        key = table.primary_key
        if not isinstance(key, str):
            continue
        if key in keys:
            ambiguous.add(key)
        keys[key] = name

    links = []
    for name, table in tables.items():
        for column in table.columns:
            target = keys.get(column)
            if target is None or target == name or column in ambiguous:
                continue
            links.append(
                Relationship(
                    source_table=name,
                    source_column=column,
                    target_table=target,
                    target_column=column,
                )
            )
    return links


def _warn_of_a_link_that_cannot_join(
    tables: Mapping[str, _TableShape], dropped: Sequence[Relationship]
) -> None:
    """Say why an inferred or declared link could not join.

    Dropping it is right: joining `int64` to `string` matches nothing, and an
    edge that matches nothing is worse than no edge. But it was dropped in
    silence, and the table then either goes unsent or the query is refused for
    a foreign key that looks as though it should exist.
    """
    for link in dropped:
        source = tables[link.source_table].columns[link.source_column]
        target_table = tables[link.target_table]
        target = target_table.columns[link.target_column]
        key = _as_key_tuple(target_table.primary_key)
        if key != (link.target_column,):
            # A declared foreign key may name a unique column that is not the
            # key, which is no fault of the dtypes.
            has = (
                f'the primary key of {link.target_table!r} is '
                f'{_and_list(list(key))}'
                if key
                else f'{link.target_table!r} has no primary key'
            )
            warnings.warn(
                f'{link.source_table}.{link.source_column} refers to '
                f'{link.target_table}.{link.target_column}, but {has}. A '
                f'relationship joins to a primary key, so none was made from '
                f'it. Pass '
                f'primary_keys={{{link.target_table!r}: '
                f'{link.target_column!r}}} if {link.target_column!r} '
                f'identifies a row of {link.target_table!r}.',
                category=SDMWarning,
                stacklevel=caller_stacklevel(),
            )
            continue
        warnings.warn(
            f'{link.source_table}.{link.source_column} names the key of '
            f'{link.target_table!r} and holds {source.dtype!r} where that key '
            f'holds {target.dtype!r}, which cannot match, so no relationship '
            f'was made from it. Store the two as one type if they are the same '
            f'identifier.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )


def _require_joinable(
    tables: Mapping[str, _TableShape], link: Relationship
) -> None:
    r"""A relationship has to be one the engine can actually build.

    An edge joins a foreign key to the *primary key* of the table it points at.
    That is the only shape the graph has, so a link naming some other
    column is not narrower, it is different: the language would resolve one
    graph while the engine built another. And the two columns have to hold
    comparable values, or the join matches nothing.
    """
    target = tables[link.target_table]
    key = target.primary_key
    keys = _as_key_tuple(key)
    if len(keys) > 1:
        raise StructuredDataError(
            f'{link} points at {link.target_table!r}, whose primary key is '
            f'{list(keys)}. An edge joins one column to one key, so a table '
            f'with a composite key cannot be the target of a relationship',
            code='INVALID_REQUEST',
        )
    if link.target_column not in keys:
        named = f'{key!r}' if key is not None else 'no primary key'
        raise StructuredDataError(
            f'{link} does not join to a primary key: table '
            f'{link.target_table!r} has {named}. A relationship points at the '
            f'key that identifies a row, so set that key first if it is not '
            f'the one inference found',
            code='INVALID_REQUEST',
        )
    source_dtype = tables[link.source_table][link.source_column].dtype
    target_dtype = target[link.target_column].dtype
    if _join_family(source_dtype) != _join_family(target_dtype):
        raise StructuredDataError(
            f'{link} joins {source_dtype!r} to {target_dtype!r}, which cannot '
            f'match. Give both columns the same kind of value',
            code='INVALID_REQUEST',
        )


def _join_family(dtype: str) -> str:
    r"""What has to agree across a join: a number matches a number."""
    if dtype.startswith(('int', 'float')):
        return 'number'
    if dtype.startswith('timestamp'):
        return 'timestamp'
    return dtype


def supports_dtype(stype: str, dtype: str) -> bool:
    r"""Whether a column of ``dtype`` can mean ``stype``.

    Answered by the language rather than by a second table kept here, so the
    two cannot drift. An unrecognised pair is allowed through: this is a guard
    against a known-wrong combination, not a whitelist.
    """
    from predictive_query.schema.typing import Dtype, Stype

    try:
        return Stype(stype).supports_dtype(
            Dtype(_WIRE_TO_PQL_DTYPE.get(dtype, dtype))
        )
    except ValueError:
        return True


class _KeyQuality(NamedTuple):
    r"""What one pass over a key column found.

    Computed once and handed on. The check, the count that gets recorded and
    the example in the message all used to derive this separately, which meant
    three full passes over the same column for every key assignment.
    """

    duplicates: int
    nulls: int
    example: Any


def _is_joinable(tables: Mapping[str, _TableShape], link: Relationship) -> bool:
    try:
        _require_joinable(tables, link)
    except StructuredDataError:
        return False
    return True


def _the_key_of(table: str, names: Sequence[str]) -> str:
    """How to name a key in a message, for one column or several.

    Joining the parts with a dot rendered `['a', 'b']` of table `t` as
    `t.a.b`, which reads as one column named `a.b`.
    """
    if len(names) == 1:
        return f'{table}.{names[0]}'
    return f'the primary key of {table!r}, {_and_list(list(names))},'


_NAME_ANOTHER_KEY = (
    ' Name a different primary key if this column is a foreign key rather '
    'than an identifier.'
)


def _too_many_repeats(duplicates: int, rows: int) -> bool:
    r"""Whether a stated key repeats more than inference would tolerate."""
    return duplicates > rows * (1 - _KEY_UNIQUENESS)


def _not_unique_enough(
    table: str, names: Sequence[str], duplicates: int, rows: int, example: Any
) -> str:
    return (
        f'{_the_key_of(table, names)} is not unique enough to identify a '
        f'row: {duplicates:,} of {rows:,} rows repeat a value already used, '
        f'e.g. {example!r}'
    )


def _a_key_missing_values(
    table: str, names: Sequence[str], nulls: int, empty: Sequence[str] | None
) -> str:
    r"""Why a stated key with a null in it is refused.

    ``empty`` names the columns holding the nulls, or is ``None`` where only
    the rows missing some part of a composite key were counted.
    """
    where = f', in {_and_list(empty)},' if empty else ' in part of it,'
    several = empty is None or len(empty) > 1
    return (
        f'{_the_key_of(table, names)} has {nulls:,} row(s) with no value'
        f'{where} so those rows cannot be identified; fill or drop them '
        f'before using {"them" if several else "it"} as a primary key'
    )


def _warn_a_key_repeats(
    table: str, names: Sequence[str], duplicates: int, example: Any, remedy: str
) -> None:
    warnings.warn(
        f'{table}.{".".join(names)} identifies a row {duplicates:,} time(s) '
        f'it has already been used, e.g. {example!r}. Those rows are '
        f'indistinguishable to the model.{remedy}',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _warn_a_key_has_no_value(table: str, column: str, missing: int) -> None:
    r"""Said of a key that was inferred rather than stated, which a null in
    it does not cost: inference is a guess, and refusing would break tables
    that worked before.
    """
    warnings.warn(
        f'{table}.{column} identifies a row, but {missing:,} row(s) have no '
        f'value for it. Those rows cannot be scored and the model will not '
        f'see them; fill or drop them, or name a different primary key.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _warn_a_key_was_declined(
    table: str,
    column: str,
    duplicates: int,
    rows: int,
    example: Any,
    remedy: str,
) -> None:
    warnings.warn(
        f'{table}.{column} is named like a key and another table names it '
        f'too, but {duplicates:,} of {rows:,} rows repeat a value already '
        f'used, e.g. {example!r}, so it does not identify a row and '
        f'{table!r} gets no primary key. Nothing can link to a table without '
        f'one: every relationship pointing at {table!r} is dropped, so '
        f'nothing is gathered through it, and a question about another table '
        f'reaches it only where {table!r} names that table itself. {remedy}',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def _key_quality(frame: pd.DataFrame, names: Sequence[str]) -> _KeyQuality:
    columns = list(names)
    if any(name not in frame.columns for name in columns):
        return _KeyQuality(0, 0, None)
    key = frame[columns[0]] if len(columns) == 1 else frame[columns]
    nulls = int(
        key.isna().any(axis=1).sum() if len(columns) > 1 else key.isna().sum()
    )
    # `duplicated` treats null as equal to null, so nulls would otherwise be
    # counted as repeats and reported as a duplicate problem, which has a
    # different fix. Count them separately and leave them out of this.
    present = (
        key[~key.isna().any(axis=1)] if len(columns) > 1 else key[key.notna()]
    )
    repeated_mask = present.duplicated()
    duplicates = int(repeated_mask.sum())
    example = None
    if duplicates:
        first = present[repeated_mask].iloc[0]
        example = first if len(columns) == 1 else tuple(first)
    return _KeyQuality(duplicates, nulls, example)


def to_schema_definition(schema: Schema) -> Any:
    r"""This schema as the language's own schema definition.

    The language resolves a query against names and types with no rows
    attached, which is what this hands it. Built here rather than in the query
    module so that a schema has exactly one translation into what PQL reads.

    `TableDefinition.pkey` is a single column name, so the language has no
    form for a composite key and cannot be given one. It is dropped rather
    than refused, because the engine does support one and the request that
    follows carries it. What the drop costs is that the language resolves the
    query against a table it believes has no identity, so a question that
    needs one is answered as though the table had none, which is why it is
    said rather than done quietly.
    """
    from predictive_query.schema.definition import (
        ColumnKey,
        ColumnKeyGroup,
        SchemaDefinition,
    )
    from predictive_query.schema.table import Column as PQLColumn
    from predictive_query.schema.table import TableDefinition
    from predictive_query.schema.typing import Dtype, Stype

    tables = {}
    for name, table in schema.tables.items():
        columns = []
        for column in table.columns.values():
            try:
                dtype = Dtype(
                    _WIRE_TO_PQL_DTYPE.get(column.dtype, column.dtype)
                )
                stype = Stype(column.stype)
            except ValueError:
                continue
            columns.append(
                PQLColumn(name=column.name, stype=stype, dtype=dtype)
            )
        key = table.primary_key
        if key is not None and not isinstance(key, str):
            say_a_composite_key_is_dropped(name, key)
        tables[name] = TableDefinition(
            cols=columns,
            pkey=key if isinstance(key, str) else None,
            time_col=table.time_column,
        )

    groups = [
        ColumnKeyGroup(
            # The group is documented as sorted and deduped, and a ColumnKey
            # has no ordering of its own, so sort on what it holds.
            columns=tuple(
                sorted(
                    (
                        ColumnKey(link.source_table, link.source_column),
                        ColumnKey(link.target_table, link.target_column),
                    ),
                    key=lambda key: (key.table_name, key.col_name),
                )
            )
        )
        for link in schema.relationships
    ]
    return SchemaDefinition(tables=tables, col_groups=groups)


def say_a_composite_key_is_dropped(name: str, key: Sequence[str]) -> None:
    r"""Say that table ``name``'s composite ``key`` is not one a query can
    name, so the query is resolved as though it had none: one place, for
    the language's schema definition and the core's query schema alike."""
    warnings.warn(
        f'table {name!r} has the composite primary key '
        f'{list(key)}, and a predictive query can only name a single '
        f'key column, so this query is resolved against {name!r} as '
        f'though it had no key. The request still carries the '
        f'composite key; only the query does not see it. Name a '
        f'single column as the primary key if the query needs to '
        f'identify a row of {name!r}.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


# The wire spells a few types differently from the language.
WIRE_TO_PQL_DTYPE = _WIRE_TO_PQL_DTYPE = {
    'int64': 'int',
    'int32': 'int',
    'float64': 'float',
    'float32': 'float',
    'timestamp[us]': 'time',
    'timestamp[ns]': 'time',
    'timestamp[ms]': 'time',
    'timestamp[s]': 'time',
}
