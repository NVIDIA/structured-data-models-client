# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The tables a request is about, and what the model should make of them.

A :class:`Schema` holds your tables, what each column means, and how the tables
relate. One table or many is the same object -- a single-table model is the
degenerate case of the same shape, not a second system, which is why this is
not called a graph.

Everything is inferred. You override only where inference is wrong:

>>> schema = Schema.from_data({'users': users_df})
>>> schema['users'].primary_key = 'user_id'
>>> schema['users']['churn'].stype = 'categorical'

**Inference reads values, so a code stored as a number reads as a number.** A
postcode, a store number or a status code is ``numerical`` here, and the model
will treat its distances as meaningful. No amount of inference settles that --
only you know the column is a label. Correct it:

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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import pandas as pd

from sdm_client.core import dates
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.wire import (
    UNTYPED,
    infer_tfm_dtype,
    require_scalar_cells,
)

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

_KEY_SUFFIXES = ('_id', '_key', 'id', 'key')


@dataclass
class Column:
    r"""One column, and what the model should treat it as.

    Attributes:
        name: The column name, as it appears in the frame.
        dtype: The wire dtype, e.g. ``'int64'`` or ``'timestamp[us]'``. Inferred
            from the values; correcting it is rarely necessary.
        stype: What the column *means* -- ``'numerical'``, ``'categorical'``,
            ``'ID'``, ``'text'``, ``'timestamp'``. This is the one worth
            correcting: it decides how the model encodes the column.
        stype_declared: Whether the meaning was assigned rather than inferred.
            An all-null column is typed ``categorical`` because a schema has to
            say something, and that placeholder is not a claim about the
            values; a meaning the caller assigned is. Recorded so the two can
            be told apart, since only the second is worth refusing a
            contradiction against.
        untyped: Whether the values carried no type at all -- an all-null
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

    def __post_init__(self) -> None:
        self._check_stype(self.stype)

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

    def _adopt_as_time_column(self) -> None:
        r"""Let a column named as a timestamp order the table it belongs to.

        `Table._check_time_column` already sets a column's stype when the
        column is named as the time column. Only that direction was wired, so
        a caller correcting an inferred schema the way the SDK's own error
        messages teach -- ``schema[table][column].stype = 'timestamp'`` -- got
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
        ``materialize()`` -- several cells after the assignment that caused it,
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


@dataclass
class Table:
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
        borrowed = [
            name
            for name, column in self.columns.items()
            if getattr(column, '_table', None) not in (None, self)
        ]
        if borrowed:
            object.__setattr__(self, 'columns', dict(self.columns))
            for name in borrowed:
                self.columns[name] = _copy_column(self.columns[name])
        for column in self.columns.values():
            object.__setattr__(column, '_table', self)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == 'primary_key':
            if value is None:
                object.__setattr__(self, 'duplicate_keys', 0)
                object.__setattr__(self, key, None)
                return
            self._check_key(value)
            object.__setattr__(self, key, value)
            self._adopt_key(value)
            return
        if key == 'time_column' and value is not None:
            self._check_time_column(value)
        object.__setattr__(self, key, value)

    def _adopt_key(self, value: str | Sequence[str]) -> None:
        r"""Make the rest of the table agree with the key it was just given.

        A key identifies a row, so its columns mean ``ID`` -- the engine's own
        setter does the same thing -- and whatever imperfection the uniqueness
        check tolerated is recorded and said out loud. Inference does all three
        after nominating a key; assignment has to as well, or the two routes
        describe the same table differently.

        A repeat is the one visible sign that the column identifies something
        coarser than a row, most often a parent on a table whose real key is
        that column together with another. Naming that possibility is what
        makes the warning actionable; saying only that the rows are
        indistinguishable describes the symptom. A column that cannot be a key
        at all has already been refused by `_check_key`.
        """
        names = _as_key_tuple(value)
        for name in names:
            self.columns[name].stype = 'ID'
        frame = getattr(self, 'data', None)
        if frame is None or frame.empty:
            return
        quality = _key_quality(frame, names)
        object.__setattr__(self, 'duplicate_keys', quality.duplicates)
        if quality.duplicates:
            together = ', '.join(
                repr(other)
                for other in self.columns
                if other not in names
                and _key_quality(frame, (*names, other)).duplicates == 0
            )
            remedy = (
                f' {".".join(names)!r} is unique together with {together}, '
                f'which is what a composite key looks like; pass that as the '
                f'primary key if so.'
                if together
                else ' Name a different primary key if this column is a '
                'foreign key rather than an identifier.'
            )
            warnings.warn(
                f'{self.name}.{".".join(names)} identifies a row '
                f'{quality.duplicates} time(s) it has already been used, '
                f'e.g. {quality.example!r}. Those rows are '
                f'indistinguishable to the model.{remedy}',
                category=SDMWarning,
                stacklevel=caller_stacklevel(),
            )

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise StructuredDataError(
                f'The name {name!r} must be a non-empty string.',
                code='INVALID_REQUEST',
            )
        invalid_chars = ['`', '\n', '\r', '.']
        name_invalid_chars = [char for char in invalid_chars if char in name]
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
        it is held to the same standard the inference applies -- and a caller
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
            raise StructuredDataError(
                f'{".".join((self.name, *names))} has {quality.nulls} row(s) '
                f'with no value, so those rows cannot be identified; fill or '
                f'drop them before using the column as a primary key',
                code='INVALID_REQUEST',
            )
        if quality.duplicates > len(frame) * (1 - _KEY_UNIQUENESS):
            raise StructuredDataError(
                f'{".".join((self.name, *names))} is not unique enough to '
                f'identify a row: {quality.duplicates} of {len(frame)} rows '
                f'repeat a value already used, e.g. {quality.example!r}',
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
        a caller runs to find out whether their schema holds together -- it
        should not be the one place that does not look.
        """
        for table in self.tables.values():
            for column in table.columns.values():
                column.check_consistent()
        self._check_relationships(self.relationships)

    def __init__(
        self,
        tables: Mapping[str, Table],
        relationships: Sequence[Relationship] = (),
    ) -> None:
        self._tables = dict(tables)
        self.relationships = list(relationships)

    def __setattr__(self, key: str, value: Any) -> None:
        r"""Hold an assigned relationship to the same standard as a passed one.

        `Schema.from_data(relationships=[link])` checks that both columns exist
        and that their dtypes can join; `schema.relationships = [link]` is the
        same statement and checked nothing, so a link that one route refuses
        the other accepted -- and `check_consistent()`, which a caller runs to
        ask exactly this, passed it.
        """
        if key == 'relationships' and hasattr(self, '_tables'):
            value = list(value)
            self._check_relationships(value)
        object.__setattr__(self, key, value)

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
        adapter reads it again at send time -- so editing a column's meaning
        after materializing changed what was sent, silently, and an agent
        reusing one schema across several questions had no way to see it.

        The rows are shared rather than copied: they are the data, they are
        what the request is about, and duplicating a warehouse extract per
        question to guard against an edit nobody makes to a frame would cost
        more than the defect. What is copied is everything a schema *says*.
        """
        return Schema(
            {name: _copy_table(table) for name, table in self._tables.items()},
            list(self.relationships),
        )

    @classmethod
    def from_data(
        cls,
        tables: Mapping[str, pd.DataFrame],
        *,
        primary_keys: Mapping[str, str | Sequence[str]] | None = None,
        time_columns: Mapping[str, str] | None = None,
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
            built[name] = _infer_table(name, frame)

        for name, key in (primary_keys or {}).items():
            table = _require_table(built, name)
            for column in _as_key_tuple(key):
                _require_column(built, name, column)
            table.primary_key = key
        for name, column in (time_columns or {}).items():
            _require_table(built, name).time_column = _require_column(
                built, name, column
            )
        if relationships is None:
            inferred = _infer_relationships(built)
            inferred = [link for link in inferred if _is_joinable(built, link)]
        else:
            inferred = list(relationships)
            for link in inferred:
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

        schema = cls(built, inferred)
        schema._mark_key_columns()
        dates = {
            name: named
            for name, table in built.items()
            if (named := _warn_dates_stored_as_text(table)) is not None
        }
        if relationships is None:
            _warn_table_cannot_be_linked_to(built, dates)
        _warn_stype_was_decided_by_the_rows_read(built)
        _warn_two_ways_to_reach_one_table(schema)
        return schema

    @property
    def tables(self) -> dict[str, Table]:
        r"""The tables, by name."""
        return self._tables

    def __getitem__(self, table: str) -> Table:
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

    def _mark_key_columns(self) -> None:
        r"""Type every key column as an identifier.

        A join key typed from its dtype alone becomes ``numerical``, which the
        model then stores as a float, and the join against an integer key fails
        downstream. Resolving keys as identifiers first is what stops that.
        """
        for table in self._tables.values():
            for name in _as_key_tuple(table.primary_key):
                if name in table.columns:
                    table.columns[name].stype = 'ID'
        for link in self.relationships:
            source = self._tables.get(link.source_table)
            if source is not None and link.source_column in source.columns:
                source.columns[link.source_column].stype = 'ID'

    def __repr__(self) -> str:
        return (
            f'Schema({len(self._tables)} tables: {sorted(self._tables)}, '
            f'{len(self.relationships)} relationships)'
        )


def _require_table(tables: Mapping[str, Table], name: str) -> Table:
    if name not in tables:
        raise StructuredDataError(
            f'no table {name!r} in this schema; it has {sorted(tables)}',
            code='INVALID_REQUEST',
        )
    return tables[name]


def _require_column(
    tables: Mapping[str, Table], table: str, column: str
) -> str:
    if column not in _require_table(tables, table).columns:
        raise StructuredDataError(
            f'table {table!r} has no column {column!r}; it has '
            f'{sorted(tables[table].columns)}',
            code='INVALID_REQUEST',
        )
    return column


def _as_key_tuple(key: str | Sequence[str] | None) -> tuple[str, ...]:
    if key is None:
        return ()
    if isinstance(key, str):
        return (key,)
    return tuple(key)


def _infer_table(name: str, frame: pd.DataFrame) -> Table:
    r"""Describe a frame as a table, inferring the types, key and time column.

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
        # worked before, including any relationship pointing at this table --
        # but the engine drops these rows without a word, so the guess is
        # said out loud.
        missing = int(frame[inferred].isna().sum())
        warnings.warn(
            f'{name}.{inferred} identifies a row, but {missing} row(s) have '
            f'no value for it. Those rows cannot be scored and the model '
            f'will not see them; fill or drop them, or name a different '
            f'primary key.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
        object.__setattr__(table, 'primary_key', inferred)
        table._adopt_key(inferred)
        table.time_column = _infer_time_column(columns)
        return table
    # Assigning the key is what marks its column, counts what the uniqueness
    # threshold tolerated and says so; doing any of it a second time here would
    # report the same duplicate twice, from inside this module.
    table.primary_key = inferred
    table.time_column = _infer_time_column(columns)
    return table


def _copy_table(table: Table) -> Table:
    r"""The same table again, field for field, judging nothing.

    Built without the constructor and without the checked setters, for two
    reasons. Inference deliberately accepts a key with a null in it -- it warns
    and goes on, because refusing would break frames that worked -- and
    assignment refuses one, so copying through the setter turned a schema
    `from_data` had accepted into a request that could not be materialized.
    And `Table.__post_init__` walks every cell of every column, which on a
    warehouse extract is a second full scan added to every single request.
    """
    copied = Table.__new__(Table)
    object.__setattr__(copied, 'name', table.name)
    object.__setattr__(copied, 'data', table.data)
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
    allowed to be inconsistent between two assignments -- that is what
    `check_consistent` exists to ask about later -- so copying one must not be
    the moment its storage and meaning are held against each other.
    """
    made = Column.__new__(Column)
    for field_name in ('name', 'dtype', 'stype', 'untyped', 'stype_declared'):
        object.__setattr__(made, field_name, getattr(column, field_name))
    return made


def _column(series: pd.Series, name: str) -> Column:
    r"""One column, with the record of whether inference had anything to go on.

    `untyped` is set after construction because it is not an init field: a
    caller must not be able to build a column that names a dtype and claims to
    have none, which is the state the adapter resolves by discarding the dtype.

    The values are read once and the answer passed down. `infer_tfm_dtype` on
    an object column walks every cell, and asking it three times -- for the
    dtype, again inside the stype, and again for `untyped` -- is what made
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
            # A key-shaped name on values that cannot identify a row -- a bool
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
    For one that does not, the count is not a property of the column at all --
    it is a property of how many rows were read, and reading more rows moves
    it. So a request written against `df.head(100)` and then pointed at the
    table describes a column two different ways, with nothing in between to
    say the description changed.

    Which is worth saying only where it can happen, and that takes both of two
    measures. The first is whether the values have run out: a column whose
    distinct count tracks its row count has not shown how many values it has,
    only how many rows were looked at. The second is whether there is anything
    within reach -- two values in four rows have not saturated either, and are
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
    dataset -- a warehouse that hands back `DATE` as `STRING`, which is
    ordinary -- infers to a schema with no time in it, and the caller learns
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
            f'it before inferring -- '
            f'frame[{name!r}] = pd.to_datetime(frame[{name!r}]{how}) -- or '
            f'set schema[{table.name!r}][{name!r}].stype = {"timestamp"!r} '
            f'afterwards.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
        return name
    return None


def _infer_time_column(columns: Mapping[str, Column]) -> str | None:
    stamps = [name for name, c in columns.items() if c.stype == 'timestamp']
    return stamps[0] if len(stamps) == 1 else None


def _warn_table_cannot_be_linked_to(
    tables: Mapping[str, Table], dates: Mapping[str, str]
) -> None:
    r"""Say when a table has no key and something wanted to point at it.

    `_infer_primary_key` requires a key-shaped name, and rightly: uniqueness
    alone would nominate any high-cardinality measurement. But a great many
    real keys are domain codes -- `IATA_CODE`, `SKU`, `ISBN`, `VIN` -- and no
    suffix rule will cover them.

    That is not merely a missing key. `_infer_relationships` can only point an
    edge at a *declared* key, so a table that has none is not a candidate
    target for anything: every relationship into it is lost, the tables are
    dropped from the request, and the model answers from less data than it was
    given with nothing on screen to say so.

    Nominating a key here was tried and withdrawn. Uniqueness within one frame
    does not distinguish a key from a foreign key that happens not to repeat in
    the rows to hand -- a two-row extract of `line_items` has a unique
    `ORDER_NO` -- so nominating turns a child into a parent and invents an edge
    between two facts. Worse, a second table keyed on a name makes that name
    ambiguous, and `_infer_relationships` then declines *every* edge using it:
    the guess does not merely risk a wrong link, it can delete the right ones
    from a schema whose keys the caller declared correctly.

    So the caller is asked instead, and only where the question is real. A
    keyless table no other table names loses no edges. Nor does one whose
    candidate column is already another table's key -- that column is a
    foreign key, naming it here would make the name ambiguous, and ambiguity
    deletes the edges that were working. A warning that talks a caller into
    the failure it was written to prevent is worse than silence.

    `dates` names, per table, a column already reported as a date stored as
    text. A column cannot be both the time column and the primary key, so
    asking for it here would contradict the other warning, and a caller who
    followed both in order would end with neither.

    How many columns could be the key and how many tables they are shared with
    are two counts, and they do not move together: several columns can be
    shared with one table, and one column with several. Agreeing either by the
    other made the sentence disagree with the graph it describes.
    """
    named_elsewhere: dict[str, set[str]] = {}
    keyed_by = set()
    for name, table in tables.items():
        for column in table.columns:
            named_elsewhere.setdefault(column, set()).add(name)
        if isinstance(table.primary_key, str):
            keyed_by.add(table.primary_key)

    for name, table in tables.items():
        if table.primary_key is not None:
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


def _unlinkable_table_reads(
    name: str, candidates: Sequence[str], tables_sharing: int
) -> str:
    r"""What to say about a table nothing can link to.

    Kept apart from the search that found it, because the two are different
    kinds of difficult. Deciding which columns could be the key is about the
    data; saying so is about English, and the sentence has to agree with the
    graph on two counts that move independently -- how many columns could be
    the key, and how many tables they are shared with. Several columns can be
    shared with one table, and one column with several.
    """
    many, shared_by_many = len(candidates) > 1, tables_sharing > 1
    subject, verb = ('they', 'are') if many else ('it', 'is')
    tables = 'tables' if shared_by_many else 'table'
    return (
        f'No column of {name!r} is named like a key, so it has no primary '
        f'key, and nothing can link to a table without one: every '
        f'relationship pointing at {name!r} is dropped and it reaches the '
        f'model unjoined. '
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


def _warn_two_ways_to_reach_one_table(schema: Schema) -> None:
    r"""Say when one table reaches another by more than one column.

    Both are real relationships and keeping both is right: a sales line names
    the customer it was sold to and the one it ships to, and they are
    different facts. But an aggregation across a pair of tables has to be
    resolved to one column, and a query that crosses an ambiguous pair is
    refused -- at the moment it is asked, which reads as a problem with the
    query rather than with the graph it was written against.

    Said here so that it is read where the cause is. Nothing is dropped: which
    of them a question means is the caller's to decide, and choosing for them
    is choosing what their model sees.
    """
    ways: dict[tuple[str, str], list[str]] = {}
    for link in schema.relationships:
        pair = (link.source_table, link.target_table)
        ways.setdefault(pair, []).append(link.source_column)

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
            f'{kept!r}, drop the others: schema.relationships = [link for link '
            f'in schema.relationships if link.source_column not in '
            f'{sorted(rest)!r}].',
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


def _infer_relationships(tables: Mapping[str, Table]) -> list[Relationship]:
    r"""Link a column to another table's primary key of the same name.

    Deliberately conservative: only an exact name match against a declared
    primary key counts. A wrong link changes what the model is shown, and is
    harder to notice than a missing one, which simply narrows the question.

    Which is why a name more than one table is keyed by links nothing. `ID` on
    four tables identifies a row in each of them and names a parent in none, so
    choosing one would be a guess in the direction this is supposed to avoid. Value overlap cannot break the tie either: a
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


def _require_joinable(tables: Mapping[str, Table], link: Relationship) -> None:
    r"""A relationship has to be one the engine can actually build.

    An edge joins a foreign key to the *primary key* of the table it points at
    -- that is the only shape the graph has -- so a link naming some other
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


def _is_joinable(tables: Mapping[str, Table], link: Relationship) -> bool:
    try:
        _require_joinable(tables, link)
    except StructuredDataError:
        return False
    return True


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


def to_graph_definition(schema: Schema) -> Any:
    r"""This schema as the language's own graph definition.

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
    from predictive_query.schema.graph import (
        ColumnKey,
        ColumnKeyGroup,
        GraphDefinition,
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
    return GraphDefinition(tables=tables, col_groups=groups)


# The wire spells a few types differently from the language.
_WIRE_TO_PQL_DTYPE = {
    'int64': 'int',
    'int32': 'int',
    'float64': 'float',
    'float32': 'float',
    'timestamp[us]': 'time',
    'timestamp[ns]': 'time',
    'timestamp[ms]': 'time',
    'timestamp[s]': 'time',
}
