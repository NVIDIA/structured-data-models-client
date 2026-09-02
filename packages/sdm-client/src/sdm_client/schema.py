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
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sdm_client.errors import StructuredDataError
from sdm_client.wire import UNTYPED, infer_tfm_dtype

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
    """

    name: str
    dtype: str
    stype: str

    def __post_init__(self) -> None:
        self._check_stype(self.stype)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == 'stype':
            self._check_stype(value)
            self._check_supports_dtype(value, getattr(self, 'dtype', None))
        object.__setattr__(self, key, value)

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
        return (
            f'Column({self.name!r}, dtype={self.dtype!r}, stype={self.stype!r})'
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
    duplicate_keys: int = 0

    def __setattr__(self, key: str, value: Any) -> None:
        if key == 'primary_key':
            if value is None:
                # No key, so nothing tolerated and nothing to report.
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
        """
        names = _as_key_tuple(value)
        for name in names:
            # `_check_key` has already refused a column that cannot be one.
            self.columns[name].stype = 'ID'
        frame = getattr(self, 'data', None)
        if frame is None or frame.empty:
            return
        object.__setattr__(
            self, 'duplicate_keys', _duplicate_key_count(frame, names)
        )
        if self.duplicate_keys:
            warnings.warn(
                f'{self.name}.{".".join(names)} identifies a row '
                f'{self.duplicate_keys} time(s) it has already been used, '
                f'e.g. {_first_duplicate(frame, names)!r}. Those rows are '
                f'indistinguishable to the model.',
                stacklevel=3,
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
        duplicates = _duplicate_key_count(frame, names)
        if duplicates > len(frame) * (1 - _KEY_UNIQUENESS):
            example = _first_duplicate(frame, names)
            raise StructuredDataError(
                f'{".".join((self.name, *names))} is not unique enough to '
                f'identify a row: {duplicates} of {len(frame)} rows repeat a '
                f'value already used, e.g. {example!r}',
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
        # A tolerated duplicate changes which row the model is shown, so it
        # belongs where a caller will actually see it.
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

    def __init__(
        self,
        tables: Mapping[str, Table],
        relationships: Sequence[Relationship] = (),
    ) -> None:
        self._tables = dict(tables)
        self.relationships = list(relationships)

    @classmethod
    def from_data(
        cls,
        tables: Mapping[str, pd.DataFrame],
        *,
        primary_keys: Mapping[str, str] | None = None,
        time_columns: Mapping[str, str] | None = None,
        relationships: Sequence[Relationship] | None = None,
    ) -> Schema:
        r"""Infer a schema from frames.

        Args:
            tables: Frames by table name. The name matters: a predictive query
                names columns through it.
            primary_keys: Overrides for the inferred key of a table.
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
            # The setter takes a composite key, so this has to as well, or the
            # two ways of saying the same thing disagree.
            for column in _as_key_tuple(key):
                _require_column(built, name, column)
            table.primary_key = key
        for name, column in (time_columns or {}).items():
            _require_table(built, name).time_column = _require_column(
                built, name, column
            )

        if relationships is None:
            inferred = _infer_relationships(built)
        else:
            inferred = list(relationships)
            for link in inferred:
                _require_column(built, link.source_table, link.source_column)
                _require_column(built, link.target_table, link.target_column)
                _require_joinable(built, link)

        schema = cls(built, inferred)
        schema._mark_key_columns()
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

    columns = {
        column: Column(
            name=column,
            dtype=_dtype_of(frame[column]),
            stype=_stype_of(frame[column]),
        )
        for column in frame.columns
    }
    table = Table(name=name, data=frame, columns=columns)
    # Assigning the key is what marks its column, counts what the uniqueness
    # threshold tolerated and says so; doing any of it a second time here would
    # report the same duplicate twice, from inside this module.
    table.primary_key = _infer_primary_key(frame, columns)
    table.time_column = _infer_time_column(columns)
    return table


def _dtype_of(series: pd.Series) -> str:
    dtype = infer_tfm_dtype(series)
    # An all-null object column carries no type. Calling it a string is the
    # safe fallback: it round-trips whatever arrives without claiming a width.
    return 'string' if dtype == UNTYPED else dtype


def _stype_of(series: pd.Series) -> str:
    dtype = _dtype_of(series)
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


def _infer_time_column(columns: Mapping[str, Column]) -> str | None:
    stamps = [name for name, c in columns.items() if c.stype == 'timestamp']
    return stamps[0] if len(stamps) == 1 else None


def _infer_relationships(tables: Mapping[str, Table]) -> list[Relationship]:
    r"""Link a column to another table's primary key of the same name.

    Deliberately conservative: only an exact name match against a declared
    primary key counts. A wrong link changes what the model is shown, and is
    harder to notice than a missing one, which simply narrows the question.
    """
    keys = {
        table.primary_key: name
        for name, table in tables.items()
        if isinstance(table.primary_key, str)
    }
    links = []
    for name, table in tables.items():
        for column in table.columns:
            target = keys.get(column)
            if target is None or target == name:
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


def _duplicate_key_count(frame: pd.DataFrame, names: Sequence[str]) -> int:
    r"""How many rows repeat a key value another row already used."""
    columns = list(names)
    if any(name not in frame.columns for name in columns):
        return 0
    key = frame[columns[0]] if len(columns) == 1 else frame[columns]
    return int(key.duplicated().sum())


def _first_duplicate(frame: pd.DataFrame, names: Sequence[str]) -> Any:
    columns = list(names)
    key = frame[columns[0]] if len(columns) == 1 else frame[columns]
    repeated = key[key.duplicated()]
    if repeated.empty:
        return None
    first = repeated.iloc[0]
    return first if len(columns) == 1 else tuple(first)


def to_graph_definition(schema: Schema) -> Any:
    r"""This schema as the language's own graph definition.

    The language resolves a query against names and types with no rows
    attached, which is what this hands it. Built here rather than in the query
    module so that a schema has exactly one translation into what PQL reads.
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
