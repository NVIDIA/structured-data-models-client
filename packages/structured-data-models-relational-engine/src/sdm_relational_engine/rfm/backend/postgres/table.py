# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import ipaddress
import json
import re
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from types import ModuleType
from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa
from predictive_query.schema.typing import Dtype, Stype
from sdm_connectors import PostgresSource, quote_ident
from sdm_connectors.backends.postgres import TYPE_OID_KEY

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.postgres import Connection, dialect
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    ColumnSpecType,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    Table,
)
from sdm_relational_engine.rfm.base.source import Statement, as_source
from sdm_relational_engine.rfm.base.table import MissingType
from sdm_relational_engine.rfm.base.utils import (
    percent_literal,
    prefilter_fraction,
)


def _decimals(values: Sequence[Any]) -> list[Decimal | None]:
    return [None if value is None else Decimal(value) for value in values]


# A `timetz` as PostgreSQL writes it, which `datetime.time.fromisoformat`
# reads only from Python 3.11: an hour-only offset (`+00`), a fraction of
# any length (`.5`), an offset with seconds (`+05:53:28`).
_TIMETZ = re.compile(
    r'(\d+):(\d+):(\d+)(?:\.(\d+))?([-+])(\d+)(?::(\d+))?(?::(\d+))?'
)


def _timetz(text: str) -> datetime.time | str:
    r"""A ``timetz`` as psycopg reads one: a time with its zone. One Python
    has no time for, such as ``24:00:00+00``, keeps its text, which psycopg
    refused to read at all.
    """
    match = _TIMETZ.fullmatch(text)
    if match is None:
        return text
    hour, minute, second, fraction, sign, *offset = match.groups()
    hours, minutes, seconds = (int(part or 0) for part in offset)
    shift = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    try:
        return datetime.time(
            int(hour),
            int(minute),
            int(second),
            int((fraction or '').ljust(6, '0')),
            datetime.timezone(-shift if sign == '-' else shift),
        )
    except ValueError:
        return text


def _inet(text: str) -> Any:
    r"""An ``inet`` as psycopg reads one: an address, or an interface where
    it has a prefix.
    """
    if '/' in text:
        return ipaddress.ip_interface(text)
    return ipaddress.ip_address(text)


# How a value is read back from the text the source holds it as, by its
# type's OID: `numeric` as its exact decimal, `json` and `jsonb` as the
# document, `timetz` as a time with its zone, each the value psycopg reads.
_READ_BACK: dict[bytes, Callable[[Any], Any]] = {
    b'1700': Decimal,
    b'1231': _decimals,
    b'114': json.loads,
    b'3802': json.loads,
    b'1266': _timetz,
}

# Read back too where a value is shown rather than sent: `uuid`, `inet` and
# `cidr` as psycopg reads them, so a message shows `UUID('...')`. A frame
# keeps their text, which is what the model is sent.
_READ_BACK_SHOWN: dict[bytes, Callable[[Any], Any]] = {
    **_READ_BACK,
    b'2950': uuid.UUID,
    b'869': _inet,
    b'650': ipaddress.ip_network,
}


def _read_back(
    table: pa.Table, read_back: Mapping[bytes, Callable[[Any], Any]]
) -> list[list[Any]]:
    r"""Each column of a result, as the values psycopg reads."""
    columns: list[list[Any]] = []
    for field, column in zip(table.schema, table.columns, strict=True):
        values = column.to_pylist()
        oid = (field.metadata or {}).get(TYPE_OID_KEY, b'')
        reader = read_back.get(oid)
        if reader is not None:
            values = [None if v is None else reader(v) for v in values]
        columns.append(values)
    return columns


def _rows(table: pa.Table) -> list[tuple[Any, ...]]:
    r"""A result's rows, as the values psycopg reads."""
    return list(zip(*_read_back(table, _READ_BACK_SHOWN), strict=True))


def _frame(table: pa.Table) -> pd.DataFrame:
    r"""A result as an ``object`` frame of the values psycopg reads."""
    helper_names = {
        '__kumo_batch__': '__KUMO_BATCH__',
        '__kumo_row__': '__KUMO_ROW__',
        '__kumo_matches__': '__KUMO_MATCHES__',
    }
    names = [helper_names.get(name, name) for name in table.column_names]
    # Preserve the Python scalars exactly until `_sanitize_postgres_frame` can
    # apply the declared PostgreSQL types. Letting pandas infer here converts
    # an integer column containing NULL to float64 and irreversibly rounds
    # values above 2**53 before the nullable Int64 conversion can run.
    rows = list(zip(*_read_back(table, _READ_BACK), strict=True))
    return pd.DataFrame(rows, columns=names, dtype=object)


def _sanitize_postgres_frame(
    df: pd.DataFrame,
    dtype_dict: Mapping[str, Dtype | None],
    stype_dict: Mapping[str, Stype | None] | None,
) -> pd.DataFrame:
    r"""Preserve nullable PostgreSQL scalar types in pandas.

    A Python ``int`` column containing ``None`` becomes ``float64`` when passed
    directly to ``pd.DataFrame``. That silently changes nullable identifiers
    into model-facing floats (for example ``285.0``) even though PostgreSQL and
    the graph both declare them as integers. Pandas' nullable extension dtypes
    retain the declared wire type without inventing sentinel values.
    """

    def stringify(value: Any) -> str | None:
        if value is None or value is pd.NA:
            return None
        if not isinstance(value, (dict, list)) and bool(pd.isna(value)):
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(',', ':'))
        return str(value)

    for name, dtype in dtype_dict.items():
        if name not in df:
            continue
        if dtype is not None and dtype.is_int():
            df[name] = pd.array(df[name], dtype='Int64')
        elif dtype is not None and dtype.is_float():
            df[name] = pd.array(df[name], dtype='Float64')
        elif dtype == Dtype.bool:
            df[name] = pd.array(df[name], dtype='boolean')
        elif dtype == Dtype.string:
            df[name] = pd.array(
                df[name].map(stringify),
                dtype='string',
            )
    return Table._sanitize(
        df=df,
        dtype_dict=dtype_dict,
        stype_dict=stype_dict,
    )


class PostgresTable(Table):
    r"""A table backed by a PostgreSQL database.

    Args:
        connection: A psycopg connection, or a
            :class:`sdm_connectors.PostgresSource` over one.
    """

    _SQL_TEXT_TYPE = 'TEXT'
    _dialect: ClassVar[ModuleType] = dialect

    #: A table without a primary key is sampled by a hash of every row, which
    #: reads every block. From this many blocks (128 MiB of 8 KiB blocks,
    #: about half a second to read), its inference sample reads a repeatable
    #: sample of its blocks first.
    _SAMPLE_PREFILTER_MIN_BLOCKS = 16_384
    #: How many blocks that sample is meant to keep, at least: 16 MiB, which
    #: holds 100,000 rows or more of any table worth sampling this way, spread
    #: over at least 2,048 blocks picked at random.
    _SAMPLE_PREFILTER_TARGET_BLOCKS = 2_048

    def _rows(self, statement: Statement) -> list[tuple]:
        return _rows(self._run(statement))

    def __init__(
        self,
        connection: Connection | PostgresSource,
        name: str,
        source_name: str | None = None,
        schema: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:
        self._source = as_source(connection, DataBackend.POSTGRES)
        if schema is None:
            schema = self._rows(dialect.current_schema())[0][0]
        if schema is None:
            raise ValueError(
                f"Unspecified 'schema' for table '{source_name or name}'"
            )

        self._schema = schema
        self._source_postgres_type_dict: dict[str, str] = {}
        super().__init__(
            name=name,
            source_name=source_name,
            columns=columns,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
        )

    @property
    def source_name(self) -> str:
        return f'{self._schema}.{self._source_name}'

    @property
    def _quoted_source_name(self) -> str:
        return '.'.join(
            quote_ident(name) for name in (self._schema, self._source_name)
        )

    @property
    def backend(self) -> DataBackend:
        return DataBackend.POSTGRES

    def _get_source_columns(self) -> list[SourceColumn]:
        rows = self._rows(dialect.columns(self._schema, self._source_name))
        if not rows:
            raise TableNotFoundError(
                f"Table '{self.source_name}' does not exist "
                f'in the PostgreSQL database'
            )
        self._schema, self._source_name = rows[0][5:7]

        constraints = self._rows(
            dialect.key_constraints(self._schema, self._source_name)
        )

        grouped_constraints: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for constraint_type, name, column, position in constraints:
            grouped_constraints.setdefault((constraint_type, name), []).append(
                (position, column)
            )
        for (
            constraint_type,
            name,
        ), positioned_columns in grouped_constraints.items():
            if constraint_type != 'PRIMARY KEY' or len(positioned_columns) == 1:
                continue
            columns = [column for _, column in sorted(positioned_columns)]
            raise ValueError(
                f"Composite primary key constraint '{name}' on PostgreSQL "
                f"table '{self.source_name}' uses columns {columns}. "
                'Automatic PostgreSQL graph discovery currently supports '
                'only single-column primary keys.'
            )

        # Postgres enforces both, so a key covering one needs no counting.
        self._unique_constraints = tuple(
            frozenset(column for _, column in positioned_columns)
            for (constraint_type, _), positioned_columns in (
                grouped_constraints.items()
            )
            if constraint_type in ('PRIMARY KEY', 'UNIQUE')
        )
        counts = Counter(name for _, name, *_ in constraints)
        primary_keys: set[str] = set()
        unique_keys: set[str] = set()
        relational_keys: set[str] = set()
        for constraint_type, name, column, _ in constraints:
            if counts[name] != 1:
                continue
            if constraint_type == 'PRIMARY KEY':
                primary_keys.add(column)
            if constraint_type in {'PRIMARY KEY', 'UNIQUE'}:
                unique_keys.add(column)
            relational_keys.add(column)

        self._source_postgres_type_dict = {
            column: postgres_type for column, *_, postgres_type in rows
        }

        return [
            SourceColumn(
                name=column,
                dtype=(
                    Dtype.string
                    if column in relational_keys
                    and dtype.strip().lower() in {'numeric', 'decimal'}
                    else self._to_dtype(dtype, udt_name, numeric_scale)
                ),
                is_primary_key=column in primary_keys,
                is_unique_key=column in unique_keys,
                is_nullable=is_nullable != 'NO',
            )
            for column, dtype, udt_name, numeric_scale, is_nullable, *_ in rows
        ]

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        rows = self._rows(dialect.foreign_keys(self._schema, self._source_name))

        counts = Counter(row[0] for row in rows)
        for constraint_name, count in counts.items():
            if count == 1:
                continue
            constraint_rows = [row for row in rows if row[0] == constraint_name]
            constraint_rows.sort(key=lambda row: row[5])
            columns = [row[1] for row in constraint_rows]
            referenced_columns = [row[4] for row in constraint_rows]
            raise ValueError(
                f"Composite foreign key constraint '{constraint_name}' on "
                f"PostgreSQL table '{self.source_name}' uses columns "
                f'{columns} referencing {referenced_columns}. Automatic '
                'PostgreSQL graph discovery currently supports only '
                'single-column foreign keys.'
            )
        return [
            SourceForeignKey(
                name=row[1],
                dst_table=f'{row[2]}.{row[3]}',
                primary_key=row[4],
            )
            for row in rows
            if counts[row[0]] == 1
        ]

    def _postgres_type(self, column: str) -> str | None:
        r"""Return PostgreSQL's own cast spelling for a source column."""
        return self._source_postgres_type_dict.get(column)

    def _sample_order_terms(self) -> list[str]:
        r"""The terms that pick and order the rows types are inferred from.

        A declared primary key, which PostgreSQL enforces and indexes, so the
        lowest keys are read without scanning the table. A table without one
        is ordered by a hash of each row's binary form, ``record_send``, which
        spreads the rows over the table, and then by that form, which orders
        them totally. The binary form sends text in the ``client_encoding``, so
        sessions with different encodings can pick different rows, but it
        follows no other session setting, such as ``DateStyle``, ``TimeZone``,
        ``IntervalStyle`` or ``extra_float_digits``.

        Only columns of a type this backend reads go into the binary form, as
        each of those types has one. Any other column, of an unknown type or
        one read as unsupported, such as ``aclitem`` or an array of it, may
        have none, so it orders only rows the rest leaves tied, by its text.
        A table with no column of a type this backend reads is ordered by a
        hash of the row's text instead, then by that text, so its sample is
        still spread over the table, though that text follows session
        settings such as ``IntervalStyle``. Hashes and text compare under the
        ``"C"`` collation, so the order doesn't follow the database's.
        """
        key = self._source_primary_key
        if key is not None:
            return [quote_ident(key)]
        known: list[str] = []
        unknown: list[str] = []
        for column in self._source_column_dict.values():
            ref = quote_ident(column.name)
            if column.dtype is None or column.dtype == Dtype.unsupported:
                unknown.append(ref)
            else:
                known.append(ref)
        terms: list[str] = []
        if len(known) > 0:
            binary = f'record_send(ROW({", ".join(known)}))'
            terms += [f'md5({binary}) COLLATE "C"', binary]
        if len(unknown) > 0:
            text = f'CAST(ROW({", ".join(unknown)}) AS text)'
            if len(known) == 0:
                terms.append(f'md5({text}) COLLATE "C"')
            terms.append(f'{text} COLLATE "C"')
        return terms

    def _read_sample(self, projections: Sequence[str]) -> pd.DataFrame:
        r"""``projections`` of the rows types are inferred from.

        Every read, of columns or of expressions, reads the same rows: the
        first read settles where they are read from, and the rest reuse it.
        A large table without a primary key reads a repeatable block sample
        first (:meth:`_prefiltered_sample_source`); if that holds fewer rows
        than a sample needs, the table is read whole instead, as is every
        other table.
        """
        order_terms = self._sample_order_terms()

        def read(source: str) -> pd.DataFrame:
            return _frame(
                self._run(
                    dialect.sample(
                        source,
                        projections,
                        order_terms,
                        self._NUM_SAMPLE_ROWS,
                    )
                )
            )

        settled = self.__dict__.get('_sample_source')
        if settled is not None:
            df = read(settled)
        else:
            source = self._quoted_source_name
            prefiltered = self._prefiltered_sample_source()
            df = read(prefiltered or source)
            if prefiltered is not None and len(df) >= self._NUM_SAMPLE_ROWS:
                source = prefiltered
            elif prefiltered is not None:
                df = read(source)
            self.__dict__['_sample_source'] = source
        return df.iloc[:, len(order_terms) :]

    def _prefiltered_sample_source(self) -> str | None:
        r"""The table with a repeatable sample of its blocks, if it is large
        and has no primary key, or ``None`` to read it whole.

        A keyed table is read in key order through its index, so it stops
        after the first rows and needs none. Any other is ordered by a hash of
        every row, which reads every block; one probe of its size, read from
        its files rather than from statistics, decides. A table of at least
        :attr:`_SAMPLE_PREFILTER_MIN_BLOCKS` blocks keeps a power-of-two
        fraction of them, at least :attr:`_SAMPLE_PREFILTER_TARGET_BLOCKS`,
        and at most an eighth. ``REPEATABLE (0)`` picks the same blocks
        every time the table is stored the same way. A view, a foreign table
        or anything else ``TABLESAMPLE`` refuses is read whole, as is a table
        whose size can't be read, on a server without ``pg_partition_tree``
        (before PostgreSQL 12) or a role that may not ask.
        """
        if self._source_primary_key is not None:
            return None
        try:
            found = self._rows(dialect.relation_size(self._quoted_source_name))
        except Exception:
            return None
        if len(found) == 0:
            return None
        kind, blocks = found[0]
        if str(kind) not in ('r', 'm', 'p') or blocks is None:
            return None
        blocks = int(blocks)
        if blocks < self._SAMPLE_PREFILTER_MIN_BLOCKS:
            return None
        fraction = prefilter_fraction(
            Fraction(self._SAMPLE_PREFILTER_TARGET_BLOCKS, blocks)
        )
        if fraction is None:
            return None
        return dialect.block_sample(
            self._quoted_source_name, percent_literal(fraction)
        )

    def _get_source_sample_df(self) -> pd.DataFrame:
        df = self._read_sample(
            [quote_ident(col) for col in self._source_column_dict]
        )
        if df.empty:
            raise RuntimeError(f"Table '{self.source_name}' is empty")
        return _sanitize_postgres_frame(
            df=df,
            dtype_dict={
                column.name: column.dtype
                for column in self._source_column_dict.values()
            },
            stype_dict=None,
        )

    def _get_num_rows(self) -> int | None:
        num_rows = self._rows(dialect.count_rows(self._quoted_source_name))[0][
            0
        ]
        if num_rows == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")
        return num_rows

    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        df = self._read_sample(
            [
                f'{column.expr} AS {quote_ident(column.name)}'
                for column in columns
            ]
        )
        if df.empty:
            raise RuntimeError(f"Table '{self.source_name}' is empty")
        return _sanitize_postgres_frame(
            df=df,
            dtype_dict={column.name: column.dtype for column in columns},
            stype_dict=None,
        )

    @staticmethod
    def _to_dtype(
        dtype: str | None,
        udt_name: str | None = None,
        numeric_scale: int | None = None,
    ) -> Dtype | None:
        if dtype is None:
            return None
        dtype = dtype.strip().lower()
        udt_name = '' if udt_name is None else udt_name.strip().lower()
        if dtype in {'smallint', 'integer', 'bigint'}:
            return Dtype.int
        if dtype in {'real', 'double precision'}:
            return Dtype.float
        if dtype in {'numeric', 'decimal'}:
            return Dtype.int if numeric_scale == 0 else Dtype.float
        if dtype == 'boolean':
            return Dtype.bool
        if dtype in {
            'character',
            'character varying',
            'text',
            'name',
            'uuid',
            'json',
            'jsonb',
            'xml',
            'inet',
            'cidr',
            'macaddr',
            'macaddr8',
        }:
            return Dtype.string
        if dtype == 'bytea':
            return Dtype.binary
        if dtype == 'date' or dtype.startswith('timestamp'):
            return Dtype.date
        if dtype.startswith('time'):
            return Dtype.time
        if dtype == 'array':
            if udt_name in {'_int2', '_int4', '_int8'}:
                return Dtype.intlist
            if udt_name in {'_float4', '_float8', '_numeric'}:
                return Dtype.floatlist
            if udt_name in {'_text', '_varchar', '_bpchar', '_uuid'}:
                return Dtype.stringlist
            return Dtype.unsupported
        if re.search(
            'interval|range|multirange|composite|record|point|line|polygon',
            dtype,
        ):
            return Dtype.unsupported
        return None
