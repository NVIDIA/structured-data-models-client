# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from predictive_query.schema.typing import Dtype, Stype
from sdm_connectors import quote_ident

from sdm_relational_engine.rfm.backend.postgres import Connection
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    ColumnSpecType,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    Table,
)
from sdm_relational_engine.rfm.base.table import MissingType


def _frame(cursor: Any) -> pd.DataFrame:
    rows = cursor.fetchall()
    names = [
        column.name if hasattr(column, 'name') else column[0]
        for column in cursor.description
    ]
    helper_names = {
        '__kumo_batch__': '__KUMO_BATCH__',
        '__kumo_row__': '__KUMO_ROW__',
    }
    names = [helper_names.get(name, name) for name in names]
    # Preserve the Python scalars exactly until `_sanitize_postgres_frame` can
    # apply the declared PostgreSQL types. Letting pandas infer here converts
    # an integer column containing NULL to float64 and irreversibly rounds
    # values above 2**53 before the nullable Int64 conversion can run.
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
    r"""A table backed by a PostgreSQL database."""

    _SQL_TEXT_TYPE = 'TEXT'

    def __init__(
        self,
        connection: Connection,
        name: str,
        source_name: str | None = None,
        schema: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:
        if schema is None:
            with connection.cursor() as cursor:
                cursor.execute('SELECT current_schema()')
                result = cursor.fetchone()
                assert result is not None
                schema = result[0]
        if schema is None:
            raise ValueError(
                f"Unspecified 'schema' for table '{source_name or name}'"
            )

        self._connection = connection
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
        with self._connection.cursor() as cursor:
            cursor.execute(
                'SELECT c.column_name, c.data_type, c.udt_name,\n'
                '       c.numeric_scale, c.is_nullable, c.table_schema,\n'
                '       c.table_name,\n'
                '       pg_catalog.format_type(a.atttypid, a.atttypmod)\n'
                'FROM information_schema.columns c\n'
                'JOIN pg_catalog.pg_namespace n\n'
                '  ON n.nspname = c.table_schema\n'
                'JOIN pg_catalog.pg_class rel\n'
                '  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\n'
                'JOIN pg_catalog.pg_attribute a\n'
                '  ON a.attrelid = rel.oid AND a.attname = c.column_name\n'
                ' AND a.attnum > 0 AND NOT a.attisdropped\n'
                'WHERE c.table_schema = %s AND c.table_name = %s\n'
                'ORDER BY c.ordinal_position',
                (self._schema, self._source_name),
            )
            rows = cursor.fetchall()
            if not rows:
                raise ValueError(
                    f"Table '{self.source_name}' does not exist "
                    f'in the PostgreSQL database'
                )
            self._schema, self._source_name = rows[0][5:7]

            cursor.execute(
                'SELECT tc.constraint_type, tc.constraint_name,\n'
                '       kcu.column_name, kcu.ordinal_position\n'
                'FROM information_schema.table_constraints tc\n'
                'JOIN information_schema.key_column_usage kcu\n'
                '  ON tc.constraint_catalog = kcu.constraint_catalog\n'
                ' AND tc.constraint_schema = kcu.constraint_schema\n'
                ' AND tc.constraint_name = kcu.constraint_name\n'
                'WHERE tc.table_schema = %s AND tc.table_name = %s\n'
                '  AND tc.constraint_type IN '
                "('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
                (self._schema, self._source_name),
            )
            constraints = cursor.fetchall()

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
        with self._connection.cursor() as cursor:
            cursor.execute(
                'SELECT tc.constraint_name, kcu.column_name,\n'
                '       target_kcu.table_schema, target_kcu.table_name,\n'
                '       target_kcu.column_name, kcu.ordinal_position\n'
                'FROM information_schema.table_constraints tc\n'
                'JOIN information_schema.referential_constraints rc\n'
                '  ON tc.constraint_catalog = rc.constraint_catalog\n'
                ' AND tc.constraint_schema = rc.constraint_schema\n'
                ' AND tc.constraint_name = rc.constraint_name\n'
                'JOIN information_schema.key_column_usage kcu\n'
                '  ON tc.constraint_catalog = kcu.constraint_catalog\n'
                ' AND tc.constraint_schema = kcu.constraint_schema\n'
                ' AND tc.constraint_name = kcu.constraint_name\n'
                'JOIN information_schema.key_column_usage target_kcu\n'
                '  ON rc.unique_constraint_catalog = '
                'target_kcu.constraint_catalog\n'
                ' AND rc.unique_constraint_schema = '
                'target_kcu.constraint_schema\n'
                ' AND rc.unique_constraint_name = '
                'target_kcu.constraint_name\n'
                ' AND target_kcu.ordinal_position = '
                'kcu.position_in_unique_constraint\n'
                "WHERE tc.constraint_type = 'FOREIGN KEY'\n"
                '  AND tc.table_schema = %s AND tc.table_name = %s\n'
                'ORDER BY tc.constraint_name, kcu.ordinal_position',
                (self._schema, self._source_name),
            )
            rows = cursor.fetchall()

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

    def _get_source_sample_df(self) -> pd.DataFrame:
        columns = [quote_ident(col) for col in self._source_column_dict]
        with self._connection.cursor() as cursor:
            cursor.execute(
                f'SELECT {", ".join(columns)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            df = _frame(cursor)
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
        with self._connection.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM {self._quoted_source_name}')
            result = cursor.fetchone()
            assert result is not None
            num_rows = result[0]
        if num_rows == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")
        return num_rows

    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        projections = [
            f'{column.expr} AS {quote_ident(column.name)}' for column in columns
        ]
        with self._connection.cursor() as cursor:
            cursor.execute(
                f'SELECT {", ".join(projections)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            df = _frame(cursor)
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
