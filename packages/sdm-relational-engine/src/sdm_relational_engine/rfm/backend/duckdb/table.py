# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from typing import cast

import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_connectors import quote_ident

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.duckdb import Connection
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


class DuckDBTable(Table):
    r"""A table backed by a :class:`duckdb` database.

    Args:
        connection: The connection to a :class:`duckdb` database.
        name: The name of this table.
        source_name: The source name of this table. If set to ``None``,
            ``name`` is being used.
        columns: The selected columns of this table.
        primary_key: The name of the primary key of this table, if it exists.
        time_column: The name of the time column of this table, if it exists.
        end_time_column: The name of the end time column of this table, if it
            exists.
    """

    _SQL_TEXT_TYPE = 'VARCHAR'

    def __init__(
        self,
        connection: Connection,
        name: str,
        source_name: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        self._connection = connection

        super().__init__(
            name=name,
            source_name=source_name,
            columns=columns,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
        )

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.DUCKDB)

    def _get_source_columns(self) -> list[SourceColumn]:
        source_columns: list[SourceColumn] = []
        with self._connection.cursor() as cursor:
            try:
                sql = f'PRAGMA table_info({self._quoted_source_name})'
                cursor.execute(sql)
            except Exception as e:
                raise TableNotFoundError(
                    f"Table '{self.source_name}' does not exist "
                    f'in the DuckDB database'
                ) from e
            columns = cursor.fetchall()

            if len(columns) == 0:
                raise TableNotFoundError(
                    f"Table '{self.source_name}' does not exist "
                    f'in the DuckDB database'
                )

            unique_keys: set[str] = set()
            primary_keys: set[str] = set()
            source_name = quote_ident(self.source_name, char="'")
            sql = (
                'SELECT constraint_type, constraint_column_names\n'
                'FROM duckdb_constraints()\n'
                f'WHERE table_name = {source_name}'
            )
            cursor.execute(sql)
            for constraint_type, column_names in cursor.fetchall():
                if len(column_names) != 1:
                    continue  # No composite keys yet.
                column = column_names[0]
                if constraint_type == 'PRIMARY KEY':
                    primary_keys.add(column)
                    unique_keys.add(column)
                elif constraint_type == 'UNIQUE':
                    unique_keys.add(column)

            for _, column, dtype, notnull, _, is_pkey in columns:
                is_primary_key = bool(is_pkey) or column in primary_keys
                source_column = SourceColumn(
                    name=column,
                    dtype=self._to_dtype(dtype),
                    is_primary_key=is_primary_key,
                    is_unique_key=column in unique_keys,
                    is_nullable=not is_primary_key and not bool(notnull),
                )
                source_columns.append(source_column)

        return source_columns

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        source_foreign_keys: list[SourceForeignKey] = []
        with self._connection.cursor() as cursor:
            source_name = quote_ident(self.source_name, char="'")
            sql = (
                'SELECT constraint_index, referenced_table,\n'
                '       constraint_column_names, referenced_column_names\n'
                'FROM duckdb_constraints()\n'
                f'WHERE table_name = {source_name}\n'
                "  AND constraint_type = 'FOREIGN KEY'"
            )
            cursor.execute(sql)
            rows = cursor.fetchall()
            counts = Counter(row[0] for row in rows)
            for idx, dst_table, foreign_keys, primary_keys in rows:
                if counts[idx] == 1 and len(foreign_keys) == 1:
                    source_foreign_key = SourceForeignKey(
                        name=foreign_keys[0],
                        dst_table=dst_table,
                        primary_key=primary_keys[0],
                    )
                    source_foreign_keys.append(source_foreign_key)
        return source_foreign_keys

    def _get_source_sample_df(self) -> pd.DataFrame:
        with self._connection.cursor() as cursor:
            columns = [quote_ident(col) for col in self._source_column_dict]
            sql = (
                f'SELECT {", ".join(columns)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            cursor.execute(sql)
            table = cursor.to_arrow_table()

        if len(table) == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        return self._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict={
                column.name: column.dtype
                for column in self._source_column_dict.values()
            },
            stype_dict=None,
        )

    def _get_num_rows(self) -> int | None:
        with self._connection.cursor() as cursor:
            sql = f'SELECT COUNT(*) FROM {self._quoted_source_name}'
            cursor.execute(sql)
            num_rows = cursor.fetchone()[0]  # type: ignore

        if num_rows == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        return num_rows

    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        with self._connection.cursor() as cursor:
            projections = [
                f'{column.expr} AS {quote_ident(column.name)}'
                for column in columns
            ]
            sql = (
                f'SELECT {", ".join(projections)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            cursor.execute(sql)
            table = cursor.to_arrow_table()

        if len(table) == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        return self._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict={column.name: column.dtype for column in columns},
            stype_dict=None,
        )

    @staticmethod
    def _to_dtype(dtype: str | None) -> Dtype | None:
        if dtype is None:
            return None
        dtype = dtype.strip().upper()
        # Tested first: DuckDB spells a list as the element type plus '[]', and
        # every scalar test below matches on a substring, so 'INTEGER[]' would
        # otherwise be claimed by the INTEGER branch and typed as a scalar.
        # The sibling Databricks and Snowflake mappings do not have this
        # problem, because 'ARRAY<...>' does not contain its element spelling
        # at the top level.
        if dtype.endswith('[]'):
            element = DuckDBTable._to_dtype(dtype[:-2])
            if element is None:
                return None
            if element.is_int():
                return Dtype.intlist
            if element.is_float():
                return Dtype.floatlist
            if element.is_string():
                return Dtype.stringlist
            return Dtype.unsupported
        if re.search('UTINYINT|USMALLINT|UINTEGER|UBIGINT|UHUGEINT', dtype):
            return Dtype.int
        if re.search('TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT', dtype):
            return Dtype.int
        if dtype.startswith('DECIMAL') or dtype.startswith('NUMERIC'):
            try:  # Parse scale from 'DECIMAL(precision, scale)'.
                scale = int(dtype.split(',')[-1].split(')')[0])
                return Dtype.int if scale == 0 else Dtype.float
            except Exception:
                return Dtype.float
        if re.search('REAL|FLOAT|DOUBLE', dtype):
            return Dtype.float
        if re.search('VARCHAR|CHAR|STRING|TEXT', dtype):
            return Dtype.string
        if dtype == 'BOOLEAN':
            return Dtype.bool
        if dtype.startswith('DATE') or dtype.startswith('TIMESTAMP'):
            return Dtype.date
        if dtype.startswith('TIME'):
            return Dtype.time
        if dtype.startswith('BLOB'):
            return Dtype.binary
        return None
