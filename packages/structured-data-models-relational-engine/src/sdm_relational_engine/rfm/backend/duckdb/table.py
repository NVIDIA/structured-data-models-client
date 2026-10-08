# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from types import ModuleType
from typing import ClassVar, cast

import pandas as pd
import pyarrow as pa
from predictive_query.schema.typing import Dtype
from sdm_connectors import DuckDBSource, quote_ident

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.duckdb import Connection, dialect
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    ColumnSpecType,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    Table,
)
from sdm_relational_engine.rfm.base.source import as_source
from sdm_relational_engine.rfm.base.table import MissingType


class DuckDBTable(Table):
    r"""A table backed by a :class:`duckdb` database.

    Args:
        connection: The connection to a :class:`duckdb` database, or a
            :class:`sdm_connectors.DuckDBSource` over one.
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
    _dialect: ClassVar[ModuleType] = dialect

    def __init__(
        self,
        connection: Connection | DuckDBSource,
        name: str,
        source_name: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        self._source = as_source(connection, DataBackend.DUCKDB)
        # Each source column's type as `PRAGMA table_info` spells it, and the
        # expression columns read as zoned timestamps, for what a `Dtype`
        # does not tell apart: a `TIMESTAMP WITH TIME ZONE` from a naive one.
        self._source_duckdb_type_dict: dict[str, str] = {}
        self._zoned_expr_columns: set[str] = set()

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
        try:
            columns = self._rows(dialect.table_info(self._quoted_source_name))
        except Exception as e:
            raise TableNotFoundError(
                f"Table '{self.source_name}' does not exist "
                f'in the DuckDB database'
            ) from e

        if len(columns) == 0:
            raise TableNotFoundError(
                f"Table '{self.source_name}' does not exist "
                f'in the DuckDB database'
            )

        unique_keys: set[str] = set()
        primary_keys: set[str] = set()
        constraints = self._rows(dialect.key_constraints(self.source_name))
        # DuckDB enforces both, composite or not, so a key covering one needs
        # no counting.
        self._unique_constraints = tuple(
            frozenset(column_names)
            for constraint_type, column_names in constraints
            if constraint_type in ('PRIMARY KEY', 'UNIQUE')
        )
        for constraint_type, column_names in constraints:
            if len(column_names) != 1:
                continue  # No composite keys yet.
            column = column_names[0]
            if constraint_type == 'PRIMARY KEY':
                primary_keys.add(column)
                unique_keys.add(column)
            elif constraint_type == 'UNIQUE':
                unique_keys.add(column)

        for _, column, dtype, notnull, _, is_pkey in columns:
            self._source_duckdb_type_dict[column] = str(dtype).strip().upper()
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
        rows = self._rows(dialect.foreign_keys(self.source_name))
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
        columns = [quote_ident(col) for col in self._source_column_dict]
        table = self._run(
            dialect.first_rows(
                self._quoted_source_name, columns, self._NUM_SAMPLE_ROWS
            )
        )

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
        rows = self._rows(dialect.count_rows(self._quoted_source_name))
        num_rows = rows[0][0]

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
        table = self._run(
            dialect.first_rows(
                self._quoted_source_name, projections, self._NUM_SAMPLE_ROWS
            )
        )

        if len(table) == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        for field in table.schema:
            if pa.types.is_timestamp(field.type) and field.type.tz is not None:
                self._zoned_expr_columns.add(field.name)
            else:
                self._zoned_expr_columns.discard(field.name)

        return self._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict={column.name: column.dtype for column in columns},
            stype_dict=None,
        )

    def _is_zoned_time(self, name: str) -> bool:
        r"""Whether column ``name`` holds ``TIMESTAMP WITH TIME ZONE`` values.

        A source column says so in its type. An expression column says so in
        the type its inference sample was read as; one not yet sampled is
        taken to be naive, as before.
        """
        if name in self._source_column_dict and self[name].is_source:
            spelled = self._source_duckdb_type_dict.get(name, '')
            return spelled in ('TIMESTAMP WITH TIME ZONE', 'TIMESTAMPTZ')
        return name in self._zoned_expr_columns

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
