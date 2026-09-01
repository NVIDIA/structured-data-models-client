# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from typing import cast

import pandas as pd
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.backend.sqlite import Connection
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    ColumnSpecType,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    Table,
)
from sdm_relational_engine.runmode import MissingType
from sdm_relational_engine.utils import quote_ident


class SQLiteTable(Table):
    r"""A table backed by a :class:`sqlite` database.

    Args:
        connection: The connection to a :class:`sqlite` database.
        name: The name of this table.
        source_name: The source name of this table. If set to ``None``,
            ``name`` is being used.
        columns: The selected columns of this table.
        primary_key: The name of the primary key of this table, if it exists.
        time_column: The name of the time column of this table, if it exists.
        end_time_column: The name of the end time column of this table, if it
            exists.
    """

    _SQL_TEXT_TYPE = 'TEXT'
    _SQL_CHR_FUNCTION = 'CHAR'

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
        return cast(DataBackend, DataBackend.SQLITE)

    def _get_source_columns(self) -> list[SourceColumn]:
        source_columns: list[SourceColumn] = []
        with self._connection.cursor() as cursor:
            sql = f'PRAGMA table_info({self._quoted_source_name})'
            cursor.execute(sql)
            columns = cursor.fetchall()

            if len(columns) == 0:
                raise ValueError(
                    f"Table '{self.source_name}' does not exist "
                    f'in the SQLite database'
                )

            unique_keys: set[str] = set()
            sql = f'PRAGMA index_list({self._quoted_source_name})'
            cursor.execute(sql)
            for _, index_name, is_unique, *_ in cursor.fetchall():
                if bool(is_unique):
                    sql = f'PRAGMA index_info({quote_ident(index_name)})'
                    cursor.execute(sql)
                    index = cursor.fetchall()
                    if len(index) == 1:
                        unique_keys.add(index[0][2])

            # Special SQLite case that creates a rowid alias for
            # `INTEGER PRIMARY KEY` annotated columns:
            rowid_candidates = [
                column
                for _, column, dtype, _, _, is_pkey in columns
                if bool(is_pkey) and dtype.strip().upper() == 'INTEGER'
            ]
            if len(rowid_candidates) == 1:
                unique_keys.add(rowid_candidates[0])

            for _, column, dtype, notnull, _, is_pkey in columns:
                source_column = SourceColumn(
                    name=column,
                    dtype=self._to_dtype(dtype),
                    is_primary_key=bool(is_pkey),
                    is_unique_key=column in unique_keys,
                    is_nullable=not bool(is_pkey) and not bool(notnull),
                )
                source_columns.append(source_column)

        return source_columns

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        source_foreign_keys: list[SourceForeignKey] = []
        with self._connection.cursor() as cursor:
            sql = f'PRAGMA foreign_key_list({self._quoted_source_name})'
            cursor.execute(sql)
            rows = cursor.fetchall()
            counts = Counter(row[0] for row in rows)
            for idx, _, dst_table, foreign_key, primary_key, *_ in rows:
                if counts[idx] == 1:
                    source_foreign_key = SourceForeignKey(
                        name=foreign_key,
                        dst_table=dst_table,
                        primary_key=primary_key,
                    )
                    source_foreign_keys.append(source_foreign_key)
        return source_foreign_keys

    def _get_source_sample_df(self) -> pd.DataFrame:
        with self._connection.cursor() as cursor:
            columns = [quote_ident(col) for col in self._source_column_dict]
            sql = (
                f'SELECT {", ".join(columns)} '
                f'FROM {self._quoted_source_name} '
                f'ORDER BY rowid '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            cursor.execute(sql)
            table = cursor.fetch_arrow_table()

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
        return None

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
                f'ORDER BY rowid '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            cursor.execute(sql)
            table = cursor.fetch_arrow_table()

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
        if re.search('INT', dtype):
            return Dtype.int
        if re.search('TEXT|CHAR|CLOB', dtype):
            return Dtype.string
        if re.search('REAL|FLOA|DOUB', dtype):
            return Dtype.float
        return None  # NUMERIC affinity.
