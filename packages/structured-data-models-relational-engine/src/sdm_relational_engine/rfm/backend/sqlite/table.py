# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from types import ModuleType
from typing import ClassVar, cast

import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_connectors import SQLiteSource, quote_ident

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.sqlite import Connection, dialect
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


class SQLiteTable(Table):
    r"""A table backed by a :class:`sqlite` database.

    Args:
        connection: The connection to a :class:`sqlite` database, or a
            :class:`sdm_connectors.SQLiteSource` over one.
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
    _dialect: ClassVar[ModuleType] = dialect

    def __init__(
        self,
        connection: Connection | SQLiteSource,
        name: str,
        source_name: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        self._source = as_source(connection, DataBackend.SQLITE)

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
        columns = self._rows(dialect.table_info(self._quoted_source_name))

        if len(columns) == 0:
            raise TableNotFoundError(
                f"Table '{self.source_name}' does not exist "
                f'in the SQLite database'
            )

        unique_keys: set[str] = set()
        constraints: list[frozenset[str]] = []
        indices = self._rows(dialect.index_list(self._quoted_source_name))
        for _, index_name, is_unique, *more in indices:
            partial = len(more) > 1 and bool(more[1])
            if bool(is_unique):
                index = self._rows(dialect.index_info(index_name))
                if len(index) == 1:
                    unique_keys.add(index[0][2])
                # A partial index is unique only over the rows it covers,
                # and an index on an expression names no column.
                names = [row[2] for row in index]
                if not partial and None not in names:
                    constraints.append(frozenset(names))

        # Special SQLite case that creates a rowid alias for
        # `INTEGER PRIMARY KEY` annotated columns:
        rowid_candidates = [
            column
            for _, column, dtype, _, _, is_pkey in columns
            if bool(is_pkey) and dtype.strip().upper() == 'INTEGER'
        ]
        rowid = rowid_candidates[0] if len(rowid_candidates) == 1 else None
        if rowid is not None:
            unique_keys.add(rowid)
            constraints.append(frozenset({rowid}))
        self._unique_constraints = tuple(constraints)

        for _, column, dtype, notnull, _, is_pkey in columns:
            source_column = SourceColumn(
                name=column,
                dtype=self._to_dtype(dtype),
                is_primary_key=bool(is_pkey),
                is_unique_key=column in unique_keys,
                # Only the rowid alias is never null: SQLite lets any
                # other primary key hold one, unless declared NOT NULL.
                is_nullable=column != rowid and not bool(notnull),
            )
            source_columns.append(source_column)

        return source_columns

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        source_foreign_keys: list[SourceForeignKey] = []
        rows = self._rows(dialect.foreign_key_list(self._quoted_source_name))
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
        return None

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
