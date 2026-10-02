# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from functools import cached_property
from typing import cast

import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_connectors import quote_ident

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.snow import Connection
from sdm_relational_engine.rfm.backend.snow.binding import paramstyle
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

# Snowflake reports an object it cannot resolve as SQL compilation error
# `002003`, which the connector surfaces as `errno=2003`:
_MISSING_OBJECT_ERRNO = 2003


def _is_missing_object_error(error: Exception) -> bool:
    r"""Whether ``error`` reports an object that Snowflake could not resolve.

    Any other failure - an expired session, a dropped connection, a missing
    privilege on the warehouse - says nothing about whether a table exists.

    Args:
        error: The error raised by the connector.
    """
    if getattr(error, 'errno', None) == _MISSING_OBJECT_ERRNO:
        return True
    return 'does not exist' in str(error).lower()


class SnowTable(Table):
    r"""A table backed by a :class:`sqlite` database.

    Args:
        connection: The connection to a :class:`snowflake` database.
        name: The name of this table.
        source_name: The source name of this table. If set to ``None``,
            ``name`` is being used.
        database: The database.
        schema: The schema.
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
        database: str | None = None,
        schema: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        if database is None or schema is None:
            with connection.cursor() as cursor:
                cursor.execute('SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()')
                result = cursor.fetchone()
                assert result is not None
                database = database or result[0]
                assert database is not None
                schema = schema or result[1]

        if schema is None:
            raise ValueError(
                f"Unspecified 'schema' for table "
                f"'{source_name or name}' in database "
                f"'{database}'"
            )

        self._connection = connection
        self._database = database
        self._schema = schema

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
        names = [self._database, self._schema, self._source_name]
        return '.'.join(names)

    @property
    def _quoted_source_name(self) -> str:
        names = [self._database, self._schema, self._source_name]
        return '.'.join([quote_ident(name) for name in names])

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.SNOWFLAKE)

    def _get_source_columns(self) -> list[SourceColumn]:
        # NOTE Quoted identifiers are case-sensitive in Snowflake, while
        # unquoted ones resolve to their upper-case form. Try the given
        # spelling first, then its folded form, and adopt whichever resolves
        # for all subsequent look-ups:
        names: tuple[str, str, str] = (
            self._database,
            self._schema,
            self._source_name,
        )
        folded: tuple[str, str, str] = (
            names[0].upper(),
            names[1].upper(),
            names[2].upper(),
        )
        candidates = [names] if names == folded else [names, folded]

        source_columns: list[SourceColumn] = []
        with self._connection.cursor() as cursor:
            error: Exception | None = None
            for candidate in candidates:
                quoted = '.'.join(quote_ident(name) for name in candidate)
                try:
                    cursor.execute(f'DESCRIBE TABLE {quoted}')
                except Exception as e:
                    if not _is_missing_object_error(e):
                        raise
                    error = error or e
                    continue
                self._database, self._schema, self._source_name = candidate
                break
            else:
                raise TableNotFoundError(
                    f"Table '{self.source_name}' does not exist "
                    f'in the remote data backend'
                ) from error

            for row in cursor.fetchall():
                column, dtype, _, null, _, is_pkey, is_unique, *_ = row

                source_column = SourceColumn(
                    name=column,
                    dtype=self._to_dtype(dtype),
                    is_primary_key=is_pkey.strip().upper() == 'Y',
                    is_unique_key=is_unique.strip().upper() == 'Y',
                    is_nullable=null.strip().upper() == 'Y',
                )
                source_columns.append(source_column)

        return source_columns

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        source_foreign_keys: list[SourceForeignKey] = []
        with self._connection.cursor() as cursor:
            sql = f'SHOW IMPORTED KEYS IN TABLE {self._quoted_source_name}'
            cursor.execute(sql)
            rows = cursor.fetchall()
            counts = Counter(row[13] for row in rows)
            for row in rows:
                if counts[row[13]] == 1:
                    source_foreign_key = SourceForeignKey(
                        name=row[8],
                        dst_table=f'{row[1]}.{row[2]}.{row[3]}',
                        primary_key=row[4],
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
            table = cursor.fetch_arrow_all(force_return_table=False)

        if table is None:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        return self._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict={
                column.name: column.dtype
                for column in self._source_column_dict.values()
            },
            stype_dict=None,
        )

    def _fetch_rows(self, sql: str) -> list[tuple]:
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            return list(cursor.fetchall())

    def _declared_primary_key(self) -> frozenset[str]:
        return frozenset(
            column.name
            for column in self._source_column_dict.values()
            if column.is_primary_key
        )

    def _enforces_key(self, names: Sequence[str]) -> tuple[bool, bool]:
        primary = self._declared_primary_key()
        # Only a hybrid table enforces its primary key, and every hybrid table
        # declares one, so a table declaring none is not asked about.
        if not primary or not primary <= set(names) or not self._is_hybrid:
            return False, False
        columns = self._source_column_dict
        never_null = all(
            name in columns and not columns[name].is_nullable for name in names
        )
        return True, never_null

    @classmethod
    def _before_counting_keys(
        cls, keys: Sequence[tuple[Table, Sequence[str]]]
    ) -> None:
        # One listing per schema answers for every table in it, rather than a
        # SHOW per table.
        asked: dict[tuple[str, str], list[SnowTable]] = {}
        for table, names in keys:
            if not isinstance(table, SnowTable) or '_is_hybrid' in vars(table):
                continue
            primary = table._declared_primary_key()
            if primary and primary <= set(names):
                schema = (table._database, table._schema)
                asked.setdefault(schema, []).append(table)
        for tables in asked.values():
            hybrid = tables[0]._hybrid_tables()
            for table in tables:
                vars(table)['_is_hybrid'] = table._source_name in hybrid

    @cached_property
    def _is_hybrid(self) -> bool:
        r"""Whether this is a hybrid table, the one kind whose primary key
        Snowflake enforces.
        """
        return self._source_name in self._hybrid_tables()

    def _hybrid_tables(self) -> set[str]:
        r"""The hybrid tables in this table's schema. A schema it cannot list
        is taken to hold none, so every key in it is counted.
        """
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f'SHOW TABLES IN SCHEMA {quote_ident(self._database)}.'
                    f'{quote_ident(self._schema)}'
                )
                fields = [
                    str(field[0]).lower() for field in cursor.description or ()
                ]
                rows = cursor.fetchall()
        except Exception:
            return set()
        if 'name' not in fields or 'is_hybrid' not in fields:
            return set()
        name, hybrid = fields.index('name'), fields.index('is_hybrid')
        return {
            row[name] for row in rows if str(row[hybrid]).strip().upper() == 'Y'
        }

    def _get_num_rows(self) -> int | None:
        with paramstyle(self._connection), self._connection.cursor() as cursor:
            sql = (
                f'SHOW TABLES LIKE ? '
                f'IN SCHEMA {quote_ident(self._database)}.'
                f'{quote_ident(self._schema)}'
            )
            cursor.execute(sql, (self._source_name,))
            result = cursor.fetchone()
            assert result is not None
            num_rows = result[7]

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
            table = cursor.fetch_arrow_all(force_return_table=False)

        if table is None:
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
        if dtype.startswith('NUMBER'):
            try:  # Parse `scale` from 'NUMBER(precision, scale)':
                scale = int(dtype.split(',')[-1].split(')')[0])
                return Dtype.int if scale == 0 else Dtype.float
            except Exception:
                return Dtype.float
        if dtype == 'FLOAT':
            return Dtype.float
        if dtype.startswith('VARCHAR'):
            return Dtype.string
        if dtype.startswith('BINARY'):
            return Dtype.binary
        if dtype == 'BOOLEAN':
            return Dtype.bool
        if dtype.startswith('DATE') or dtype.startswith('TIMESTAMP'):
            return Dtype.date
        if dtype.startswith('TIME'):
            return Dtype.time
        if dtype.startswith('VECTOR'):
            try:  # Parse element data type from 'VECTOR(dtype, dimension)':
                dtype = dtype.split(',')[0].split('(')[1].strip()
                if dtype == 'INT':
                    return Dtype.intlist
                if dtype == 'FLOAT':
                    return Dtype.floatlist
            except Exception:
                pass
            return Dtype.unsupported
        if dtype.startswith('ARRAY'):
            try:  # Parse element data type from 'ARRAY(dtype)':
                dtype = dtype.split('(', maxsplit=1)[1]
                dtype = dtype.rsplit(')', maxsplit=1)[0]
                _dtype = SnowTable._to_dtype(dtype)
                if _dtype is not None and _dtype.is_int():
                    return Dtype.intlist
                if _dtype is not None and _dtype.is_float():
                    return Dtype.floatlist
                if _dtype is not None and _dtype.is_string():
                    return Dtype.stringlist
            except Exception:
                pass
            return Dtype.unsupported
        # Unsupported data types:
        if re.search(
            'DECFLOAT|VARIANT|OBJECT|MAP|FILE|GEOGRAPHY|GEOMETRY',
            dtype,
        ):
            return Dtype.unsupported
        return None
