# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_connectors import quote_ident

from sdm_relational_engine.exceptions import (
    MetricViewAsTableError,
    TableNotFoundError,
)
from sdm_relational_engine.rfm.backend.databricks import AnyConnection
from sdm_relational_engine.rfm.backend.databricks.metric_view import (
    SOURCE_ALIAS,
)
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
from sdm_relational_engine.rfm.view_expr import _SESSION_WORDS

# Databricks quotes SQL identifiers with backticks rather than double quotes:
BACKTICK = '`'


class DatabricksTable(Table):
    r"""A table backed by a :class:`databricks` SQL warehouse.

    Args:
        connection: The connection to a :class:`databricks` SQL warehouse.
        name: The name of this table.
        source_name: The source name of this table. If set to ``None``,
            ``name`` is being used.
        catalog: The Unity Catalog catalog.
        schema: The schema.
        columns: The selected columns of this table.
        primary_key: The name of the primary key of this table, if it exists.
        time_column: The name of the time column of this table, if it exists.
        end_time_column: The name of the end time column of this table, if it
            exists.
        row_filter: A condition on this table's columns, such as a metric
            view's ``filter``. Every read of the table's rows sees only the
            rows it holds for. Columns may be qualified by ``source``.
    """

    _SQL_TEXT_TYPE = 'STRING'

    def _quote_key_column(self, name: str) -> str:
        return quote_ident(name, BACKTICK)

    def __init__(
        self,
        connection: AnyConnection,
        name: str,
        source_name: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
        row_filter: str | None = None,
    ) -> None:

        if catalog is None or schema is None:
            with connection.cursor() as cursor:
                cursor.execute('SELECT current_catalog(), current_schema()')
                result = cursor.fetchone()
                assert result is not None
                catalog = catalog or result[0]
                assert catalog is not None
                schema = schema or result[1]

        if schema is None:
            raise ValueError(
                f"Unspecified 'schema' for table "
                f"'{source_name or name}' in catalog "
                f"'{catalog}'"
            )

        self._connection = connection
        self._catalog = catalog
        self._schema = schema
        self._row_filter = row_filter
        # Columns named otherwise than the source spells them, by name.
        self._source_names: dict[str, str] = {}

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
        names = [self._catalog, self._schema, self._source_name]
        return '.'.join(names)

    @property
    def row_filter(self) -> str | None:
        r"""The condition every read of this table's rows is limited to.

        The setter replaces it, and raises :class:`ValueError` once a row has
        been read, since what was read and counted then would hold rows the
        new condition leaves out.
        """
        return self._row_filter

    @row_filter.setter
    def row_filter(self, row_filter: str | None) -> None:
        if self._rows_read:
            raise ValueError(
                f"Cannot change the row filter of table '{self.name}' since "
                f'its rows have already been read'
            )
        self._row_filter = row_filter

    @property
    def _rows_read(self) -> bool:
        return (
            '_source_sample_df' in vars(self)
            or '_num_rows' in vars(self)
            or len(self._expr_sample_df.columns) > 0
        )

    def _name_columns(self, names: Mapping[str, str]) -> None:
        r"""Names source columns otherwise, as a metric view spells them.

        Databricks matches a name to a column without regard to case, so a
        view may call `SalesOrderID` `SALESORDERID`, and a question about the
        view uses the view's spelling. Every read then names the column so.

        Args:
            names: Each column as the source spells it, and its new name.

        Raises:
            ValueError: If a row has already been read, under the old names.
        """
        if self._rows_read:
            raise ValueError(
                f"Cannot rename the columns of table '{self.name}' since its "
                f'rows have already been read'
            )
        columns: dict[str, SourceColumn] = {}
        for name, column in self._source_column_dict.items():
            new = names.get(name, name)
            if new != name:
                self._source_names[new] = self._source_names.pop(name, name)
            columns[new] = replace(column, name=new)
        self.__dict__['_source_column_dict'] = columns

    @property
    def _quoted_source_name(self) -> str:
        r"""What every read of this table's rows names in its ``FROM`` or
        ``JOIN``: the table, or the rows of it its filter holds for.
        """
        names = [self._catalog, self._schema, self._source_name]
        name = '.'.join([quote_ident(name, BACKTICK) for name in names])
        # Databricks reads a column named like a session word, `user` say,
        # quoted or not, as the session value once the column is renamed or
        # dropped, rather than failing. Whatever was decided on the columns
        # read here, such as which words of a view's expression are columns,
        # would then silently stop holding. So once they are read, each is
        # named through the table, and a missing one fails the statement.
        # A column renamed is renamed the same way.
        read = self.__dict__.get('_source_column_dict', {})
        columns = '*'
        if self._source_names or any(
            col.lower() in _SESSION_WORDS for col in read
        ):
            columns = ', '.join(self._projection(col) for col in read)
        elif self._row_filter is None:
            return name
        where = ''
        if self._row_filter is not None:
            where = f' WHERE ({self._row_filter})'
        # Left unaliased, since Databricks refuses `(...) AS x TABLESAMPLE`.
        return f'(SELECT {columns} FROM {name} AS {SOURCE_ALIAS}{where})'

    def _projection(self, column: str) -> str:
        source = self._source_names.get(column, column)
        projection = f'{SOURCE_ALIAS}.{quote_ident(source, BACKTICK)}'
        if source != column:
            projection += f' AS {quote_ident(column, BACKTICK)}'
        return projection

    @property
    def _quoted_catalog(self) -> str:
        return quote_ident(self._catalog, BACKTICK)

    @property
    def backend(self) -> DataBackend:
        return cast(DataBackend, DataBackend.DATABRICKS)

    def _get_source_columns(self) -> list[SourceColumn]:
        r"""Read this table's columns and key constraints from Unity Catalog.

        Databricks resolves identifiers case-insensitively, while
        ``information_schema`` compares them as case-sensitive string literals.
        Unity Catalog stores ``table_schema`` and ``table_name`` folded to
        lower case, so folding the *literal* matches whatever spelling the
        caller used, and the canonical names read back here are adopted for
        every subsequent look-up.

        Folding the column with ``lower(table_schema) = lower(:schema)`` is
        logically equivalent but not sargable. It cannot be pushed down, so
        the point look-up scans every column of every table in the catalog. On
        a 27k-column catalog, this took 31 seconds per table instead of 0.5
        seconds, and it grows with catalog size.
        """
        with self._connection.cursor() as cursor:
            # The relation's type rides along so that a metric view is named
            # as one here, rather than failing on its first read of rows.
            sql = (
                f'SELECT c.column_name, c.full_data_type, c.is_nullable,\n'
                f'       c.table_catalog, c.table_schema, c.table_name,\n'
                f'       t.table_type\n'
                f'FROM {self._quoted_catalog}.information_schema.columns c\n'
                f'JOIN {self._quoted_catalog}.information_schema.tables t\n'
                f'  ON t.table_catalog = c.table_catalog\n'
                f' AND t.table_schema = c.table_schema\n'
                f' AND t.table_name = c.table_name\n'
                f'WHERE c.table_schema = lower(?)\n'
                f'  AND c.table_name = lower(?)\n'
                f'ORDER BY c.ordinal_position'
            )
            cursor.execute(sql, parameters=[self._schema, self._source_name])
            rows = cursor.fetchall()

            if len(rows) == 0:
                raise TableNotFoundError(
                    f"Table '{self.source_name}' does not exist "
                    f'in the remote data backend'
                )

            if rows[0][6] == 'METRIC_VIEW':
                raise MetricViewAsTableError(
                    f"'{self.source_name}' is a metric view, not a table. "
                    f'Its measures are aggregations with no value per row, '
                    f'so it cannot be read as a table. Load it with '
                    f"Graph.from_databricks_metric_view('{self._source_name}', "
                    f"connection, catalog='{self._catalog}', "
                    f"schema='{self._schema}') instead.",
                    name=self._source_name,
                )

            self._catalog, self._schema, self._source_name = rows[0][3:6]

            # Primary key and unique key constraints are informational only and
            # exposed via the Unity Catalog `information_schema`. They may be
            # absent (older runtime or missing privileges), in which case we
            # silently fall back to heuristic key inference:
            primary_keys: set[str] = set()
            unique_keys: set[str] = set()
            try:
                sql = (
                    f'SELECT tc.constraint_type, tc.constraint_name,\n'
                    f'       kcu.column_name\n'
                    f'FROM {self._quoted_catalog}.information_schema.'
                    f'table_constraints tc\n'
                    f'JOIN {self._quoted_catalog}.information_schema.'
                    f'key_column_usage kcu\n'
                    f'  ON tc.constraint_catalog = kcu.constraint_catalog\n'
                    f' AND tc.constraint_schema = kcu.constraint_schema\n'
                    f' AND tc.constraint_name = kcu.constraint_name\n'
                    f'WHERE tc.table_schema = lower(?)\n'
                    f'  AND tc.table_name = lower(?)\n'
                    f"  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')"
                )
                cursor.execute(
                    sql, parameters=[self._schema, self._source_name]
                )
                constraint_rows = cursor.fetchall()
                # Only consider single-column keys (no composite support yet).
                # Count columns per constraint (by name), so that two distinct
                # single-column UNIQUE constraints are both kept rather than
                # discarded as if they were one composite key:
                counts = Counter(name for _, name, _ in constraint_rows)
                for constraint_type, name, column in constraint_rows:
                    if counts[name] != 1:
                        continue
                    if constraint_type.strip().upper() == 'PRIMARY KEY':
                        primary_keys.add(column)
                    unique_keys.add(column)
            except Exception:
                pass

        source_columns: list[SourceColumn] = []
        for column, dtype, is_nullable, *_ in rows:
            source_column = SourceColumn(
                name=column,
                dtype=self._to_dtype(dtype),
                is_primary_key=column in primary_keys,
                is_unique_key=column in unique_keys,
                is_nullable=is_nullable.strip().upper() != 'NO',
            )
            source_columns.append(source_column)

        return source_columns

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        source_foreign_keys: list[SourceForeignKey] = []
        with self._connection.cursor() as cursor:
            try:
                sql = (
                    f'SELECT tc.constraint_name, kcu.column_name,\n'
                    f'       ccu.table_catalog, ccu.table_schema,\n'
                    f'       ccu.table_name, ccu.column_name\n'
                    f'FROM {self._quoted_catalog}.information_schema.'
                    f'table_constraints tc\n'
                    f'JOIN {self._quoted_catalog}.information_schema.'
                    f'referential_constraints rc\n'
                    f'  ON tc.constraint_catalog = rc.constraint_catalog\n'
                    f' AND tc.constraint_schema = rc.constraint_schema\n'
                    f' AND tc.constraint_name = rc.constraint_name\n'
                    f'JOIN {self._quoted_catalog}.information_schema.'
                    f'key_column_usage kcu\n'
                    f'  ON tc.constraint_catalog = kcu.constraint_catalog\n'
                    f' AND tc.constraint_schema = kcu.constraint_schema\n'
                    f' AND tc.constraint_name = kcu.constraint_name\n'
                    f'JOIN {self._quoted_catalog}.information_schema.'
                    f'constraint_column_usage ccu\n'
                    f'  ON rc.unique_constraint_catalog = '
                    f'ccu.constraint_catalog\n'
                    f' AND rc.unique_constraint_schema = '
                    f'ccu.constraint_schema\n'
                    f' AND rc.unique_constraint_name = ccu.constraint_name\n'
                    f"WHERE tc.constraint_type = 'FOREIGN KEY'\n"
                    f'  AND tc.table_schema = lower(?)\n'
                    f'  AND tc.table_name = lower(?)'
                )
                cursor.execute(
                    sql, parameters=[self._schema, self._source_name]
                )
                rows = cursor.fetchall()
            except Exception:
                return []

            # Only keep single-column foreign keys (no composite support yet).
            # A single-column key references a single primary key column, hence
            # yields exactly one row per constraint:
            counts = Counter(row[0] for row in rows)
            for row in rows:
                if counts[row[0]] != 1:
                    continue
                source_foreign_key = SourceForeignKey(
                    name=row[1],
                    dst_table=f'{row[2]}.{row[3]}.{row[4]}',
                    primary_key=row[5],
                )
                source_foreign_keys.append(source_foreign_key)

        return source_foreign_keys

    def _get_source_sample_df(self) -> pd.DataFrame:
        with self._connection.cursor() as cursor:
            columns = [
                quote_ident(col, BACKTICK) for col in self._source_column_dict
            ]
            sql = (
                f'SELECT {", ".join(columns)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            cursor.execute(sql)
            table = cursor.fetchall_arrow()

        if table.num_rows == 0:
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

    def _get_num_rows(self) -> int | None:
        with self._connection.cursor() as cursor:
            sql = f'SELECT COUNT(*) FROM {self._quoted_source_name}'
            cursor.execute(sql)
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
        with self._connection.cursor() as cursor:
            projections = [
                f'{column.expr} AS {quote_ident(column.name, BACKTICK)}'
                for column in columns
            ]
            sql = (
                f'SELECT {", ".join(projections)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            cursor.execute(sql)
            table = cursor.fetchall_arrow()

        if table.num_rows == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        return self._sanitize(
            df=table.to_pandas(types_mapper=pd.ArrowDtype),
            dtype_dict={column.name: column.dtype for column in columns},
            stype_dict=None,
        )

    @staticmethod
    def _to_dtype(dtype: str | None) -> Dtype | None:
        # https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-datatypes
        if dtype is None:
            return None
        dtype = dtype.strip().lower()
        if dtype in {
            'tinyint',
            'byte',
            'smallint',
            'short',
            'int',
            'integer',
            'bigint',
            'long',
        }:
            return Dtype.int
        if dtype in {'float', 'real', 'double'}:
            return Dtype.float
        if dtype.startswith(('decimal', 'numeric', 'dec')):
            try:  # Parse `scale` from 'decimal(precision, scale)':
                inside = dtype[dtype.index('(') + 1 : dtype.index(')')]
                parts = inside.split(',')
                scale = int(parts[1]) if len(parts) > 1 else 0
                return Dtype.int if scale == 0 else Dtype.float
            except Exception:
                return Dtype.float
        if dtype in {'boolean', 'bool'}:
            return Dtype.bool
        if dtype.startswith(('string', 'varchar', 'char')):
            return Dtype.string
        if dtype.startswith('binary'):
            return Dtype.binary
        if dtype == 'date':
            return Dtype.date
        if dtype.startswith('timestamp'):  # TIMESTAMP / _NTZ / _LTZ
            return Dtype.date
        if dtype.startswith('array'):
            try:  # Parse element data type from 'array<dtype>':
                inner = dtype[dtype.index('<') + 1 : dtype.rindex('>')]
                _dtype = DatabricksTable._to_dtype(inner)
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
            'interval|map|struct|variant|object|geography|geometry|void', dtype
        ):
            return Dtype.unsupported
        return None
