# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from functools import cached_property
from types import ModuleType
from typing import ClassVar, cast

import pandas as pd
import pyarrow as pa
from predictive_query.schema.typing import Dtype
from sdm_connectors import DatabricksSource, quote_ident

from sdm_relational_engine.exceptions import (
    MetricViewAsTableError,
    TableNotFoundError,
)
from sdm_relational_engine.rfm.backend.databricks import (
    AnyConnection,
    dialect,
)
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
from sdm_relational_engine.rfm.base.source import as_source
from sdm_relational_engine.rfm.base.table import MissingType
from sdm_relational_engine.rfm.base.utils import sample_order_terms
from sdm_relational_engine.rfm.view_expr import _SESSION_WORDS

# Databricks quotes SQL identifiers with backticks rather than double quotes:
BACKTICK = '`'


class DatabricksTable(Table):
    r"""A table backed by a :class:`databricks` SQL warehouse.

    Args:
        connection: The connection to a :class:`databricks` SQL warehouse,
            a notebook's Spark session held as a
            :class:`~sdm_connectors.backends.spark.SparkConnection`, or a
            :class:`sdm_connectors.DatabricksSource` over either.
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
    _dialect: ClassVar[ModuleType] = dialect

    #: The inference sample takes the rows with the lowest hash, which reads
    #: every file of the columns it reads. From this size, 1 GiB of
    #: compressed files, about ten million rows of a typical table and
    #: seconds of a SQL warehouse's time, it reads a bounded subset first.
    _SAMPLE_BOUND_MIN_BYTES = 1 << 30
    #: How many rows that subset holds: a hundred times the sample, so the
    #: hash still picks the sample's rows from many.
    _SAMPLE_BOUND_ROWS = 100_000

    def _quote_key_column(self, name: str) -> str:
        return quote_ident(name, BACKTICK)

    def __init__(
        self,
        connection: 'AnyConnection | DatabricksSource',
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

        self._source = as_source(connection, DataBackend.DATABRICKS)
        if catalog is None or schema is None:
            result = self._rows(dialect.current_namespace())[0]
            catalog = catalog or result[0]
            assert catalog is not None
            schema = schema or result[1]

        if schema is None:
            raise ValueError(
                f"Unspecified 'schema' for table "
                f"'{source_name or name}' in catalog "
                f"'{catalog}'"
            )

        self._catalog = catalog
        self._schema = schema
        self._row_filter = row_filter

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
        if (
            '_source_sample_df' in vars(self)
            or '_num_rows' in vars(self)
            or len(self._expr_sample_df.columns) > 0
        ):
            raise ValueError(
                f"Cannot change the row filter of table '{self.name}' since "
                f'its rows have already been read'
            )
        self._row_filter = row_filter

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
        read = self.__dict__.get('_source_column_dict', {})
        columns = '*'
        if any(col.lower() in _SESSION_WORDS for col in read):
            columns = ', '.join(
                f'{SOURCE_ALIAS}.{quote_ident(col, BACKTICK)}' for col in read
            )
        elif self._row_filter is None:
            return name
        where = ''
        if self._row_filter is not None:
            where = f' WHERE ({self._row_filter})'
        # Left unaliased, since Databricks refuses `(...) AS x TABLESAMPLE`.
        return f'(SELECT {columns} FROM {name} AS {SOURCE_ALIAS}{where})'

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
        rows = self._rows(
            dialect.columns(
                self._quoted_catalog, self._schema, self._source_name
            )
        )

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
            constraint_rows = self._rows(
                dialect.key_constraints(
                    self._quoted_catalog, self._schema, self._source_name
                )
            )
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
        try:
            rows = self._rows(
                dialect.foreign_keys(
                    self._quoted_catalog, self._schema, self._source_name
                )
            )
        except Exception:
            return []

        # Only keep single-column foreign keys (no composite support yet). A
        # single-column key references a single primary key column, hence
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

    def _sample_order_terms(self) -> list[str]:
        r"""The terms that pick and order the rows types are inferred from.

        ``xxhash64`` of the declared primary key, or of a few scalar columns,
        then every scalar column, as :func:`sample_order_terms` says. Spark
        cannot hash a map, which ``_to_dtype`` reads as unsupported.
        """
        return sample_order_terms(
            list(self._source_column_dict.values()),
            self._source_primary_key,
            lambda name: quote_ident(name, BACKTICK),
            'xxhash64',
        )

    def _read_sample(self, projections: Sequence[str]) -> pa.Table:
        order_terms = self._sample_order_terms()
        table = self._run(
            dialect.sample(
                self._sample_source,
                projections,
                order_terms,
                self._NUM_SAMPLE_ROWS,
            )
        )
        return table.select(range(len(order_terms), table.num_columns))

    @cached_property
    def _sample_source(self) -> str:
        r"""What the inference sample reads, settled once, so every read of
        columns or of expressions reads the same rows.

        Databricks has no sample that reads less of a table and picks the
        same rows every time: ``TABLESAMPLE`` reads every file. A table at
        least :attr:`_SAMPLE_BOUND_MIN_BYTES` large is therefore read through
        a subset of its first :attr:`_SAMPLE_BOUND_ROWS` rows, as Databricks
        reads them, and the sample is hashed out of those. That is fast, and
        the same while the table's files are, but Databricks does not
        guarantee it. A view is read the same way, whatever its size: it has
        none to read, and evaluating all of it can cost far more than reading
        a table. Any other table, and one whose size can't be read, is read
        whole, as before.
        """
        source = self._quoted_source_name
        if self._is_large():
            return dialect.bounded(source, self._SAMPLE_BOUND_ROWS)
        return source

    def _is_large(self) -> bool:
        r"""Whether the inference sample reads a bounded subset of this
        table: whether it is a view, or holds at least
        :attr:`_SAMPLE_BOUND_MIN_BYTES`.

        A Delta table's size is in ``DESCRIBE DETAIL``. Any other says
        whether it is a view, and the size its statistics hold if it was
        analyzed, in ``DESCRIBE TABLE EXTENDED``.
        """
        names = [self._catalog, self._schema, self._source_name]
        name = '.'.join(quote_ident(name, BACKTICK) for name in names)
        size: int | None = None
        try:
            detail = self._run(dialect.describe_detail(name))
            if detail.num_rows > 0 and 'sizeInBytes' in detail.column_names:
                size = detail.column('sizeInBytes')[0].as_py()
        except Exception:
            size = None
        if size is None:
            try:
                described = {
                    str(row[0]).strip(): str(row[1]).strip()
                    for row in self._rows(dialect.describe_extended(name))
                    if len(row) >= 2 and row[0] is not None
                }
            except Exception:
                return False
            if described.get('Type', '').upper() == 'VIEW':
                return True
            found = re.match(r'(\d+) bytes', described.get('Statistics', ''))
            size = None if found is None else int(found.group(1))
        return size is not None and int(size) >= self._SAMPLE_BOUND_MIN_BYTES

    def _get_source_sample_df(self) -> pd.DataFrame:
        table = self._read_sample(
            [quote_ident(col, BACKTICK) for col in self._source_column_dict]
        )

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
        table = self._read_sample(
            [
                f'{column.expr} AS {quote_ident(column.name, BACKTICK)}'
                for column in columns
            ]
        )

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
