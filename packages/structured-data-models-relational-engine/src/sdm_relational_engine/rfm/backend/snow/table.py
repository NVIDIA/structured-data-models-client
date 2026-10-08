# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from functools import cached_property
from types import ModuleType
from typing import ClassVar, cast

import pandas as pd
import pyarrow as pa
from predictive_query.schema.typing import Dtype
from sdm_connectors import SnowflakeSource, quote_ident

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.snow import Connection, dialect
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
from sdm_relational_engine.rfm.base.source import rows as rows_of
from sdm_relational_engine.rfm.base.table import MissingType
from sdm_relational_engine.rfm.base.utils import (
    percent_literal,
    prefilter_fraction,
    sample_order_terms,
)

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
    r"""A table backed by a :class:`snowflake` database.

    Args:
        connection: The connection to a :class:`snowflake` database, or a
            :class:`sdm_connectors.SnowflakeSource` over one.
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
    _dialect: ClassVar[ModuleType] = dialect

    #: The rows a block prefilter is meant to keep, at least, for the
    #: inference sample; a seeded draw of ``n`` rows keeps ten times ``n``
    #: if that is more.
    _SAMPLE_PREFILTER_ROWS = 100_000
    #: The micro-partitions a block prefilter keeps, at least, so that a
    #: clustered table's sample still comes from many clusters.
    _SAMPLE_PREFILTER_PARTITIONS = 128
    #: The compressed size a micro-partition is taken to have.
    _MICRO_PARTITION_BYTES = 16 << 20
    #: The kinds of table ``SAMPLE SYSTEM ... SEED`` reads: standard tables,
    #: which keep their rows in micro-partitions.
    _SAMPLED_KINDS = frozenset({'TABLE', 'TRANSIENT', 'TEMPORARY'})
    #: What ``SHOW TABLES`` says of a table that decides whether it is
    #: sampled by its blocks.
    _LISTING_FIELDS = (
        'rows',
        'bytes',
        'kind',
        'is_external',
        'is_hybrid',
        'is_iceberg',
        'is_dynamic',
    )

    def __init__(
        self,
        connection: Connection | SnowflakeSource,
        name: str,
        source_name: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:

        self._source = as_source(connection, DataBackend.SNOWFLAKE)
        if database is None or schema is None:
            result = self._rows(dialect.current_namespace())[0]
            database = database or result[0]
            assert database is not None
            schema = schema or result[1]

        if schema is None:
            raise ValueError(
                f"Unspecified 'schema' for table "
                f"'{source_name or name}' in database "
                f"'{database}'"
            )

        self._database = database
        self._schema = schema
        # Each source column's type as `DESCRIBE TABLE` spells it, for what
        # its `Dtype` does not tell apart, such as `DATE` from `TIMESTAMP`,
        # and the expression columns read as zoned timestamps:
        self._source_snowflake_type_dict: dict[str, str] = {}
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
        error: Exception | None = None
        for candidate in candidates:
            quoted = '.'.join(quote_ident(name) for name in candidate)
            try:
                rows = self._rows(dialect.describe_table(quoted))
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

        for row in rows:
            column, dtype, _, null, _, is_pkey, is_unique, *_ = row
            self._source_snowflake_type_dict[column] = (
                str(dtype).strip().upper()
            )

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
        # `SHOW` takes no `ORDER BY`, and the keys become edges in the order
        # they are listed:
        rows = sorted(
            self._rows(dialect.imported_keys(self._quoted_source_name)),
            key=lambda row: tuple(str(value) for value in row),
        )
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

    def _sample_order_terms(self) -> list[str]:
        r"""The terms that pick and order the rows types are inferred from.

        ``HASH`` of the declared primary key, or of a few scalar columns, then
        every scalar column, as :func:`sample_order_terms` says. ``HASH``
        hashes the typed values, so no session setting changes the rows
        picked.
        """
        return sample_order_terms(
            list(self._source_column_dict.values()),
            self._source_primary_key,
            lambda name: quote_ident(name),
            'HASH',
        )

    def _read_sample(self, projections: Sequence[str]) -> pa.Table | None:
        r"""``projections`` of the rows types are inferred from.

        Every read, of columns or of expressions, reads the same rows: the
        first read settles where they are read from, and the rest reuse it.
        A large standard table reads a seeded sample of its micro-partitions
        first (:meth:`_prefilter`); if that holds fewer rows than a sample
        needs, the table is read whole instead, as is every other table.
        """
        order_terms = self._sample_order_terms()

        def read(source: str) -> pa.Table:
            return self._run(
                dialect.sample(
                    source,
                    projections,
                    order_terms,
                    self._NUM_SAMPLE_ROWS,
                )
            )

        settled = self.__dict__.get('_sample_source')
        if settled is not None:
            table = read(settled)
        else:
            source = self._quoted_source_name
            clause = self._prefilter(self._SAMPLE_PREFILTER_ROWS, seed=0)
            prefiltered = None if clause is None else f'{source} {clause}'
            table = read(prefiltered or source)
            if (
                prefiltered is not None
                and table.num_rows >= self._NUM_SAMPLE_ROWS
            ):
                source = prefiltered
            elif prefiltered is not None:
                table = read(source)
            self.__dict__['_sample_source'] = source
        if table.num_rows == 0:
            return None
        return table.select(range(len(order_terms), table.num_columns))

    def _prefilter(self, rows: int, seed: int) -> str | None:
        r"""The clause that samples this table's micro-partitions so a read
        still sees ``rows`` rows, or ``None`` to read it whole.

        Only a listed standard table is sampled, as Snowflake refuses a seed
        on a view, a subquery or any other kind of table. It keeps the larger
        of ``rows`` of its rows and :attr:`_SAMPLE_PREFILTER_PARTITIONS` of its
        micro-partitions, rounded up to a power of two, and only when that is
        at most an eighth of it; a table of about 16 GiB compressed or less
        is read whole.

        Args:
            rows: The rows a read through the sample needs, at least.
            seed: The sample's seed, kept below 2^31 as Snowflake requires.
        """
        listed = self._listing
        if listed is None:
            return None
        if str(listed['kind']).strip().upper() not in self._SAMPLED_KINDS:
            return None
        flags = ('is_external', 'is_hybrid', 'is_iceberg', 'is_dynamic')
        if any(str(listed[flag]).strip().upper() == 'Y' for flag in flags):
            return None
        try:
            num_rows, num_bytes = int(listed['rows']), int(listed['bytes'])
        except (TypeError, ValueError):
            return None
        if num_rows <= 0:
            return None
        partitions = max(
            Fraction(1), Fraction(num_bytes, self._MICRO_PARTITION_BYTES)
        )
        fraction = prefilter_fraction(
            max(
                Fraction(rows, num_rows),
                Fraction(self._SAMPLE_PREFILTER_PARTITIONS) / partitions,
            )
        )
        if fraction is None:
            return None
        return dialect.block_sample(percent_literal(fraction), seed % (1 << 31))

    def _get_source_sample_df(self) -> pd.DataFrame:
        table = self._read_sample(
            [quote_ident(col) for col in self._source_column_dict]
        )

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
            listed = self._run(
                dialect.tables_in_schema(self._database, self._schema)
            )
        except Exception:
            return set()
        fields = [str(field).lower() for field in listed.column_names]
        rows = rows_of(listed)
        if 'name' not in fields or 'is_hybrid' not in fields:
            return set()
        name, hybrid = fields.index('name'), fields.index('is_hybrid')
        return {
            row[name] for row in rows if str(row[hybrid]).strip().upper() == 'Y'
        }

    @cached_property
    def _listed(self) -> list[dict[str, object]]:
        r"""What ``SHOW TABLES`` lists of the tables named like this one, by
        field, read once.
        """
        listed = self._run(
            dialect.tables_like(self._database, self._schema, self._source_name)
        )
        fields = [str(field).lower() for field in listed.column_names]
        return [dict(zip(fields, row, strict=True)) for row in rows_of(listed)]

    def _named(self) -> list[dict[str, object]]:
        r"""The listing of this table: ``LIKE`` matches case-insensitively,
        and ``_`` matches any character, so ``ORDERS`` also lists ``orders``
        and ``ORDERS_`` lists ``ORDERSX``. The row named exactly is this
        table's, or else the one row named alike but for case.
        """
        rows = self._listed
        named = [row for row in rows if row.get('name') == self._source_name]
        if len(named) == 0:
            folded = self._source_name.upper()
            named = [
                row for row in rows if str(row.get('name')).upper() == folded
            ]
        return named

    @cached_property
    def _listing(self) -> dict[str, object] | None:
        r"""What ``SHOW TABLES`` says of this table that decides whether it
        is sampled by its blocks, or ``None`` where it lists no one table
        (a view, a semantic view, an ambiguous name) or leaves a field out.
        """
        try:
            named = self._named()
        except Exception:
            return None
        if len(named) != 1:
            return None
        row = named[0]
        if any(field not in row for field in self._LISTING_FIELDS):
            return None
        return {field: row[field] for field in self._LISTING_FIELDS}

    def _get_num_rows(self) -> int | None:
        named = self._named()
        if len(named) == 0:
            raise RuntimeError(
                f"Could not count the rows of '{self.source_name}': "
                f'SHOW TABLES does not list it'
            )
        if len(named) > 1:
            names = sorted(str(row.get('name')) for row in named)
            raise RuntimeError(
                f"Could not count the rows of '{self.source_name}': it names "
                f'each of the tables {names}. Spell the name exactly.'
            )
        num_rows = named[0].get('rows')

        if num_rows == 0:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        return cast(int | None, num_rows)

    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        table = self._read_sample(
            [
                f'{column.expr} AS {quote_ident(column.name)}'
                for column in columns
            ]
        )

        if table is None:
            raise RuntimeError(f"Table '{self.source_name}' is empty")

        # The connector reads a `TIMESTAMP_LTZ` or `TIMESTAMP_TZ` as a
        # timestamp in the session's zone, and a `TIMESTAMP_NTZ` as a naive
        # one.
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
