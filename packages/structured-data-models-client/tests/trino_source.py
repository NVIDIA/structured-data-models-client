# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Your own :class:`~sdm_client.Source` over a query engine: Trino.

A complete example of a source for an engine the client has no built-in
source for, written against the public API only (:mod:`sdm_client` and
:mod:`sdm_client.provider`). It talks to Trino over its HTTP protocol with
the standard library, and turns each typed request the client sends into
one Trino query:

- ``describe``: the tables, columns and namespace, from Trino's
  ``information_schema``; Trino declares no keys, so the schema names them.
- ``scan`` and ``preview``: a ``SELECT`` with the request's filters.
- ``count_rows``: a ``count(*)`` with the request's filters.
- ``sample_entities``: the rows ordered by a hash of each key and the seed,
  so the same seed draws the same rows.
- ``lookup``: the seeds' key set joined to the table, one row per seed.

Every other request (``neighbors`` among them) is answered by the client
in memory, from scans, within the declared fallback. :meth:`failure` reads
Trino's error names into what they mean to the client, such as a table
that does not exist.

>>> source = TrinoSource('http://localhost:8080', 'memory', 'shop')
>>> schema = Schema.from_source(source, primary_keys={'users': 'id'})

The Java version is ``java/src/test/java/com/nvidia/sdm/examples/
TrinoSource.java``; ``test_trino_source.py`` runs this one against a Trino
server where ``SDM_TRINO_TEST_URL`` names one.
"""

from __future__ import annotations

import base64
import datetime
import json
import math
import urllib.request
from collections.abc import Iterable
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa

from sdm_client import Source
from sdm_client.provider import (
    BATCH_COLUMN,
    MATCHES_COLUMN,
    SEED_BATCH,
    SEED_KEY,
    AllOf,
    AnyOf,
    Capabilities,
    Compare,
    CompareOp,
    CountRows,
    DataRequest,
    Describe,
    DescribePart,
    FailureFacts,
    FailureReason,
    Fallback,
    Filter,
    FilterKind,
    HasRows,
    In,
    IsNull,
    KeyIn,
    KeySet,
    Lookup,
    NotNull,
    OptionalRequest,
    Preview,
    ProviderFailure,
    Relation,
    SampleEntities,
    Scan,
    TimeRange,
    Unusable,
    Value,
)

__all__ = ['TrinoError', 'TrinoSource']


class TrinoError(Exception):
    r"""A query Trino failed, with the error it named.

    Attributes:
        name: Trino's error name, such as ``TABLE_NOT_FOUND``.
        code: Trino's error code.
        kind: Trino's error type, such as ``USER_ERROR``.
    """

    def __init__(self, message: str, name: str, code: int, kind: str) -> None:
        super().__init__(message)
        self.name, self.code, self.kind = name, code, kind


class TrinoSource(Source):
    r"""The tables of one Trino catalog and schema.

    Args:
        url: The Trino server, such as ``http://localhost:8080``.
        catalog: The catalog the tables are in.
        schema: The schema the tables are in.
        user: The user Trino runs the queries as.
        filters: The value filters to have Trino apply; by default every
            kind. The client applies every value filter itself as well, so
            leaving a kind out only widens the reads.
        max_keys_per_request: The most seeds one ``lookup`` carries; more
            reach the source as several lookups.
    """

    def __init__(
        self,
        url: str,
        catalog: str,
        schema: str,
        *,
        user: str = 'sdm',
        filters: Iterable[FilterKind] = FilterKind,
        max_keys_per_request: int = 1_000,
    ) -> None:
        self._url = url.rstrip('/')
        self._catalog = catalog
        self._schema = schema
        self._user = user
        self._capabilities = Capabilities(
            optional=frozenset(
                {
                    OptionalRequest.PREVIEW,
                    OptionalRequest.COUNT_ROWS,
                    OptionalRequest.SAMPLE_ENTITIES,
                    OptionalRequest.LOOKUP,
                }
            ),
            filters=frozenset(filters),
            max_keys_per_request=max_keys_per_request,
            # The client answers every other request from scans, within
            # these limits.
            fallback=Fallback(max_rows=10_000_000, max_bytes=2**31),
            namespace=(catalog, schema),
        )

    # --- the two methods every source implements ---------------------------

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def answer(
        self, request: DataRequest, expected: pa.Schema | None
    ) -> pa.Table:
        if isinstance(request, Describe):
            assert expected is not None  # a describe answer's shape is fixed
            return pa.Table.from_pylist(
                self._describe(request), schema=expected
            )
        if isinstance(request, Scan):
            limit = '' if request.limit is None else f' LIMIT {request.limit}'
            return self._query(
                f'SELECT {_names(request.columns)} FROM {self._table(request.table)}'
                f'{self._where(request.filters)}{limit}'
            )
        if isinstance(request, Preview):
            return self._query(
                f'SELECT {_names(request.columns)} FROM {self._table(request.table)} '
                f'LIMIT {request.limit}'
            )
        if isinstance(request, CountRows):
            # A count is exact, so every filter must be written.
            return self._query(
                f'SELECT count(*) AS "rows" FROM {self._table(request.table)}'
                f'{self._where(request.filters, exact=True)}'
            )
        if isinstance(request, SampleEntities):
            key = ", '|', ".join(
                f'CAST({_quoted(k)} AS VARCHAR)' for k in request.key
            )
            seed = _literal(Value(str(request.seed.value), pa.string()))
            return self._query(
                f'SELECT {_names(request.columns)} FROM {self._table(request.table)}'
                f'{self._where(request.filters)} '
                f"ORDER BY xxhash64(to_utf8(concat({key}, '|', {seed}))) "
                f'LIMIT {request.n}'
            )
        if isinstance(request, Lookup):
            return self._lookup(request)
        # The client sends no request this source did not declare.
        raise NotImplementedError(request.name)

    def failure(self, error: Exception) -> ProviderFailure | None:
        r"""What a Trino error means to the client; ``None`` for any other
        failure, whose message the client keeps as it is."""
        if not isinstance(error, TrinoError):
            return None
        return ProviderFailure(
            vendor_code=str(error.code),
            error_class=error.name,
            message=str(error),
            facts=_FACTS.get(error.name),
        )

    # --- describe ------------------------------------------------------------

    def _describe(self, request: Describe) -> list[dict[str, Any]]:
        in_schema = f'table_schema = {_text(self._schema)}'
        catalog = f'{_quoted(self._catalog)}.information_schema'
        if request.part is DescribePart.TABLES:
            names = [relation.table for relation in request.tables]
            only = (
                f' AND table_name IN ({", ".join(map(_text, names))})'
                if names
                else ''
            )
            rows = self._rows(
                f'SELECT table_name FROM {catalog}.tables WHERE {in_schema}{only} '
                'ORDER BY table_name'
            )
            return [{'table': name} for (name,) in rows]
        if request.part is DescribePart.COLUMNS:
            rows = self._rows(
                f'SELECT column_name, data_type, is_nullable FROM {catalog}.columns '
                f'WHERE {in_schema} AND table_name = {_text(request.tables[0].table)} '
                'ORDER BY ordinal_position'
            )
            return [
                {
                    'name': name,
                    'type': kind,
                    'not_null': nullable == 'NO',
                    'primary_key': False,
                }
                for name, kind, nullable in rows
            ]
        if request.part is DescribePart.NAMESPACE:
            return [{'name': self._catalog}, {'name': self._schema}]
        # Trino declares no keys, foreign keys or indexes: an empty answer.
        return []

    # --- lookup ----------------------------------------------------------------

    def _lookup(self, request: Lookup) -> pa.Table:
        r"""One row per seed whose key the table holds: the seed's batch
        position, the rows holding the key, and the columns of one of them."""
        seeds = request.seeds.reader.read_all()
        width = len(request.key)
        # The seed key set's columns: the key's (key, or key_1, key_2, ...)
        # and each seed's batch position.
        keys = (
            [SEED_KEY]
            if width == 1
            else [f'{SEED_KEY}_{i}' for i in range(1, width + 1)]
        )
        # A VALUES list of no row is no SQL: no seed, an empty seed table.
        rows = (
            ', '.join(
                '(' + ', '.join(map(_literal, row)) + ')'
                for row in _key_rows(request.seeds, [*keys, SEED_BATCH], seeds)
            )
            or '(' + ', '.join(['NULL'] * (width + 1)) + ')'
        )
        seeded = ', '.join(f'k{i}' for i in range(width))
        on = ' AND '.join(
            f't.{_quoted(k)} = s.k{i}' for i, k in enumerate(request.key)
        )
        picked = ''.join(
            f', arbitrary(t.{_quoted(c)}) AS {_quoted(c)}'
            for c in request.columns
        )
        answer = self._query(
            f'SELECT s.batch AS {_quoted(BATCH_COLUMN)}, count(*) AS {_quoted(MATCHES_COLUMN)}'
            f'{picked} FROM (VALUES {rows}) AS s({seeded}, batch) '
            f'JOIN {self._table(request.table)} AS t ON {on} GROUP BY s.batch ORDER BY s.batch'
        )
        # The batch column is of the seed key set's batch type.
        batch = seeds.schema.field(SEED_BATCH).type
        return answer.set_column(0, BATCH_COLUMN, answer.column(0).cast(batch))

    # --- filters ---------------------------------------------------------------

    def _where(self, filters: Iterable[Filter], *, exact: bool = False) -> str:
        written = []
        for each in filters:
            condition = self._condition(each)
            if condition is None and exact:
                raise NotImplementedError(
                    f'an exact read with {type(each).__name__}'
                )
            if condition is not None:
                written.append(condition)
        return f' WHERE {" AND ".join(written)}' if written else ''

    def _condition(self, filter: Filter) -> str | None:
        r"""``filter`` in Trino's SQL; ``None`` for a value filter left to
        the client, which only widens the read."""
        if isinstance(filter, KeyIn):
            return self._key_in(filter)
        if isinstance(filter, NotNull):
            return f'{_quoted(filter.column)} IS NOT NULL'
        if isinstance(filter, TimeRange):
            bounds = [
                f'{_quoted(filter.column)} {op if bound.inclusive else op[0]} {_literal(bound.value)}'
                for bound, op in ((filter.lower, '>='), (filter.upper, '<='))
                if bound is not None
            ]
            return ' AND '.join(bounds) or None
        if isinstance(filter, IsNull):
            return f'{_quoted(filter.column)} IS NULL'
        if isinstance(filter, Compare):
            return f'{_quoted(filter.column)} {_OPERATORS[filter.op]} {_literal(filter.value)}'
        if isinstance(filter, In):
            if not filter.values:
                return 'FALSE'
            return f'{_quoted(filter.column)} IN ({", ".join(map(_literal, filter.values))})'
        if isinstance(filter, AllOf):
            parts = [
                c for c in map(self._condition, filter.filters) if c is not None
            ]
            return f'({" AND ".join(parts)})' if parts else None
        if isinstance(filter, AnyOf):
            parts = [self._condition(each) for each in filter.filters]
            # An OR with any part left out is left out whole.
            if not parts or any(part is None for part in parts):
                return None
            return f'({" OR ".join(parts)})'  # type: ignore[arg-type]
        if isinstance(filter, HasRows):
            key = _quoted(filter.key)
            within = [f'{key} IS NOT NULL'] + [
                c for c in map(self._condition, filter.filters) if c is not None
            ]
            counted = (
                f' GROUP BY {key} HAVING count(*) >= {filter.at_least}'
                if filter.at_least > 1
                else ''
            )
            return (
                f'{_quoted(filter.column)} IN (SELECT {key} FROM {self._table(filter.table)} '
                f'WHERE {" AND ".join(within)}{counted})'
            )
        return None

    def _key_in(self, filter: KeyIn) -> str:
        r"""The row's columns equal one of the key set's rows."""
        table = filter.keys.reader.read_all()
        if table.num_rows == 0:
            return 'FALSE'
        rows = list(_key_rows(filter.keys, filter.keys.columns, table))
        if len(filter.columns) == 1:
            return f'{_quoted(filter.columns[0])} IN ({", ".join(_literal(v) for (v,) in rows)})'
        tuples = ', '.join(
            '(' + ', '.join(map(_literal, row)) + ')' for row in rows
        )
        return f'({_names(filter.columns)}) IN (VALUES {tuples})'

    # --- running a query -------------------------------------------------------

    def _table(self, relation: Relation) -> str:
        r"""A table by its name's parts; one part is a table of this
        source's catalog and schema."""
        parts = (
            relation.parts
            if len(relation.parts) > 1
            else (self._catalog, self._schema, *relation.parts)
        )
        return '.'.join(map(_quoted, parts))

    def _query(self, sql: str) -> pa.Table:
        r"""The query's rows as Arrow, typed by the columns Trino says it
        returns."""
        columns, rows = self._run(sql)
        fields = [pa.field(c['name'], _arrow_type(c['type'])) for c in columns]
        arrays = [
            pa.array([_arrow_value(row[at], f.type) for row in rows], f.type)
            for at, f in enumerate(fields)
        ]
        return pa.Table.from_arrays(arrays, schema=pa.schema(fields))

    def _rows(self, sql: str) -> list[list[Any]]:
        return self._run(sql)[1]

    def _run(self, sql: str) -> tuple[list[dict[str, Any]], list[list[Any]]]:
        r"""Trino's client protocol: the statement posted, then its results
        followed page by page."""
        headers = {'X-Trino-User': self._user}
        request = urllib.request.Request(
            f'{self._url}/v1/statement', data=sql.encode(), headers=headers
        )
        columns: list[dict[str, Any]] = []
        rows: list[list[Any]] = []
        while True:
            with urllib.request.urlopen(request) as response:
                page = json.load(response)
            if 'error' in page:
                error = page['error']
                raise TrinoError(
                    error['message'],
                    error['errorName'],
                    error['errorCode'],
                    error['errorType'],
                )
            columns = page.get('columns', columns)
            rows += page.get('data', [])
            if 'nextUri' not in page:
                return columns, rows
            request = urllib.request.Request(page['nextUri'], headers=headers)


# --- what Trino's errors mean ------------------------------------------------

_MISSING = FailureFacts(
    not_found=True,
    unusable=Unusable.MISSING,
    reason=FailureReason.TABLE_UNRESOLVED,
)
_FACTS = {
    'TABLE_NOT_FOUND': _MISSING,
    'SCHEMA_NOT_FOUND': _MISSING,
    'CATALOG_NOT_FOUND': _MISSING,
    'COLUMN_NOT_FOUND': FailureFacts(reason=FailureReason.COLUMN_UNRESOLVED),
    'PERMISSION_DENIED': FailureFacts(
        denied=True,
        unusable=Unusable.FORBIDDEN,
        reason=FailureReason.PERMISSION_DENIED,
    ),
    'SYNTAX_ERROR': FailureFacts(reason=FailureReason.SYNTAX_REJECTED),
    'TYPE_MISMATCH': FailureFacts(reason=FailureReason.TYPES_INCOMPATIBLE),
    'DIVISION_BY_ZERO': FailureFacts(reason=FailureReason.DIVISION_BY_ZERO),
    'NUMERIC_VALUE_OUT_OF_RANGE': FailureFacts(
        reason=FailureReason.OUT_OF_RANGE
    ),
    'INVALID_CAST_ARGUMENT': FailureFacts(
        reason=FailureReason.VALUE_NOT_CONVERTED
    ),
}


# --- Trino's types and values as Arrow's -----------------------------------------


def _arrow_type(trino: str) -> pa.DataType:
    if trino.startswith('decimal('):
        precision, scale = trino[len('decimal(') : -1].split(',')
        return pa.decimal128(int(precision), int(scale))
    if trino.startswith('timestamp') and 'with time zone' in trino:
        return pa.timestamp('us', tz='UTC')
    if trino.startswith('timestamp'):
        return pa.timestamp('us')
    if trino.startswith(('varchar', 'char')):
        return pa.string()
    if trino not in _SIMPLE_TYPES:
        raise NotImplementedError(f'a Trino column of type {trino}')
    return _SIMPLE_TYPES[trino]


_SIMPLE_TYPES = {
    'boolean': pa.bool_(),
    'tinyint': pa.int8(),
    'smallint': pa.int16(),
    'integer': pa.int32(),
    'bigint': pa.int64(),
    'real': pa.float32(),
    'double': pa.float64(),
    'date': pa.date32(),
    'varbinary': pa.binary(),
}


def _arrow_value(value: Any, kind: pa.DataType) -> Any:
    r"""A value as Trino's protocol writes it in JSON, as Arrow's."""
    if value is None:
        return None
    if pa.types.is_floating(kind) and isinstance(value, str):
        return float(value)  # 'NaN', 'Infinity' and '-Infinity'
    if pa.types.is_decimal(kind):
        return Decimal(value)
    if pa.types.is_date(kind):
        return datetime.date.fromisoformat(value)
    if pa.types.is_timestamp(kind) and kind.tz is not None:
        # '2024-01-01 10:00:00.000000 Europe/Paris': its instant, in UTC.
        wall, zone = value.rsplit(' ', 1)
        moment = datetime.datetime.fromisoformat(wall).replace(
            tzinfo=ZoneInfo(zone)
        )
        return moment.astimezone(datetime.timezone.utc)
    if pa.types.is_timestamp(kind):
        return datetime.datetime.fromisoformat(value)
    if pa.types.is_binary(kind):
        return base64.b64decode(value)
    return value


def _key_rows(
    keys: KeySet, columns: Iterable[str], table: pa.Table
) -> Iterable[tuple[Value, ...]]:
    r"""Each row of a key set as typed values, its columns in order."""
    names = list(columns)
    types = [keys.schema.field(name).type for name in names]
    for row in table.select(names).to_pylist():
        yield tuple(
            Value(row[name], kind)
            for name, kind in zip(names, types, strict=True)
        )


# --- SQL text ----------------------------------------------------------------------

_OPERATORS = {
    CompareOp.EQ: '=',
    CompareOp.NE: '<>',
    CompareOp.LT: '<',
    CompareOp.LE: '<=',
    CompareOp.GT: '>',
    CompareOp.GE: '>=',
}


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _names(names: Iterable[str]) -> str:
    return ', '.join(map(_quoted, names))


def _text(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _literal(given: Value) -> str:
    r"""A request's value as a Trino literal of its type: the type the
    client holds it as says how to write it exactly."""
    value, kind = given.value, given.type
    if value is None:
        return 'NULL'
    if pa.types.is_boolean(kind):
        return 'TRUE' if value else 'FALSE'
    if pa.types.is_integer(kind):
        return (
            f'CAST({value} AS DECIMAL(20, 0))'
            if value > 2**63 - 1
            else str(int(value))
        )
    if pa.types.is_floating(kind):
        name = 'REAL' if pa.types.is_float32(kind) else 'DOUBLE'
        if math.isnan(value):
            return f'CAST(nan() AS {name})'
        if math.isinf(value):
            return f'CAST({"-" if value < 0 else ""}infinity() AS {name})'
        return f"{name} '{value!r}'"
    if pa.types.is_decimal(kind):
        return f"CAST('{value}' AS DECIMAL({kind.precision}, {kind.scale}))"
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return _text(value)
    if pa.types.is_binary(kind):
        return f"X'{bytes(value).hex()}'"
    if pa.types.is_date(kind):
        return f"DATE '{value.isoformat()}'"
    if pa.types.is_timestamp(kind):
        if kind.tz is not None:
            # An instant, compared as the UTC wall clock Trino stores.
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if pa.types.is_time(kind):
        return f"TIME '{value.isoformat()}'"
    raise NotImplementedError(f'a value of type {kind}')
