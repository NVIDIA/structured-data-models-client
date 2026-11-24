# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A caller's own :class:`~sdm_client.Source` over a SQL
engine, as a caller writes one for an engine the client has no connector
for: each request it declares rendered as that engine's SQL, by an
:class:`Engine` naming how the engine runs a query, qualifies a table and
hashes a value. The tests run it over DuckDB in process and over Trino.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pyarrow as pa

from sdm_client import provider as p


@dataclass(frozen=True)
class Engine:
    r"""What differs between engines.

    Attributes:
        query: Runs a query, returning its rows as a pyarrow table.
        table: A table's name as the engine reads it, from its last part.
        hashed: A SQL expression hashing a text expression to an integer.
        schema_name: The schema whose catalog ``describe`` reads.
        catalog: The engine's ``information_schema``, qualified as the
            engine needs it.
    """

    query: Callable[[str], pa.Table]
    table: Callable[[str], str]
    hashed: Callable[[str], str]
    schema_name: str
    catalog: str = 'information_schema'


class SqlSource(p.Source):
    r"""Answers ``describe``, ``scan`` and the optional requests declared,
    leaving the rest to the core's in-memory fallback."""

    def __init__(
        self,
        engine: Engine,
        *,
        filters: Iterable[p.FilterKind] = p.FilterKind,
        optional: Iterable[p.OptionalRequest] = (
            p.OptionalRequest.SAMPLE_ENTITIES,
        ),
        max_keys_per_request: int | None = None,
    ) -> None:
        self._engine = engine
        self._declared = p.Capabilities(
            optional=frozenset(optional),
            filters=frozenset(filters),
            max_keys_per_request=max_keys_per_request,
            fallback=p.Fallback(5_000_000, 1 << 30),
        )
        #: Each request answered, in order.
        self.asked: list[Any] = []
        #: The keys of each lookup asked, in order.
        self.lookup_keys: list[int] = []

    def capabilities(self) -> p.Capabilities:
        return self._declared

    def answer(self, request: Any, expected: pa.Schema | None) -> pa.Table:
        self.asked.append(request)
        if isinstance(request, p.Describe):
            assert expected is not None
            return pa.Table.from_pylist(
                [
                    dict(zip(expected.names, row, strict=True))
                    for row in self._describe(request)
                ],
                schema=expected,
            )
        if isinstance(request, p.Scan):
            limit = '' if request.limit is None else f' LIMIT {request.limit}'
            return self._engine.query(
                f'SELECT {_columns(request.columns)} FROM '
                f'{self._table(request.table)}{self._where(request.filters)}{limit}'
            )
        if isinstance(request, p.SampleEntities):
            key = ", '|', ".join(
                f'CAST({_quoted(k)} AS VARCHAR)' for k in request.key
            )
            order = self._engine.hashed(
                f"concat({key}, '|{request.seed.value}')"
            )
            return self._engine.query(
                f'SELECT {_columns(request.columns)} FROM '
                f'{self._table(request.table)}{self._where(request.filters)} '
                f'ORDER BY {order} LIMIT {request.n}'
            )
        if isinstance(request, p.Lookup):
            return self._lookup(request)
        raise NotImplementedError(request.name)

    def _describe(self, request: p.Describe) -> list[tuple[Any, ...]]:
        schema = _literal(self._engine.schema_name)
        if request.part is p.DescribePart.TABLES:
            rows = self._engine.query(
                f'SELECT table_name FROM {self._engine.catalog}.tables '
                f'WHERE table_schema = {schema} ORDER BY table_name'
            )
            return [(name,) for name in rows.column(0).to_pylist()]
        if request.part is p.DescribePart.COLUMNS:
            table = _literal(request.tables[0].table)
            rows = self._engine.query(
                'SELECT column_name, data_type, is_nullable FROM '
                f'{self._engine.catalog}.columns WHERE table_schema = {schema} '
                f'AND table_name = {table} ORDER BY ordinal_position'
            ).to_pylist()
            return [
                (
                    row['column_name'],
                    row['data_type'],
                    row['is_nullable'] == 'NO',
                    False,
                )
                for row in rows
            ]
        # No keys, foreign keys or indexes are declared.
        return []

    def _lookup(self, request: p.Lookup) -> pa.Table:
        # One row per seed whose key the table holds: its batch, the rows
        # holding the key, and the columns of one of them.
        seeds = request.seeds.reader.read_all()
        self.lookup_keys.append(seeds.num_rows)
        key_columns = (
            [p.SEED_KEY]
            if len(request.key) == 1
            else [f'{p.SEED_KEY}_{i}' for i in range(1, len(request.key) + 1)]
        )
        rows = seeds.to_pylist()
        if not rows:
            values = 'SELECT NULL AS batch WHERE FALSE'
        else:
            values = ' UNION ALL '.join(
                'SELECT '
                + ', '.join(
                    f'{_literal(row[c])} AS {_quoted("k" + str(i))}'
                    for i, c in enumerate(key_columns)
                )
                + f', {_literal(row[p.SEED_BATCH])} AS batch'
                for row in rows
            )
        joined = ' AND '.join(
            f't.{_quoted(k)} = s.{_quoted("k" + str(i))}'
            for i, k in enumerate(request.key)
        )
        picked = ', '.join(
            f'min(t.{_quoted(c)}) AS {_quoted(c)}' for c in request.columns
        )
        query = (
            f'SELECT s.batch AS {_quoted(p.BATCH_COLUMN)}, '
            f'COUNT(*) AS {_quoted(p.MATCHES_COLUMN)}'
            + (f', {picked}' if picked else '')
            + f' FROM ({values}) s JOIN {self._table(request.table)} t ON {joined}'
            ' GROUP BY s.batch ORDER BY s.batch'
        )
        answer = self._engine.query(query)
        batch_type = seeds.schema.field(p.SEED_BATCH).type
        return answer.set_column(
            0, p.BATCH_COLUMN, answer.column(0).cast(batch_type)
        )

    def _table(self, relation: p.Relation) -> str:
        return self._engine.table(relation.table)

    def _where(self, filters: Iterable[Any]) -> str:
        written = [c for c in (self._condition(f) for f in filters) if c]
        return f' WHERE {" AND ".join(written)}' if written else ''

    def _condition(self, filter: Any) -> str | None:
        r"""The filter as SQL; none for one left to the core, which only
        widens the read."""
        if isinstance(filter, p.NotNull):
            return f'{_quoted(filter.column)} IS NOT NULL'
        if isinstance(filter, p.IsNull):
            return f'{_quoted(filter.column)} IS NULL'
        if isinstance(filter, p.Compare):
            return f'{_quoted(filter.column)} {_OPERATORS[filter.op]} {_literal(filter.value.value)}'
        if isinstance(filter, p.In):
            values = ', '.join(_literal(v.value) for v in filter.values)
            return f'{_quoted(filter.column)} IN ({values})'
        if isinstance(filter, p.TimeRange):
            bounds = []
            if filter.lower is not None:
                sign = '>=' if filter.lower.inclusive else '>'
                bounds.append(
                    f'{_quoted(filter.column)} {sign} {_literal(filter.lower.value.value)}'
                )
            if filter.upper is not None:
                sign = '<=' if filter.upper.inclusive else '<'
                bounds.append(
                    f'{_quoted(filter.column)} {sign} {_literal(filter.upper.value.value)}'
                )
            return ' AND '.join(bounds) or None
        if isinstance(filter, p.AllOf):
            parts = [
                c for c in (self._condition(f) for f in filter.filters) if c
            ]
            return f'({" AND ".join(parts)})' if parts else None
        if isinstance(filter, p.AnyOf):
            parts = [self._condition(f) for f in filter.filters]
            # An OR any part of which is not written is not written at all.
            return (
                None
                if not parts or None in parts
                else f'({" OR ".join(parts)})'
            )  # type: ignore[arg-type]
        if isinstance(filter, p.HasRows):
            key = _quoted(filter.key)
            within = [f'{key} IS NOT NULL'] + [
                c for c in (self._condition(f) for f in filter.filters) if c
            ]
            counted = (
                f' GROUP BY {key} HAVING COUNT(*) >= {filter.at_least}'
                if filter.at_least > 1
                else ''
            )
            return (
                f'{_quoted(filter.column)} IN (SELECT {key} FROM '
                f'{self._table(filter.table)} WHERE {" AND ".join(within)}{counted})'
            )
        if isinstance(filter, p.KeyIn):
            rows = filter.keys.reader.read_all().to_pylist()
            if not rows:
                return 'FALSE'
            names = filter.keys.columns
            tuples = ', '.join(
                '(' + ', '.join(_literal(row[n]) for n in names) + ')'
                for row in rows
            )
            return f'({_columns(filter.columns)}) IN ({tuples})'
        return None


_OPERATORS = {
    p.CompareOp.EQ: '=',
    p.CompareOp.NE: '<>',
    p.CompareOp.LT: '<',
    p.CompareOp.LE: '<=',
    p.CompareOp.GT: '>',
    p.CompareOp.GE: '>=',
}


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(names: Iterable[str]) -> str:
    return ', '.join(_quoted(n) for n in names)


def _literal(value: Any) -> str:
    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"CAST('{value!r}' AS DOUBLE)"
    if isinstance(value, Decimal):
        return f"CAST('{value}' AS DECIMAL(38, {max(0, -value.as_tuple().exponent)}))"  # type: ignore[operator]
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return f"TIMESTAMP '{value.strftime('%Y-%m-%d %H:%M:%S.%f')}'"
    if isinstance(value, datetime.date):
        return f"DATE '{value.isoformat()}'"
    return "'" + str(value).replace("'", "''") + "'"
