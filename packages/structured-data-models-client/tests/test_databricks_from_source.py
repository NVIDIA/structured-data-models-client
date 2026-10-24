# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``Schema.from_source`` over a Databricks SQL warehouse, through the
public API on both routes, the warehouse a fake connection answering the
old client's recorded statements (``contract/fixtures/databricks``,
base.binary's): the core route describes the source as the old route does
(T3.4 fix 1, round 1's P1-1 to P1-3).

- A source naming its catalog and schema, and one naming its catalog
  alone, its schema the connection's current one (``DatabricksSource``'s
  own second example; the old client reads ``current_schema()``).
- A warehouse whose key and foreign-key reads fail (an older runtime, or
  missing privileges): the old client carries on with no declared keys
  (DB/table.py:271-303, :318-352), and so does the core.
- The schema's tables keep the names the source holds them under, of one
  part, as the caller's host gives them and takes them back.
"""

from __future__ import annotations

import base64
import copy
import json
import re
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from sdm_connectors import DatabricksSource

from sdm_client import Schema, _routing

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT / 'contract' / 'fixtures' / 'databricks' / 'capture.json'


def _normal(text: str) -> str:
    return ' '.join(text.split())


def _recorded() -> dict[tuple[str, str], pa.Table]:
    r"""base.binary's statements on the warehouse, by their text (each run
    of whitespace one space) and parameters, with the Arrow fetched."""
    capture = json.loads(CAPTURE.read_text(encoding='utf-8'))
    found: dict[tuple[str, str], pa.Table] = {}
    for case in capture['cases']:
        if case['case'] != 'base.binary':
            continue
        for statement in case['statements']:
            data = base64.b64decode(statement['result']['arrow'])
            found[
                (
                    _normal(statement['text']),
                    json.dumps(statement['parameters']),
                )
            ] = pa.ipc.open_stream(pa.py_buffer(data)).read_all()
    return found


class _Cursor:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._result: pa.Table | None = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        return None

    def execute(
        self, operation: str, parameters: Any = None, **_: Any
    ) -> _Cursor:
        self._result = self._connection.answer(
            operation, list(parameters or [])
        )
        return self

    def fetchall_arrow(self) -> pa.Table:
        assert self._result is not None
        return self._result

    def fetchall(self) -> list[tuple[Any, ...]]:
        assert self._result is not None
        # By position: a catalog answer may name two columns alike.
        columns = [column.to_pylist() for column in self._result.columns]
        return list(zip(*columns, strict=True))

    def fetchone(self) -> tuple[Any, ...] | None:
        rows = self.fetchall()
        return rows[0] if rows else None


class _Warehouse:
    r"""A Databricks SQL connection answering base.binary's recorded
    statements, the session's current catalog ``main`` and schema ``shop``;
    with ``fail_constraints``, every key and foreign-key read fails as a
    warehouse without the privilege fails it."""

    def __init__(self, *, fail_constraints: bool = False) -> None:
        self._recorded = _recorded()
        self._fail_constraints = fail_constraints
        self.asked: list[str] = []
        #: Each statement asked with its parameters.
        self.calls: list[tuple[str, list[Any]]] = []
        #: The session's current schema, which a caller may switch.
        self.schema = 'shop'

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def close(self) -> None:
        return None

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        text = _normal(operation)
        self.asked.append(text)
        self.calls.append((text, list(parameters)))
        if (
            self._fail_constraints
            and 'information_schema.table_constraints' in text
        ):
            raise RuntimeError(
                '[INSUFFICIENT_PERMISSIONS] Insufficient privileges SQLSTATE: 42501'
            )
        if text == 'SELECT current_catalog(), current_schema()':
            return pa.table(
                {
                    'current_catalog()': ['main'],
                    'current_schema()': [self.schema],
                }
            )
        if 'ORDER BY p.o' in text:
            # The core's read of the session's namespace: one row per part,
            # each given (bound) or the session's current one.
            given = iter(parameters)
            parts = [
                next(given) if f'SELECT {at} AS o, ? AS n' in text else current
                for at, current in enumerate(['main', self.schema])
            ]
            return pa.table({'name': parts})
        key = (text, json.dumps(parameters))
        if key not in self._recorded:
            raise AssertionError(
                f'no recorded answer for {text!r} {parameters!r}'
            )
        return self._recorded[key]


@pytest.fixture(autouse=True)
def _driver(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    r"""The Databricks SQL connector's modules as stand-ins naming its
    ``Connection`` the fake warehouse, whether or not the connector is
    installed: the old route takes only its driver's connection, which
    these runs never reach (round 2's Codex P2). Every module that read
    the driver is imported anew over the stand-ins, and forgotten after,
    so a later test sees the driver as it is installed."""
    for name in list(sys.modules):
        if 'databricks' in name:
            monkeypatch.delitem(sys.modules, name)
    before = set(sys.modules)
    client = types.ModuleType('databricks.sql.client')
    client.Connection = _Warehouse  # type: ignore[attr-defined]
    sql = types.ModuleType('databricks.sql')
    sql.client = client  # type: ignore[attr-defined]
    root = types.ModuleType('databricks')
    root.sql = sql  # type: ignore[attr-defined]
    for name, module in (
        ('databricks', root),
        ('databricks.sql', sql),
        ('databricks.sql.client', client),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    yield
    for name in set(sys.modules) - before:
        if 'databricks' in name:
            sys.modules.pop(name, None)


#: Recognised as a Databricks SQL connection by its module.
_Connection = type(
    'Connection', (_Warehouse,), {'__module__': 'databricks.sql.client'}
)


def _projection(schema: Schema) -> dict[str, Any]:
    return {
        'tables': {
            name: {
                'source_name': table.source_name,
                'columns': [
                    [c, col.dtype, col.stype]
                    for c, col in table.columns.items()
                ],
                'primary_key': table.primary_key,
                'time_column': table.time_column,
            }
            for name, table in schema.tables.items()
        },
        'links': sorted(
            [r.source_table, r.source_column, r.target_table, r.target_column]
            for r in schema.relationships
        ),
    }


def _both(source: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.FROM_SOURCE_DATABRICKS_SQL)
    )
    on_the_core = Schema.from_source(source())
    monkeypatch.delenv(_routing.ENVIRONMENT)
    on_the_old_route = Schema.from_source(source())
    return on_the_core, on_the_old_route


@pytest.mark.parametrize(
    'namespace', [{'catalog': 'main', 'schema': 'shop'}, {'catalog': 'main'}]
)
def test_a_warehouse_source_is_described_on_the_core_as_on_the_old_route(
    namespace: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    core_schema, old_schema = _both(
        lambda: DatabricksSource(_Connection(), **namespace), monkeypatch
    )
    assert _projection(core_schema) == _projection(old_schema)
    # Each table by the one name the source holds it under, as given.
    assert {t.source_name for t in core_schema.tables.values()} == {
        'orders',
        'users',
    }
    assert core_schema.relationships


def test_failed_key_reads_leave_the_schema_without_declared_keys_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_schema, old_schema = _both(
        lambda: DatabricksSource(
            _Connection(fail_constraints=True), catalog='main', schema='shop'
        ),
        monkeypatch,
    )
    assert _projection(core_schema) == _projection(old_schema)


class _SessionWordWarehouse(_Warehouse):
    r"""base.binary's warehouse whose ``users`` table also has a column
    ``user``, a word Databricks reads as the session user once the column
    is gone; a statement not recorded fails, after it is logged."""

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        found = super().answer(operation, parameters)
        if 'information_schema.columns' in _normal(operation) and parameters[
            -1:
        ] == ['users']:
            extra = {
                name: [found[name][0].as_py()] for name in found.schema.names
            }
            extra['column_name'] = ['user']
            extra['full_data_type'] = ['string']
            found = pa.concat_tables(
                [found, pa.table(extra, schema=found.schema)]
            )
        return found


_SessionWordConnection = type(
    'Connection',
    (_SessionWordWarehouse,),
    {'__module__': 'databricks.sql.client'},
)


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_column_named_like_a_session_word_is_read_through_the_table(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Each read of ``users`` names its columns through the table, so a
    ``user`` column since dropped fails the read rather than reading the
    session user (round 1's Codex P0; ``_quoted_source_name``)."""
    if route == 'core':
        monkeypatch.setenv(
            _routing.ENVIRONMENT, ','.join(_routing.FROM_SOURCE_DATABRICKS_SQL)
        )
    connection = _SessionWordConnection()
    # The preview of ``users`` is not recorded with the column added.
    with pytest.raises(Exception):  # noqa: B017, PT011
        Schema.from_source(
            DatabricksSource(connection, catalog='main', schema='shop')
        )
    previews = [
        text
        for text in connection.asked
        if 'LIMIT 1000' in text and '`users`' in text
    ]
    assert previews, connection.asked
    assert (
        'FROM (SELECT source.`user_id`, source.`age`, source.`country`, '
        'source.`user` FROM `main`.`shop`.`users` AS source)'
    ) in previews[0]


class _DeclaredKeyWarehouse(_Warehouse):
    r"""base.binary's warehouse whose ``orders.user_id`` is named ``buyer``
    and declared a foreign key to ``main.shop.users.user_id``, a link name
    inference would not find; each statement naming ``buyer`` is answered
    as the recorded one naming ``user_id``, the column renamed back."""

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        text = _normal(operation)
        renamed = '`buyer`' in text
        found = super().answer(
            operation.replace('`buyer`', '`user_id`'), parameters
        )
        if renamed:
            return found.rename_columns(
                ['buyer' if c == 'user_id' else c for c in found.column_names]
            )
        if 'information_schema.columns' in text and parameters[-1:] == [
            'orders'
        ]:
            names = [
                'buyer' if c == 'user_id' else c
                for c in found['column_name'].to_pylist()
            ]
            return found.set_column(
                found.schema.get_field_index('column_name'),
                found.schema.field('column_name'),
                pa.array(names, found.schema.field('column_name').type),
            )
        if "constraint_type = 'FOREIGN KEY'" in text and parameters[-1:] == [
            'orders'
        ]:
            row = ['fk_buyer', 'buyer', 'main', 'shop', 'users', 'user_id']
            return pa.Table.from_arrays(
                [
                    pa.array([value], field.type)
                    for value, field in zip(row, found.schema, strict=True)
                ],
                schema=found.schema,
            )
        return found


_DeclaredKeyConnection = type(
    'Connection',
    (_DeclaredKeyWarehouse,),
    {'__module__': 'databricks.sql.client'},
)


@pytest.mark.parametrize(
    'namespace',
    [
        {},
        {'schema': 'shop'},
        {'catalog': 'main'},
        {'catalog': 'main', 'schema': 'shop'},
    ],
    ids=['none', 'schema', 'catalog', 'both'],
)
def test_a_declared_foreign_key_is_linked_whatever_namespace_the_source_names(
    namespace: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A declared foreign key names its table in full (catalog, schema,
    table); the parts the source leaves to the connection are read
    (``current_catalog()``, ``current_schema()``), so the link is kept as
    the old client keeps it (round 2's Claude P0-1)."""
    core_schema, old_schema = _both(
        lambda: DatabricksSource(_DeclaredKeyConnection(), **namespace),
        monkeypatch,
    )
    assert _projection(core_schema) == _projection(old_schema)
    assert ['orders', 'buyer', 'users', 'user_id'] in _projection(core_schema)[
        'links'
    ]


@pytest.mark.parametrize('route', ['core', 'old'])
def test_materialize_after_the_session_moved_reads_where_the_session_says_now(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema described under the session's current schema ``shop`` and
    materialized after the session moved to ``other`` reads ``other``'s
    tables on the core, which keeps no table's location between runs (plan
    B), where the old route reads ``shop``'s, which it kept."""
    _route(route, monkeypatch)
    connection = _SessionConnection()
    schema = Schema.from_source(DatabricksSource(connection, catalog='main'))
    connection.schema = 'other'
    read = _reads(schema, connection)
    assert read, connection.asked
    assert all(where[:2] == _KEPT[route] for where in read), read


def _route(route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if route == 'core':
        monkeypatch.setenv(
            _routing.ENVIRONMENT,
            ','.join(
                _routing.FROM_SOURCE_DATABRICKS_SQL
                + _routing.FROM_SOURCE_DATABRICKS_SQL_TABLES
                + _routing.MATERIALIZE_DATABRICKS_SQL
            ),
        )


_TABLES = ('orders', 'users', 'new_orders')

#: Where a run reads its tables once the session moved from ``shop`` to
#: ``other``, by route: the old route reads a table it kept where it kept
#: it; the core keeps no table's shape or location between runs (the
#: owner's plan B, 2026-10-24) and reads each where the session says now.
_KEPT = {'old': ('main', 'shop'), 'core': ('main', 'other')}


def _reads(schema: Schema, *connections: Any) -> list[tuple[str, str, str]]:
    r"""Where materializing ``schema`` reads each table on ``connections``
    before it stops (not every sampling statement of either route is
    recorded): ``(catalog, schema, table)`` of each catalog read (its
    ``information_schema``'s catalog, the schema and table bound) and each
    read of a table's rows."""
    import pandas as pd

    from sdm_client import PredictiveQuery

    marks = [len(connection.calls) for connection in connections]
    try:
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))
    except Exception:  # noqa: BLE001, S110 - the reads are what is held
        pass
    found = []
    for connection, mark in zip(connections, marks, strict=True):
        for text, parameters in connection.calls[mark:]:
            catalog = re.search(r'`(\w+)`\.information_schema', text)
            if catalog and parameters[-1:] and parameters[-1] in _TABLES:
                found.append((catalog.group(1), parameters[0], parameters[-1]))
                continue
            found.extend(re.findall(r'`(\w+)`\.`(\w+)`\.`(\w+)`', text))
    return [read for read in found if read[2] in _TABLES]


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_described_schema_rebuilt_on_another_catalog_reads_that_catalog(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built anew from a described schema's tables on a source
    naming another catalog reads that catalog's tables, as the old route
    reads a new schema's tables afresh (round 3's P0, Claude's case)."""
    _route(route, monkeypatch)
    connection = _Connection()
    described = Schema.from_source(DatabricksSource(connection))
    rebuilt = Schema(
        described.tables,
        described.relationships,
        source=DatabricksSource(connection, catalog='other'),
    )
    read = _reads(rebuilt, connection)
    assert read, connection.asked
    assert all(where[:2] == ('other', 'shop') for where in read), read


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_described_schema_rebuilt_on_another_connection_reads_its_session(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built anew on another connection, whose session's schema is
    ``other``, reads ``other``'s tables (round 3's P0, Claude's case)."""
    _route(route, monkeypatch)
    described = Schema.from_source(DatabricksSource(_Connection()))
    elsewhere = _Connection()
    elsewhere.schema = 'other'
    rebuilt = Schema(
        described.tables,
        described.relationships,
        source=DatabricksSource(elsewhere),
    )
    read = _reads(rebuilt, elsewhere)
    assert read, elsewhere.asked
    assert all(where[:2] == ('main', 'other') for where in read), read


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_retargeted_after_the_session_moved_is_read_under_the_session(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A described table pointed at another source table after the session
    moved to ``other`` is read under ``other``, as the old route reads a
    table it has not read yet; a table left as described is read where it
    was described (round 3's P0, Codex's case).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _Connection()
    schema = Schema.from_source(DatabricksSource(connection, catalog='main'))
    connection.schema = 'other'
    schema['orders'].source_name = 'new_orders'
    read = _reads(schema, connection)
    retargeted = [where for where in read if where[2] == 'new_orders']
    assert retargeted, read
    assert all(where[:2] == ('main', 'other') for where in retargeted), read
    kept = [where for where in read if where[2] == 'users']
    assert all(where[:2] == _KEPT[route] for where in kept), read


class _SessionWarehouse(_Warehouse):
    r"""base.binary's warehouse holding the same tables in every schema of
    ``main``: a statement naming the session's schema (``other``,
    ``third``) is answered as the recorded one naming ``shop``, the schema
    it names in each answer the session's."""

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        text = _normal(operation)
        if 'current_catalog()' in text or 'ORDER BY p.o' in text:
            return super().answer(operation, parameters)
        asked = (text, list(parameters))
        for part in ('other', 'third'):
            operation = operation.replace(f'`main`.`{part}`.', '`main`.`shop`.')
        moved = ['shop' if p in ('other', 'third') else p for p in parameters]
        try:
            found = super().answer(operation, moved)
        finally:
            self.calls[-1] = asked
        if 'table_schema' in found.column_names:
            at = found.schema.get_field_index('table_schema')
            found = found.set_column(
                at,
                found.schema.field(at),
                pa.array(
                    [self.schema] * found.num_rows, found.schema.field(at).type
                ),
            )
        return found


_SessionConnection = type(
    'Connection', (_SessionWarehouse,), {'__module__': 'databricks.sql.client'}
)


def _described_under(connection: Any) -> set[str]:
    r"""The schemas ``connection``'s catalog reads were bound to."""
    return {
        parameters[0]
        for text, parameters in connection.calls
        if 'information_schema.columns' in text and parameters
    }


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_schema_mixing_two_descriptions_reads_the_session_it_is_built_in(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built from tables described under ``shop`` and under
    ``other`` is a new schema, read under its source's session, ``third``,
    as the old route reads it (round 3's P0; round 4's P2-1)."""
    _route(route, monkeypatch)
    at_shop = _SessionConnection()
    under_shop = Schema.from_source(DatabricksSource(at_shop))
    at_other = _SessionConnection()
    at_other.schema = 'other'
    under_other = Schema.from_source(DatabricksSource(at_other))
    assert _described_under(at_shop) == {'shop'}
    assert _described_under(at_other) == {'other'}
    at_third = _SessionConnection()
    at_third.schema = 'third'
    mixed = Schema(
        {'orders': under_shop['orders'], 'users': under_other['users']},
        under_shop.relationships,
        source=DatabricksSource(at_third),
    )
    read = _reads(mixed, at_third)
    assert read, at_third.asked
    assert all(where[:2] == ('main', 'third') for where in read), read


@pytest.mark.parametrize('connections', ['one', 'two'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_adopted_in_place_is_read_where_the_receiving_schema_read_it(
    route: str, connections: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A table assigned into a schema from another schema described under
    another session is read where the receiving schema read its table of
    that name, as the old route reads it (its own reading of the source,
    by source name): on one connection or two (round 4's P0, Claude's
    case)."""
    _route(route, monkeypatch)
    dev_connection = _SessionConnection()
    dev = Schema.from_source(DatabricksSource(dev_connection))
    if connections == 'one':
        prod_connection = dev_connection
    else:
        prod_connection = _SessionConnection()
    prod_connection.schema = 'other'
    prod = Schema.from_source(DatabricksSource(prod_connection))
    prod.tables['orders'] = dev['orders']
    prod.check_consistent()
    read = _reads(prod, prod_connection)
    assert read, prod_connection.asked
    assert all(where[:2] == ('main', 'other') for where in read), read


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_adopted_from_a_later_reading_keeps_the_receiving_schema_s(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema read under ``shop`` that adopts its ``users`` from a later
    reading of the same source under ``other`` reads ``users`` where it
    read it itself, ``shop``, as the old route keeps its own reading
    (round 4's P0-1, Codex's case).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _SessionConnection()
    source = DatabricksSource(connection, catalog='main')
    schema = Schema.from_source(source)
    connection.schema = 'other'
    elsewhere = Schema.from_source(source)
    schema.tables['users'] = elsewhere['users']
    schema.check_consistent()
    read = _reads(schema, connection)
    assert read, connection.asked
    assert all(where[:2] == _KEPT[route] for where in read), read


@pytest.mark.parametrize('edit', ['relationship', 'stype'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_an_edit_keeps_each_table_where_it_was_read(
    route: str, edit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema read under ``shop`` for ``orders`` alone that takes a
    ``users`` read nowhere yet (a schema built anew), then is edited (a
    relationship assigned, an unrelated column's meaning changed), reads
    ``orders`` under ``shop`` and ``users`` under the session's ``other``,
    as the old route does: an edit spreads no table's reading onto another
    (round 4's P0-2, Codex's case).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    from sdm_client import Relationship

    _route(route, monkeypatch)
    connection = _SessionConnection()
    source = DatabricksSource(connection, catalog='main')
    schema = Schema.from_source(
        source, tables={'orders': 'orders'}, check_keys=False
    )
    connection.schema = 'other'
    users = Schema.from_source(
        source, tables={'users': 'users'}, check_keys=False
    )
    rebuilt = Schema(users.tables, source=source)
    schema.tables['users'] = rebuilt['users']
    schema.check_consistent()
    if edit == 'relationship':
        schema.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id')
        ]
    else:
        schema.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id')
        ]
        schema['orders']['price'].stype = 'numerical'
    read = _reads(schema, connection)
    assert read, connection.asked
    orders = [where for where in read if where[2] == 'orders']
    users_read = [where for where in read if where[2] == 'users']
    assert orders, read
    assert users_read, read
    assert all(where[:2] == _KEPT[route] for where in orders), read
    assert all(where[:2] == ('main', 'other') for where in users_read), read


class _CompletingWarehouse(_SessionWarehouse):
    r"""The session warehouse, which also answers the core's neighbor reads
    the capture does not hold, by running them over the capture's own rows
    in DuckDB (only Spark's JSON-to-table syntax replaced), so a
    ``materialize`` completes on both routes; ``other``'s ``users`` holds a
    country of its own, ``OTHER``, which only a read of it puts in a
    request."""

    def __init__(self, *, fail_constraints: bool = False) -> None:
        import duckdb

        super().__init__(fail_constraints=fail_constraints)
        self._db = duckdb.connect()
        for name in ('users', 'orders'):
            rows = next(
                found
                for (text, _), found in self._recorded.items()
                if f'FROM `main`.`shop`.`{name}` LIMIT 1000' in text
            )
            self._db.register('rows', rows)
            self._db.execute(f'CREATE TABLE {name} AS SELECT * FROM rows')
            self._db.unregister('rows')

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        import pandas as pd

        try:
            found = super().answer(operation, parameters)
        except AssertionError:
            text = _normal(operation)
            if text.startswith('SELECT ') and not parameters:
                # A read the capture does not hold, such as the folded key
                # of a table keyed by several columns, over the same rows.
                text = re.sub(r'`main`\.`\w+`\.', '', text).replace('`', '"')
                found = self._db.execute(text).to_arrow_table()
                self.calls[-1] = (_normal(operation), list(parameters))
                return found
            if not text.startswith('WITH TMP AS ( SELECT pos AS '):
                raise
            keys = json.loads(parameters[0])
            seeds = pd.DataFrame(
                keys
                if keys and isinstance(keys[0], dict)
                else [{'key': key} for key in keys]
            )
            seeds['batch'] = range(len(seeds))
            for column in ('start', 'end', 'time'):
                if column in seeds:
                    seeds[column] = pd.to_datetime(seeds[column])
            self._db.register(
                'seeds', pa.Table.from_pandas(seeds, preserve_index=False)
            )
            text = re.sub(
                r'^WITH TMP AS \( .*? \) SELECT ',
                'WITH TMP AS (SELECT * FROM seeds) SELECT ',
                text,
                count=1,
            )
            text = re.sub(r'`main`\.`\w+`\.', '', text).replace('`', '"')
            found = self._db.execute(text).to_arrow_table()
            self.calls[-1] = (_normal(operation), list(parameters))
        if (
            '`main`.`other`.`users`' in operation
            and 'country' in found.column_names
        ):
            at = found.schema.get_field_index('country')
            found = found.set_column(
                at,
                found.schema.field(at),
                pa.array(
                    ['OTHER'] * found.num_rows, found.schema.field(at).type
                ),
            )
        return found


_CompletingConnection = type(
    'Connection',
    (_CompletingWarehouse,),
    {'__module__': 'databricks.sql.client'},
)


def _materialized(schema: Schema) -> Any:
    import pandas as pd

    from sdm_client import PredictiveQuery

    return PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        schema,
    ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))


def _read_since(connection: Any, mark: int) -> set[tuple[str, str, str]]:
    return {
        where
        for text, _ in connection.calls[mark:]
        for where in re.findall(r'`(\w+)`\.`(\w+)`\.`(\w+)`', text)
    }


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_rebuilt_schema_keeps_where_its_first_materialize_read_its_tables(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built anew is read where the session says at its first
    ``materialize``, and there again at the next after the session moved,
    as the old route keeps the tables its first ``materialize`` read
    (``EngineTables.keep``; round 5's P0).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _CompletingConnection()
    source = DatabricksSource(connection, catalog='main')
    described = Schema.from_source(source)
    schema = Schema(described.tables, described.relationships, source=source)
    mark = len(connection.calls)
    first = _materialized(schema)
    assert {w[:2] for w in _read_since(connection, mark)} == {('main', 'shop')}
    connection.schema = 'other'
    mark = len(connection.calls)
    second = _materialized(schema)
    assert {w[:2] for w in _read_since(connection, mark)} == {_KEPT[route]}
    assert 'OTHER' not in repr(first.payloads)
    assert ('OTHER' in repr(second.payloads)) == (route == 'core')


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_added_to_a_described_schema_keeps_where_it_was_first_read(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A table the described schema never read, added by hand, is read
    where the session says at the first ``materialize`` and there again
    after the session moved (round 5's P0).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    from sdm_client import Relationship

    connection = _CompletingConnection()
    source = DatabricksSource(connection, catalog='main')
    schema = Schema.from_source(
        source, tables={'orders': 'orders'}, check_keys=False
    )
    users = Schema(
        Schema.from_source(
            source, tables={'users': 'users'}, check_keys=False
        ).tables,
        source=source,
    )
    schema.tables['users'] = users['users']
    schema.relationships = [
        Relationship('orders', 'user_id', 'users', 'user_id')
    ]
    _materialized(schema)
    connection.schema = 'other'
    mark = len(connection.calls)
    second = _materialized(schema)
    assert {w[:2] for w in _read_since(connection, mark)} == {_KEPT[route]}
    assert ('OTHER' in repr(second.payloads)) == (route == 'core')


class _FailingOnceWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, whose first neighbor read fails once where
    ``fail_once`` is set, as a transient warehouse failure does."""

    fail_once = False

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        if self.fail_once and 'WITH TMP AS' in operation:
            self.fail_once = False
            raise RuntimeError('a transient warehouse failure')
        return super().answer(operation, parameters)


_FailingOnceConnection = type(
    'Connection',
    (_FailingOnceWarehouse,),
    {'__module__': 'databricks.sql.client'},
)


def _where_read(schema: Schema, connection: Any) -> set[tuple[str, str]]:
    r"""Where a completed ``materialize`` of ``schema`` read its tables,
    asserting the request holds no row of ``other``'s ``users``."""
    mark = len(connection.calls)
    request = _materialized(schema)
    read = {where[:2] for where in _read_since(connection, mark)}
    assert read, connection.calls[mark:]
    if read == {('main', 'shop')}:
        assert 'OTHER' not in repr(request.payloads)
    return read


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_retry_after_a_failure_partway_reads_where_the_failed_run_read(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built anew whose first ``materialize`` fails after reading
    its tables reads them there again when retried after the session moved,
    as the old route keeps the tables a run read before it failed (round
    6's P0, Codex's ``retry``).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _FailingOnceConnection()
    source = DatabricksSource(connection, catalog='main')
    described = Schema.from_source(source)
    schema = Schema(described.tables, described.relationships, source=source)
    connection.fail_once = True
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure
        _materialized(schema)
    assert not connection.fail_once
    connection.schema = 'other'
    assert _where_read(schema, connection) == {_KEPT[route]}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_removed_edited_around_and_restored_is_read_where_it_was(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A described table taken out of the schema, an edit made meanwhile,
    then put back, is read where the schema read it, after the session
    moved, as the old route keeps every table it read (round 6's P0,
    Codex's ``remove_edit_restore``).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _CompletingConnection()
    schema = Schema.from_source(DatabricksSource(connection, catalog='main'))
    links = list(schema.relationships)
    schema.relationships = []
    users = schema.tables.pop('users')
    schema['orders']['price'].stype = 'numerical'
    schema.tables['users'] = users
    schema.relationships = links
    connection.schema = 'other'
    assert _where_read(schema, connection) == {_KEPT[route]}


@pytest.mark.parametrize('first', ['original', 'snapshot'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_snapshot_shares_the_tables_its_original_reads_later(
    route: str, first: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema and a snapshot of it share the tables either reads after
    the snapshot was taken, as the old route shares its engine reading: a
    table the one reads first is read there by the other after the session
    moved, whichever reads first (round 6's P0, Codex's ``snapshot``).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    from sdm_client import Relationship

    _route(route, monkeypatch)
    connection = _CompletingConnection()
    source = DatabricksSource(connection, catalog='main')
    described = Schema.from_source(source)
    schema = Schema.from_source(
        source, tables={'orders': 'orders'}, check_keys=False
    )
    snapshot = schema.snapshot()

    def add_users(target: Schema) -> None:
        target.tables['users'] = described['users']
        target.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id')
        ]

    earlier, later = (
        (schema, snapshot) if first == 'original' else (snapshot, schema)
    )
    add_users(earlier)
    assert _where_read(earlier, connection) == {('main', 'shop')}
    connection.schema = 'other'
    add_users(later)
    assert _where_read(later, connection) == {_KEPT[route]}


class _ArmedWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, with one failure armed: the first statement
    naming ``fail_on`` raises ``failure`` (a transient error, or an
    interrupt); and a table read under schema ``missing`` fails as a table
    the warehouse does not hold."""

    fail_on: str | None = None
    #: The parameter the armed statement binds last, where one is named.
    fail_with: str | None = None
    failure: type[BaseException] = RuntimeError

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        if '`main`.`missing`.' in operation:
            raise RuntimeError('[TABLE_OR_VIEW_NOT_FOUND] SQLSTATE: 42P01')
        if (
            self.fail_on is not None
            and self.fail_on in operation
            and (self.fail_with is None or parameters[-1:] == [self.fail_with])
        ):
            self.fail_on = None
            raise self.failure('an armed failure')
        return super().answer(operation, parameters)


_ArmedConnection = type(
    'Connection', (_ArmedWarehouse,), {'__module__': 'databricks.sql.client'}
)


def _rebuilt(connection: Any) -> Schema:
    r"""A schema built anew from a description of ``connection``'s source
    under ``shop``: it has read none of its tables yet."""
    source = DatabricksSource(connection, catalog='main')
    described = Schema.from_source(source)
    return Schema(described.tables, described.relationships, source=source)


#: The same statement on both routes at each moment of a first run: the
#: catalog read of each table the run reads anew (each route's engine reads
#: a table's catalog before it keeps it), and the first count of rows (each
#: route's sampler counts a table's rows once its tables are kept).
_PHASES = {
    'first_table': ('information_schema.columns', 'orders'),
    'second_table': ('information_schema.columns', 'users'),
    'after_reading': ('SELECT COUNT(*) FROM `main`.`shop`.`orders`', None),
}


@pytest.mark.parametrize(
    ('phase', 'kept'),
    [
        ('first_table', False),
        ('second_table', False),
        ('after_reading', True),
    ],
)
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_retry_after_an_early_failure_reads_where_the_old_route_reads(
    route: str, phase: str, kept: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` failing at the same statement on both routes
    keeps its tables exactly when the old route does: none where it failed
    reading a table's catalog, all once its tables were read and its
    sampler's count failed; a retry after the session moved reads
    ``other`` or ``shop`` accordingly (round 7's and round 8's P0).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    connection.fail_on, connection.fail_with = _PHASES[phase]
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure
        _materialized(schema)
    assert connection.fail_on is None, 'the failure was not reached'
    connection.schema = 'other'
    expected = _KEPT[route] if kept else ('main', 'other')
    assert _where_read(schema, connection) == {expected}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_first_run_in_the_wrong_schema_keeps_nothing_and_the_fixed_retry_runs(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` with the session in a schema that holds no
    such table fails and keeps no reading, so a retry once the session is
    fixed reads ``shop`` (round 7's P0 and P1, Claude's wrong session)."""
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    connection.schema = 'missing'
    with pytest.raises(Exception):  # noqa: B017, PT011 - the table is not held
        _materialized(schema)
    connection.schema = 'shop'
    assert _where_read(schema, connection) == {('main', 'shop')}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_snapshot_after_an_early_failure_shares_the_retry_s_readings(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` failing at its namespace read leaves the
    schema an engine reading of its own, empty, which a snapshot taken then
    shares, so the original's retry's readings reach the snapshot (round 7's
    P0, Codex's early failure).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    connection.fail_on = 'current_schema()'
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure
        _materialized(schema)
    assert connection.fail_on is None
    snapshot = schema.snapshot()
    assert _where_read(schema, connection) == {('main', 'shop')}
    connection.schema = 'other'
    assert _where_read(snapshot, connection) == {_KEPT[route]}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_an_interrupted_first_run_keeps_the_readings_it_made(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` interrupted while it samples, after it read
    its tables, keeps where it read them, so a retry after the session
    moved reads them there (round 7's P0, Codex's interrupt).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    connection.fail_on = 'WITH TMP AS'
    connection.failure = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        _materialized(schema)
    connection.schema = 'other'
    assert _where_read(schema, connection) == {_KEPT[route]}


class _ReservedWarehouse(_CompletingWarehouse):
    r"""The completing warehouse whose ``users`` also holds a column named as
    a request names its own, ``instance_id``."""

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        text = _normal(operation)
        if '`instance_id`' in text:
            found = super().answer(
                operation.replace(', `instance_id`', ''), parameters
            )
            return found.append_column(
                'instance_id', pa.array(range(found.num_rows), pa.int64())
            )
        found = super().answer(operation, parameters)
        if 'information_schema.columns' in text and parameters[-1:] == [
            'users'
        ]:
            extra = {
                name: [found[name][0].as_py()] for name in found.schema.names
            }
            extra['column_name'] = ['instance_id']
            extra['full_data_type'] = ['bigint']
            found = pa.concat_tables(
                [found, pa.table(extra, schema=found.schema)]
            )
        return found


_ReservedConnection = type(
    'Connection', (_ReservedWarehouse,), {'__module__': 'databricks.sql.client'}
)


@pytest.mark.parametrize('refusal', ['no_indices', 'reserved_column'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_refused_first_run_reads_nothing_and_leaves_no_engine_reading(
    route: str, refusal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` refused before it reads (no entities, a
    column named as a request names its own) reads nothing and leaves the
    schema no engine reading, so a snapshot taken then reads its tables
    where its session says, not where the original's later run read them
    (round 8's P0, both reviewers)."""
    import pandas as pd

    from sdm_client import PredictiveQuery

    _route(route, monkeypatch)
    connection = (
        _ReservedConnection()
        if refusal == 'reserved_column'
        else _CompletingConnection()
    )
    schema = _rebuilt(connection)
    if refusal == 'reserved_column':
        assert 'instance_id' in schema['users'].columns
    mark = len(connection.calls)
    with pytest.raises(Exception):  # noqa: B017, PT011 - refused
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            schema,
        ).materialize(
            indices=[] if refusal == 'no_indices' else [0, 1],
            anchor_time=pd.Timestamp('2024-03-01'),
        )
    assert connection.calls[mark:] == []
    assert schema._engine_tables is None
    if refusal == 'reserved_column':
        del schema['users'].columns['instance_id']
    snapshot = schema.snapshot()
    assert _where_read(schema, connection) == {('main', 'shop')}
    connection.schema = 'other'
    assert _where_read(snapshot, connection) == {('main', 'other')}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_metric_view_read_as_a_table_is_refused_in_the_old_client_s_words(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A metric view named as a table is refused with the old client's
    words and its advice to read it as a semantic model (round 8's P2)."""
    from sdm_client.errors import StructuredDataError

    class _MetricViews(_CompletingWarehouse):
        def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
            found = super().answer(operation, parameters)
            if 'information_schema.columns' in operation:
                at = found.schema.get_field_index('table_type')
                found = found.set_column(
                    at,
                    found.schema.field(at),
                    pa.array(
                        ['METRIC_VIEW'] * found.num_rows,
                        found.schema.field(at).type,
                    ),
                )
            return found

    _route(route, monkeypatch)
    connection = type(
        'Connection', (_MetricViews,), {'__module__': 'databricks.sql.client'}
    )()
    with pytest.raises(StructuredDataError) as refused:
        Schema.from_source(
            DatabricksSource(connection, catalog='main', schema='shop'),
            tables={'orders': 'orders'},
        )
    assert str(refused.value) == (
        "[INVALID_REQUEST] 'orders' is a metric view, not a table: its "
        'measures are aggregations with no value per row. Read it as a '
        "semantic model instead, with source.semantic_model('orders')."
    )


def _fresh(schema: Schema, table: str) -> None:
    r"""``table``'s columns replaced by new ones of the same names and
    types, which hold nothing of what the source said of them."""
    from sdm_client import Column

    for name, column in list(schema[table].columns.items()):
        schema[table].columns[name] = Column(
            column.name, column.dtype, column.stype
        )


class _RenamedWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, which also answers for ``users`` named
    ``Users``, as Unity Catalog does."""

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        return super().answer(
            operation.replace('`Users`', '`users`'),
            ['users' if p == 'Users' else p for p in parameters],
        )


_RenamedConnection = type(
    'Connection', (_RenamedWarehouse,), {'__module__': 'databricks.sql.client'}
)


@pytest.mark.parametrize('change', ['fresh_columns', 'retargeted'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_refused_run_over_columns_with_no_facts_reads_nothing(
    route: str, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` refused for its inputs (no entities) over a
    table whose columns hold no facts of the source (new columns, or the
    table pointed at another spelling of its name) reads nothing and leaves
    no engine reading, so a snapshot reads its own session's tables (round
    9's P0, both reviewers)."""
    import pandas as pd

    from sdm_client import PredictiveQuery

    _route(route, monkeypatch)
    connection = _RenamedConnection()
    schema = _rebuilt(connection)
    if change == 'fresh_columns':
        _fresh(schema, 'users')
    else:
        schema['users'].source_name = 'Users'
    mark = len(connection.calls)
    with pytest.raises(Exception):  # noqa: B017, PT011 - refused
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            schema,
        ).materialize(indices=[], anchor_time=pd.Timestamp('2024-03-01'))
    assert connection.calls[mark:] == []
    assert schema._engine_tables is None
    snapshot = schema.snapshot()
    assert _where_read(schema, connection) == {('main', 'shop')}
    connection.schema = 'other'
    assert _where_read(snapshot, connection) == {('main', 'other')}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_preview_failing_before_the_tables_are_located_keeps_none(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` over a table whose columns hold no facts,
    failing at that table's preview (the same statement on both routes),
    keeps no table's location: a retry after the session moved reads
    ``other`` (round 9's P0, Codex)."""
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    _fresh(schema, 'users')
    connection.fail_on = 'SELECT `user_id`, `age`, `country` FROM `main`.`shop`.`users` LIMIT 1000'
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure
        _materialized(schema)
    assert connection.fail_on is None, 'the failure was not reached'
    connection.schema = 'other'
    assert _where_read(schema, connection) == {('main', 'other')}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_meaning_the_source_refuses_is_refused_after_the_tables_are_located(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` over columns holding no facts, one given a
    meaning its source type does not carry, is refused once the tables are
    read and kept, as the old route checks meanings after it keeps its
    tables (``relational.py:586``): a retry after the meaning is put right
    and the session moved reads ``shop`` (round 10's P0, Codex).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    from sdm_client.errors import StructuredDataError

    _route(route, monkeypatch)
    connection = _CompletingConnection()
    schema = _rebuilt(connection)
    _fresh(schema, 'users')
    schema['users']['age'].dtype = 'string'
    schema['users']['age'].stype = 'text'
    with pytest.raises(StructuredDataError) as refused:
        _materialized(schema)
    assert refused.value.code == 'INVALID_REQUEST'
    schema['users']['age'].dtype = 'int64'
    schema['users']['age'].stype = 'numerical'
    connection.schema = 'other'
    assert _where_read(schema, connection) == {_KEPT[route]}


@pytest.mark.parametrize('column', ['__kumo_extra', 'user'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_column_the_source_does_not_hold_is_left_out_of_every_read(
    route: str, column: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A column added by hand that the source does not hold, one named as
    the sampler reserves (``__kumo_``) or like a session word (``user``), is
    never read: the old route leaves it out silently, the core refuses it
    by name, the source holding no such column (plan B, the owner's item
    1)."""
    from sdm_client import Column
    from sdm_client.errors import StructuredDataError

    _route(route, monkeypatch)
    connection = _CompletingConnection()
    schema = _rebuilt(connection)
    stype = 'numerical' if column.startswith('__') else 'text'
    dtype = 'int64' if column.startswith('__') else 'string'
    schema['users'].columns[column] = Column(column, dtype, stype)
    if route == 'old':
        assert _where_read(schema, connection) == {('main', 'shop')}
        return
    mark = len(connection.calls)
    with pytest.raises(StructuredDataError) as refused:
        _materialized(schema)
    _assert_no_such_column(refused.value, 'users', column)
    assert not any(f'`{column}`' in text for text, _ in connection.calls[mark:])


@pytest.mark.parametrize('dtype', ['int8', 'int128', 'float16', 'float8'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_link_over_a_type_name_the_client_reads_joins_before_the_source_is_read(
    route: str, dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A foreign key given by hand a data type name the client reads as a
    number joins its key before the source is read, as the old client reads
    the name, and the run completes (round 10's P1, both reviewers)."""
    from sdm_client import Column

    _route(route, monkeypatch)
    connection = _CompletingConnection()
    schema = _rebuilt(connection)
    _fresh(schema, 'orders')
    schema['orders'].columns['user_id'] = Column('user_id', dtype, 'ID')
    assert _where_read(schema, connection) == {('main', 'shop')}


def _asked(schema: Schema, **how: Any) -> Any:
    r"""``_materialized``'s question, asked with the call's arguments
    changed as ``how`` says."""
    import pandas as pd

    from sdm_client import PredictiveQuery

    asked: dict[str, Any] = {
        'indices': [0, 1],
        'anchor_time': pd.Timestamp('2024-03-01'),
    }
    asked.update(how)
    return PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        schema,
    ).materialize(**asked)


def _retyped(
    table: str, column: str, dtype: str, stype: str, *, unlinked: bool = False
) -> Any:
    r"""A change of ``table.column`` to ``dtype`` meaning ``stype``. Where
    its setter would refuse the meaning on the type it holds, the meaning
    is set first, so that only the request's checks see the pair; where
    ``unlinked``, the links are taken off for the change and put back past
    the schema's setter, so that only the request's checks see them."""

    def change(schema: Schema) -> None:
        links = list(schema.relationships)
        if unlinked:
            schema.relationships = []
        held = schema[table][column]
        from sdm_client.schema import supports_dtype

        if supports_dtype(stype, held.dtype):
            held.stype = stype
            held.dtype = dtype
        else:
            held.dtype = dtype
            held.stype = stype
        if unlinked:
            for link in links:
                list.append(schema.relationships, link)

    return change


def _linked(*link: str) -> Any:
    r"""A relationship appended past the checks the schema's list makes,
    as the old client's ``check_consistent`` is there to find one."""
    from sdm_client import Relationship

    def change(schema: Schema) -> None:
        list.append(schema.relationships, Relationship(*link))

    return change


def _keyed(table: str, key: Any) -> Any:
    def change(schema: Schema) -> None:
        schema[table].primary_key = key

    return change


def _added(table: str, column: str, link: bool) -> Any:
    r"""A column the source does not hold added by hand, and linked to its
    own table's key where ``link`` says."""
    from sdm_client import Column, Relationship

    def change(schema: Schema) -> None:
        schema[table].columns[column] = Column(column, 'int64', 'ID')
        if link:
            key = schema[table].primary_key
            schema.relationships.append(Relationship(table, column, table, key))

    return change


def _asking(**how: Any) -> Any:
    def change(schema: Schema) -> dict[str, Any]:
        return how

    return change


#: Each refusal the old route makes from ``PredictiveQuery.materialize`` to
#: its sampler's first count, by the side of its engine's reads it makes it
#: on (the tracker's table): ``before`` refuses having read nothing, so a
#: retry after the session moved reads there; ``after`` refuses once its
#: engine read and kept the tables, so a retry reads where it read them.
_SIDES = {
    'unknown_option': (_asking(no_such_option=1), 'before'),
    'empty_indices': (_asking(indices=[]), 'before'),
    'statistic': (_asking(prediction_statistic='mode'), 'before'),
    'rows_per_call': (_asking(rows_per_call=0), 'before'),
    'reserved_wire_name': (_added('users', 'instance_id', False), 'before'),
    'empty_num_neighbors': (_asking(num_neighbors=[]), 'before'),
    'meaning_on_the_caller_s_type': (
        _retyped('users', 'age', 'string', 'numerical'),
        'before',
    ),
    'link_to_a_column_not_held': (
        _linked('orders', 'nothing', 'users', 'user_id'),
        'before',
    ),
    'link_to_a_column_not_the_key': (
        _linked('orders', 'price', 'users', 'age'),
        'before',
    ),
    'link_types_on_the_caller_s_types': (
        _retyped('orders', 'user_id', 'string', 'ID', unlinked=True),
        'before',
    ),
    'primary_key_as_foreign_key': (_keyed('orders', 'user_id'), 'after'),
    'time_column_type': (_retyped('orders', 'ts', 'string', 'text'), 'after'),
    'meaning_on_the_source_s_type': (
        _retyped('users', 'age', 'string', 'text'),
        'after',
    ),
    'composite_key': (_keyed('orders', ['order_id', 'user_id']), 'after'),
    'link_from_a_column_the_source_lacks': (
        _added('orders', 'buyer', True),
        'after',
    ),
}


@pytest.mark.parametrize('case', sorted(_SIDES))
@pytest.mark.parametrize('route', ['core', 'old'])
def test_each_refusal_is_made_on_the_old_route_s_side_of_the_reads(
    route: str, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Each refusal the old route makes before its engine reads the
    source refuses having read nothing, and a retry after the session moved
    reads where the session says; each it makes once its engine read and
    kept the tables refuses having read them, and a retry reads them where
    it did (round 11: Codex's P0 primary key and time column retries, and
    one case per row of the tracker's table that refuses).
    On the core, which keeps no table between runs (plan B), each run
    reads where the session says then (``_KEPT``).
    """
    from sdm_client.errors import StructuredDataError

    _route(route, monkeypatch)
    connection = _CompletingConnection()
    schema = _rebuilt(connection)
    columns = {
        name: {
            column: copy.copy(held) for column, held in table.columns.items()
        }
        for name, table in schema.tables.items()
    }
    keys = {name: table.primary_key for name, table in schema.tables.items()}
    links = list(schema.relationships)
    change, side = _SIDES[case]
    how = change(schema) or {}
    mark = len(connection.calls)
    with pytest.raises(StructuredDataError) as refused:
        _asked(schema, **how)
    assert refused.value.code != 'INTERNAL_ERROR', str(refused.value)
    assert (len(connection.calls) > mark) == (side == 'after'), str(
        refused.value
    )
    # The schema put right, its record of where it read its tables kept.
    schema.relationships = []
    for name, held in columns.items():
        for column in [c for c in schema[name].columns if c not in held]:
            del schema[name][column]
        for column, original in held.items():
            schema[name].columns[column] = copy.copy(original)
        schema[name].primary_key = keys[name]
    schema.relationships = links
    connection.schema = 'other'
    where = _KEPT[route] if side == 'after' else ('main', 'other')
    assert _where_read(schema, connection) == {where}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_too_many_entities_are_refused_before_the_source_is_read(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""More entities than one ``materialize`` holds are refused from the
    count alone, before the old route's engine reads anything
    (``relational.py:339``): the tracker's entity-count row."""
    from sdm_client.errors import StructuredDataError
    from sdm_client.materializers import relational

    _route(route, monkeypatch)
    monkeypatch.setattr(relational, 'MOST_ENTITIES', 1)
    connection = _CompletingConnection()
    schema = _rebuilt(connection)
    mark = len(connection.calls)
    with pytest.raises(StructuredDataError) as refused:
        _asked(schema)
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert len(connection.calls) == mark


class _EmptyingWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, whose ``shop.users`` holds no row while
    ``empty`` says so: every read of its rows answers none, and its count
    zero."""

    empty = False

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        found = super().answer(operation, parameters)
        if self.empty and '`main`.`shop`.`users`' in operation:
            if operation.startswith('SELECT COUNT(*)'):
                return pa.table({found.column_names[0]: [0]})
            return found.slice(0, 0)
        return found


_EmptyingConnection = type(
    'Connection', (_EmptyingWarehouse,), {'__module__': 'databricks.sql.client'}
)


@pytest.mark.parametrize('route', ['core', 'old'])
def test_an_empty_table_is_refused_before_the_tables_are_kept(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built anew whose columns all hold their facts reads each
    table it has not kept in full, its preview included, before keeping
    any, as the old route's engine reads every table it keeps: a table with
    no rows is refused there (``databricks/table.py:385-386``), nothing
    kept, so a retry after the session moved reads there (round 12's P0-1,
    Codex)."""
    from sdm_client.errors import StructuredDataError

    _route(route, monkeypatch)
    connection = _EmptyingConnection()
    schema = _rebuilt(connection)
    connection.empty = True
    with pytest.raises(StructuredDataError) as refused:
        _materialized(schema)
    assert str(refused.value) == (
        "[INVALID_REQUEST] Table 'main.shop.users' is empty"
    )
    connection.empty = False
    connection.schema = 'other'
    mark = len(connection.calls)
    request = _materialized(schema)
    assert {where[:2] for where in _read_since(connection, mark)} == {
        ('main', 'other')
    }
    assert 'OTHER' in repr(request.payloads)


class _DroppingWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, from whose ``users`` the column ``age``
    is dropped while ``dropped`` says so: its catalog and preview answer
    without it, and every read of its rows runs over the table without it,
    so a read that names it fails as the warehouse fails it."""

    dropped = False

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        import pandas as pd
        import pyarrow.compute as pc

        text = _normal(operation)
        if self.dropped and text.startswith('WITH TMP AS ( SELECT pos AS '):
            self.calls.append((text, list(parameters)))
            keys = json.loads(parameters[0])
            seeds = pd.DataFrame(
                keys
                if keys and isinstance(keys[0], dict)
                else [{'key': key} for key in keys]
            )
            seeds['batch'] = range(len(seeds))
            for column in ('start', 'end', 'time', 'e', 's'):
                if column in seeds:
                    seeds[column] = pd.to_datetime(seeds[column])
            for given, named in (
                ('batch', '__KUMO_BATCH__'),
                ('key', '__KUMO_ID__'),
                ('id', '__KUMO_ID__'),
                ('e', '__KUMO_END_TIME__'),
                ('s', '__KUMO_START_TIME__'),
            ):
                if given in seeds:
                    seeds[named] = seeds[given]
            self._db.register(
                'seeds', pa.Table.from_pandas(seeds, preserve_index=False)
            )
            text = re.sub(
                r'^WITH TMP AS \( .*? \) SELECT ',
                'WITH TMP AS (SELECT * FROM seeds) SELECT ',
                text,
                count=1,
            )
            text = re.sub(r'`main`\.`\w+`\.', '', text).replace('`', '"')
            return self._db.execute(text, parameters[1:]).to_arrow_table()
        found = super().answer(operation, parameters)
        if (
            self.dropped
            and 'information_schema.columns' in text
            and parameters[-1:] == ['users']
        ):
            found = found.filter(pc.not_equal(found['column_name'], 'age'))
        if (
            self.dropped
            and 'FROM `main`.`shop`.`users` LIMIT 1000' in text
            and 'age' in found.column_names
        ):
            found = found.drop(['age'])
        return found


_DroppingConnection = type(
    'Connection', (_DroppingWarehouse,), {'__module__': 'databricks.sql.client'}
)


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_column_the_source_dropped_is_never_read(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema built anew whose columns all hold their facts, one of them
    since dropped from the source, never reads the dropped column: the old
    route's engine leaves it out silently (``sources.py:81-88``) and the run
    completes; the core reads each table's columns as the source holds
    them now and refuses the column the schema names by name (plan B, the
    owner's item 1)."""
    from sdm_client.errors import StructuredDataError

    _route(route, monkeypatch)
    connection = _DroppingConnection()
    schema = _rebuilt(connection)
    connection._db.execute('ALTER TABLE users DROP COLUMN age')
    connection.dropped = True
    mark = len(connection.calls)
    if route == 'old':
        assert _materialized(schema).payloads
    else:
        with pytest.raises(StructuredDataError) as refused:
            _materialized(schema)
        _assert_no_such_column(refused.value, 'users', 'age')
    assert not any(
        re.search(r'\bage\b', text) for text, _ in connection.calls[mark:]
    )


def test_a_column_dropped_between_two_runs_is_refused_by_name_on_the_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A column the schema names, dropped from the source between two
    ``materialize`` runs, is refused by name on the second: each run reads
    each table's columns as the source holds them then (plan B, the owner's
    item 1), and the dropped column is never read."""
    from sdm_client.errors import StructuredDataError

    _route('core', monkeypatch)
    connection = _DroppingConnection()
    schema = _rebuilt(connection)
    assert _materialized(schema).payloads
    connection._db.execute('ALTER TABLE users DROP COLUMN age')
    connection.dropped = True
    mark = len(connection.calls)
    with pytest.raises(StructuredDataError) as refused:
        _materialized(schema)
    _assert_no_such_column(refused.value, 'users', 'age')
    assert not any(
        re.search(r'\bage\b', text) for text, _ in connection.calls[mark:]
    )


class _AddingWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, whose ``users`` gains a column ``plan``
    while ``added`` says so: its catalog lists it, and its rows hold
    ``NEW_FEATURE`` in it."""

    added = False

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        found = super().answer(operation, parameters)
        if (
            self.added
            and 'information_schema.columns' in operation
            and parameters[-1:] == ['users']
        ):
            row = dict(found.to_pylist()[-1])
            row['column_name'] = 'plan'
            for name in ('data_type', 'full_data_type'):
                if name in row:
                    row[name] = 'STRING'
            if 'ordinal_position' in row:
                row['ordinal_position'] = 4
            found = pa.concat_tables(
                [found, pa.Table.from_pylist([row], schema=found.schema)]
            )
        return found


_AddingConnection = type(
    'Connection', (_AddingWarehouse,), {'__module__': 'databricks.sql.client'}
)


@pytest.mark.parametrize('named', [False, True], ids=['unnamed', 'named'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_column_the_source_gained_is_sent_only_where_the_schema_names_it(
    route: str, named: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A column the source gained after the schema was described is no
    column of any request the schema does not name it in, on both routes;
    named by hand, it is sent on the core, which reads each table as the
    source holds it at each ``materialize`` (plan B), where the old route
    leaves it out of the table it kept (round 13's P0-1, Codex)."""
    from sdm_client import Column

    _route(route, monkeypatch)
    connection = _AddingConnection()
    schema = Schema.from_source(DatabricksSource(connection, catalog='main'))
    connection._db.execute(
        "ALTER TABLE users ADD COLUMN plan VARCHAR DEFAULT 'NEW_FEATURE'"
    )
    connection.added = True
    if named:
        schema['users'].columns['plan'] = Column(
            'plan', 'string', 'categorical'
        )
    request = _materialized(schema)
    users = request.payloads[0]['schema']['related_tables']['users']['columns']
    sent = named and route == 'core'
    assert ('plan' in list(users)) == sent
    assert ('NEW_FEATURE' in repr(request.payloads)) == sent


def _assert_no_such_column(refused: Any, table: str, column: str) -> None:
    r"""``refused`` is the core's refusal of ``table.column``, a column the
    schema names that the source does not hold (plan B, the owner's item
    1): named, an invalid request, saying the source has no such column."""
    assert refused.code == 'INVALID_REQUEST', str(refused)
    assert f'{table}.{column}' in str(refused), str(refused)
    assert 'the source has no such column' in str(refused), str(refused)


def test_a_column_taken_out_of_the_schema_and_put_back_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A column taken out of the schema and put back by hand is a column of
    the request, the source holding it (round 14's case under plan B; the
    old route's statements for it are not all in this warehouse's
    recording, so the core's run alone is held here)."""
    from sdm_client import Column

    _route('core', monkeypatch)
    connection = _CompletingConnection()
    schema = _rebuilt(connection)
    assert _materialized(schema).payloads
    del schema['users']['country']
    assert 'country' not in list(
        _materialized(schema).payloads[0]['schema']['related_tables']['users'][
            'columns'
        ]
    )
    schema['users'].columns['country'] = Column(
        'country', 'string', 'categorical'
    )
    request = _materialized(schema)
    users = request.payloads[0]['schema']['related_tables']['users']['columns']
    assert 'country' in list(users)
