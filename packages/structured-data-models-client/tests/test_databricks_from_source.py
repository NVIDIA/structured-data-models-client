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
def test_materialize_reads_the_tables_described_whatever_the_session_s_schema_is_now(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema described under the session's current schema ``shop``
    reads ``shop``'s tables when it is materialized after the session moved
    to ``other``, as the old route keeps the tables it described (round 2's
    Codex P0-2): the namespace the description was read under travels with
    it."""
    import pandas as pd

    from sdm_client import PredictiveQuery

    if route == 'core':
        monkeypatch.setenv(
            _routing.ENVIRONMENT,
            ','.join(
                _routing.FROM_SOURCE_DATABRICKS_SQL
                + _routing.MATERIALIZE_DATABRICKS_SQL
            ),
        )
    connection = _Connection()
    schema = Schema.from_source(DatabricksSource(connection, catalog='main'))
    connection.schema = 'other'
    described = len(connection.asked)
    # Not every sampling statement of either route is recorded; what the
    # run reads before it stops is what is held here.
    try:
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))
    except Exception:  # noqa: BLE001, S110 - the reads are what is held
        pass
    read = [
        text
        for text in connection.asked[described:]
        if '`orders`' in text or '`users`' in text
    ]
    assert read, connection.asked[described:]
    assert all('`main`.`shop`.' in text for text in read), read


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
    was described (round 3's P0, Codex's case)."""
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
    assert all(where[:2] == ('main', 'shop') for where in kept), read


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
    (round 4's P0-1, Codex's case)."""
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
    assert all(where[:2] == ('main', 'shop') for where in read), read


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
    (round 4's P0-2, Codex's case)."""
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
    assert all(where[:2] == ('main', 'shop') for where in orders), read
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
    (``EngineTables.keep``; round 5's P0)."""
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
    assert {w[:2] for w in _read_since(connection, mark)} == {('main', 'shop')}
    assert 'OTHER' not in repr(first.payloads)
    assert 'OTHER' not in repr(second.payloads)


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_added_to_a_described_schema_keeps_where_it_was_first_read(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A table the described schema never read, added by hand, is read
    where the session says at the first ``materialize`` and there again
    after the session moved (round 5's P0)."""
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
    assert {w[:2] for w in _read_since(connection, mark)} == {('main', 'shop')}
    assert 'OTHER' not in repr(second.payloads)


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
    6's P0, Codex's ``retry``)."""
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
    assert _where_read(schema, connection) == {('main', 'shop')}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_table_removed_edited_around_and_restored_is_read_where_it_was(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A described table taken out of the schema, an edit made meanwhile,
    then put back, is read where the schema read it, after the session
    moved, as the old route keeps every table it read (round 6's P0,
    Codex's ``remove_edit_restore``)."""
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
    assert _where_read(schema, connection) == {('main', 'shop')}


@pytest.mark.parametrize('first', ['original', 'snapshot'])
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_snapshot_shares_the_tables_its_original_reads_later(
    route: str, first: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A schema and a snapshot of it share the tables either reads after
    the snapshot was taken, as the old route shares its engine reading: a
    table the one reads first is read there by the other after the session
    moved, whichever reads first (round 6's P0, Codex's ``snapshot``)."""
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
    assert _where_read(later, connection) == {('main', 'shop')}


class _ArmedWarehouse(_CompletingWarehouse):
    r"""The completing warehouse, with one failure armed: the first statement
    naming ``fail_on`` raises ``failure`` (a transient error, or an
    interrupt); and a table read under schema ``missing`` fails as a table
    the warehouse does not hold."""

    fail_on: str | None = None
    failure: type[BaseException] = RuntimeError

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        if '`main`.`missing`.' in operation:
            raise RuntimeError('[TABLE_OR_VIEW_NOT_FOUND] SQLSTATE: 42P01')
        if self.fail_on is not None and self.fail_on in operation:
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


#: Where each route first reads each table, and its first read after it
#: has read them all: the old route's engine reads a table's catalog and
#: preview, the core counts its rows; then the old route samples, and the
#: core reads the time bounds.
_PHASES = {
    'old': {
        'first_table': 'FROM `main`.`shop`.`orders` LIMIT 1000',
        'second_table': 'FROM `main`.`shop`.`users` LIMIT 1000',
        'after_reading': 'SELECT COUNT(*) FROM `main`.`shop`.`orders`',
    },
    'core': {
        'first_table': 'SELECT COUNT(*) FROM `main`.`shop`.`orders`',
        'second_table': 'SELECT COUNT(*) FROM `main`.`shop`.`users`',
        'after_reading': 'as min_date',
    },
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
    r"""A first ``materialize`` that fails before it has read every table it
    reads anew keeps none of them, and one that fails after keeps them all,
    as the old route keeps its engine's reading only once its engine read
    every table: a retry after the session moved reads ``other`` or ``shop``
    accordingly (round 7's P0, Claude's early failure)."""
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    connection.fail_on = _PHASES[route][phase]
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure
        _materialized(schema)
    assert connection.fail_on is None, 'the failure was not reached'
    connection.schema = 'other'
    expected = ('main', 'shop') if kept else ('main', 'other')
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
    P0, Codex's early failure)."""
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
    assert _where_read(snapshot, connection) == {('main', 'shop')}


@pytest.mark.parametrize('route', ['core', 'old'])
def test_an_interrupted_first_run_keeps_the_readings_it_made(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A first ``materialize`` interrupted while it samples, after it read
    its tables, keeps where it read them, so a retry after the session
    moved reads them there (round 7's P0, Codex's interrupt)."""
    _route(route, monkeypatch)
    connection = _ArmedConnection()
    schema = _rebuilt(connection)
    connection.fail_on = 'WITH TMP AS'
    connection.failure = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        _materialized(schema)
    connection.schema = 'other'
    assert _where_read(schema, connection) == {('main', 'shop')}
