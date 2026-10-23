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
        #: The session's current schema, which a caller may switch.
        self.schema = 'shop'

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def close(self) -> None:
        return None

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        text = _normal(operation)
        self.asked.append(text)
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
