# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake semantic models through the public API and installed native core.

The fake provider feeds the genuine S2 Snowflake catalog/preview observations
back to the native run, and supports the bounded single-table key reads in
this test. It executes no SQL and reaches no account or old engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from sdm_connectors import SnowflakeSource
from sdm_connectors.providers import text as binding_text

from sdm_client import PredictiveQuery, Schema, StructuredDataError, _routing
from sdm_client._rust import providers

CAPTURE = (
    Path(__file__).resolve().parents[3]
    / 'contract/captures/s2/snowflake/view.semantic.from-source'
)


def _public_text(value: str) -> str:
    return value.replace('{catalog}', 'DB').replace('{schema}', 'SHOP')


def _recorded_rows(path: str) -> pa.Table:
    with (CAPTURE / path).open('rb') as stream:
        table = pa.ipc.open_stream(stream).read_all()
    for at, column in enumerate(table.columns):
        if pa.types.is_string(column.type) or pa.types.is_large_string(
            column.type
        ):
            column = pa.array(
                [
                    None if v is None else _public_text(v)
                    for v in column.to_pylist()
                ],
                column.type,
            )
            table = table.set_column(at, table.schema.field(at), column)
    return table


class _Connection:
    __module__ = 'snowflake.connector.connection'

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.plans: list[dict[str, Any]] = []
        self.failure: Exception | None = None
        self.definition: Any = ...


# Source recognition uses the connector's public class name and module.
_Connection.__name__ = 'SnowflakeConnection'


class _RecordedSemanticProvider:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        capture = json.loads((CAPTURE / 'capture.json').read_text())
        self.catalogs: dict[str, pa.Table] = {}
        self.previews: dict[str, pa.Table] = {}
        for statement in capture['statements']:
            sql = _public_text(statement['text'])
            rows = _recorded_rows(statement['result'])
            if 'SYSTEM$READ_YAML' in sql:
                self.definition = rows
            for table in ('ORDERS', 'USERS'):
                if sql.startswith('DESCRIBE TABLE') and sql.endswith(
                    f'."{table}"'
                ):
                    self.catalogs[table] = rows
                elif ' LIMIT 1000' in sql and f'."{table}"' in sql:
                    self.previews[table] = rows

    def __call__(self, step: dict[str, Any], keys: Any = None) -> pa.Table:
        request = step['request']
        self.connection.requests.append(request)
        self.connection.plans.append(step['plan'])
        assert step['plan']['plan'] == 'sql'
        assert all(
            s['dialect'] == 'snowflake' for s in step['plan']['statements']
        )
        kind = request['request']
        if kind == 'describe':
            if request['part'] == 'view_definition':
                if self.connection.failure is not None:
                    raise self.connection.failure
                if self.connection.definition is not ...:
                    return pa.table(
                        {'definition': [self.connection.definition]}
                    )
                return self.definition
            assert request['part'] in ('columns', 'keys')
            return self.catalogs[request['tables'][0]['source'][-1]]
        table = request['table']['source'][-1]
        rows = self.previews[table]
        if kind in ('preview', 'sample_entities'):
            return rows.select(request['columns'])
        if kind == 'count_rows':
            # Both captured previews contain every source row in this fixture.
            return pa.table({'rows': pa.array([rows.num_rows], pa.int64())})
        if kind == 'lookup':
            param = next(
                p
                for statement in step['plan']['statements']
                for p in statement['params']
                if p['param'] == 'text'
            )
            values = json.loads(binding_text(param, keys)[1])
            by_key = {
                value: at
                for at, value in enumerate(rows[request['key'][0]].to_pylist())
            }
            pairs = [
                (batch, by_key[value])
                for batch, value in enumerate(values)
                if value in by_key
            ]
            result = rows.take(
                pa.array([row for _, row in pairs], pa.int64())
            ).select(request['columns'])
            result = result.add_column(
                0,
                '__KUMO_BATCH__',
                pa.array([batch for batch, _ in pairs], pa.int64()),
            )
            if any(
                '__kumo_matches__' in statement['text']
                for statement in step['plan']['statements']
            ):
                result = result.append_column(
                    '__kumo_matches__', pa.array([1] * len(pairs), pa.int64())
                )
            return result
        raise AssertionError(f'unexpected request {request!r}')


def _source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = 'view_semantic_from_source',
    database: str | None = 'DB',
    schema: str | None = 'SHOP',
) -> tuple[_Connection, Any]:
    paths = (
        _routing.FROM_SOURCE_SNOWFLAKE_SEMANTIC_VIEW
        + _routing.MATERIALIZE_SNOWFLAKE_SEMANTIC_VIEW
    )
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(paths))
    monkeypatch.setattr(
        providers, 'SnowflakeProvider', _RecordedSemanticProvider
    )
    connection = _Connection()
    model = SnowflakeSource(
        connection, database=database, schema=schema
    ).semantic_model(name)
    return connection, model


@pytest.mark.parametrize(
    ('database', 'namespace', 'name', 'bound'),
    [
        (None, None, 'V', 'V'),
        ('DB', None, 'V', 'DB.V'),
        (None, 'SHOP', 'V', 'SHOP.V'),
        ('DB', 'SHOP', 'V', 'DB.SHOP.V'),
        (None, None, 'SHOP.V', 'SHOP.V'),
        ('DB', 'SHOP', 'SHOP.V', 'SHOP.V'),
        ('DB', 'SHOP', 'OTHER.SHOP.V', 'OTHER.SHOP.V'),
        ('DB', 'SHOP', '"Sales View"', 'DB.SHOP."Sales View"'),
        ('DB', 'SHOP', '"Sales.View"', '"Sales.View"'),
        ('DB', 'SHOP', 'OTHER.SHOP."Sales View"', 'OTHER.SHOP."Sales View"'),
        (
            'DB',
            'SHOP',
            'OTHER."Shop.Name"."Sales View"',
            'OTHER."Shop.Name"."Sales View"',
        ),
        ('DB', 'SHOP', '"Customer\'s View"', 'DB.SHOP."Customer\'s View"'),
    ],
)
def test_semantic_lookup_binds_the_supplied_name_and_leaves_resolution_to_snowflake(
    monkeypatch: pytest.MonkeyPatch,
    database: str | None,
    namespace: str | None,
    name: str,
    bound: str,
) -> None:
    connection, model = _source(
        monkeypatch, name=name, database=database, schema=namespace
    )
    schema = Schema.from_source(
        model, tables={'people': 'USERS'}, relationships=[]
    )
    PredictiveQuery(
        'PREDICT people.AGE FOR EACH people.USER_ID', schema
    ).materialize(
        indices=[0],
        num_context_samples=4,
        model='kumo-relational',
        random_seed=42,
    )
    definitions = [
        plan['statements'][0]
        for request, plan in zip(
            connection.requests, connection.plans, strict=True
        )
        if request['request'] == 'describe'
        and request['part'] == 'view_definition'
    ]
    assert len(definitions) == 2
    for statement in definitions:
        assert (
            statement['text'] == 'SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)'
        )
        assert statement['params'] == [
            {'param': 'value', 'value': {'t': 'utf8', 'v': bound}}
        ]
    assert not any(
        request['request'] == 'describe' and request['part'] == 'namespace'
        for request in connection.requests
    )


def test_recorded_semantic_view_describes_and_materializes_on_the_native_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, model = _source(monkeypatch)
    schema = Schema.from_source(
        model, tables={'people': 'USERS'}, relationships=[]
    )
    assert list(schema.tables) == ['people']
    assert schema['people'].source_name == 'USERS'
    assert schema['people'].primary_key == 'USER_ID'
    assert schema.relationships == []
    described = connection.requests.copy()
    assert [
        r['tables'][0]['source'][-1]
        for r in described
        if r['request'] == 'describe' and r['part'] == 'columns'
    ] == ['ORDERS', 'USERS']
    assert len([r for r in described if r['request'] == 'preview']) == 2
    assert not any(
        r['request'] == 'describe' and r['part'] == 'keys' for r in described
    )
    request = PredictiveQuery(
        'PREDICT people.AGE FOR EACH people.USER_ID', schema
    ).materialize(
        indices=[0, 1],
        num_context_samples=4,
        model='kumo-relational',
        random_seed=42,
    )
    assert request.is_relational
    assert request.payloads
    assert any(
        r['request'] == 'lookup' for r in connection.requests[len(described) :]
    )
    assert (
        sum(
            r['request'] == 'describe' and r['part'] == 'view_definition'
            for r in connection.requests
        )
        == 2
    )
    # Caller aliases are resolved back to the physical table before sampling.
    assert all(
        r['table']['source'][-1] == 'USERS'
        for r in connection.requests[len(described) :]
        if r['request'] in ('count_rows', 'sample_entities', 'lookup')
    )


@pytest.mark.parametrize('definition', [None, '', 42])
def test_unreadable_semantic_definitions_have_a_named_native_error(
    monkeypatch: pytest.MonkeyPatch,
    definition: Any,
) -> None:
    connection, model = _source(monkeypatch)
    connection.definition = definition
    with pytest.raises(
        StructuredDataError,
        match='Could not read the definition of semantic view',
    ) as raised:
        Schema.from_source(model)
    assert raised.value.code == 'INVALID_REQUEST'
    assert len(connection.requests) == 1


def test_missing_semantic_view_has_a_named_error_without_driver_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, model = _source(monkeypatch)
    connection.failure = RuntimeError(
        'row-canary SELECT driver-sql-canary does not exist'
    )
    with pytest.raises(StructuredDataError, match='Semantic view') as raised:
        Schema.from_source(model)
    assert raised.value.code == 'INVALID_REQUEST'
    assert 'row-canary' not in str(raised.value)
    assert 'driver-sql-canary' not in str(raised.value)
    assert len(connection.requests) == 1
