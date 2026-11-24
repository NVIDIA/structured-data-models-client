# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A caller's own data provider (:mod:`sdm_client.provider`).

The core's messages are read into the public request types as the shared
vectors (``contract/host-provider-vectors.json``) say, which the Java host's
tests read too. A provider over a SQL engine then answers the context
scenarios of the compatibility manifest, over DuckDB in process and over
Trino where ``SDM_TRINO_TEST_URL`` names a server: each keeps exactly the
examples the fixture holds for it, whether the provider declares every
filter kind or only those of an engine without ``OR`` or subqueries.
"""

from __future__ import annotations

import base64
import datetime
import enum
import json
import os
import pickle
import typing
import urllib.request
import uuid
import warnings
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from sql_source import Engine, SqlSource

from sdm_client import (
    PredictiveQuery,
    Relationship,
    Schema,
    StructuredDataError,
)
from sdm_client import provider as p
from sdm_client._rust import host_providers as host

ROOT = Path(__file__).resolve().parents[3]
VECTORS = json.loads((ROOT / 'contract/host-provider-vectors.json').read_text())


# --- the shared vectors ----------------------------------------------------


def _keys() -> dict[int, pa.Table]:
    tables = {}
    for attachment, key_set in VECTORS['key_sets'].items():
        fields = [
            pa.field(name, getattr(pa, kind)())
            for name, kind in key_set['fields']
        ]
        tables[int(attachment)] = pa.Table.from_pylist(
            [
                dict(zip([f.name for f in fields], row, strict=True))
                for row in key_set['rows']
            ],
            schema=pa.schema(fields),
        )
    return tables


def _neutral(value: Any) -> Any:
    r"""A public request, filter or value in the vectors' neutral form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, p.Relation):
        return list(value.parts)
    if isinstance(value, p.KeySet):
        table = value.reader.read_all()
        return {
            'columns': list(value.columns),
            'rows': [list(row.values()) for row in table.to_pylist()],
        }
    if isinstance(value, p.Count):
        return {'count': value.count}
    if isinstance(value, p.AllBut):
        return {'all_but': value.value}
    if isinstance(value, (tuple, list)):
        return [_neutral(v) for v in value]
    if hasattr(value, '__dataclass_fields__'):
        out: dict[str, Any] = {}
        if isinstance(value, typing.get_args(p.DataRequest)):
            out['request'] = value.name
        elif type(value).__name__ in _FILTERS:
            out['filter'] = _FILTERS[type(value).__name__]
        for name in value.__dataclass_fields__:
            field = getattr(value, name)
            out[name] = (
                (
                    field.value
                    if isinstance(field, enum.Enum)
                    else _neutral(field)
                )
                if name not in ('value', 'seed', 'values')
                else _scalars(field)
            )
        return out
    raise TypeError(type(value))


_FILTERS = {
    'KeyIn': 'key_in',
    'TimeRange': 'time_range',
    'NotNull': 'not_null',
    'Compare': 'compare',
    'In': 'in',
    'IsNull': 'is_null',
    'AnyOf': 'any_of',
    'AllOf': 'all_of',
    'HasRows': 'has_rows',
}


def _scalars(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_scalar(v) for v in value]
    return _scalar(value)


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return {'bool': value}
    if isinstance(value, int):
        return {'int': value}
    if isinstance(value, float):
        return {'float': value}
    if isinstance(value, Decimal):
        return {'decimal': str(value)}
    if isinstance(value, str):
        return {'text': value}
    if isinstance(value, bytes):
        return {'binary': base64.b64encode(value).decode()}
    if isinstance(value, pd.Timestamp):
        zone = None if value.tz is None else str(value.tz)
        wall = value.tz_convert('UTC').tz_localize(None) if zone else value
        text = (
            wall.strftime('%Y-%m-%dT%H:%M:%S')
            + f'.{wall.microsecond:06d}{wall.nanosecond:03d}'
        )
        return {'timestamp': text, 'zone': zone}
    if isinstance(value, datetime.datetime):
        zone = None if value.tzinfo is None else str(value.tzinfo)
        if zone:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        text = (
            value.strftime('%Y-%m-%dT%H:%M:%S') + f'.{value.microsecond:06d}000'
        )
        return {'timestamp': text, 'zone': zone}
    if isinstance(value, datetime.date):
        return {'date': value.isoformat()}
    if isinstance(value, datetime.time):
        return {
            'time': value.strftime('%H:%M:%S') + f'.{value.microsecond:06d}000'
        }
    if isinstance(value, pd.Timedelta):
        # The vectors hold no duration: a Timedelta is a time of day with
        # nanoseconds, since midnight.
        seconds, nanos = divmod(value.value, 10**9)
        clock = (
            f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'
        )
        return {'time': f'{clock}.{nanos:09d}'}
    raise TypeError(type(value))


@pytest.mark.parametrize(
    'case', VECTORS['requests'], ids=[c['name'] for c in VECTORS['requests']]
)
def test_each_request_is_read_as_the_shared_vectors_say(
    case: dict[str, Any],
) -> None:
    read = host.read_request(case['wire'], _keys())
    assert _neutral(read) == case['decoded']


@pytest.mark.parametrize(
    'case', VECTORS['expected'], ids=[c['name'] for c in VECTORS['expected']]
)
def test_each_expected_schema_is_read_as_the_shared_vectors_say(
    case: dict[str, Any],
) -> None:
    schema = host.expected_schema(case['wire'])
    if case['decoded'] is None:
        assert schema is None
        return
    assert [[f.name, _type(f.type), f.nullable] for f in schema] == case[
        'decoded'
    ]


def _type(kind: pa.DataType) -> str:
    r"""A pyarrow type by the contract's name of it, as the vectors write it."""
    if pa.types.is_list(kind):
        return f'list<{_type(kind.value_type)}>'
    if pa.types.is_timestamp(kind):
        return f'timestamp[{kind.unit}]'
    return {pa.string(): 'utf8', pa.bool_(): 'bool'}.get(kind, str(kind))


@pytest.mark.parametrize(
    'case',
    VECTORS['capabilities'],
    ids=[c['name'] for c in VECTORS['capabilities']],
)
def test_each_capabilities_declared_is_sent_as_the_shared_vectors_say(
    case: dict[str, Any],
) -> None:
    declared = dict(case['declared'])
    if 'fallback' in declared:
        declared['fallback'] = p.Fallback(**declared['fallback'])
    assert host.capabilities(p.Capabilities(**declared)) == case['wire']


# --- the public types ----------------------------------------------------


class _Answers(p.Source):
    def __init__(self, answer: Any = None, failure: Any = None) -> None:
        self._answer = answer
        self._failure = failure

    def capabilities(self) -> p.Capabilities:
        return p.Capabilities()

    def answer(self, request: Any, expected: Any) -> Any:
        return self._answer

    def failure(self, error: Exception) -> Any:
        return self._failure


def test_capabilities_refuse_what_no_provider_declares() -> None:
    with pytest.raises(ValueError):
        p.Capabilities(max_keys_per_request=0)
    with pytest.raises(ValueError):
        p.Capabilities(max_seed=-1)
    with pytest.raises(ValueError):
        p.Capabilities(filters={'regex'})
    with pytest.raises(ValueError):
        p.Fallback(0, 1)
    assert p.Capabilities(filters={'in'}).filters == {p.FilterKind.IN}


def test_a_source_is_abstract_and_is_not_pickled() -> None:
    with pytest.raises(TypeError, match='abstract'):
        p.Source()  # type: ignore[abstract]

    class Half(p.Source):
        def capabilities(self) -> p.Capabilities:
            return p.Capabilities()

    with pytest.raises(TypeError, match='answer'):
        Half()  # type: ignore[abstract]
    with pytest.raises(TypeError, match='pickled'):
        pickle.dumps(_Answers())


class _Picklable(_Answers):
    def __reduce__(self) -> Any:
        return (_Picklable, ())


def test_a_schema_over_a_source_is_not_pickled_whatever_the_source() -> None:
    pickle.dumps(_Picklable())
    schema = Schema.__new__(Schema)
    object.__setattr__(schema, '_source', _Picklable())
    with pytest.raises(TypeError, match='cannot be pickled'):
        pickle.dumps(schema)


def test_an_object_that_only_looks_like_a_source_is_refused() -> None:
    class Looks:
        def capabilities(self) -> p.Capabilities:
            return p.Capabilities()

        def answer(self, request: Any, expected: Any) -> Any:
            return None

    with pytest.raises(StructuredDataError, match='needs a Source'):
        Schema.from_source(Looks())  # type: ignore[arg-type]


def test_an_answer_that_exports_no_arrow_is_the_provider_s_error() -> None:
    _, run = host.chosen(_Answers(answer=[1, 2]))
    step = {
        'request': VECTORS['requests'][1]['wire'],
        'plan': {'plan': 'host', 'expect': {'expect': 'discover'}},
    }
    with pytest.raises(TypeError, match='Arrow stream'):
        run(step, {})


def test_a_failure_tells_the_core_what_the_provider_says_of_it() -> None:
    told = p.ProviderFailure(
        sqlstate='42P01', vendor_code='7', message='no table'
    )
    _, run = host.chosen(_Answers(failure=told))
    assert run.error_fields(RuntimeError('x')) == {
        'sqlstate': '42P01',
        'vendor_code': '7',
        'driver_message': 'no table',
    }
    _, silent = host.chosen(_Answers())
    assert silent.error_fields(RuntimeError('x')) is None


def test_capabilities_that_are_not_capabilities_are_refused() -> None:
    class Wrong(_Answers):
        def capabilities(self) -> Any:
            return {'optional': []}

    with pytest.raises(TypeError, match='Capabilities'):
        host.chosen(Wrong())


# --- a provider over a SQL engine ----------------------------------------

FIXTURE = ROOT / 'tests/compatibility/fixtures/context.json'
TYPES = {
    'users': 'id BIGINT, value DOUBLE, class VARCHAR, flag BOOLEAN',
    'events': 'event BIGINT, id BIGINT, feature DOUBLE, event_time TIMESTAMP(6), item BIGINT',
    'items': 'item BIGINT, price DOUBLE',
}
NO_OR_NO_SUBQUERY = frozenset(
    {
        p.FilterKind.COMPARE,
        p.FilterKind.IN,
        p.FilterKind.IS_NULL,
        p.FilterKind.ALL_OF,
    }
)


def _scenarios() -> list[dict[str, Any]]:
    manifest = json.loads(
        (ROOT / 'tests/compatibility/manifest.json').read_text()
    )
    return [s for s in manifest['scenarios'] if s['id'].startswith('context.')]


def _insert_sql(
    fixture: dict[str, Any], table: str, name: str
) -> Iterator[str]:
    rows = fixture['tables'][table]['rows']
    for start in range(0, len(rows), 500):
        values = ', '.join(
            '(' + ', '.join(_fixture_literal(v) for v in row) + ')'
            for row in rows[start : start + 500]
        )
        yield f'INSERT INTO {name} VALUES {values}'


def _fixture_literal(value: Any) -> str:
    if (
        isinstance(value, str)
        and len(value) == 19
        and value[4] == '-'
        and value[10] == 'T'
    ):
        return f"TIMESTAMP '{value.replace('T', ' ')}'"
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


@pytest.fixture(scope='module')
def duckdb_engine() -> Iterator[Engine]:
    duckdb = pytest.importorskip('duckdb')
    connection = duckdb.connect()
    fixture = json.loads(FIXTURE.read_text())
    for table, columns in TYPES.items():
        connection.execute(
            f'CREATE TABLE "{table}" ({columns.replace("TIMESTAMP(6)", "TIMESTAMP")})'
        )
        for statement in _insert_sql(fixture, table, f'"{table}"'):
            connection.execute(statement)
    yield Engine(
        query=lambda sql: connection.sql(sql).to_arrow_table(),
        table=lambda name: f'"{name}"',
        hashed=lambda text: f'hash({text})',
        schema_name='main',
    )
    connection.close()


class _Trino:
    def __init__(self, url: str) -> None:
        self._url = url

    def run(self, sql: str) -> tuple[list[dict[str, Any]], list[list[Any]]]:
        request = urllib.request.Request(
            self._url + '/v1/statement',
            data=sql.encode(),
            headers={'X-Trino-User': 'sdm'},
        )
        columns: list[dict[str, Any]] = []
        rows: list[list[Any]] = []
        while True:
            with urllib.request.urlopen(request) as response:
                body = json.load(response)
            if 'error' in body:
                raise RuntimeError(f'Trino: {body["error"]["message"]}\n{sql}')
            columns = body.get('columns', columns)
            rows += body.get('data', [])
            if 'nextUri' not in body:
                return columns, rows
            request = urllib.request.Request(
                body['nextUri'], headers={'X-Trino-User': 'sdm'}
            )

    def table(self, sql: str) -> pa.Table:
        columns, rows = self.run(sql)
        fields = [pa.field(c['name'], _trino_type(c['type'])) for c in columns]
        data = [
            [_trino_value(row[i], f.type) for row in rows]
            for i, f in enumerate(fields)
        ]
        return pa.table(data, schema=pa.schema(fields))


def _trino_type(name: str) -> pa.DataType:
    if name.startswith('timestamp'):
        return pa.timestamp('us')
    return {
        'bigint': pa.int64(),
        'integer': pa.int32(),
        'double': pa.float64(),
        'boolean': pa.bool_(),
    }.get(name, pa.string())


def _trino_value(value: Any, kind: pa.DataType) -> Any:
    if value is not None and pa.types.is_timestamp(kind):
        return datetime.datetime.fromisoformat(value)
    return value


@pytest.fixture(scope='module')
def trino_engine() -> Iterator[Engine]:
    url = os.environ.get('SDM_TRINO_TEST_URL')
    if not url:
        pytest.skip('SDM_TRINO_TEST_URL names no Trino server')
    trino = _Trino(url)
    schema = 'sdm_' + uuid.uuid4().hex
    trino.run(f'CREATE SCHEMA memory.{schema}')
    fixture = json.loads(FIXTURE.read_text())
    try:
        for table, columns in TYPES.items():
            name = f'memory.{schema}."{table}"'
            trino.run(f'CREATE TABLE {name} ({columns})')
            for statement in _insert_sql(fixture, table, name):
                trino.run(statement)
        yield Engine(
            query=trino.table,
            table=lambda name: f'memory.{schema}."{name}"',
            hashed=lambda text: f'xxhash64(to_utf8({text}))',
            schema_name=schema,
            catalog='memory.information_schema',
        )
    finally:
        for table in TYPES:
            trino.run(f'DROP TABLE IF EXISTS memory.{schema}."{table}"')
        trino.run(f'DROP SCHEMA IF EXISTS memory.{schema}')


def _materialize(provider: SqlSource, scenario: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            provider,
            # The engine declares no keys: the caller names them.
            primary_keys={'users': 'id', 'events': 'event', 'items': 'item'},
            time_columns={'events': 'event_time'},
            relationships=[
                Relationship('events', 'id', 'users', 'id'),
                Relationship('events', 'item', 'items', 'item'),
            ],
        )
        schema['users']['value'].stype = 'numerical'
        options = dict(scenario['options'])
        if 'anchor_time' in options:
            options['anchor_time'] = pd.Timestamp(options['anchor_time'])
        return PredictiveQuery(scenario['pql'], schema).materialize(**options)


def _expected(scenario: dict[str, Any], declared: str) -> list[tuple[Any, ...]]:
    # A filter reaches the provider only where it declares the kinds the
    # scenario's filter is written with (``sent_with``).
    context = scenario['context']
    kinds = {kind.value for kind in _kinds(declared)}
    sent = context['filtered_at_source'] and set(context['sent_with']) <= kinds
    return [
        (kept, context['asked'], context['stopped'], sent)
        for kept in context['kept']
    ]


def _drawn(request: Any) -> list[tuple[Any, ...]]:
    report = request.context_report
    assert report.status == 'available'
    return [
        (d.kept, d.asked, d.stopped, d.filtered_at_source) for d in report.parts
    ]


CASES = [
    pytest.param(s, declared, id=f'{s["id"]} ({declared})')
    for s in _scenarios()
    for declared in ('every kind', 'no OR or subquery')
]


def _kinds(declared: str) -> frozenset[p.FilterKind]:
    return (
        frozenset(p.FilterKind)
        if declared == 'every kind'
        else NO_OR_NO_SUBQUERY
    )


@pytest.mark.parametrize(('scenario', 'declared'), CASES)
def test_a_provider_over_duckdb_keeps_exactly_the_examples_the_source_holds(
    duckdb_engine: Engine, scenario: dict[str, Any], declared: str
) -> None:
    provider = SqlSource(
        duckdb_engine,
        filters=_kinds(declared),
        optional=(p.OptionalRequest.SAMPLE_ENTITIES, p.OptionalRequest.LOOKUP),
        max_keys_per_request=7,
    )
    request = _materialize(provider, scenario)
    assert _drawn(request) == _expected(scenario, declared)
    # A key set past the provider's limit reaches it in parts.
    assert all(keys <= 7 for keys in provider.lookup_keys)


@pytest.mark.parametrize(('scenario', 'declared'), CASES)
def test_a_provider_over_trino_keeps_exactly_the_examples_the_source_holds(
    trino_engine: Engine, scenario: dict[str, Any], declared: str
) -> None:
    provider = SqlSource(trino_engine, filters=_kinds(declared))
    assert _drawn(_materialize(provider, scenario)) == _expected(
        scenario, declared
    )


def test_a_lookup_asked_in_parts_is_answered_as_one(
    duckdb_engine: Engine,
) -> None:
    scenario = next(
        s for s in _scenarios() if s['id'] == 'context.static-exhausted'
    )
    parts = SqlSource(
        duckdb_engine,
        optional=(p.OptionalRequest.SAMPLE_ENTITIES, p.OptionalRequest.LOOKUP),
        max_keys_per_request=2,
    )
    whole = SqlSource(
        duckdb_engine,
        optional=(p.OptionalRequest.SAMPLE_ENTITIES, p.OptionalRequest.LOOKUP),
    )
    split, single = _materialize(parts, scenario), _materialize(whole, scenario)
    assert parts.lookup_keys and max(parts.lookup_keys) <= 2
    assert len(parts.lookup_keys) > len(whole.lookup_keys)
    assert split.payloads == single.payloads


def test_a_provider_s_failure_is_the_operation_s_provider_error(
    duckdb_engine: Engine,
) -> None:
    class Failing(SqlSource):
        def answer(self, request: Any, expected: Any) -> Any:
            if isinstance(request, p.SampleEntities):
                raise RuntimeError('the engine is down')
            return super().answer(request, expected)

    scenario = next(
        s for s in _scenarios() if s['id'] == 'context.where-entity'
    )
    with pytest.raises(StructuredDataError) as raised:
        _materialize(Failing(duckdb_engine), scenario)
    assert raised.value.code == 'PROVIDER_ERROR'


def test_the_documented_example_runs_as_written(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The first Python block of docs/reference/data-providers.md, exactly
    # as a reader copies it.
    page = (ROOT / 'docs/reference/data-providers.md').read_text()
    example = page.split('```python\n', 1)[1].split('```', 1)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exec(compile(example, 'data-providers.md', 'exec'), {})  # noqa: S102
    assert "ContextReport(status='available'" in capsys.readouterr().out
