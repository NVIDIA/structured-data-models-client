# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A caller's own data provider (:mod:`sdm_client.provider`).

The core's messages are read into the public request types as the shared
vectors (``contract/host-provider-vectors.json``) say, which the Java host's
tests read too. A provider over a SQL engine then answers the context
scenarios of the compatibility manifest over DuckDB in process
(``context_scenarios.py``): each keeps exactly the examples the fixture
holds for it, whether the provider declares every filter kind or only those
of an engine without ``OR`` or subqueries. The Trino example source runs
them over Trino in ``test_trino_source.py``.
"""

from __future__ import annotations

import base64
import datetime
import enum
import json
import pickle
import traceback
import typing
import warnings
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from context_scenarios import (
    CASES,
    FIXTURE,
    TYPES,
    drawn,
    expected,
    insert_sql,
    kinds,
    materialize,
    scenarios,
)
from sql_source import Engine, SqlSource

from sdm_client import (
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
            if name in _LATER and field is None:
                # Fields the shared vectors' first cases predate.
                continue
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


#: Fields of the public types the vectors' decoded forms leave out where
#: they say nothing; the round trips hold them.
_LATER = ('key_dtype', 'source_type')

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
    if isinstance(value, p.Value):
        return _scalar(value.value)
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


ROUND_TRIPS = VECTORS['round_trips']


def _named_alike(
    written: Any, keys: dict[int, Any], given: dict[int, Any]
) -> Any:
    r"""``written`` with each key set's attachment the one the core gave
    its rows under: a written request numbers its key sets anew."""
    by_reader = {id(reader): at for at, reader in given.items()}
    renamed = {at: by_reader[id(reader)] for at, reader in keys.items()}

    def rename(node: Any) -> Any:
        if isinstance(node, dict):
            out = {name: rename(item) for name, item in node.items()}
            if 'attachment' in node and 'columns' in node:
                out['attachment'] = renamed[node['attachment']]
            return out
        if isinstance(node, list):
            return [rename(item) for item in node]
        return node

    return rename(written)


def _readers() -> dict[int, pa.RecordBatchReader]:
    return {at: table.to_reader() for at, table in _keys().items()}


@pytest.mark.parametrize(
    'case',
    VECTORS['requests'] + ROUND_TRIPS['requests'],
    ids=[c['name'] for c in VECTORS['requests'] + ROUND_TRIPS['requests']],
)
def test_each_request_read_and_written_again_is_the_request_sent(
    case: dict[str, Any],
) -> None:
    given = _readers()
    read = host.read_request(case['wire'], given)
    written, keys = host.write_request(read)
    assert _named_alike(written, keys, given) == case['wire']
    # Each key set written is the one read, rows and all.
    for reader in keys.values():
        assert any(reader is r for r in given.values())


@pytest.mark.parametrize(
    'case',
    ROUND_TRIPS['capabilities'],
    ids=[c['name'] for c in ROUND_TRIPS['capabilities']],
)
def test_each_capabilities_read_and_written_again_are_the_ones_sent(
    case: dict[str, Any],
) -> None:
    read = host.read_capabilities(case['wire'])
    assert host.capabilities(read) == case['wire']


@pytest.mark.parametrize(
    'case',
    ROUND_TRIPS['failures'],
    ids=[c['name'] for c in ROUND_TRIPS['failures']],
)
def test_each_failure_read_and_written_again_tells_the_same(
    case: dict[str, Any],
) -> None:
    told = {k: v for k, v in case['wire'].items() if v is not None}
    assert host.failure(host.read_failure(case['wire'])) == told


def test_a_value_keeps_its_exact_type() -> None:
    read = host.read_request(ROUND_TRIPS['requests'][0]['wire'], _readers())
    assert isinstance(read, p.Scan)
    first = read.filters[9]
    assert isinstance(first, p.Compare)
    assert first.value == p.Value(2**64 - 1, pa.uint64())
    zoned = read.filters[27]
    assert isinstance(zoned, p.Compare)
    assert zoned.value.type == pa.timestamp('ns', tz='America/New_York')
    assert read.table.columns[1] == p.RelationColumn(
        'total', expression='price * quantity'
    )


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
    # The core's seed is an unsigned 64-bit integer, as Java's BigInteger
    # maxSeed bounds it.
    assert p.Capabilities(max_seed=2**64 - 1).max_seed == 2**64 - 1
    with pytest.raises(ValueError):
        p.Capabilities(max_seed=2**64)
    with pytest.raises(ValueError):
        p.Capabilities(engine='')
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


@pytest.fixture(scope='module')
def duckdb_engine() -> Iterator[Engine]:
    duckdb = pytest.importorskip('duckdb')
    connection = duckdb.connect()
    fixture = json.loads(FIXTURE.read_text())
    for table, columns in TYPES.items():
        connection.execute(
            f'CREATE TABLE "{table}" ({columns.replace("TIMESTAMP(6)", "TIMESTAMP")})'
        )
        for statement in insert_sql(fixture, table, f'"{table}"'):
            connection.execute(statement)
    yield Engine(
        query=lambda sql: connection.sql(sql).to_arrow_table(),
        table=lambda name: f'"{name}"',
        hashed=lambda text: f'hash({text})',
        schema_name='main',
    )
    connection.close()


@pytest.mark.parametrize(('scenario', 'declared'), CASES)
def test_a_provider_over_duckdb_keeps_exactly_the_examples_the_source_holds(
    duckdb_engine: Engine, scenario: dict[str, Any], declared: str
) -> None:
    provider = SqlSource(
        duckdb_engine,
        filters=kinds(declared),
        optional=(p.OptionalRequest.SAMPLE_ENTITIES, p.OptionalRequest.LOOKUP),
        max_keys_per_request=7,
    )
    request = materialize(provider, scenario)
    assert drawn(request) == expected(scenario, declared)
    # A key set past the provider's limit reaches it in parts.
    assert all(keys <= 7 for keys in provider.lookup_keys)


def test_a_lookup_asked_in_parts_is_answered_as_one(
    duckdb_engine: Engine,
) -> None:
    scenario = next(
        s for s in scenarios() if s['id'] == 'context.static-exhausted'
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
    split, single = materialize(parts, scenario), materialize(whole, scenario)
    assert parts.lookup_keys and max(parts.lookup_keys) <= 2
    assert len(parts.lookup_keys) > len(whole.lookup_keys)
    assert split.payloads == single.payloads


def test_a_provider_s_failure_is_the_operation_s_provider_error(
    duckdb_engine: Engine,
) -> None:
    failure = ValueError("cannot parse date '2025-13-01' in column ts")

    class Failing(SqlSource):
        def answer(self, request: Any, expected: Any) -> Any:
            if isinstance(request, p.SampleEntities):
                raise failure
            return super().answer(request, expected)

    scenario = next(s for s in scenarios() if s['id'] == 'context.where-entity')
    with pytest.raises(StructuredDataError) as raised:
        materialize(Failing(duckdb_engine), scenario)
    assert raised.value.code == 'PROVIDER_ERROR'
    assert raised.value.details['driver_class'] == 'builtins.ValueError'
    # The source's own exception, its message whole, is on the error for
    # the caller to read; its message, details and traceback say only the
    # core's.
    assert raised.value.source_error is failure
    assert raised.value.__cause__ is None
    assert '2025-13-01' not in str(raised.value)
    assert '2025-13-01' not in repr(raised.value.details)
    assert '2025-13-01' not in ''.join(traceback.format_exception(raised.value))


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
