# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Databricks providers' own code, over a fake cursor (T3.4).

Both connections a Databricks source takes are faked here: a SQL warehouse
connection, whose cursor records ``execute(sql, parameters)`` and answers
``fetchall_arrow()``, and a Spark session (``.sql()``, ``.version``, a frame
with ``toArrow()``) wrapped in ``sdm_connectors``' ``SparkConnection``. What
is checked is what the provider binds, as the reference's sampler binds it
(``DB/sampler.py``: ``_dumps``, ``_chunk_rows``, ``_flatten``), what it
answers, how the client tells the connections apart and routes them, and
how a Databricks failure reaches the core.
"""

from __future__ import annotations

import datetime
import json
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest

from sdm_client import _routing
from sdm_client._rust import materialize as core_materialize
from sdm_client._rust import schema as core_schema
from sdm_client._rust.providers import databricks
from sdm_client._rust.providers.databricks import (
    BATCH_COLUMN,
    DatabricksProvider,
    DatabricksSparkProvider,
    DatabricksWarehouseProvider,
    PlanNotRun,
)
from sdm_client._rust.run import _driver

connectors = pytest.importorskip('sdm_connectors')
from sdm_connectors import DatabricksSource, Source  # noqa: E402
from sdm_connectors.backends.spark import SparkConnection  # noqa: E402

Answer = Any


# The reference's own key-set rules (DB/sampler.py:32-44, 116-136 at
# 43e55f4f), transcribed as the oracle.
def _reference_default(obj: Any) -> Any:
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def _reference_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(',', ':'), default=_reference_default)


def _reference_chunk_rows(rows: list, max_bytes: int) -> list[list]:
    chunks: list[list] = []
    current: list = []
    current_bytes = 2
    for row in rows:
        size = len(_reference_dumps(row)) + 1
        if current and current_bytes + size > max_bytes:
            chunks.append(current)
            current, current_bytes = [], 2
        current.append(row)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks or [[]]


class _Cursor:
    def __init__(self, connection: _Warehouse) -> None:
        self._connection = connection
        self._result: Any = None

    def execute(self, operation: str, parameters: Any = None) -> None:
        self._connection.executed.append((operation, parameters))
        self._result = self._connection.answer(operation, parameters)

    def fetchall_arrow(self) -> Any:
        return self._result

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self._connection.closed += 1


class _Warehouse:
    """A SQL warehouse connection's cursor, as the provider uses it."""

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.executed: list[tuple[str, Any]] = []
        self.cursors = 0
        self.closed = 0

    def cursor(self) -> _Cursor:
        self.cursors += 1
        return _Cursor(self)


class _Frame:
    def __init__(self, table: Any) -> None:
        self._table = table

    def toArrow(self) -> Any:  # noqa: N802 - Spark's name
        return self._table


class _Session:
    """A notebook's Spark session, as ``SparkConnection`` uses it."""

    version = '4.0.0'

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.executed: list[tuple[str, Any]] = []

    def sql(self, query: str, args: Any = None) -> _Frame:
        self.executed.append((query, args))
        return _Frame(self.answer(query, args))


def _both(answer: Answer) -> list[tuple[Any, list[tuple[str, Any]]]]:
    r"""A warehouse connection and a wrapped Spark session, each answering
    with ``answer``, and the list each records its statements in."""
    warehouse = _Warehouse(answer)
    session = _Session(answer)
    return [
        (warehouse, warehouse.executed),
        (SparkConnection(session), session.executed),
    ]


CONNECTIONS = ['warehouse', 'spark']


def _statement(text: str, *params: dict[str, Any]) -> dict[str, Any]:
    return {'dialect': 'databricks', 'text': text, 'params': list(params)}


def _step(*statements: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': 1,
        'plan': {'plan': 'sql', 'statements': list(statements), 'expect': {}},
    }


def _key_set(attachment: int = 7) -> dict[str, Any]:
    return {
        'param': 'key_set',
        'keys': {'attachment': attachment, 'columns': []},
        'relation': 'TMP',
    }


def _value(scalar: dict[str, Any]) -> dict[str, Any]:
    return {'param': 'value', 'value': scalar}


def _batches(sql: str, parameters: Any) -> Any:
    r"""An answer numbering each seed of the key set bound first, as the
    statement's ``posexplode`` does: its position, and the seed."""
    seeds = json.loads(parameters[0])
    return pa.table(
        {
            BATCH_COLUMN: pa.array(range(len(seeds)), pa.int32()),
            'seed': pa.array(seeds, pa.int64()),
        }
    )


@pytest.mark.parametrize('which', [0, 1], ids=CONNECTIONS)
def test_statements_run_in_order_and_the_last_one_s_arrow_is_the_answer(
    which: int,
) -> None:
    answers = {
        'SELECT current_catalog(), current_schema()': pa.table({'a': [1]}),
        'SELECT 2 AS b': pa.table({'b': pa.array([2], pa.int64())}),
    }
    connection, executed = _both(lambda sql, _: answers[sql])[which]
    answer = DatabricksProvider(connection)(
        _step(
            _statement('SELECT current_catalog(), current_schema()'),
            _statement('SELECT 2 AS b'),
        )
    )
    assert executed == [
        ('SELECT current_catalog(), current_schema()', None),
        ('SELECT 2 AS b', None),
    ]
    # The driver's Arrow table, handed over as it came.
    assert answer is answers['SELECT 2 AS b']
    if isinstance(connection, _Warehouse):
        assert (connection.cursors, connection.closed) == (1, 1)


@pytest.mark.parametrize('which', [0, 1], ids=CONNECTIONS)
def test_an_empty_answer_keeps_its_schema(which: int) -> None:
    empty = pa.table({'x': pa.array([], pa.decimal128(38, 6))})
    connection, _ = _both(lambda sql, _: empty)[which]
    answer = DatabricksProvider(connection)(_step(_statement('SELECT x')))
    assert answer.num_rows == 0
    assert answer.schema == empty.schema


@pytest.mark.parametrize('which', [0, 1], ids=CONNECTIONS)
def test_scalars_are_bound_as_python_values_in_order(which: int) -> None:
    connection, executed = _both(lambda sql, _: pa.table({'a': [1]}))[which]
    scalars = [
        ({'t': 'i64', 'v': '-9223372036854775808'}, -(2**63)),
        ({'t': 'utf8', 'v': 'a`b"c'}, 'a`b"c'),
        ({'t': 'f64', 'v': '0.30000000000000004'}, 0.30000000000000004),
        ({'t': 'bool', 'v': True}, True),
        ({'t': 'null'}, None),
        (
            {'t': 'd128', 'v': '-123456', 'p': 38, 's': 6},
            Decimal('-0.123456'),
        ),
        ({'t': 'date32', 'v': '19000'}, datetime.date(2022, 1, 8)),
        (
            {'t': 'ts', 'v': '1700000000123456', 'unit': 'us'},
            datetime.datetime(2023, 11, 14, 22, 13, 20, 123456),
        ),
        (
            {
                't': 'ts',
                'v': '1700000000000000000',
                'unit': 'ns',
                'zone': 'UTC',
            },
            datetime.datetime(
                2023, 11, 14, 22, 13, 20, tzinfo=datetime.timezone.utc
            ),
        ),
        (
            {
                't': 'ts',
                'v': '1700000000000000',
                'unit': 'us',
                'zone': '+05:30',
            },
            datetime.datetime(
                2023,
                11,
                15,
                3,
                43,
                20,
                tzinfo=datetime.timezone(datetime.timedelta(hours=5.5)),
            ),
        ),
    ]
    DatabricksProvider(connection)(
        _step(
            _statement(
                'SELECT ' + ', '.join('?' for _ in scalars),
                *(_value(scalar) for scalar, _ in scalars),
            )
        )
    )
    [(_, bound)] = executed
    assert bound == [value for _, value in scalars]
    assert [type(value) for value in bound] == [
        type(value) for _, value in scalars
    ]
    zoned = bound[-1]
    assert zoned.utcoffset() == datetime.timedelta(hours=5, minutes=30)


@pytest.mark.parametrize(
    'columns, rows',
    [
        (
            {'id': pa.array([1, 2, None, 2**62], pa.int64())},
            [1, 2, None, 2**62],
        ),
        (
            {'id': pa.array(['a', 'é', 'line\nbreak', '"q"', None])},
            ['a', 'é', 'line\nbreak', '"q"', None],
        ),
        ({'id': pa.array([1.5, 0.1, -0.0])}, [1.5, 0.1, -0.0]),
        (
            {
                'id': pa.array([1, 2], pa.int64()),
                'at': pa.array(['x', None]),
            },
            [{'id': 1, 'at': 'x'}, {'id': 2, 'at': None}],
        ),
    ],
    ids=['ints', 'text', 'floats', 'two-columns'],
)
@pytest.mark.parametrize('which', [0, 1], ids=CONNECTIONS)
def test_a_key_set_is_bound_as_the_reference_s_compact_json(
    which: int, columns: dict[str, Any], rows: list[Any]
) -> None:
    connection, executed = _both(lambda sql, _: pa.table({'a': [1]}))[which]
    keys = pa.table(columns)
    DatabricksProvider(connection)(
        _step(
            _statement(
                'SELECT * FROM posexplode(from_json(?, ...)) WHERE t <= ?',
                _key_set(),
                _value({'t': 'i64', 'v': '3'}),
            )
        ),
        {7: keys},
    )
    [(_, bound)] = executed
    payload = json.dumps(rows, separators=(',', ':'))
    # Byte for byte, the key set first where its marker is, the bound after.
    assert bound == [payload, 3]
    assert bound[0] == _reference_dumps(rows)
    assert bound[0].isascii()


@pytest.mark.parametrize('which', [0, 1], ids=CONNECTIONS)
def test_a_key_set_past_the_budget_runs_once_per_chunk_its_batches_moved(
    which: int,
) -> None:
    connection, executed = _both(_batches)[which]
    seeds = [7, 123, 4567, 89, 1, 23456, 78, 9, 1011, 12, 0, 99999]
    budget = 20
    expected = _reference_chunk_rows(seeds, budget)
    assert len(expected) > 2
    answer = DatabricksProvider(connection, budget=budget)(
        _step(
            _statement(
                'SELECT * FROM posexplode(from_json(?, ...)) WHERE t <= ?',
                _key_set(),
                _value({'t': 'i64', 'v': '5'}),
            )
        ),
        {7: pa.table({'id': pa.array(seeds, pa.int64())})},
    )
    # One statement per chunk, each bound the reference's chunk's text.
    assert [bound for _, bound in executed] == [
        [_reference_dumps(chunk), 5] for chunk in expected
    ]
    assert all(len(bound[0]) <= budget for _, bound in executed)
    # The batches in one space: each seed at its position in the key set.
    assert answer.column(BATCH_COLUMN).type == pa.int64()
    assert answer.column(BATCH_COLUMN).to_pylist() == list(range(len(seeds)))
    assert answer.column('seed').to_pylist() == seeds
    assert answer.schema.names == [BATCH_COLUMN, 'seed']


def test_one_chunk_runs_once_and_keeps_the_driver_s_answer() -> None:
    connection, executed = _both(_batches)[0]
    keys = pa.table({'id': pa.array([1, 2, 3], pa.int64())})
    answer = DatabricksProvider(connection)(
        _step(_statement('SELECT ?', _key_set())), {7: keys}
    )
    assert executed == [('SELECT ?', ['[1,2,3]'])]
    # Not moved, so its batch keeps the driver's type.
    assert answer.column(BATCH_COLUMN).type == pa.int32()


def test_an_empty_key_set_is_bound_as_an_empty_array() -> None:
    connection, executed = _both(_batches)[0]
    keys = pa.table({'id': pa.array([], pa.int64())})
    DatabricksProvider(connection)(
        _step(_statement('SELECT ?', _key_set())), {7: keys}
    )
    assert executed == [('SELECT ?', ['[]'])]


def test_chunking_is_the_reference_s_on_a_wide_key_set() -> None:
    rows = [{'a': index, 'b': 'x' * (index % 17)} for index in range(4000)]
    keys = pa.table(
        {
            'a': pa.array([row['a'] for row in rows], pa.int64()),
            'b': pa.array([row['b'] for row in rows]),
        }
    )
    sent: list[str] = []

    def answer(sql: str, parameters: Any) -> Any:
        sent.append(parameters[0])
        return _batches(sql, [json.dumps([0] * len(json.loads(parameters[0])))])

    budget = 10_000
    DatabricksProvider(_Warehouse(answer), budget=budget)(
        _step(_statement('SELECT ?', _key_set())), {7: keys}
    )
    assert sent == [
        _reference_dumps(chunk) for chunk in _reference_chunk_rows(rows, budget)
    ]
    assert databricks.PARAMETER_BUDGET == 900_000


@pytest.mark.parametrize(
    'step, keys, match',
    [
        ({'plan': {'plan': 'host', 'expect': {}}}, None, 'host plan'),
        (_step(), None, 'no statement'),
        (
            _step({'dialect': 'duckdb', 'text': 'SELECT 1', 'params': []}),
            None,
            'another dialect',
        ),
        (_step(_statement('SELECT ?', _key_set())), {}, 'not handed over'),
        (
            _step(_statement('SELECT ?, ?', _key_set(), _key_set())),
            {7: pa.table({'id': [1]})},
            'more than one key set',
        ),
        (
            _step(_statement('SELECT ?', _value({'t': 'nope', 'v': '1'}))),
            None,
            'does not hold',
        ),
        (
            _step(
                _statement(
                    'SELECT ?',
                    _value({'t': 'ts', 'v': '1', 'unit': 'ns'}),
                )
            ),
            None,
            'finer than a microsecond',
        ),
    ],
    ids=[
        'host',
        'empty',
        'dialect',
        'missing',
        'two-key-sets',
        'bad-scalar',
        'nanos',
    ],
)
def test_a_plan_the_provider_cannot_run_raises_before_running(
    step: dict[str, Any], keys: Any, match: str
) -> None:
    connection = _Warehouse(lambda sql, _: pa.table({'a': [1]}))
    with pytest.raises(PlanNotRun, match=match):
        DatabricksProvider(connection)(step, keys)
    assert connection.executed == []


def test_a_chunked_key_set_before_the_answer_is_not_run() -> None:
    connection = _Warehouse(_batches)
    keys = pa.table({'id': pa.array(range(100), pa.int64())})
    with pytest.raises(PlanNotRun, match='not the answer'):
        DatabricksProvider(connection, budget=20)(
            _step(
                _statement('SELECT ?', _key_set()),
                _statement('SELECT 1'),
            ),
            {7: keys},
        )
    assert connection.executed == []


def test_a_chunked_answer_without_batches_is_refused() -> None:
    connection = _Warehouse(lambda sql, _: pa.table({'a': [1]}))
    keys = pa.table({'id': pa.array(range(100), pa.int64())})
    with pytest.raises(PlanNotRun, match='no batch'):
        DatabricksProvider(connection, budget=20)(
            _step(_statement('SELECT ?', _key_set())), {7: keys}
        )


#: A notebook's Spark session as ``sdm_connectors`` recognises one, by its
#: class's name and a ``pyspark`` module (not ``pyspark.sql``, so no
#: installed pyspark is read for its DataFrame class).
_SparkSession = type(
    'SparkSession', (_Session,), {'__module__': 'pyspark.fake_for_tests'}
)


def _databricks_connection() -> Any:
    return type('Connection', (), {'__module__': 'databricks.sql.client'})()


def _sources() -> dict[str, Any]:
    warehouse = DatabricksSource(
        _databricks_connection(), catalog='main', schema='sales'
    )
    spark = DatabricksSource(_SparkSession(lambda sql, _: None), schema='sales')
    return {
        'warehouse': warehouse,
        'spark': spark,
        'metric-view': warehouse.semantic_model('sales_metrics'),
    }


def test_the_client_tells_the_databricks_connections_apart() -> None:
    sources = _sources()
    assert {name: s.backend for name, s in sources.items()} == dict.fromkeys(
        sources, 'databricks'
    )
    assert {
        name: _routing.route_key(source) for name, source in sources.items()
    } == {
        'warehouse': 'databricks-sql',
        'spark': 'databricks-spark',
        'metric-view': 'databricks-metric-view',
    }
    # Anything else is its backend, a Source or not.
    assert _routing.route_key(type('S', (), {'backend': 'duckdb'})()) == (
        'duckdb'
    )
    assert _routing.route_key(object()) is None
    assert isinstance(sources['spark'], Source)


@pytest.mark.parametrize(
    'name, choice, made',
    [
        (
            'warehouse',
            ['main', 'sales'],
            DatabricksWarehouseProvider,
        ),
        ('spark', [None, 'sales'], DatabricksSparkProvider),
    ],
)
def test_the_provider_choice_names_the_dialect_and_the_namespace(
    name: str, choice: list[Any], made: type
) -> None:
    source = _sources()[name]
    for chooser in (core_schema._provider, core_materialize._provider):
        chosen, provider = chooser(source)
        assert chosen == {
            'provider': 'sql',
            'dialect': 'databricks',
            databricks.NAMESPACE_FIELD: choice,
        }
        assert type(provider) is made
        assert provider._connection is source.connection


def test_every_databricks_default_is_old_and_each_connection_its_own_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    groups = (
        _routing.FROM_SOURCE,
        _routing.FROM_SOURCE_TABLES,
        _routing.MATERIALIZE,
        _routing.PREDICT,
    )
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    for paths in groups:
        assert not any(_routing.on_the_core(s, paths) for s in sources.values())
    assert set(_routing.MATERIALIZE['databricks-sql']) == {
        f'query.materialize/indices-{kind}/databricks-sql'
        for kind in (
            'binary',
            'multiclass',
            'regression',
            'link-prediction',
            'ranking',
            'forecast',
        )
    }
    forced = (
        _routing.FROM_SOURCE_DATABRICKS_SQL
        + _routing.MATERIALIZE_DATABRICKS_SQL
        + _routing.PREDICT_DATABRICKS_SQL
    )
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(forced))
    for paths in (_routing.FROM_SOURCE, _routing.MATERIALIZE, _routing.PREDICT):
        assert _routing.on_the_core(sources['warehouse'], paths)
        assert not _routing.on_the_core(sources['spark'], paths)
        # A metric view is no table of its warehouse: T3.5 routes it.
        assert not _routing.on_the_core(sources['metric-view'], paths)
    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.PREDICT_DATABRICKS_SPARK)
    )
    assert _routing.on_the_core(sources['spark'], _routing.PREDICT)
    assert not _routing.on_the_core(sources['warehouse'], _routing.PREDICT)


def _error(module: str, name: str, **attributes: Any) -> type[Exception]:
    return type(name, (Exception,), {'__module__': module, **attributes})


SERVER = _error('databricks.sql.exc', 'ServerOperationError')
ANALYSIS = _error(
    'pyspark.errors.exceptions.captured',
    'AnalysisException',
    getSqlState=lambda self: '42703',
)
CONNECT = _error(
    'pyspark.errors.exceptions.connect',
    'SparkConnectGrpcException',
    getSqlState=lambda self: None,
)


@pytest.mark.parametrize(
    'error, sqlstate, refuses',
    [
        (
            SERVER(
                '[TABLE_OR_VIEW_NOT_FOUND] t cannot be found. SQLSTATE: 42P01'
            ),
            '42P01',
            True,
        ),
        (
            SERVER(
                "[CAST_INVALID_INPUT] 'SQLSTATE: 08001' cannot be cast. "
                'SQLSTATE: 22018\n== SQL (line 1) ==\nSELECT 1 -- SQLSTATE: 08001'
            ),
            '22018',
            True,
        ),
        (
            SERVER('[INSUFFICIENT_PERMISSIONS] no. SQLSTATE: 42501'),
            '42501',
            True,
        ),
        (SERVER('the connection reset. SQLSTATE: 08006'), '08006', False),
        (SERVER('no state at all'), None, False),
        (ANALYSIS('[UNRESOLVED_COLUMN] SQLSTATE: 08001'), '42703', True),
        (CONNECT('lost. SQLSTATE: 08S01'), '08S01', False),
    ],
    ids=[
        'not-found',
        'last-before-the-statement',
        'permission',
        'connection',
        'no-state',
        'spark-answers',
        'spark-connect-message',
    ],
)
def test_a_databricks_failure_reaches_the_core_with_its_state(
    error: Exception, sqlstate: str | None, refuses: bool
) -> None:
    provided, driver = _driver(error)
    assert provided['sqlstate'] == sqlstate
    assert provided.get('refuses_expression', False) is refuses
    if not refuses:
        assert 'refuses_expression' not in provided
    assert provided['kind'] == 'data_source'
    assert driver == f'{type(error).__module__}.{type(error).__qualname__}'
    # The driver's message is never in what the core is told.
    assert not any(
        isinstance(value, str) and ('[' in value or ' ' in value)
        for value in provided.values()
    )


def test_another_driver_s_failure_is_read_as_before() -> None:
    other = _error('psycopg.errors', 'UndefinedTable')(
        'relation does not exist. SQLSTATE: 42P01'
    )
    other.sqlstate = '42P01'  # type: ignore[attr-defined]
    provided, _ = _driver(other)
    assert provided == {
        'kind': 'data_source',
        'retryable': False,
        'sqlstate': '42P01',
        'vendor_code': None,
    }
    plain, _ = _driver(ValueError('SQLSTATE: 42000'))
    assert plain['sqlstate'] is None
    assert 'refuses_expression' not in plain


class _Once:
    """A key set's Arrow stream, as the core hands one over: read once."""

    def __init__(self, table: Any) -> None:
        self._table = table
        self.reads = 0

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        self.reads += 1
        assert self.reads == 1, 'a key set read twice'
        return self._table.__arrow_c_stream__(requested_schema)


def test_a_key_set_two_statements_bind_is_read_once() -> None:
    connection = _Warehouse(_batches)
    keys = _Once(pa.table({'id': pa.array([4, 5], pa.int64())}))
    DatabricksProvider(connection)(
        _step(
            _statement('CACHE ?', _key_set()),
            _statement('SELECT ?', _key_set()),
        ),
        {7: keys},
    )
    assert connection.executed == [
        ('CACHE ?', ['[4,5]']),
        ('SELECT ?', ['[4,5]']),
    ]
    assert keys.reads == 1
