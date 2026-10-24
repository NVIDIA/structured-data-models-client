# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Databricks providers' own code, over a fake cursor (T3.4).

Both connections a Databricks source takes are faked here: a SQL warehouse
connection, whose cursor records ``execute(sql, parameters)`` and answers
``fetchall_arrow()``, and a Spark session (``.sql()``, ``.version``, a frame
with ``toArrow()``) wrapped in ``sdm_connectors``' ``SparkConnection``. What
is checked is what the provider binds, as the reference's sampler binds it
(the key set's text the core wrote, T3.28, and the scalars), what it
answers, how the client tells the connections apart and routes them, and
how a Databricks failure reaches the core.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest
from sdm_connectors import DatabricksSource, Source
from sdm_connectors.backends.spark import SparkConnection

from sdm_client import _routing
from sdm_client._rust import providers
from sdm_client._rust.providers.databricks import (
    DatabricksProvider,
    DatabricksSparkProvider,
    DatabricksSqlProvider,
    PlanNotRun,
)
from sdm_client._rust.run import _driver

Answer = Any


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


def _text(attachment: int = 7) -> dict[str, Any]:
    return {'param': 'text', 'attachment': attachment}


def _key_set(attachment: int = 7) -> dict[str, Any]:
    return {
        'param': 'key_set',
        'keys': {'attachment': attachment, 'columns': []},
        'relation': 'TMP',
    }


def _value(scalar: dict[str, Any]) -> dict[str, Any]:
    return {'param': 'value', 'value': scalar}


def _cell(text: str) -> Any:
    r"""A key set's text as the core hands it over: one text cell."""
    return pa.table({'keys': pa.array([text], pa.string())})


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


@pytest.mark.parametrize('which', [0, 1], ids=CONNECTIONS)
def test_a_key_set_s_text_is_bound_as_the_core_wrote_it(which: int) -> None:
    # The core writes the key set's JSON (T3.28): the provider binds the
    # one cell it is handed where the text parameter is, as it is.
    connection, executed = _both(lambda sql, _: pa.table({'a': [1]}))[which]
    text = '[{"key":1,"end":"2024-03-01 00:00:00"},{"key":2,"end":null}]'
    answer = DatabricksProvider(connection)(
        _step(
            _statement(
                'SELECT * FROM posexplode(from_json(?, ...)) WHERE t <= ?',
                _text(),
                _value({'t': 'i64', 'v': '3'}),
            )
        ),
        {7: _cell(text)},
    )
    assert executed == [
        (
            'SELECT * FROM posexplode(from_json(?, ...)) WHERE t <= ?',
            [text, 3],
        )
    ]
    # One statement, one answer: the driver's, as it came.
    assert answer.to_pydict() == {'a': [1]}


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
        (_step(_statement('SELECT ?', _text())), {}, 'not handed over'),
        (
            _step(_statement('SELECT ?', _key_set())),
            {7: pa.table({'id': [1]})},
            'unknown kind',
        ),
        (
            _step(_statement('SELECT ?', _text())),
            {7: pa.table({'keys': ['[1]', '[2]']})},
            'not one text',
        ),
        (
            _step(_statement('SELECT ?', _text())),
            {7: pa.table({'keys': pa.array([1], pa.int64())})},
            'not one text',
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
        'arrow-key-set',
        'two-cells',
        'not-text',
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
    assert {name: source.kind for name, source in sources.items()} == {
        'warehouse': 'databricks-sql',
        'spark': 'databricks-spark',
        'metric-view': 'databricks-metric-view',
    }
    # An object that is no source has no route, whatever paths are asked.
    assert not _routing.on_the_core(object(), _routing.FROM_SOURCE)
    assert isinstance(sources['spark'], Source)


@pytest.mark.parametrize(
    'name, choice, made',
    [
        (
            'warehouse',
            ['main', 'sales'],
            DatabricksSqlProvider,
        ),
        ('spark', [None, 'sales'], DatabricksSparkProvider),
    ],
)
def test_the_provider_choice_names_the_dialect_and_the_namespace(
    name: str, choice: list[Any], made: type
) -> None:
    source = _sources()[name]
    chosen, provider = providers.for_source(source)
    assert chosen == {
        'provider': 'sql',
        'dialect': 'databricks',
        'namespace': choice,
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
CONDITION = _error(
    'pyspark.errors.exceptions.captured',
    'AnalysisException',
    getSqlState=lambda self: '42501',
    getCondition=lambda self: 'PERMISSION_DENIED',
)


@pytest.mark.parametrize(
    'error, sqlstate, error_class',
    [
        (
            SERVER(
                '[TABLE_OR_VIEW_NOT_FOUND] t cannot be found. SQLSTATE: 42P01'
            ),
            '42P01',
            'TABLE_OR_VIEW_NOT_FOUND',
        ),
        (
            SERVER(
                "[CAST_INVALID_INPUT] 'SQLSTATE: 08001' cannot be cast. "
                'SQLSTATE: 22018\n== SQL (line 1) ==\nSELECT 1 -- SQLSTATE: 08001'
            ),
            '22018',
            'CAST_INVALID_INPUT',
        ),
        (
            SERVER('[INSUFFICIENT_PERMISSIONS] no. SQLSTATE: 42501'),
            '42501',
            'INSUFFICIENT_PERMISSIONS',
        ),
        (SERVER('the connection reset. SQLSTATE: 08006'), '08006', None),
        (SERVER('no state at all'), None, None),
        (
            SERVER('[UNRESOLVED_COLUMN.WITH_SUGGESTION] no state'),
            None,
            'UNRESOLVED_COLUMN.WITH_SUGGESTION',
        ),
        (
            ANALYSIS('[UNRESOLVED_COLUMN] SQLSTATE: 08001'),
            '42703',
            'UNRESOLVED_COLUMN',
        ),
        (
            CONDITION('[NOT_THIS_ONE] SQLSTATE: 42501'),
            '42501',
            'PERMISSION_DENIED',
        ),
        (CONNECT('lost. SQLSTATE: 08S01'), '08S01', None),
    ],
    ids=[
        'not-found',
        'last-before-the-statement',
        'permission',
        'connection',
        'no-state',
        'class-without-state',
        'spark-answers',
        'spark-answers-its-condition',
        'spark-connect-message',
    ],
)
def test_a_databricks_failure_reaches_the_core_with_its_state_and_class(
    error: Exception, sqlstate: str | None, error_class: str | None
) -> None:
    # Read by the provider whose driver failed (its ``error_fields``, the
    # provider protocol's hook), never by the run loop naming a backend;
    # what the failure means is the core's to say (T3.27: the data layer's
    # ``failure``), so the provider hands over facts, never a judgement.
    provided, driver = _driver(error, DatabricksSqlProvider(None))
    assert provided['sqlstate'] == sqlstate
    assert provided.get('error_class') == error_class
    if error_class is None:
        assert 'error_class' not in provided
    assert 'refuses_expression' not in provided
    assert provided['kind'] == 'data_source'
    assert driver == f'{type(error).__module__}.{type(error).__qualname__}'
    # The internal diagnostic is passed unread, independently of the
    # structured state and class. Public errors sanitize it in the core.
    assert provided['driver_message'] == str(error)


def test_another_driver_s_failure_is_read_as_before() -> None:
    other = _error('psycopg.errors', 'UndefinedTable')(
        'relation does not exist. SQLSTATE: 42P01'
    )
    other.sqlstate = '42P01'  # type: ignore[attr-defined]
    provided, _ = _driver(other, DatabricksSqlProvider(None))
    assert provided == {
        'kind': 'data_source',
        'retryable': False,
        'sqlstate': '42P01',
        'vendor_code': None,
        'driver_message': str(other),
    }
    plain, _ = _driver(
        ValueError('SQLSTATE: 42000'), DatabricksSqlProvider(None)
    )
    assert plain['sqlstate'] is None
    assert 'error_class' not in plain


def test_a_provider_with_no_reading_of_failures_leaves_them_as_drivers_set_them() -> (
    None
):
    error = _error('databricks.sql.exc', 'ServerOperationError')(
        '[TABLE_OR_VIEW_NOT_FOUND] SQLSTATE: 42P01'
    )
    provided, _ = _driver(error, lambda step, keys=None: None)
    assert provided['sqlstate'] is None
    assert 'error_class' not in provided


class _Once:
    """A text's Arrow stream, as the core hands one over: read once."""

    def __init__(self, table: Any) -> None:
        self._table = table
        self.reads = 0

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        self.reads += 1
        assert self.reads == 1, 'a text read twice'
        return self._table.__arrow_c_stream__(requested_schema)


def test_a_text_two_statements_bind_is_read_once() -> None:
    connection = _Warehouse(lambda sql, _: pa.table({'a': [1]}))
    keys = _Once(_cell('[4,5]'))
    DatabricksProvider(connection)(
        _step(
            _statement('CACHE ?', _text()),
            _statement('SELECT ?', _text()),
        ),
        {7: keys},
    )
    assert connection.executed == [
        ('CACHE ?', ['[4,5]']),
        ('SELECT ?', ['[4,5]']),
    ]
    assert keys.reads == 1
