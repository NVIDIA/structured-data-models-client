# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL and Spark adapters preserve Arrow data, bindings and caller ownership.

Driver doubles accept exact authored plan text and never use pandas.
These are local adapter tests, not live warehouse validation.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest


def _providers() -> tuple[Any, Any]:
    from sdm_client._rust.providers.databricks import (
        DatabricksSparkProvider,
        DatabricksSqlProvider,
    )

    return DatabricksSqlProvider, DatabricksSparkProvider


def _statement(case: str, kind: str) -> dict[str, Any]:
    """Small input tables test transport, not SQL generation or sampling."""
    if kind == 'preview':
        text, parameters = 'SELECT id, age, ts, label FROM shop.users', []
        result = pa.table(
            {
                'id': pa.array([1, 2], pa.int64()),
                'age': pa.array([20, None], pa.int32()),
                'ts': pa.array([0, None], pa.timestamp('us')),
                'label': pa.array(['café', None], pa.string()),
            }
        )
    elif kind == 'describe/columns':
        text, parameters = (
            'SELECT column_name, data_type FROM catalog WHERE schema_name = ? AND table_name = ?',
            ['shop', 'users'],
        )
        result = pa.table(
            {'column_name': ['id', 'age'], 'data_type': ['BIGINT', 'INT']}
        )
    elif kind == 'lookup':
        text, parameters = (
            'SELECT batch, id FROM lookup_keys(?)',
            ['["2", "1", "2"]'],
        )
        result = pa.table(
            {
                'batch': pa.array([0, 1, 2], pa.int64()),
                'id': pa.array([2, 1, 2], pa.int64()),
            }
        )
    elif kind == 'count_rows':
        text, parameters = 'SELECT COUNT(*) AS n FROM shop.users', []
        result = pa.table({'n': pa.array([2], pa.int64())})
    else:
        assert kind == 'describe/keys'
        declared = case == 'key.declared-int32'
        text, parameters = (
            f'SELECT constraint_name FROM keys_{"declared" if declared else "empty"} WHERE schema_name = ? AND table_name = ?',
            ['shop', 'users'],
        )
        result = pa.table(
            {
                'constraint_name': pa.array(
                    ['users_pk'] if declared else [], pa.string()
                )
            }
        )
    return {'text': text, 'parameters': parameters, 'result': result}


def _answers() -> dict[str, tuple[list[Any], pa.Table]]:
    statements = [
        _statement('base.binary', kind)
        for kind in (
            'preview',
            'describe/columns',
            'lookup',
            'count_rows',
            'describe/keys',
        )
    ]
    statements.append(_statement('key.declared-int32', 'describe/keys'))
    return {s['text']: (s['parameters'], s['result']) for s in statements}


# --- the fakes ------------------------------------------------------------------


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._result: Any = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.closed_cursors += 1

    def execute(
        self, operation: str, parameters: Any = None, **options: Any
    ) -> _Cursor:
        self._connection.executed.append((operation, parameters))
        if operation not in self._connection.recorded:
            raise AssertionError(f'no recorded result for:\n{operation}')
        self._result = self._connection.recorded[operation][1]
        return self

    def fetchall_arrow(self) -> Any:
        self._connection.fetched += 1
        return self._result

    def fetchall(self) -> Any:
        raise AssertionError('rows read where Arrow is the fetch')

    def fetchone(self) -> Any:
        raise AssertionError('rows read where Arrow is the fetch')


class _Connection:
    r"""A ``databricks.sql`` connection over the capture."""

    def __init__(self, recorded: dict[str, tuple[list[Any], Any]]) -> None:
        self.recorded = recorded
        self.executed: list[tuple[str, Any]] = []
        self.fetched = 0
        self.closed_cursors = 0

    def cursor(self, *args: Any, **kwargs: Any) -> _Cursor:
        return _Cursor(self)

    def close(self) -> None:
        raise AssertionError("a provider closed the caller's connection")


class _Frame:
    def __init__(self, table: Any, *, arrow: bool) -> None:
        self._table = table
        if arrow:
            self.toArrow = lambda: table

    def toPandas(self) -> Any:  # noqa: N802 - Spark's name
        raise AssertionError('pandas read where Arrow is the fetch')

    def collect(self) -> Any:
        raise AssertionError('rows read where Arrow is the fetch')


class _Session:
    r"""A notebook's Spark session over the capture: ``sql`` and
    ``version``, all a provider may use; its frames answer ``toArrow`` only
    where ``arrow`` (Spark 4, DBR 17.3)."""

    def __init__(
        self, recorded: dict[str, tuple[list[Any], Any]], *, arrow: bool = True
    ) -> None:
        self.recorded = recorded
        self.version = '4.0.0' if arrow else '3.5.0'
        self._arrow = arrow
        self.ran: list[tuple[str, Any]] = []

    def sql(self, sqlQuery: str, args: Any = None, **kwargs: Any) -> _Frame:  # noqa: N803 - Spark's name
        self.ran.append((sqlQuery, args))
        if sqlQuery not in self.recorded:
            raise AssertionError(f'no recorded result for:\n{sqlQuery}')
        return _Frame(self.recorded[sqlQuery][1], arrow=self._arrow)

    def stop(self) -> None:
        raise AssertionError("a provider stopped the notebook's session")


# --- plans as the core hands them over ----------------------------------------------


def _text(value: str) -> dict[str, Any]:
    return {'param': 'value', 'value': {'t': 'utf8', 'v': value}}


def _step(*statements: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        'plan': {
            'plan': 'sql',
            'statements': [
                {'dialect': 'databricks', 'text': text, 'params': params}
                for text, params in statements
            ],
        }
    }


def _key_text(attachment: int) -> dict[str, Any]:
    r"""A key set's parameter as the core plans it on Databricks:
    the JSON text the core wrote, a one-cell attachment."""
    return {'param': 'text', 'attachment': attachment}


def _cell(text: str) -> Any:
    return pa.table({'keys': pa.array([text], pa.string())})


def _table(answer: Any) -> Any:
    if isinstance(answer, pa.Table):
        return answer
    reader = pa.RecordBatchReader._import_from_c_capsule(
        answer.__arrow_c_stream__()
    )
    return reader.read_all()


def _providers_over_answers(
    recorded: dict[str, tuple[list[Any], Any]],
) -> list[tuple[str, Any, Any]]:
    r"""Each provider over its fake: its name, the provider, the fake."""
    sql_provider, spark_provider = _providers()
    connection = _Connection(recorded)
    session = _Session(recorded)
    return [
        ('databricks-sql', sql_provider(connection), connection),
        ('databricks-spark', spark_provider(session), session),
    ]


def _ran(fake: Any) -> list[tuple[str, Any]]:
    return fake.executed if isinstance(fake, _Connection) else fake.ran


# --- the tests --------------------------------------------------------------------


@pytest.mark.parametrize('kind', ['preview', 'describe/columns', 'lookup'])
def test_each_provider_preserves_arrow_values_types_and_nulls(
    kind: str,
) -> None:
    r"""The driver's Arrow crosses unchanged: the same schema, types and
    authored values (a naive time in microseconds, ``int32`` beside
    ``int64``, a catalog of strings), with no pandas round trip."""
    recorded = _answers()
    statement = _statement('base.binary', kind)
    expected = statement['result']
    assert expected.num_rows > 0
    for leg, provider, fake in _providers_over_answers(recorded):
        if kind == 'lookup':
            answer = provider(
                _step((statement['text'], [_key_text(7)])),
                {7: _cell(statement['parameters'][0])},
            )
        else:
            params = [_text(p) for p in statement['parameters']]
            answer = provider(_step((statement['text'], params)))
        got = _table(answer)
        assert got.schema == expected.schema, leg
        assert got.equals(expected), leg
        assert [text for text, _ in _ran(fake)] == [statement['text']], leg


def test_the_catalog_statements_bind_schema_and_table_as_answers() -> None:
    """Schema/table bindings preserve their order on both driver APIs."""
    recorded = _answers()
    statement = _statement('key.declared-int32', 'describe/keys')
    parameters = statement['parameters']
    assert parameters[0] == 'shop' and len(parameters) == 2
    for leg, provider, fake in _providers_over_answers(recorded):
        provider(_step((statement['text'], [_text(p) for p in parameters])))
        ((text, bound),) = _ran(fake)
        assert text == statement['text'], leg
        assert list(bound) == parameters, leg


def test_a_statement_without_parameters_binds_none() -> None:
    """Unparameterized plans do not introduce Spark argument markers."""
    recorded = _answers()
    statement = _statement('base.binary', 'count_rows')
    for leg, provider, fake in _providers_over_answers(recorded):
        provider(_step((statement['text'], [])))
        ((_, bound),) = _ran(fake)
        assert not bound, leg


def test_a_key_set_s_text_is_bound_as_one_parameter_as_written() -> None:
    """The attachment is bound byte-for-byte as one JSON text parameter."""
    recorded = _answers()
    statement = _statement('base.binary', 'lookup')
    (payload,) = statement['parameters']
    for leg, provider, fake in _providers_over_answers(recorded):
        provider(
            _step((statement['text'], [_key_text(3)])), {3: _cell(payload)}
        )
        ((_, bound),) = _ran(fake)
        assert list(bound) == [payload], leg


def test_an_empty_result_keeps_its_schema() -> None:
    r"""A statement answering no rows (a catalog that declares no key) is an
    empty table of the declared schema, not an error and not schema-less."""
    recorded = _answers()
    statement = _statement('base.binary', 'describe/keys')
    expected = statement['result']
    assert expected.num_rows == 0
    for leg, provider, _ in _providers_over_answers(recorded):
        got = _table(
            provider(
                _step(
                    (
                        statement['text'],
                        [_text(p) for p in statement['parameters']],
                    )
                )
            )
        )
        assert got.num_rows == 0, leg
        assert got.schema == expected.schema, leg


def test_the_last_statement_s_rows_are_the_answer() -> None:
    r"""A plan of several statements runs each, in order, on one session,
    and answers with the last one's rows (the contract's ``FetchPlan``)."""
    recorded = _answers()
    first = _statement('base.binary', 'count_rows')
    last = _statement('base.binary', 'preview')
    for leg, provider, fake in _providers_over_answers(recorded):
        got = _table(provider(_step((first['text'], []), (last['text'], []))))
        assert got.equals(last['result']), leg
        assert [text for text, _ in _ran(fake)] == [
            first['text'],
            last['text'],
        ], leg


def test_a_plan_for_another_dialect_or_a_host_is_not_run() -> None:
    r"""As the other providers refuse it: a host plan, or a statement of
    another dialect, raises before any statement runs."""
    recorded = _answers()
    text = _statement('base.binary', 'count_rows')['text']
    for leg, provider, fake in _providers_over_answers(recorded):
        with pytest.raises(Exception):  # noqa: B017, PT011 - the provider's own class
            provider({'plan': {'plan': 'host', 'expect': {}}})
        other = _step((text, []))
        other['plan']['statements'][0]['dialect'] = 'snowflake'
        with pytest.raises(Exception):  # noqa: B017, PT011 - the provider's own class
            provider(other)
        assert _ran(fake) == [], leg


def test_the_warehouse_provider_leaves_the_connection_open() -> None:
    """Close request cursors while retaining the caller-owned connection."""
    recorded = _answers()
    sql_provider, _ = _providers()
    connection = _Connection(recorded)
    provider = sql_provider(connection)
    statement = _statement('base.binary', 'count_rows')
    provider(_step((statement['text'], [])))
    provider(_step((statement['text'], [])))
    assert connection.fetched == 2
    assert connection.closed_cursors >= 2


def test_a_spark_session_without_arrow_is_refused_by_name() -> None:
    r"""A session whose frames cannot return Arrow (before Spark 4: DBR 17.3
    LTS, serverless environment 4) is refused with an actionable error,
    naming the runtime to use, never read through pandas."""
    recorded = _answers()
    _, spark_provider = _providers()
    session = _Session(recorded, arrow=False)
    statement = _statement('base.binary', 'count_rows')
    with pytest.raises(Exception, match=r'17\.3') as raised:  # noqa: PT011 - adapter-specific connector error
        spark_provider(session)(_step((statement['text'], [])))
    assert 'serverless environment 4' in str(raised.value)
