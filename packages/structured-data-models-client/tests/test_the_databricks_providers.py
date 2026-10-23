# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Both Databricks providers' own code over a fake cursor returning the
warehouse's recorded Arrow (T3.4, the brief's exit 5; blind spot B3).

The SQL-warehouse provider runs a plan's statements on one cursor of a
``databricks.sql`` connection and hands back ``cursor.fetchall_arrow()``
(the old client's fetch, DB/sampler.py:152,160,275, DB/table.py:383,430);
the Spark provider runs each through the notebook's session,
``spark.sql(text, args=[...])``, and hands back ``frame.toArrow()``
(CN/backends/spark.py:58-91), refusing by name a session whose frames
cannot (DBR 17.3 or later, serverless environment 4 or later,
CN/backends/spark.py:33-35, :132). Neither reads pandas or converts
anything: the recorded Arrow crosses as recorded (design choice 5).

The recorded Arrow is ``contract/fixtures/databricks/capture.json``'s,
shaped as the design's ``sdm-capture/1``, a synthetic unit fixture recorded
once from the reference on an offline stand-in connection since removed (owner
B-lite); C-1's capture replaces it.
Each fake answers only a statement the capture holds, by its exact text, so
a provider that rewrites a statement fails.

The providers' names are this test's reading of the brief ("the SQL-warehouse
provider (``fetchall_arrow``) and the Spark provider (``spark.sql(...)
.toArrow()``)", in ``sdm_client/_rust/providers/``), beside the DuckDB,
SQLite and Postgres providers' ``<Name>Provider``; a plan is the core's
fetch step as those take it.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT / 'contract' / 'fixtures' / 'databricks' / 'capture.json'


def _providers() -> tuple[Any, Any]:
    from sdm_client._rust.providers.databricks import (
        DatabricksSparkProvider,
        DatabricksSqlProvider,
    )

    return DatabricksSqlProvider, DatabricksSparkProvider


# --- the recorded capture -----------------------------------------------------


def _capture() -> dict[str, Any]:
    capture = json.loads(CAPTURE.read_text(encoding='utf-8'))
    assert capture['format'] == 'sdm-capture/1'
    return capture


def _arrow(result: dict[str, Any]) -> Any:
    data = base64.b64decode(result['arrow'])
    return pa.ipc.open_stream(pa.py_buffer(data)).read_all()


def _recorded() -> dict[str, tuple[list[Any], Any]]:
    r"""Each statement the capture holds, by its text: its parameters and
    its result as the driver fetched it."""
    found: dict[str, tuple[list[Any], Any]] = {}
    for case in _capture()['cases']:
        for statement in case['statements']:
            found.setdefault(
                statement['text'],
                (statement['parameters'], _arrow(statement['result'])),
            )
    return found


def _statement(case: str, kind: str, *, containing: str = '') -> dict[str, Any]:
    for recorded in _capture()['cases']:
        if recorded['case'] != case:
            continue
        for statement in recorded['statements']:
            if statement['kind'] == kind and containing in statement['text']:
                return statement
    raise AssertionError(f'the capture holds no {kind} of {case}')


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


def _key_set(
    attachment: int, relation: str = '__sdm_seeds__'
) -> dict[str, Any]:
    return {
        'param': 'key_set',
        'keys': {
            'attachment': attachment,
            'columns': [
                {'name': 'key', 'nullable': True, 'type': {'type': 'int64'}},
                {'name': 'batch', 'nullable': False, 'type': {'type': 'int64'}},
            ],
        },
        'relation': relation,
    }


def _table(answer: Any) -> Any:
    if isinstance(answer, pa.Table):
        return answer
    reader = pa.RecordBatchReader._import_from_c_capsule(
        answer.__arrow_c_stream__()
    )
    return reader.read_all()


def _providers_over_the_capture(
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
def test_each_provider_hands_back_the_recorded_arrow_as_recorded(
    kind: str,
) -> None:
    r"""The driver's Arrow crosses unchanged: the same schema, types and
    values as recorded (a naive time in microseconds, ``int32`` beside
    ``int64``, a catalog of strings), with no pandas round trip."""
    recorded = _recorded()
    statement = _statement('base.binary', kind)
    expected = _arrow(statement['result'])
    assert expected.num_rows > 0
    for leg, provider, fake in _providers_over_the_capture(recorded):
        if kind == 'lookup':
            keys = json.loads(statement['parameters'][0])
            attachment = pa.table(
                {
                    'key': pa.array(keys, pa.int64()),
                    'batch': pa.array(range(len(keys)), pa.int64()),
                }
            )
            answer = provider(
                _step((statement['text'], [_key_set(7)])), {7: attachment}
            )
        else:
            params = [_text(p) for p in statement['parameters']]
            answer = provider(_step((statement['text'], params)))
        got = _table(answer)
        assert got.schema == expected.schema, leg
        assert got.equals(expected), leg
        assert [text for text, _ in _ran(fake)] == [statement['text']], leg


def test_the_catalog_statements_bind_schema_and_table_as_recorded() -> None:
    r"""A plan's text parameters reach the driver as the old client bound
    them (``lower(?)``, DB/table.py:240), in order: the warehouse cursor's
    ``parameters``, the session's ``args``."""
    recorded = _recorded()
    statement = _statement('key.declared-int32', 'describe/keys')
    parameters = statement['parameters']
    assert parameters[0] == 'shop' and len(parameters) == 2
    for leg, provider, fake in _providers_over_the_capture(recorded):
        provider(_step((statement['text'], [_text(p) for p in parameters])))
        ((text, bound),) = _ran(fake)
        assert text == statement['text'], leg
        assert list(bound) == parameters, leg


def test_a_statement_without_parameters_binds_none() -> None:
    r"""As the old client runs one (``spark.sql(text)``, CN/backends/spark.py:64):
    no parameters, never an empty list Spark would read as markers."""
    recorded = _recorded()
    statement = _statement('base.binary', 'count_rows')
    for leg, provider, fake in _providers_over_the_capture(recorded):
        provider(_step((statement['text'], [])))
        ((_, bound),) = _ran(fake)
        assert not bound, leg


def test_a_key_set_is_bound_as_one_compact_json_parameter() -> None:
    r"""A key set is bound as the old client binds it (DB/sampler.py:39,
    :116-160): one JSON text parameter, written with no whitespace, an array
    of one element per key-set row, in the key set's order, each holding its
    row's key."""
    recorded = _recorded()
    statement = _statement('base.binary', 'lookup')
    keys = json.loads(statement['parameters'][0])
    attachment = pa.table(
        {
            'key': pa.array(keys, pa.int64()),
            'batch': pa.array(range(len(keys)), pa.int64()),
        }
    )
    for leg, provider, fake in _providers_over_the_capture(recorded):
        provider(_step((statement['text'], [_key_set(3)])), {3: attachment})
        ((_, bound),) = _ran(fake)
        (payload,) = list(bound)
        assert isinstance(payload, str), leg
        decoded = json.loads(payload)
        assert json.dumps(decoded, separators=(',', ':')) == payload, leg
        assert isinstance(decoded, list) and len(decoded) == len(keys), leg
        for element, key in zip(decoded, keys, strict=True):
            if isinstance(element, dict):
                element = list(element.values())
            if isinstance(element, list):
                assert key in element, leg
            else:
                assert element == key, leg


def test_an_empty_result_keeps_its_schema() -> None:
    r"""A statement answering no rows (a catalog that declares no key) is an
    empty table of the recorded schema, not an error and not schema-less."""
    recorded = _recorded()
    statement = _statement('base.binary', 'describe/keys')
    expected = _arrow(statement['result'])
    assert expected.num_rows == 0
    for leg, provider, _ in _providers_over_the_capture(recorded):
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
    recorded = _recorded()
    first = _statement('base.binary', 'count_rows')
    last = _statement('base.binary', 'preview')
    for leg, provider, fake in _providers_over_the_capture(recorded):
        got = _table(provider(_step((first['text'], []), (last['text'], []))))
        assert got.equals(_arrow(last['result'])), leg
        assert [text for text, _ in _ran(fake)] == [
            first['text'],
            last['text'],
        ], leg


def test_a_plan_for_another_dialect_or_a_host_is_not_run() -> None:
    r"""As the other providers refuse it: a host plan, or a statement of
    another dialect, raises before any statement runs."""
    recorded = _recorded()
    text = _statement('base.binary', 'count_rows')['text']
    for leg, provider, fake in _providers_over_the_capture(recorded):
        with pytest.raises(Exception):  # noqa: B017, PT011 - the provider's own class
            provider({'plan': {'plan': 'host', 'expect': {}}})
        other = _step((text, []))
        other['plan']['statements'][0]['dialect'] = 'snowflake'
        with pytest.raises(Exception):  # noqa: B017, PT011 - the provider's own class
            provider(other)
        assert _ran(fake) == [], leg


def test_the_warehouse_provider_leaves_the_connection_open() -> None:
    r"""The connection is the caller's (CN/source.py: a source never closes
    the connection it was given): each request's cursor is closed, the
    connection never."""
    recorded = _recorded()
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
    LTS, serverless environment 4) is refused with the old client's reason,
    naming the runtime to use, never read through pandas."""
    recorded = _recorded()
    _, spark_provider = _providers()
    session = _Session(recorded, arrow=False)
    statement = _statement('base.binary', 'count_rows')
    with pytest.raises(Exception, match=r'17\.3') as raised:  # noqa: PT011 - the old client's ConnectorError
        spark_provider(session)(_step((statement['text'], [])))
    assert 'serverless environment 4' in str(raised.value)
