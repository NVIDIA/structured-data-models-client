# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Snowflake serving transport: naming, the payload cap, and the reply.

A service is named rather than addressed by URL, the payload travels as a bind
parameter, and the reply arrives as a SQL row rather than a prediction
envelope.
"""

from __future__ import annotations

import importlib.util
import json
from unittest import mock

import pytest
from sdm_api.generated.tfm_api import TFMOperations
from sdm_relational_engine.client.snowflake_serving import (
    RESPONSE_COLUMN,
    SnowflakeServingClient,
)
from sdm_relational_engine.client.transport import (
    RFMTransport,
    ServingResponse,
)
from sdm_relational_engine.exceptions import HTTPException

_TOK = 'tok3n'

PREDICTION = TFMOperations.run_prediction.endpoint


def _connector_installed() -> bool:
    """Whether snowflake.connector can be imported.

    ``find_spec`` on a dotted name imports the parent package, so it raises
    rather than returning None when ``snowflake`` is absent altogether, which
    is exactly the environment this guard exists for.
    """
    try:
        return importlib.util.find_spec('snowflake.connector') is not None
    except ModuleNotFoundError:
        return False


# The transport imports snowflake.connector lazily, so it works without it and
# falls back to the pyformat placeholder. Only the tests that read the real
# module's paramstyle need it installed.
requires_connector = pytest.mark.skipif(
    not _connector_installed(),
    reason='snowflake-connector-python is not installed',
)


class _Row:
    """A Snowpark row, which exposes its columns through ``as_dict``."""

    def __init__(self, body: str) -> None:
        self._body = body

    def as_dict(self) -> dict[str, str]:
        return {RESPONSE_COLUMN.upper(): self._body}


class _Session:
    """A Snowpark session that records the statement and its bind parameters."""

    def __init__(self, rows=None) -> None:
        self.statements: list[tuple[str, list | None]] = []
        self._rows = rows if rows is not None else [_Row('{"ok": true}')]

    def sql(self, statement, params=None):
        self.statements.append((statement, params))
        session = self

        class _Result:
            def collect(self):
                return [] if statement.startswith('ALTER') else session._rows

        return _Result()


def _client(session=None, **kwargs) -> SnowflakeServingClient:
    return SnowflakeServingClient('DB.SC.SVC', session or _Session(), **kwargs)


def test_it_satisfies_the_transport_protocol() -> None:
    assert isinstance(_client(), RFMTransport)


@pytest.mark.parametrize(
    'name',
    [
        'https://acct.snowflakecomputing.com',
        'acct.snowflakecomputing.com/api',
        'has space',
        'drops;table',
        'quoted"name',
        'a.b.c.d',
        '',
        '   ',
    ],
)
def test_anything_but_a_service_name_is_refused(name: str) -> None:
    """A service is named, not addressed, and the name is the one part of the
    statement that cannot be bound.
    """
    with pytest.raises(ValueError):
        SnowflakeServingClient(name, _Session())


def test_the_rejection_never_echoes_the_value() -> None:
    """The likeliest wrong input is a pasted connection string, which can carry
    credentials.
    """
    secret = f'https://user:{_TOK}@acct.snowflakecomputing.com'
    with pytest.raises(ValueError) as caught:
        SnowflakeServingClient(secret, _Session())
    assert 'tok3n' not in str(caught.value)
    assert secret not in str(caught.value)


@pytest.mark.parametrize('name', ['SVC', 'SC.SVC', 'DB.SC.SVC', '_svc$1'])
def test_bare_and_qualified_names_are_accepted(name: str) -> None:
    assert SnowflakeServingClient(name, _Session()).service == name


@pytest.mark.parametrize('bad', [True, False, 0, -1, 'many'])
def test_a_boolean_or_out_of_range_cap_is_refused(bad) -> None:
    """``True`` is an ``int``: an unguarded boolean cap rejects every request."""
    with pytest.raises(ValueError):
        _client(max_request_bytes=bad)


@pytest.mark.parametrize('bad', [True, False, 0, -1, 'soon'])
def test_a_boolean_or_out_of_range_timeout_is_refused(bad) -> None:
    """``float(True)`` is one second, which is the cold start this exists for."""
    with pytest.raises(ValueError):
        _client(timeout=bad)


def test_the_payload_is_bound_never_interpolated() -> None:
    """A canonical request is megabytes of sampled customer rows, and query text
    is logged and shown in history.
    """
    session = _Session()
    _client(session)._request(PREDICTION, json={'model': 'kumo-relational'})
    statement, params = session.statements[-1]
    assert '?' in statement
    assert 'kumo-relational' not in statement
    assert params == [
        json.dumps({'model': 'kumo-relational'}, separators=(',', ':'))
    ]


@requires_connector
def test_snowpark_is_bound_with_a_question_mark() -> None:
    """Snowpark binds with ``?`` whatever the connector paramstyle is set to."""
    import snowflake.connector as connector_module

    session = _Session()
    with mock.patch.object(connector_module, 'paramstyle', 'pyformat'):
        _client(session)._request(PREDICTION, json={})
    statement, _ = session.statements[-1]
    assert '(?)' in statement
    assert '%s' not in statement


def test_the_statement_names_the_service_and_method() -> None:
    session = _Session()
    SnowflakeServingClient('DB.SC.SVC', session, method='PREDICT')._request(
        PREDICTION, json={}
    )
    statement, _ = session.statements[-1]
    assert 'DB.SC.SVC!PREDICT(?)' in statement


def test_an_oversized_payload_is_refused_before_the_statement() -> None:
    session = _Session()
    client = _client(session, max_request_bytes=64)
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'pad': 'x' * 256})
    assert caught.value.status_code == 413
    assert not any(not s.startswith('ALTER') for s, _ in session.statements), (
        'paid to send it anyway'
    )


def test_only_one_shot_prediction_is_supported() -> None:
    """A caller reaching for a session should fail loudly rather than be
    reinterpreted as a prediction.
    """
    from sdm_api.endpoints import Endpoint, HTTPMethod

    with pytest.raises(HTTPException) as caught:
        _client()._request(
            Endpoint(path='/v1/sessions', method=HTTPMethod.POST)
        )
    assert caught.value.status_code == 501


def test_a_non_object_request_is_refused() -> None:
    with pytest.raises(HTTPException) as caught:
        _client()._request(PREDICTION, json=['not', 'an', 'object'])
    assert caught.value.status_code == 400


def test_the_response_column_is_returned_verbatim() -> None:
    session = _Session(rows=[_Row('{"predictions": []}')])
    response = _client(session)._request(PREDICTION, json={})
    assert isinstance(response, ServingResponse)
    assert response.ok
    assert response.json() == {'predictions': []}


@pytest.mark.parametrize(
    'rows', [[], [_Row('a'), _Row('b')]], ids=['none', 'two']
)
def test_anything_but_one_row_is_a_bad_gateway(rows) -> None:
    with pytest.raises(HTTPException) as caught:
        _client(_Session(rows=rows))._request(PREDICTION, json={})
    assert caught.value.status_code == 502


def test_a_row_without_the_response_column_is_a_bad_gateway() -> None:
    class _Empty:
        def as_dict(self):
            return {'SOMETHING_ELSE': 'x'}

    with pytest.raises(HTTPException) as caught:
        _client(_Session(rows=[_Empty()]))._request(PREDICTION, json={})
    assert caught.value.status_code == 502


def test_a_driver_failure_never_echoes_the_provider_message() -> None:
    """A Snowflake error can name the account, the warehouse and the statement,
    and the statement is where the customer rows were bound.
    """

    class _Failing(_Session):
        def sql(self, statement, params=None):
            if statement.startswith('ALTER'):
                return super().sql(statement, params)
            raise RuntimeError('acct-xy12345.snowflakecomputing.com refused')

    with pytest.raises(HTTPException) as caught:
        _client(_Failing())._request(PREDICTION, json={})
    assert 'snowflakecomputing.com' not in caught.value.detail
    assert 'DB.SC.SVC' in caught.value.detail


@requires_connector
@pytest.mark.parametrize(
    ('paramstyle', 'expected'),
    [('pyformat', '%s'), ('qmark', '?'), ('format', '%s')],
)
def test_the_connector_placeholder_follows_its_paramstyle(
    paramstyle: str, expected: str
) -> None:
    """The connector default is ``pyformat``, so a hardcoded ``?`` fails.

    It does not fail as a binding error either. The connector falls through to
    string formatting and reports "not all arguments converted during string
    formatting", which names neither the placeholder nor the paramstyle, so
    this is pinned rather than left to be rediscovered against a live account.
    """
    import snowflake.connector as connector_module

    class _Cursor:
        statement = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, statement, params=None):
            if not statement.startswith('ALTER'):
                type(self).statement = statement

        def fetchall(self):
            return [('{"ok": true}',)]

    class _Connection:
        def cursor(self):
            return _Cursor()

    with mock.patch.object(connector_module, 'paramstyle', paramstyle):
        _client(_Connection())._request(PREDICTION, json={})
    assert f'({expected})' in _Cursor.statement


def test_a_raw_connector_connection_is_accepted() -> None:
    """A notebook holds a Snowpark session; a script more often holds a
    connector connection.
    """

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, statement, params=None):
            self.statement, self.params = statement, params

        def fetchall(self):
            return [('{"ok": true}',)]

    class _Connection:
        def cursor(self):
            return _Cursor()

    response = _client(_Connection())._request(PREDICTION, json={})
    assert response.json() == {'ok': True}


def test_the_service_function_envelope_is_unwrapped() -> None:
    """A service method returns the model's output row as an object keyed by
    column name, so the canonical response arrives nested one level.

    The SQL column and the model's column share a name, so the nesting is
    invisible until a parser reports a missing 'id' and blames the model.
    """
    canonical = '{"id":"pred_1","model":"kumo-relational","predictions":[]}'
    session = _Session(rows=[_Row(json.dumps({RESPONSE_COLUMN: canonical}))])
    response = _client(session)._request(PREDICTION, json={})
    assert response.json()['id'] == 'pred_1'


def test_an_unwrapped_response_is_left_alone() -> None:
    """A canonical response is also a JSON object; it just has no
    'response_json' member, so it must survive untouched.
    """
    canonical = (
        '{"id":"pred_2","predictions":[{"id":"1000","prediction":94.0}]}'
    )
    session = _Session(rows=[_Row(canonical)])
    response = _client(session)._request(PREDICTION, json={})
    assert response.json()['id'] == 'pred_2'
    assert response.json()['predictions'][0]['prediction'] == 94.0


def test_a_non_json_body_is_returned_verbatim() -> None:
    session = _Session(rows=[_Row('not json at all')])
    assert (
        _client(session)._request(PREDICTION, json={}).text == 'not json at all'
    )


def test_the_placeholder_falls_back_when_the_connector_is_absent() -> None:
    """The transport must import and work without snowflake-connector-python,
    which is what the lazy import inside the placeholder lookup is for.
    """
    with mock.patch.dict('sys.modules', {'snowflake.connector': None}):
        assert SnowflakeServingClient._connector_placeholder() == '%s'


def test_a_prediction_survives_a_refused_alter_session() -> None:
    """A stored procedure may not run ALTER SESSION.

    Snowflake rejects it with "Unsupported statement type", which said nothing
    about the request; the transport mapped it to 503 "temporarily unavailable,
    likely at capacity" and no prediction could be made from a procedure at all.
    """

    class _NoAlter(_Session):
        def sql(self, statement, params=None):
            if statement.startswith('ALTER'):
                raise RuntimeError("Unsupported statement type 'ALTER_SESSION'")
            return super().sql(statement, params)

    response = _client(_NoAlter())._request(PREDICTION, json={})
    assert response.json() == {'ok': True}


def test_the_connector_path_also_survives_a_refused_alter_session() -> None:
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, statement, params=None):
            if statement.startswith('ALTER'):
                raise RuntimeError("Unsupported statement type 'ALTER_SESSION'")

        def fetchall(self):
            return [('{"ok": true}',)]

    class _Connection:
        def cursor(self):
            return _Cursor()

    assert _client(_Connection())._request(PREDICTION, json={}).json() == {
        'ok': True
    }
