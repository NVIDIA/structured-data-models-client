# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sdm_connectors import connect
from sdm_connectors.backends import mark_owned, owns_connection
from sdm_connectors.sql import (
    ConnectorError,
    check_connect_args,
)


class _Conn:
    pass


class _SlottedConn:
    r"""Weakref-able but rejects arbitrary attribute writes (like a borrowed
    Snowpark session that only exposes a fixed API)."""

    __slots__ = ('__weakref__',)


def test_unmarked_connection_is_treated_as_owned():
    assert owns_connection(_Conn()) is True


def test_marked_owned_connection_is_owned():
    connection = _Conn()
    mark_owned(connection, True)
    assert owns_connection(connection) is True


def test_borrowed_connection_is_not_owned():
    connection = _Conn()
    mark_owned(connection, False)
    assert owns_connection(connection) is False


def test_borrowed_connection_rejecting_attribute_writes_is_not_owned():
    connection = _SlottedConn()
    mark_owned(connection, False)
    assert owns_connection(connection) is False


def test_check_connect_args_suggests_the_intended_name():
    with pytest.raises(ConnectorError) as excinfo:
        check_connect_args('snowflake', {'shema': 'X'}, {'schema', 'database'})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert "did you mean 'schema'" in excinfo.value.message
    assert excinfo.value.details['arguments'] == ['shema']


def test_check_connect_args_accepts_known_arguments():
    check_connect_args('snowflake', {'schema': 'X'}, {'schema', 'database'})


@pytest.mark.parametrize(
    'backend,module,typo,intended',
    [
        ('snowflake', 'snowflake.connector', 'shema', 'schema'),
        ('databricks', 'databricks.sql', 'catalg', 'catalog'),
    ],
)
def test_connect_rejects_unknown_kwargs(backend, module, typo, intended):
    """Both drivers accept ``**kwargs`` and silently drop what they do not know,
    so the typo must be rejected before a connection is attempted.
    """
    pytest.importorskip(module)
    with pytest.raises(ConnectorError) as excinfo:
        connect(backend, **{typo: 'X'})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert f'did you mean {intended!r}' in excinfo.value.message


def test_databricks_driver_options_bypass_the_allow_list(monkeypatch):
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend

    seen = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or _Conn(),
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    backend.connect(catalog='c', driver_options={'use_cloud_fetch': True})

    assert seen == {'catalog': 'c', 'use_cloud_fetch': True}


@pytest.mark.parametrize(
    'backend,module,argument',
    [
        ('snowflake', 'snowflake.connector', 'password'),
        ('databricks', 'databricks.sql', 'catalog'),
    ],
)
def test_driver_options_cannot_restate_a_validated_argument(
    backend, module, argument
):
    r"""The escape hatch must not override a validated argument.

    ``driver_options`` bypasses the allow-list by design, so silently letting
    either side win would let it replace a credential or redirect a read to
    another catalog -- the failure this backend validation exists to prevent.
    """
    pytest.importorskip(module)
    with pytest.raises(ConnectorError) as excinfo:
        connect(
            backend,
            driver_options={argument: 'from-options'},
            **{argument: 'from-caller'},
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['arguments'] == [argument]
    assert 'driver_options' in excinfo.value.message


def test_snowflake_driver_options_credentials_suppress_borrowing(monkeypatch):
    r"""Credentials given through the escape hatch still count.

    Anchoring the borrow decision to the top-level arguments alone would
    silently ignore them, which is what the guard exists to prevent.
    """
    pytest.importorskip('snowflake.connector')
    from sdm_connectors.backends import snowflake as backend

    opened, seen = _Conn(), {}
    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: _Conn())
    monkeypatch.setattr(
        backend.snowflake_connector,
        'connect',
        lambda **kwargs: seen.update(kwargs) or opened,
    )

    result = backend.connect(
        driver_options={'account': 'a', 'user': 'u', 'password': 'p'}
    )

    assert result is opened
    assert seen == {'account': 'a', 'user': 'u', 'password': 'p'}


def test_snowflake_borrows_session_when_no_auth_arguments(monkeypatch):
    pytest.importorskip('snowflake.connector')
    from sdm_connectors.backends import snowflake as backend

    borrowed = _Conn()
    monkeypatch.setattr(
        backend, '_active_snowpark_connection', lambda: borrowed
    )

    assert backend.connect() is borrowed
    assert owns_connection(borrowed) is False


def test_snowflake_rejects_session_arguments_on_a_borrowed_session(monkeypatch):
    """Previously this fell through to a credential-less connect that failed with
    the driver's unrelated "User is empty" message.
    """
    pytest.importorskip('snowflake.connector')
    from sdm_connectors.backends import snowflake as backend

    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: _Conn())

    with pytest.raises(ConnectorError) as excinfo:
        backend.connect(schema='OTHER')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'Snowpark session' in excinfo.value.message
    assert excinfo.value.details['arguments'] == ['schema']


def test_snowflake_credentials_still_open_their_own_connection(monkeypatch):
    pytest.importorskip('snowflake.connector')
    from sdm_connectors.backends import snowflake as backend

    opened = _Conn()
    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: _Conn())
    monkeypatch.setattr(
        backend.snowflake_connector, 'connect', lambda **kwargs: opened
    )

    assert backend.connect(account='a', user='u', password='p') is opened
    assert owns_connection(opened) is True


def test_snowflake_no_arguments_and_no_session_explains_requirements(
    monkeypatch,
):
    pytest.importorskip('snowflake.connector')
    from sdm_connectors.backends import snowflake as backend

    def _fail(**kwargs):
        raise RuntimeError(
            "Default connection with name 'default' cannot be found"
        )

    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: None)
    monkeypatch.setattr(backend.snowflake_connector, 'connect', _fail)

    with pytest.raises(ConnectorError) as excinfo:
        connect('snowflake')
    assert excinfo.value.code == 'CONNECT_FAILED'
    assert 'Snowpark session' in excinfo.value.message
    assert "'account'" in excinfo.value.message


@pytest.mark.parametrize(
    'argument', ['connection_name', 'connections_file_path']
)
def test_snowflake_allows_named_connection_arguments(argument):
    """``DEFAULT_CONFIGURATION`` is the driver's config-parameter table and omits
    the two constructor-only parameters that select a named connection from a
    TOML file -- the secrets-file-free auth path. The guard rejected them as
    typos, and ``connection_name`` was already listed in the connector's own
    ``_AUTH_ARGS``, so the allow-list contradicted the auth logic behind it.
    """
    snowflake = pytest.importorskip('sdm_connectors.backends.snowflake')
    check_connect_args('snowflake', {argument: 'x'}, snowflake._CONNECT_ARGS)


def test_snowflake_allow_list_covers_every_auth_argument():
    snowflake = pytest.importorskip('sdm_connectors.backends.snowflake')
    assert snowflake._AUTH_ARGS <= snowflake._CONNECT_ARGS


def test_snowflake_still_rejects_a_misspelled_argument():
    snowflake = pytest.importorskip('sdm_connectors.backends.snowflake')
    with pytest.raises(ConnectorError) as excinfo:
        check_connect_args(
            'snowflake', {'accont': 'x'}, snowflake._CONNECT_ARGS
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'account' in str(excinfo.value)


@pytest.mark.parametrize(
    'argument',
    [
        'auth_type',
        'credentials_provider',
        'use_cloud_fetch',
        'user_agent_entry',
    ],
)
def test_databricks_allows_kwargs_routed_arguments(argument):
    """The driver routes its documented OAuth parameters through ``**kwargs``,
    where ``inspect.signature`` cannot see them, so a signature-only
    allow-list rejected the whole non-PAT auth path.
    """
    databricks = pytest.importorskip('sdm_connectors.backends.databricks')
    check_connect_args('databricks', {argument: 'x'}, databricks._CONNECT_ARGS)


def test_databricks_still_rejects_a_misspelled_argument():
    databricks = pytest.importorskip('sdm_connectors.backends.databricks')
    with pytest.raises(ConnectorError) as excinfo:
        check_connect_args(
            'databricks', {'cattalog': 'x'}, databricks._CONNECT_ARGS
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'catalog' in str(excinfo.value)


# --- postgres ---------------------------------------------------------------


def test_postgres_is_a_supported_backend():
    """The list is what `connect` validates against, so an unlisted backend
    is refused before any driver is imported."""
    from sdm_connectors.backends import _SQL_BACKENDS

    assert 'postgres' in _SQL_BACKENDS


def test_an_unknown_backend_names_postgres_among_the_supported():
    with pytest.raises(ConnectorError) as error:
        connect('postgresql', 'postgresql://localhost/db')
    assert 'postgres' in str(error.value)
    assert error.value.code == 'UNKNOWN_CONNECTOR'


def test_postgres_refuses_a_connect_argument_it_does_not_take():
    """Caught here rather than by libpq, which reports an unknown keyword as
    a connection failure and hides the typo."""
    postgres = pytest.importorskip('sdm_connectors.backends.postgres')

    with pytest.raises(ConnectorError) as error:
        postgres.connect(nosuchoption='x')
    assert 'nosuchoption' in str(error.value)


def test_postgres_accepts_the_libpq_connection_parameters():
    """A representative few, so a future trim of the set is deliberate."""
    from sdm_connectors.backends.postgres import _CONNECT_ARGS

    assert {'host', 'port', 'dbname', 'user', 'password', 'sslmode'} <= (
        _CONNECT_ARGS
    )


def test_postgres_reads_the_standard_environment_variables(monkeypatch):
    """`PGHOST` and friends are how a Postgres user expects to configure a
    client, and libpq only reads them when the argument is absent."""
    from sdm_connectors.backends import postgres

    monkeypatch.setenv('PGHOST', 'example.invalid')
    monkeypatch.setenv('PGDATABASE', 'from_env')
    captured: dict = {}

    def _fake_connect(*args, **kwargs):
        captured.update(kwargs)
        captured['args'] = args
        raise RuntimeError('stop before the socket')

    monkeypatch.setattr(postgres.psycopg, 'connect', _fake_connect)
    with pytest.raises(RuntimeError):
        postgres.connect(dbname='explicit')

    assert captured['host'] == 'example.invalid'
    assert captured['dbname'] == 'explicit', 'an argument must win over PG*'
