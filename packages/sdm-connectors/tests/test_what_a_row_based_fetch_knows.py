# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the row-based fetch reads, and what it threw away.

Everything a driver that answers in rows can say about a column beyond its
values is in ``cursor.description``, and only the name was being read. A
column of nothing but nulls was built as ``object`` and described to the model
as text, though the server had called it ``bigint``; every other backend gets
its type through Arrow, so postgres was the only one that lost it.

Three more of the same shape. Rows were indexed by position, which a mapping
row factory refuses -- and ``row_factory`` is on this connector's own
allow-list, so the connection opens and every read fails with ``KeyError: 0``
surfacing as a query failure naming ``0``. A result set of no columns kept no
rows. A null-free integer came back nullable where every other path narrows
it.

And the guard called every failure a failure to execute, including the ones
where the statement executed and the *fetch* is what failed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_connectors.reader import _frame_by_position, _frame_from_rows


class Described:
    """A ``cursor.description`` entry the way psycopg builds one."""

    def __init__(self, name: str, type_display: str | None = None) -> None:
        self.name = name
        self.type_display = type_display

    def __getitem__(self, index: int) -> object:
        return (self.name, None)[index]


def test_a_column_of_nothing_but_nulls_is_typed_from_what_the_server_called_it() -> (
    None
):
    frame = _frame_from_rows(
        [(1, None, None), (2, None, None)],
        [
            Described('a', 'int8'),
            Described('b', 'int8'),
            Described('c', 'text'),
        ],
    )

    assert frame['a'].dtype == 'int64'
    assert frame['b'].dtype == 'Int64'
    assert str(frame['c'].dtype) == 'string'


def test_nothing_is_claimed_where_the_driver_names_no_type() -> None:
    frame = _frame_from_rows([(None,), (None,)], [('a', None, None)])
    assert frame['a'].dtype == object


@pytest.mark.parametrize(
    ('declared', 'dtype'),
    [
        ('int8', 'Int64'),
        ('bigint', 'Int64'),
        ('float8', 'float64'),
        ('bool', 'boolean'),
        ('date', 'datetime64[ns]'),
        ('timestamp', 'datetime64[ns]'),
        ('timestamptz', 'datetime64[ns, UTC]'),
        ('text', 'string'),
        ('varchar(42)', 'string'),
    ],
)
def test_every_declared_type_this_knows_survives_a_column_of_nulls(
    declared: str, dtype: str
) -> None:
    """The defect was filed about an integer and is the same for every other
    type the row path can be handed: the values say nothing, so the server's
    own name is all there is."""
    frame = _frame_from_rows([(None,), (None,)], [Described('a', declared)])
    assert str(frame['a'].dtype) == dtype


@pytest.mark.parametrize('declared', ['int8[]', 'numeric', 'json', 'point'])
def test_a_type_this_does_not_know_is_left_alone(declared: str) -> None:
    frame = _frame_from_rows([(None,)], [Described('a', declared)])
    assert frame['a'].dtype == object


def test_a_column_the_server_calls_integral_but_that_holds_values_is_left_alone() -> (
    None
):
    frame = _frame_from_rows([('x',), ('y',)], [Described('a', 'int8')])
    assert frame['a'].tolist() == ['x', 'y']


def test_rows_that_are_mappings_are_read_by_name() -> None:
    frame = _frame_by_position(
        [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], ['a', 'b']
    )

    assert list(frame.columns) == ['a', 'b']
    assert frame['a'].tolist() == [1, 2]
    assert frame['b'].tolist() == ['x', 'y']


def test_mapping_rows_that_lost_a_duplicate_name_say_so() -> None:
    from sdm_connectors.sql import ConnectorError

    with pytest.raises(ConnectorError) as raised:
        _frame_by_position([{'a': 2}], ['a', 'a'])

    said = str(raised.value)
    assert 'row factory' in said
    assert 'AS' in said


def test_rows_of_tuples_still_keep_two_columns_of_one_name() -> None:
    frame = _frame_by_position([(1, 2)], ['a', 'a'])

    assert list(frame.columns) == ['a', 'a']
    assert frame.iloc[0].tolist() == [1, 2]


def test_a_result_of_no_columns_keeps_its_rows() -> None:
    assert _frame_by_position([(), ()], []).shape == (2, 0)
    assert _frame_by_position([], []).shape == (0, 0)


def test_a_null_free_integer_comes_back_the_way_every_other_path_gives_it() -> (
    None
):
    frame = _frame_from_rows([(1,), (2,)], [Described('a', 'int8')])
    assert frame['a'].dtype == 'int64'

    with_a_null = _frame_from_rows([(1,), (None,)], [Described('a', 'int8')])
    assert with_a_null['a'].dtype == 'Int64'


def test_a_wide_integer_survives_the_narrowing() -> None:
    frame = _frame_from_rows([(2**70,), (1,)], [Described('a', 'numeric')])
    assert frame['a'].tolist() == [2**70, 1]


class _Cursor:
    description = [Described('a', 'int8')]

    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    def execute(self, sql: str) -> None:
        if self._fail_on == 'execute':
            raise RuntimeError('relation "nope" does not exist')

    def fetchall(self) -> list[tuple[int]]:
        if self._fail_on == 'fetch':
            raise RuntimeError("Can't unify schema with duplicate field names.")
        return [(1,)]

    def close(self) -> None:
        pass


class _Connection:
    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    def cursor(self) -> _Cursor:
        return _Cursor(self._fail_on)


@pytest.mark.parametrize(
    ('fail_on', 'said'),
    [('execute', 'query execution failed'), ('fetch', 'reading the result')],
)
def test_a_query_that_executed_is_not_reported_as_one_that_failed_to(
    fail_on: str, said: str
) -> None:
    from sdm_connectors.reader import read_table
    from sdm_connectors.sql import ConnectorError

    with pytest.raises(ConnectorError) as raised:
        read_table(_Connection(fail_on), query='select 1')

    assert said in str(raised.value)
    if fail_on == 'fetch':
        assert 'query execution failed' not in str(raised.value)


def test_a_query_that_works_still_answers() -> None:
    from sdm_connectors.reader import read_table

    frame = read_table(_Connection('neither'), query='select 1')
    assert frame['a'].tolist() == [1]
    assert isinstance(frame, pd.DataFrame)


def test_a_databricks_connect_with_no_credential_is_refused_not_opened(
    monkeypatch,
) -> None:
    """The driver's answer to no credential is to open a browser and wait for
    a human, binding a listening socket while it does, so a script that forgot
    the token never returned."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    def refuse(**kwargs: object) -> object:
        raise AssertionError('the driver was called with no credential')

    monkeypatch.setattr(backend.databricks_sql, 'connect', refuse)
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    for absent in ({}, {'access_token': None}):
        with pytest.raises(ConnectorError) as raised:
            backend.connect(server_hostname='h', http_path='p', **absent)
        assert 'needs a credential' in str(raised.value)


@pytest.mark.parametrize(
    'credential',
    [
        {
            'use_kernel': True,
            'oauth_client_id': 'i',
            'oauth_client_secret': 's',
        },
        {
            'auth_type': 'azure-sp-m2m',
            'azure_client_id': 'i',
            'azure_client_secret': 's',
        },
        {
            'auth_type': 'databricks-oauth',
            'oauth_client_id': 'i',
            'oauth_redirect_port': 8020,
        },
        {
            'use_kernel': True,
            'oauth_client_id': 'i',
            'oauth_jwt_key_file': '/keys/client.pem',
            'oauth_jwt_kid': 'kid-1',
        },
    ],
)
def test_the_installed_driver_s_own_auth_options_are_accepted(
    monkeypatch, credential: dict
) -> None:
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend

    seen: dict = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    backend.connect(server_hostname='h', http_path='p', **credential)
    assert set(credential) <= set(seen)


@pytest.mark.parametrize(
    'given',
    [{'auth_type': ''}, {'access_token': ''}, {'_use_cert_as_auth': False}],
)
def test_a_credential_that_is_empty_is_no_credential(
    monkeypatch, given: dict
) -> None:
    """The guard read the key and not the value, so an empty string said how
    to authenticate no more than leaving it out did -- and the driver's answer
    to that is a browser."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    def refuse(**kwargs: object) -> object:
        raise AssertionError('the driver was called with no credential')

    monkeypatch.setattr(backend.databricks_sql, 'connect', refuse)
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    with pytest.raises(ConnectorError) as raised:
        backend.connect(server_hostname='h', http_path='p', **given)
    assert 'needs a credential' in str(raised.value)


def test_a_type_error_creating_a_cursor_is_not_called_a_bad_argument() -> None:
    from sdm_connectors.reader import read_table
    from sdm_connectors.sql import ConnectorError

    class Connection:
        def cursor(self) -> object:
            raise TypeError('driver failed creating cursor')

    with pytest.raises(ConnectorError) as raised:
        read_table(Connection(), query='select 1')

    assert raised.value.code == 'QUERY_FAILED'


def test_only_the_names_the_driver_reads_count_as_a_credential(
    monkeypatch,
) -> None:
    """The driver reads the underscored spelling for certificate
    authentication and the public one only as a transport setting, so counting
    the public one satisfied the guard while leaving the driver with no
    certificate and a browser to open."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    seen: dict = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    with pytest.raises(ConnectorError):
        backend.connect(
            server_hostname='h', http_path='p', tls_client_cert_file='c.pem'
        )

    backend.connect(
        server_hostname='h',
        http_path='p',
        _use_cert_as_auth=True,
        _tls_client_cert_file='c.pem',
    )
    assert seen['_tls_client_cert_file'] == 'c.pem'


def test_the_driver_s_jwt_private_key_auth_is_accepted(monkeypatch) -> None:
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend

    seen: dict = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    backend.connect(
        server_hostname='h',
        http_path='p',
        use_kernel=True,
        oauth_client_id='i',
        oauth_jwt_key_file='/keys/client.pem',
        oauth_jwt_kid='kid-1',
    )

    assert seen['oauth_jwt_key_file'] == '/keys/client.pem'


def test_a_query_that_is_not_a_statement_is_named_as_the_caller_s() -> None:
    """Splitting the query guard so a fetch failure stops being called an
    execution failure also stopped a `TypeError` naming the caller's own
    argument, and `query` is the one such argument the region holds."""
    import sqlite3

    from sdm_connectors.reader import read_table
    from sdm_connectors.sql import ConnectorError

    with pytest.raises(ConnectorError) as raised:
        read_table(sqlite3.connect(':memory:'), query=123)

    assert raised.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'must be a SQL statement' in str(raised.value)


@pytest.mark.parametrize(
    ('given', 'accepted'),
    [
        ({'access_token': 't'}, True),
        ({'credentials_provider': object()}, True),
        ({'auth_type': 'databricks-oauth'}, True),
        (
            {
                'auth_type': 'azure-sp-m2m',
                'azure_client_id': 'i',
                'azure_client_secret': 's',
            },
            True,
        ),
        ({'_use_cert_as_auth': True, '_tls_client_cert_file': 'c.pem'}, True),
        (
            {
                'use_kernel': True,
                'oauth_client_id': 'i',
                'oauth_client_secret': 's',
            },
            True,
        ),
        (
            {
                'use_kernel': True,
                'oauth_client_id': 'i',
                'oauth_jwt_key_file': '/k.pem',
                'oauth_jwt_kid': 'k1',
            },
            True,
        ),
        (
            {
                'use_kernel': True,
                'oauth_client_id': 'i',
                'oauth_jwt_key_file': '/k.pem',
            },
            True,
        ),
        (
            {
                'use_kernel': True,
                'azure_client_id': 'i',
                'azure_client_secret': 's',
            },
            True,
        ),
        ({'identity_federation_client_id': 'x', 'access_token': 't'}, True),
        ({'oauth_token_cache_enabled': True, 'access_token': 't'}, True),
        ({'azure_client_id': 'i', 'azure_client_secret': 's'}, False),
        ({'oauth_client_id': 'i', 'oauth_client_secret': 's'}, False),
        ({'oauth_client_id': 'i', 'oauth_jwt_key_file': '/k.pem'}, False),
        ({'oauth_client_secret': 's'}, False),
        ({'oauth_client_id': 'app', 'oauth_redirect_port': 8020}, True),
        (
            {
                'oauth_client_id': 'app',
                'oauth_redirect_port_range': [8020, 8021],
            },
            True,
        ),
        ({'oauth_client_id': 'app'}, True),
        ({'_use_cert_as_auth': True}, False),
        ({'azure_client_secret': 's'}, False),
        ({'tls_client_cert_file': 'c.pem'}, False),
        ({'auth_type': ''}, False),
        ({}, False),
    ],
)
def test_half_a_credential_is_not_a_credential(
    monkeypatch, given: dict, accepted: bool
) -> None:
    """Only the thrift route waits for a human: handed nothing it
    recognises it opens a browser and binds a listening socket, so a script
    that forgot its token never returns. That is what is refused here.
    Everything else -- a machine-to-machine credential on the wrong route, a
    certificate without the flag that turns certificate authentication on --
    the driver refuses itself, immediately and with a better message."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    monkeypatch.setattr(
        backend.databricks_sql, 'connect', lambda **kwargs: object()
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    if accepted:
        backend.connect(server_hostname='h', http_path='p', **given)
    else:
        with pytest.raises(ConnectorError):
            backend.connect(server_hostname='h', http_path='p', **given)


@pytest.mark.parametrize(
    ('given', 'reaches_the_driver'),
    [
        ({'use_kernel': True, 'credentials_provider': object()}, True),
        (
            {
                'use_kernel': True,
                'oauth_client_id': 'i',
                'oauth_jwt_key_file': '/k.pem',
            },
            True,
        ),
        ({'username': 'u', 'password': 'p'}, True),
        ({'oauth_client_id': 'i', 'oauth_client_secret': 's'}, False),
    ],
)
def test_the_guard_refuses_only_what_would_reach_the_browser(
    monkeypatch, given: dict, reaches_the_driver: bool
) -> None:
    """The driver refuses everything else itself, immediately and with a
    better message: a machine-to-machine credential on the wrong route, a JWT
    key with no key id, username and password at all. Only the thrift
    fallthrough waits for a human, and only that is refused here."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    monkeypatch.setattr(
        backend.databricks_sql, 'connect', lambda **kwargs: object()
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    if reaches_the_driver:
        backend.connect(server_hostname='h', http_path='p', **given)
    else:
        with pytest.raises(ConnectorError):
            backend.connect(server_hostname='h', http_path='p', **given)


@pytest.mark.parametrize(
    ('named', 'accepted'),
    [
        ('_tls_client_cert_key_file', True),
        ('_tls_client_cert_key_password', True),
        ('tls_client_cert_key_file', False),
        ('tls_client_cert_key_password', False),
    ],
)
def test_the_mtls_key_names_are_the_ones_the_driver_reads(
    monkeypatch, named: str, accepted: bool
) -> None:
    """The public spellings were taken from documentation; the installed
    driver reads the underscored ones, so a separately stored or encrypted
    client key was accepted and then silently omitted."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    seen: dict = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    def connect() -> None:
        backend.connect(
            server_hostname='h',
            http_path='p',
            _use_cert_as_auth=True,
            _tls_client_cert_file='c.pem',
            **{named: 'value'},
        )

    if accepted:
        connect()
        assert seen[named] == 'value'
    else:
        with pytest.raises(ConnectorError):
            connect()


@pytest.mark.parametrize(
    ('given', 'reaches_the_driver'),
    [
        ({'auth_type': 'databricks-oauth'}, True),
        ({'auth_type': 'azure-oauth'}, True),
        ({'auth_type': 'not-a-driver-auth-type'}, False),
        ({'auth_type': 'typo', 'access_token': 't'}, True),
        ({'auth_type': 'typo', 'use_kernel': True}, True),
    ],
)
def test_only_an_auth_type_the_driver_knows_counts_as_one(
    monkeypatch, given: dict, reaches_the_driver: bool
) -> None:
    """The driver dispatches on three values and falls through to the browser
    for anything else, so accepting `auth_type` on sight let a typo produce
    exactly the wait this guard exists to prevent."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    monkeypatch.setattr(
        backend.databricks_sql, 'connect', lambda **kwargs: object()
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    assert backend._KNOWN_AUTH_TYPES == frozenset(
        {'databricks-oauth', 'azure-oauth', 'azure-sp-m2m'}
    )

    if reaches_the_driver:
        backend.connect(server_hostname='h', http_path='p', **given)
    else:
        with pytest.raises(ConnectorError):
            backend.connect(server_hostname='h', http_path='p', **given)


def test_naming_an_oauth_app_is_choosing_the_browser_flow_on_purpose(
    monkeypatch,
) -> None:
    """The driver resolves to its browser flow for anything it does not
    recognise, including an empty call, so it cannot tell a deliberate U2M
    setup from a forgotten token. Naming an OAuth application can: the guard
    is for the accident, not the choice. Not beside a machine-to-machine
    secret, which asks for something else and would be quietly downgraded."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    monkeypatch.setattr(
        backend.databricks_sql, 'connect', lambda **kwargs: object()
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    backend.connect(
        server_hostname='h',
        http_path='p',
        oauth_client_id='app',
        oauth_redirect_port=8020,
    )

    with pytest.raises(ConnectorError):
        backend.connect(
            server_hostname='h',
            http_path='p',
            oauth_client_id='app',
            oauth_client_secret='secret',
        )
