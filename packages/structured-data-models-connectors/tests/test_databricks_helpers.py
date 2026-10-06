# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``warehouse`` and ``lakebase``: a native connection opened as the caller's
own workspace client, which here is a stand-in with just the calls they make.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from sdm_connectors.databricks import lakebase, warehouse
from sdm_connectors.sql import ConnectorError

ENDPOINT = 'projects/p/branches/b/endpoints/e'

#: What Databricks requires on every connection structured-data-models-client opens.
PARTNER = 'nvidia_sdm-client/1.0.0'


def _authenticate() -> dict[str, str]:
    return {'Authorization': 'Bearer from-the-workspace'}


def _workspace(
    *,
    host: str | None = 'lakebase.example.com',
    token: str = 'minted',
    user: str = 'someone@example.com',
    calls: list[str] | None = None,
) -> SimpleNamespace:
    calls = [] if calls is None else calls

    def get_warehouse(warehouse_id: str) -> SimpleNamespace:
        calls.append(f'warehouses.get:{warehouse_id}')
        return SimpleNamespace(
            odbc_params=SimpleNamespace(
                hostname='adb-1.azuredatabricks.net',
                path=f'/sql/1.0/warehouses/{warehouse_id}',
            )
        )

    def get_endpoint(name: str) -> SimpleNamespace:
        calls.append(f'get_endpoint:{name}')
        return SimpleNamespace(
            status=SimpleNamespace(hosts=SimpleNamespace(host=host))
        )

    def generate_database_credential(endpoint: str) -> SimpleNamespace:
        calls.append(f'generate_database_credential:{endpoint}')
        return SimpleNamespace(token=token)

    return SimpleNamespace(
        warehouses=SimpleNamespace(get=get_warehouse),
        postgres=SimpleNamespace(
            get_endpoint=get_endpoint,
            generate_database_credential=generate_database_credential,
        ),
        current_user=SimpleNamespace(
            me=lambda: SimpleNamespace(user_name=user)
        ),
        config=SimpleNamespace(authenticate=_authenticate),
    )


# --- warehouse --------------------------------------------------------------


@pytest.fixture
def driver_connect(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    databricks_sql = pytest.importorskip('databricks.sql')
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or SimpleNamespace(),
    )
    return seen


def test_warehouse_connects_as_the_workspace_and_names_sdm_client(
    driver_connect: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The backend's own connect would read this beside credentials_provider.
    monkeypatch.setenv('DATABRICKS_TOKEN', 'dapi-from-the-shell')
    calls: list[str] = []

    warehouse(
        'abc123',
        workspace=_workspace(calls=calls),
        use_cloud_fetch=False,
    )

    provider = driver_connect.pop('credentials_provider')
    assert provider() is _authenticate
    # Exactly these: no access token from the shell, and the partner
    # identifier and sampling tag Databricks attributes the connection by.
    assert driver_connect == {
        'server_hostname': 'adb-1.azuredatabricks.net',
        'http_path': '/sql/1.0/warehouses/abc123',
        'use_cloud_fetch': False,
        'user_agent_entry': PARTNER,
        'session_configuration': {'QUERY_TAGS': 'sdm_client_purpose:sampling'},
    }
    assert calls == ['warehouses.get:abc123']


def test_warehouse_keeps_a_caller_s_entry_and_tags_after_its_own(
    driver_connect: dict[str, Any],
) -> None:
    warehouse(
        'abc123',
        workspace=_workspace(),
        user_agent_entry='acme/2.1',
        session_configuration={'QUERY_TAGS': 'team:growth'},
    )

    assert driver_connect['user_agent_entry'] == f'{PARTNER} acme/2.1'
    assert driver_connect['session_configuration'] == {
        'QUERY_TAGS': 'team:growth,sdm_client_purpose:sampling'
    }


def test_warehouse_takes_a_none_entry_as_none_given(
    driver_connect: dict[str, Any],
) -> None:
    """As connect() does, and as the driver reads None: its default."""
    warehouse('abc123', workspace=_workspace(), user_agent_entry=None)

    assert driver_connect['user_agent_entry'] == PARTNER


def test_warehouse_refuses_an_entry_no_header_can_carry(
    driver_connect: dict[str, Any],
) -> None:
    calls: list[str] = []

    with pytest.raises(ConnectorError) as excinfo:
        warehouse(
            'abc123',
            workspace=_workspace(calls=calls),
            user_agent_entry='bad\x01entry',
        )

    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert calls == []
    assert driver_connect == {}


@pytest.mark.parametrize(
    'option,refusal',
    [
        ({'http_path': '/sql/1.0/warehouses/other'}, 'itself'),
        ({'access_token': 'dapi-other'}, 'itself'),
        ({'catalg': 'main'}, "did you mean 'catalog'"),
    ],
)
def test_warehouse_refuses_an_option_it_sets_or_the_driver_ignores(
    driver_connect: dict[str, Any], option: dict[str, Any], refusal: str
) -> None:
    with pytest.raises(ConnectorError) as excinfo:
        warehouse('abc123', workspace=_workspace(), **option)

    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['arguments'] == list(option)
    assert refusal in excinfo.value.message
    assert driver_connect == {}


def test_a_connection_the_caller_opened_is_used_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """structured-data-models-client did not open it, so it neither reopens it to name itself
    nor sets a tag on its session: only the read runs."""
    from sdm_connectors import DatabricksSource, read_table

    databricks_sql = pytest.importorskip('databricks.sql')
    monkeypatch.setattr(
        databricks_sql, 'connect', lambda **kwargs: pytest.fail('opened')
    )
    executed: list[str] = []

    class Cursor:
        description = [('n',)]

        def execute(self, sql: str) -> None:
            executed.append(sql)

        def fetchall(self) -> list[tuple[int]]:
            return [(1,)]

        def close(self) -> None:
            pass

    connection = type(
        'Connection',
        (),
        {'__module__': 'databricks.sql.client', 'cursor': lambda _: Cursor()},
    )()

    source = DatabricksSource(connection, catalog='main', schema='s')
    frame = read_table(source.connection, table='t')

    assert source.connection is connection
    assert frame['n'].tolist() == [1]
    assert executed == ['SELECT * FROM t']
    assert vars(connection) == {}


# --- the workspace's own requests -------------------------------------------


@pytest.fixture
def workspace_host(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """A local stand-in for a workspace that knows one warehouse, recording
    the headers of each request it is sent."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    # A proxied runner would otherwise send these local requests to its proxy.
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    received: list[dict[str, str]] = []
    body = json.dumps(
        {
            'id': 'abc123',
            'odbc_params': {
                'hostname': 'adb-1.azuredatabricks.net',
                'path': '/sql/1.0/warehouses/abc123',
            },
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received.append({'path': self.path, **self.headers})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}', received
    server.shutdown()
    server.server_close()


def _api_calls(received: list[dict[str, str]]) -> list[dict[str, str]]:
    """The warehouse lookup, without the SDK's own discovery of the host."""
    return [sent for sent in received if sent['path'].startswith('/api/')]


def test_a_caller_s_workspace_client_is_copied_to_name_sdm_client(
    driver_connect: dict[str, Any],
    workspace_host: tuple[str, list[dict[str, str]]],
) -> None:
    """The warehouse lookup is a Databricks API call structured-data-models-client makes, so it
    carries the identifier too; the caller's own client is left as it was."""
    sdk = pytest.importorskip('databricks.sdk')
    host, received = workspace_host
    workspace = sdk.WorkspaceClient(
        config=sdk.core.Config(host=host, token='dapi-test', auth_type='pat')
    )
    before = workspace.config.user_agent

    warehouse('abc123', workspace=workspace)

    [sent] = _api_calls(received)
    assert PARTNER in sent['User-Agent'].split()
    assert sent['Authorization'] == 'Bearer dapi-test'
    assert workspace.config.user_agent == before
    assert 'nvidia_sdm-client' not in workspace.config.user_agent
    # The connection authenticates as the caller's workspace still.
    assert driver_connect['credentials_provider']()() == {
        'Authorization': 'Bearer dapi-test'
    }


def test_the_ambient_workspace_client_names_sdm_client(
    driver_connect: dict[str, Any],
    workspace_host: tuple[str, list[dict[str, str]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    pytest.importorskip('databricks.sdk')
    host, received = workspace_host
    for name in list(os.environ):
        if name.startswith(('DATABRICKS_', 'ARM_', 'GOOGLE_')):
            monkeypatch.delenv(name)
    monkeypatch.setenv('DATABRICKS_CONFIG_FILE', str(tmp_path / 'absent.cfg'))
    monkeypatch.setenv('DATABRICKS_HOST', host)
    monkeypatch.setenv('DATABRICKS_TOKEN', 'dapi-ambient')
    monkeypatch.setenv('DATABRICKS_AUTH_TYPE', 'pat')

    warehouse('abc123')

    [sent] = _api_calls(received)
    assert PARTNER in sent['User-Agent'].split()
    assert sent['Authorization'] == 'Bearer dapi-ambient'
    assert driver_connect['user_agent_entry'] == PARTNER


def test_a_workspace_client_subclass_is_used_as_given(
    driver_connect: dict[str, Any],
    workspace_host: tuple[str, list[dict[str, str]]],
) -> None:
    """A copy would lose what the subclass adds, so its requests go as the
    caller set them up; the connection itself still names sdm-client."""
    sdk = pytest.importorskip('databricks.sdk')
    host, received = workspace_host

    class Audited(sdk.WorkspaceClient):
        pass

    workspace = Audited(
        config=sdk.core.Config(host=host, token='dapi-test', auth_type='pat')
    )

    warehouse('abc123', workspace=workspace)

    [sent] = _api_calls(received)
    assert 'nvidia_sdm-client' not in sent['User-Agent']
    assert driver_connect['user_agent_entry'] == PARTNER


# --- lakebase ---------------------------------------------------------------


@pytest.fixture(scope='module')
def postgres_dsn() -> str:
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if dsn:
        return dsn
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        pytest.fail(
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
    pytest.skip('SDM_POSTGRES_TEST_DSN is not set')


@pytest.fixture
def lakebase_role(postgres_dsn: str) -> Iterator[tuple[str, str]]:
    r"""A login named like a Databricks user, whose password is a token.

    Neither is anything libpq would pick by default, so a connection that
    works was given both.
    """
    # Where a server is required, so is its driver: a skip reads as a pass.
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        import psycopg
    else:
        psycopg = pytest.importorskip('psycopg')
    from psycopg import sql

    user = f'sdm-{uuid.uuid4().hex[:8]}@example.com'
    token = uuid.uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as admin:
        admin.execute(
            sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(
                sql.Identifier(user), sql.Literal(token)
            )
        )
        try:
            yield user, token
        finally:
            admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(user)))


def test_lakebase_connects_as_the_current_user_with_a_minted_token(
    postgres_dsn: str, lakebase_role: tuple[str, str]
) -> None:
    from psycopg.conninfo import conninfo_to_dict

    server = conninfo_to_dict(postgres_dsn)
    user, token = lakebase_role
    calls: list[str] = []
    workspace = _workspace(
        host=str(server['host']), token=token, user=user, calls=calls
    )

    connection = lakebase(
        ENDPOINT,
        database=str(server['dbname']),
        workspace=workspace,
        port=server.get('port', 5432),
        # The test server speaks no TLS; Lakebase always does.
        sslmode='disable',
    )
    try:
        row = connection.execute('SELECT current_user, 1').fetchone()
        assert row == (user, 1)
        assert connection.autocommit
        # What the server records the session as, where Databricks reads it.
        name = connection.execute('SHOW application_name').fetchone()
        assert name == (PARTNER,)
    finally:
        connection.close()
    assert sorted(calls) == [
        f'generate_database_credential:{ENDPOINT}',
        f'get_endpoint:{ENDPOINT}',
    ]


def test_lakebase_takes_a_given_host_without_asking_for_the_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg = pytest.importorskip('psycopg')
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        psycopg, 'connect', lambda **kwargs: seen.update(kwargs)
    )
    calls: list[str] = []

    lakebase(
        ENDPOINT,
        database='databricks_postgres',
        workspace=_workspace(host=None, calls=calls),
        host='given.example.com',
    )

    assert seen == {
        'host': 'given.example.com',
        'dbname': 'databricks_postgres',
        'user': 'someone@example.com',
        'password': 'minted',
        'sslmode': 'require',
        'autocommit': True,
        'application_name': PARTNER,
    }
    assert calls == [f'generate_database_credential:{ENDPOINT}']


@pytest.mark.parametrize(
    'theirs,sent',
    [
        ('etl', f'{PARTNER} etl'),
        ('   ', PARTNER),
        # Postgres keeps 63 bytes, and a cut through a character would leave
        # half of it.
        ('é' * 40, f'{PARTNER} ' + 'é' * 19),
    ],
    ids=['kept-after', 'blank', 'cut-to-63-bytes'],
)
def test_lakebase_names_sdm_client_ahead_of_a_caller_s_application_name(
    monkeypatch: pytest.MonkeyPatch, theirs: str, sent: str
) -> None:
    psycopg = pytest.importorskip('psycopg')
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        psycopg, 'connect', lambda **kwargs: seen.update(kwargs)
    )

    lakebase(
        ENDPOINT,
        database='databricks_postgres',
        workspace=_workspace(),
        application_name=theirs,
    )

    assert seen['application_name'] == sent
    assert len(seen['application_name'].encode()) <= 63


def _no_postgres_api() -> SimpleNamespace:
    workspace = _workspace()
    del workspace.postgres
    return workspace


@pytest.mark.parametrize(
    'workspace,code',
    [
        (_no_postgres_api, 'MISSING_BACKEND'),
        (lambda: _workspace(host=None), 'CONNECT_FAILED'),
    ],
    ids=['sdk-before-0.89', 'no-host'],
)
def test_lakebase_refuses_a_workspace_that_cannot_say_where_to_connect(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Callable[[], SimpleNamespace],
    code: str,
) -> None:
    psycopg = pytest.importorskip('psycopg')
    monkeypatch.setattr(
        psycopg, 'connect', lambda **kwargs: pytest.fail('connected')
    )

    with pytest.raises(ConnectorError) as excinfo:
        lakebase(
            ENDPOINT, database='databricks_postgres', workspace=workspace()
        )

    assert excinfo.value.code == code


# --- what fails is a ConnectorError -----------------------------------------


def _no_ambient_credentials(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip('databricks.sdk')
    request.getfixturevalue('driver_connect')
    for name in list(os.environ):
        if name.startswith('DATABRICKS_'):
            monkeypatch.delenv(name)
    home = request.getfixturevalue('tmp_path')
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('DATABRICKS_CONFIG_FILE', str(home / 'absent.cfg'))
    warehouse('abc123')


def _unknown_warehouse(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors = pytest.importorskip('databricks.sdk.errors')
    request.getfixturevalue('driver_connect')
    workspace = _workspace()

    def get(warehouse_id: str) -> None:
        raise errors.NotFound(f'Warehouse {warehouse_id} does not exist.')

    workspace.warehouses.get = get
    warehouse('abc123', workspace=workspace)


def _no_address(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    request.getfixturevalue('driver_connect')
    workspace = _workspace()
    workspace.warehouses.get = lambda warehouse_id: SimpleNamespace(
        odbc_params=None
    )
    warehouse('abc123', workspace=workspace)


def _real_time(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the driver's kernel backend raises for any credentials_provider.
    exc = pytest.importorskip('databricks.sql.exc')

    def connect(**kwargs: Any) -> None:
        raise exc.NotSupportedError(
            'use_kernel=True does not support a custom credentials_provider.'
        )

    monkeypatch.setattr(
        pytest.importorskip('databricks.sql'), 'connect', connect
    )
    warehouse('abc123', workspace=_workspace())


def _real_time_without_the_kernel(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the driver raises when its fallback for a Real-Time warehouse
    # cannot import the kernel, chained from the warehouse's rejection.
    exc = pytest.importorskip('databricks.sql.exc')
    if not hasattr(exc, 'ReydenThriftUnsupportedError'):
        pytest.skip('this driver does not fall back for a Real-Time warehouse')

    def connect(**kwargs: Any) -> None:
        rejection = exc.ReydenThriftUnsupportedError('KP001')
        raise ImportError('use_kernel=True requires ...') from rejection

    monkeypatch.setattr(
        pytest.importorskip('databricks.sql'), 'connect', connect
    )
    warehouse('abc123', workspace=_workspace())


def _refused_login(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from psycopg.conninfo import conninfo_to_dict

    server = conninfo_to_dict(request.getfixturevalue('postgres_dsn'))
    user, _ = request.getfixturevalue('lakebase_role')
    lakebase(
        ENDPOINT,
        database=str(server['dbname']),
        workspace=_workspace(
            host=str(server['host']), token='not-the-token', user=user
        ),
        port=server.get('port', 5432),
        sslmode='disable',
    )


@pytest.mark.parametrize(
    'failing,phrase',
    [
        (_no_ambient_credentials, 'pass workspace='),
        (_unknown_warehouse, "could not look up Databricks warehouse 'abc123'"),
        (_no_address, 'reports no host or HTTP path'),
        (_real_time, 'Real-Time warehouse'),
        (_real_time_without_the_kernel, 'Real-Time warehouse'),
        (
            _refused_login,
            f'could not connect to Lakebase endpoint {ENDPOINT!r}',
        ),
    ],
    ids=[
        'no-ambient-credentials',
        'unknown-warehouse',
        'no-address',
        'real-time',
        'real-time-without-the-kernel',
        'refused-login',
    ],
)
def test_a_workspace_or_driver_failure_is_a_connector_error(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    failing: Callable[[pytest.FixtureRequest, pytest.MonkeyPatch], None],
    phrase: str,
) -> None:
    r"""``except ConnectorError`` covers the connector layer, these included:
    otherwise a missing credential, a wrong id or a refused login reaches
    the caller as whichever SDK or driver error it happened to be.
    """
    with pytest.raises(ConnectorError) as excinfo:
        failing(request, monkeypatch)

    assert excinfo.value.code == 'CONNECT_FAILED'
    assert phrase in excinfo.value.message
