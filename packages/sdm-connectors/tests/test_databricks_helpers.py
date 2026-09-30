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


def test_warehouse_connects_as_the_workspace_and_tags_nothing(
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
    # Exactly these: no access token from the shell and no partner tag.
    assert driver_connect == {
        'server_hostname': 'adb-1.azuredatabricks.net',
        'http_path': '/sql/1.0/warehouses/abc123',
        'use_cloud_fetch': False,
    }
    assert calls == ['warehouses.get:abc123']


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
    }
    assert calls == [f'generate_database_credential:{ENDPOINT}']


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
