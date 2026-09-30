# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Connections to Databricks, opened as the caller's own workspace client.

Each helper returns the driver's own connection, which the caller owns and
closes, and hands it to a source like any other. Neither tags the connection:
sdm-client names itself on prediction requests only, never on the queries that
sample a warehouse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdm_connectors.sql import (
    ConnectorError,
    check_connect_args,
    require_driver,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.postgres import PostgresAPI

    from sdm_connectors.backends.databricks import (
        Connection as DatabricksConnection,
    )
    from sdm_connectors.backends.postgres import (
        Connection as PostgresConnection,
    )

__all__ = ['lakebase', 'warehouse']

# Taken from the workspace client. The driver lets `credentials_provider` win
# over any other credential, so a restated one would be silently ignored.
_WAREHOUSE_SETS = frozenset(
    {
        'access_token',
        'auth_type',
        'credentials_provider',
        'experimental_oauth_persistence',
        'http_path',
        'oauth_client_id',
        'oauth_client_secret',
        'oauth_redirect_port_range',
        'oauth_scopes',
        'password',
        'server_hostname',
        'username',
    }
)

_LAKEBASE_SETS = frozenset({'dbname', 'host', 'password', 'user'})


def warehouse(
    warehouse_id: str,
    *,
    workspace: WorkspaceClient | None = None,
    **driver_options: Any,
) -> DatabricksConnection:
    r"""Open a Databricks SQL warehouse connection as ``workspace``'s identity.

    The host and HTTP path are looked up by the warehouse's id, and every
    request authenticates through ``workspace``, an ambient
    ``WorkspaceClient()`` when omitted. ``driver_options`` go to
    ``databricks.sql.connect`` as they are, and the caller closes the
    connection it gets back::

        from databricks.sdk import WorkspaceClient
        from sdm_connectors import DatabricksSource
        from sdm_connectors.databricks import warehouse

        connection = warehouse(warehouse_id, workspace=WorkspaceClient())
        source = DatabricksSource(connection, catalog='main', schema='sales')

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for an option the driver
            does not take or one the workspace already sets, and
            ``CONNECT_FAILED`` for a warehouse that reports no address.
        MissingBackendError: Without the ``databricks`` extra.
    """
    from sdm_connectors.backends import databricks as backend

    _refuse_restated('warehouse', driver_options, _WAREHOUSE_SETS)
    check_connect_args(
        'databricks', driver_options, backend._CONNECT_ARGS, hint=None
    )
    if 'user_agent_entry' in driver_options:
        backend._check_user_agent_entry(driver_options['user_agent_entry'])
    client = _workspace(workspace, 'databricks')
    odbc = client.warehouses.get(warehouse_id).odbc_params
    if odbc is None or not odbc.hostname or not odbc.path:
        raise ConnectorError(
            f'Databricks warehouse {warehouse_id!r} reports no host or HTTP '
            f'path to connect to',
            code='CONNECT_FAILED',
            details={'warehouse_id': warehouse_id},
        )
    # The driver itself, not the backend's connect, which would also read
    # DATABRICKS_TOKEN from the environment beside the workspace's own auth.
    return backend.databricks_sql.connect(
        server_hostname=odbc.hostname,
        http_path=odbc.path,
        credentials_provider=lambda: client.config.authenticate,
        **driver_options,
    )


def lakebase(
    endpoint: str,
    *,
    database: str,
    workspace: WorkspaceClient | None = None,
    host: str | None = None,
    **driver_options: Any,
) -> PostgresConnection:
    r"""Open a psycopg connection to a Lakebase endpoint as ``workspace``'s
    user.

    The password is a database credential minted for ``endpoint`` (its full
    ``projects/.../branches/.../endpoints/...`` name), and the host is the
    endpoint's own unless ``host`` is given. ``workspace`` is an ambient
    ``WorkspaceClient()`` when omitted. ``sslmode='require'`` and
    ``autocommit=True`` are defaults that ``driver_options`` can override. The
    caller closes the connection it gets back::

        from databricks.sdk import WorkspaceClient
        from sdm_connectors import PostgresSource
        from sdm_connectors.databricks import lakebase

        workspace = WorkspaceClient()
        connection = lakebase(
            endpoint, database='databricks_postgres', workspace=workspace
        )
        source = PostgresSource(connection, schema='public')

    The credential is short-lived: an open connection outlives it, and a new
    connection needs a fresh call.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for an option libpq does
            not take or one this sets itself, ``MISSING_BACKEND`` for a
            Databricks SDK older than 0.89, and ``CONNECT_FAILED`` for an
            endpoint that reports no host.
        MissingBackendError: Without the ``lakebase`` extra.
    """
    psycopg = require_driver('lakebase', 'psycopg', 'psycopg')
    from sdm_connectors.backends import postgres as backend

    _refuse_restated('lakebase', driver_options, _LAKEBASE_SETS)
    check_connect_args(
        'lakebase', driver_options, backend._CONNECT_ARGS, hint=None
    )
    client = _workspace(workspace, 'lakebase')
    api: PostgresAPI | None = getattr(client, 'postgres', None)
    if api is None:
        raise ConnectorError(
            'Lakebase needs databricks-sdk 0.89 or later, whose '
            "WorkspaceClient has 'postgres'. Install it via the 'lakebase' "
            "extra, e.g. `pip install 'sdm-connectors[lakebase]'`",
            code='MISSING_BACKEND',
            details={'backend': 'lakebase', 'driver': 'databricks-sdk>=0.89'},
        )
    if host is None:
        host = _endpoint_host(api, endpoint)
    credential = api.generate_database_credential(endpoint=endpoint)
    options = {'sslmode': 'require', 'autocommit': True, **driver_options}
    connection: PostgresConnection = psycopg.connect(
        host=host,
        dbname=database,
        user=client.current_user.me().user_name,
        password=credential.token,
        **options,
    )
    return connection


def _endpoint_host(api: PostgresAPI, endpoint: str) -> str:
    status = api.get_endpoint(name=endpoint).status
    hosts = status.hosts if status is not None else None
    host = hosts.host if hosts is not None else None
    if not host:
        raise ConnectorError(
            f'Lakebase endpoint {endpoint!r} reports no host to connect to; '
            f'pass host= explicitly',
            code='CONNECT_FAILED',
            details={'endpoint': endpoint},
        )
    return host


def _workspace(
    workspace: WorkspaceClient | None, backend: str
) -> WorkspaceClient:
    if workspace is not None:
        return workspace
    sdk = require_driver(
        backend,
        'databricks-sdk',
        'databricks.sdk',
        absent=('databricks', 'databricks.sdk'),
    )
    client: WorkspaceClient = sdk.WorkspaceClient()
    return client


def _refuse_restated(
    helper: str, options: dict[str, Any], taken: frozenset[str]
) -> None:
    restated = sorted(set(options) & taken)
    if restated:
        raise ConnectorError(
            f'{helper}() sets {", ".join(map(repr, restated))} itself, from '
            f'its own arguments and the workspace client, so a driver option '
            f'cannot restate it',
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': restated},
        )
