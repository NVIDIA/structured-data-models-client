# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Connections to Databricks, opened as the caller's own workspace client.

Each helper returns the driver's own connection, which the caller owns and
closes, and hands it to a source like any other. Databricks requires a partner
product to identify every connection and API call it opens, so both name
sdm-client as ``nvidia_sdm-client/<version>``: in the warehouse driver's
User-Agent, in Lakebase's ``application_name``, and on the workspace client's
own requests. A connection the caller opens and passes to a source directly
is theirs, and is used as it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdm_connectors._databricks_partner import (
    partner_identifier,
    partner_user_agent,
)
from sdm_connectors.sql import (
    ConnectorError,
    check_connect_args,
    driver_guard,
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

    The connection names sdm-client in its User-Agent, ahead of any
    ``user_agent_entry`` given, and tags its statements
    ``sdm_client_purpose: sampling`` beside any query tags given. The
    warehouse lookup names sdm-client too, through a copy of ``workspace``
    rather than the caller's own.

    A Real-Time warehouse is refused: the driver serves it through a backend
    that takes no workspace-client credentials, so connect to it with
    ``databricks.sql.connect`` and a token or OAuth M2M credentials instead.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for an option the driver
            does not take or one the workspace already sets, and
            ``CONNECT_FAILED`` when the workspace cannot be reached or does
            not know the warehouse, or the warehouse refuses the connection.
        MissingBackendError: Without the ``databricks`` extra.
    """
    from sdm_connectors.backends import databricks as backend

    _refuse_restated('warehouse', driver_options, _WAREHOUSE_SETS)
    check_connect_args(
        'databricks', driver_options, backend._CONNECT_ARGS, hint=None
    )
    connect_args = backend._as_sdm_client(driver_options)
    client = _workspace(workspace, 'databricks')
    with driver_guard(
        'CONNECT_FAILED',
        f'could not look up Databricks warehouse {warehouse_id!r}',
        warehouse_id=warehouse_id,
    ):
        odbc = client.warehouses.get(warehouse_id).odbc_params
    if odbc is None or not odbc.hostname or not odbc.path:
        raise ConnectorError(
            f'Databricks warehouse {warehouse_id!r} reports no host or HTTP '
            f'path to connect to',
            code='CONNECT_FAILED',
            details={'warehouse_id': warehouse_id},
        )
    with driver_guard(
        'CONNECT_FAILED',
        f'could not connect to Databricks warehouse {warehouse_id!r}',
        warehouse_id=warehouse_id,
    ):
        try:
            # The driver itself, not the backend's connect, which would also
            # read DATABRICKS_TOKEN from the environment beside the workspace.
            connection: DatabricksConnection = backend.databricks_sql.connect(
                server_hostname=odbc.hostname,
                http_path=odbc.path,
                credentials_provider=lambda: client.config.authenticate,
                **connect_args,
            )
        except Exception as error:
            if not _refused_as_real_time(error):
                raise
            raise ConnectorError(
                f'Databricks warehouse {warehouse_id!r} does not accept a '
                f"workspace client's credentials: the driver serves its type "
                f'(a Real-Time warehouse, for instance) through a backend that '
                f'refuses them. Connect with databricks.sql.connect and a '
                f'token or OAuth M2M credentials',
                code='CONNECT_FAILED',
                details={
                    'warehouse_id': warehouse_id,
                    'driver_error': type(error).__name__,
                },
            ) from error
    return connection


def _refused_as_real_time(error: Exception) -> bool:
    r"""Whether the driver's kernel backend, which it falls back to for a
    Real-Time warehouse, refused the connection.

    With the kernel installed it refuses any ``credentials_provider`` as not
    supported; without it, the fallback fails to import, caused by the
    warehouse's rejection of the default protocol.
    """
    from databricks.sql import exc

    if isinstance(error, exc.NotSupportedError):
        return True
    rejection = getattr(exc, 'ReydenThriftUnsupportedError', None)
    return (
        isinstance(error, ImportError)
        and rejection is not None
        and isinstance(error.__cause__, rejection)
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

    ``application_name`` is ``nvidia_sdm-client/<version>``, with any
    ``application_name`` given after it, cut to the 63 bytes Postgres keeps.
    The workspace's calls name structured-data-models-client too, through a copy of
    ``workspace`` rather than the caller's own.

    The credential is valid for at most an hour: an open connection outlives
    it, and a new connection needs a fresh call. The login user is whatever
    ``current_user.me().user_name`` reports, for a service principal too.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for an option libpq does
            not take or one this sets itself, ``MISSING_BACKEND`` for a
            Databricks SDK older than 0.89, and ``CONNECT_FAILED`` when the
            workspace cannot be reached, cannot say where the endpoint is or
            mint a credential for it, or the endpoint refuses the login.
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
            "extra, e.g. `pip install 'structured-data-models-connectors[lakebase]'`",
            code='MISSING_BACKEND',
            details={'backend': 'lakebase', 'driver': 'databricks-sdk>=0.89'},
        )
    if host is None:
        host = _endpoint_host(api, endpoint)
    with driver_guard(
        'CONNECT_FAILED',
        f'could not get a login for Lakebase endpoint {endpoint!r}',
        endpoint=endpoint,
    ):
        token = api.generate_database_credential(endpoint=endpoint).token
        user = client.current_user.me().user_name
    options = {'sslmode': 'require', 'autocommit': True, **driver_options}
    options['application_name'] = _application_name(
        options.get('application_name')
    )
    with driver_guard(
        'CONNECT_FAILED',
        f'could not connect to Lakebase endpoint {endpoint!r}',
        endpoint=endpoint,
    ):
        connection: PostgresConnection = psycopg.connect(
            host=host, dbname=database, user=user, password=token, **options
        )
    return connection


def _endpoint_host(api: PostgresAPI, endpoint: str) -> str:
    with driver_guard(
        'CONNECT_FAILED',
        f'could not look up Lakebase endpoint {endpoint!r}',
        endpoint=endpoint,
    ):
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


#: What Postgres keeps of an ``application_name``: NAMEDATALEN less one.
_APPLICATION_NAME_BYTES = 63


def _application_name(theirs: object) -> str:
    r"""``nvidia_sdm-client/<version>``, then the caller's own name.

    Cut to the bytes Postgres keeps, on a character boundary, since the
    server would otherwise cut it wherever the limit falls.
    """
    name = partner_identifier()
    if theirs is not None and str(theirs).strip():
        name = f'{name} {theirs}'
    return name.encode()[:_APPLICATION_NAME_BYTES].decode(errors='ignore')


def _workspace(
    workspace: WorkspaceClient | None, backend: str
) -> WorkspaceClient:
    r"""A workspace client whose requests name structured-data-models-client.

    A caller's ``WorkspaceClient`` is copied for that, not changed; a
    subclass or test double is used as given, since a copy would lose what
    it adds.
    """
    if workspace is not None:
        return _named_copy(workspace)
    sdk = require_driver(
        backend,
        'databricks-sdk',
        'databricks.sdk',
        absent=('databricks', 'databricks.sdk'),
    )
    from databricks.sdk.config import Config

    try:
        # Named before the client is built, which reads the User-Agent once.
        config = Config().with_user_agent_extra(*partner_user_agent())
        client: WorkspaceClient = sdk.WorkspaceClient(config=config)
    except Exception as error:
        # The type only: the SDK's text can quote the configuration it read.
        raise ConnectorError(
            f'could not authenticate to Databricks from the ambient '
            f'configuration ({type(error).__name__}). Run this in a Databricks '
            f'notebook, set DATABRICKS_HOST and a credential, or pass '
            f'workspace=',
            code='CONNECT_FAILED',
            details={'driver_error': type(error).__name__},
        ) from None
    return client


def _named_copy(workspace: WorkspaceClient) -> WorkspaceClient:
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return workspace
    if type(workspace) is not WorkspaceClient:
        return workspace
    # copy() gives the copy its own User-Agent extras and shares the rest,
    # credentials included.
    config = workspace.config.copy()
    return WorkspaceClient(
        config=config.with_user_agent_extra(*partner_user_agent())
    )


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
