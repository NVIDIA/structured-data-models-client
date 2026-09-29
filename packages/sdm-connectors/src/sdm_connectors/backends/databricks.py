# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from typing import Any, TypeAlias

from sdm_connectors._databricks_telemetry import (
    DatabricksTelemetryVersionError,
    DatabricksUserAgentEntryError,
    databricks_user_agent,
)
from sdm_connectors._version import __version__
from sdm_connectors.sql import (
    ConnectorError,
    check_connect_args,
    merge_driver_options,
    require_driver,
)

_ABSENT = ('databricks', 'databricks.sql', 'databricks.sql.client')

databricks_sql = require_driver(
    'databricks',
    'databricks-sql-connector',
    'databricks.sql',
    absent=_ABSENT,
)
_client = require_driver(
    'databricks',
    'databricks-sql-connector',
    'databricks.sql.client',
    absent=_ABSENT,
)

if not hasattr(_client, 'Connection'):
    raise ImportError(
        "cannot import name 'Connection' from 'databricks.sql.client'"
    )

Connection: TypeAlias = _client.Connection

_ENV_BY_ARG = {
    'server_hostname': 'DATABRICKS_SERVER_HOSTNAME',
    'http_path': 'DATABRICKS_HTTP_PATH',
    'access_token': 'DATABRICKS_TOKEN',
    'catalog': 'DATABRICKS_CATALOG',
    'schema': 'DATABRICKS_SCHEMA',
}


_DOCUMENTED_KWARGS = frozenset(
    {
        'auth_type',
        'azure_client_id',
        'azure_client_secret',
        'azure_tenant_id',
        'azure_workspace_resource_id',
        'credentials_provider',
        'experimental_oauth_persistence',
        'oauth_client_id',
        'oauth_client_secret',
        'oauth_jwt_key_file',
        'oauth_jwt_kid',
        'oauth_redirect_port',
        'oauth_redirect_port_range',
        'oauth_scopes',
        'password',
        'session_configuration',
        'tls_client_cert_file',
        'tls_client_cert_key_file',
        'tls_client_cert_key_password',
        'tls_verify',
        'use_cloud_fetch',
        'use_kernel',
        'user_agent_entry',
        'username',
        '_tls_client_cert_file',
        '_use_cert_as_auth',
    }
)


# What counts as having said how to authenticate, taken from the names the
# installed driver actually reads when it resolves an auth provider. Anything
# here is the caller's answer to that question; nothing here means there is no
# answer, and the driver's reply to no answer is to wait for a browser.
#
# `tls_client_cert_file` is deliberately not here. The driver reads the
# underscored spelling for certificate authentication and the public one only
# as a transport setting, so counting the public one satisfied this guard
# while leaving the driver with no certificate and a browser to open.
_CREDENTIAL_ARGS = frozenset(
    {
        'access_token',
        'auth_type',
        'credentials_provider',
        'azure_client_id',
        'azure_client_secret',
        'oauth_client_id',
        'oauth_client_secret',
        'oauth_jwt_key_file',
        'password',
        'username',
        '_use_cert_as_auth',
    }
)


def _declared_connect_args() -> frozenset[str]:
    r"""Return the Databricks driver's accepted connection keywords.

    The driver accepts ``**kwargs`` and silently ignores unknown names, so this
    guard rejects typos before they fall back to ambient defaults. OAuth
    options are added explicitly because the driver routes them through
    ``**kwargs``, where ``inspect.signature`` cannot see them.
    """
    parameters = inspect.signature(Connection.__init__).parameters.values()
    return frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind is not parameter.VAR_KEYWORD
        and parameter.name != 'self'
    )


_CONNECT_ARGS = (
    _declared_connect_args() | frozenset(_ENV_BY_ARG) | _DOCUMENTED_KWARGS
)


def connect(
    *,
    driver_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Connect to a Databricks SQL warehouse.

    ``DATABRICKS_*`` environment variables are used when their corresponding
    keyword arguments are absent -- including ``DATABRICKS_CATALOG`` and
    ``DATABRICKS_SCHEMA``, which decide which table a bare name resolves to.
    Passing one explicitly as ``None`` stops this function reading the variable
    for it, and the key is then not sent at all; for ``catalog`` and ``schema``
    that leaves the choice to the workspace, and for a credential it leaves no
    credential, which is refused rather than sent.

    A connect with no credential at all is refused here. The driver's own
    answer is to open a browser and wait for a human to finish an OAuth flow,
    binding a listening socket while it does, so a script that simply forgot
    the token never returned.
    """
    check_connect_args('databricks', kwargs, _CONNECT_ARGS)
    given = set(kwargs)
    kwargs = merge_driver_options('databricks', kwargs, driver_options)
    for arg, env in _ENV_BY_ARG.items():
        if arg not in given and arg not in kwargs:
            kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    # The value, not the key. A credential given as an empty string, or a
    # `_use_cert_as_auth` given as False, says how to authenticate no more than
    # leaving it out does, and the driver's answer to that is a browser.
    if not any(kwargs.get(name) for name in _CREDENTIAL_ARGS):
        raise ConnectorError(
            'connecting to Databricks needs a credential and none was given: '
            'pass access_token=..., set DATABRICKS_TOKEN, or name one of '
            f'{sorted(_CREDENTIAL_ARGS - {"access_token"})} to authenticate '
            'another way',
            code='INVALID_CONNECTOR_ARGS',
        )
    # Databricks requires a partner product to identify every connection it
    # creates. Set after the caller's options are merged, so a connection we
    # own cannot quietly lose the attribution, and any tag the caller supplied
    # is kept after the required identifier rather than replaced.
    try:
        kwargs['user_agent_entry'] = databricks_user_agent(
            __version__, kwargs.get('user_agent_entry')
        )
    except DatabricksTelemetryVersionError as error:
        raise ConnectorError(
            f'installed version {__version__!r} cannot be expressed as the '
            f'Databricks telemetry version: {error}',
            code='INVALID_CONNECTOR_ARGS',
            details={'package_version': __version__},
        ) from error
    except (DatabricksUserAgentEntryError, TypeError) as error:
        raise ConnectorError(
            f'invalid Databricks user_agent_entry: {error}',
            code='INVALID_CONNECTOR_ARGS',
        ) from error
    return databricks_sql.connect(**kwargs)
