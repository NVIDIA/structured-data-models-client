# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
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
        'identity_federation_client_id',
        'oauth_jwt_algorithm',
        'oauth_jwt_key_file',
        'oauth_jwt_kid',
        'oauth_jwt_passphrase',
        'oauth_redirect_port',
        'oauth_redirect_port_range',
        'oauth_scopes',
        'oauth_token_cache_enabled',
        'password',
        'token_url',
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


# What counts as having said how to authenticate, read from the branches the
# installed driver dispatches on rather than from documentation. Each entry is
# a set of names that have to be present together: the driver reads a client
# secret only beside its client id, and a certificate only beside the flag that
# turns certificate authentication on, so counting either half on its own let a
# call through that the driver then resolves to its browser flow.
#
# `tls_client_cert_file` is deliberately absent. The driver reads the
# underscored spelling for certificate authentication and the public one only
# as a transport setting.
#
# Naming an `auth_type` counts on its own: it is the caller choosing a flow,
# including the browser one, which is theirs to choose.
_WAYS_TO_AUTHENTICATE = (
    frozenset({'access_token'}),
    frozenset({'credentials_provider'}),
    frozenset({'auth_type'}),
    frozenset({'azure_client_id', 'azure_client_secret'}),
    frozenset({'oauth_client_id', 'oauth_client_secret'}),
    frozenset({'oauth_client_id', 'oauth_jwt_key_file'}),
    frozenset({'_use_cert_as_auth', '_tls_client_cert_file'}),
    # The driver refuses these itself, clearly and immediately, which is a
    # better message than this one; they are here so it gets to give it.
    frozenset({'username'}),
    frozenset({'password'}),
)


def _says_how_to_authenticate(given: Mapping[str, Any]) -> bool:
    r"""Whether these arguments name a way the driver can authenticate."""
    named = {name for name, value in given.items() if value}
    return any(way <= named for way in _WAYS_TO_AUTHENTICATE)


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
    if not _says_how_to_authenticate(kwargs):
        named = ', '.join(
            ' + '.join(sorted(way)) for way in _WAYS_TO_AUTHENTICATE[1:-2]
        )
        raise ConnectorError(
            f'connecting to Databricks needs a credential and none was given: '
            f'pass access_token=..., set DATABRICKS_TOKEN, or name {named} to '
            f'authenticate another way. Half of a pair is not a credential -- '
            f'the driver resolves it to its browser flow, which waits for a '
            f'human and never returns',
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
