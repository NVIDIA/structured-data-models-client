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
        'tls_verify',
        'use_cloud_fetch',
        'use_kernel',
        'user_agent_entry',
        'username',
        '_tls_client_cert_file',
        '_tls_client_cert_key_file',
        '_tls_client_cert_key_password',
        '_use_cert_as_auth',
    }
)


# What this guard is for: the driver's thrift route, handed nothing it
# recognises, falls through to an OAuth flow that opens a browser and waits for
# a human, binding a listening socket while it does -- so a script that simply
# forgot its token never returns. Everything else the driver refuses itself,
# immediately and with a better message than this one could give.
#
# So the question is narrow: would this call reach that fallthrough? The names
# below are the branches the installed driver dispatches on before it, each
# with the names it reads together. Anything that matches one of them, and
# anything on the kernel route -- which raises rather than waiting -- is left
# to the driver.
_WAYS_THE_THRIFT_ROUTE_AUTHENTICATES = (
    frozenset({'access_token'}),
    frozenset({'credentials_provider'}),
    frozenset({'auth_type'}),
    frozenset({'_use_cert_as_auth', '_tls_client_cert_file'}),
    # Refused by the driver before it reaches any flow, and its sentence --
    # that this is no longer supported, use OAuth or an access token -- says
    # more than this one could.
    frozenset({'username'}),
    frozenset({'password'}),
)


def _an_auth_type_the_driver_knows() -> frozenset[str]:
    r"""The flows the installed driver dispatches ``auth_type`` on.

    Read from the driver. Any other value falls through to the browser, so
    accepting `auth_type` on sight let a typo produce exactly the wait this
    guard exists to prevent.
    """
    try:
        from databricks.sql.auth.auth import AuthType

        return frozenset(member.value for member in AuthType)
    except Exception:  # noqa: BLE001 - an unknown driver is not a credential
        return frozenset()


_KNOWN_AUTH_TYPES = _an_auth_type_the_driver_knows()


#: Naming an OAuth application is choosing the browser flow on purpose. The
#: driver resolves to it for anything it does not recognise -- including an
#: empty call -- so the driver cannot tell a deliberate U2M setup from a
#: forgotten token, and this is the only thing that can: one of them says the
#: word `oauth`. Not beside a machine-to-machine secret, which asks for
#: something else and would be quietly downgraded to a browser.
_AN_OAUTH_APP = frozenset(
    {'oauth_client_id', 'oauth_redirect_port', 'oauth_redirect_port_range'}
)
_MACHINE_TO_MACHINE = frozenset(
    {'oauth_client_secret', 'oauth_jwt_key_file', 'azure_client_secret'}
)


def _says_how_to_authenticate(given: Mapping[str, Any]) -> bool:
    r"""Whether this call can reach the driver's browser flow by accident."""
    named = {name for name, value in given.items() if value}
    if 'use_kernel' in named:
        return True
    if 'auth_type' in named and given['auth_type'] not in _KNOWN_AUTH_TYPES:
        named = named - {'auth_type'}
    if _AN_OAUTH_APP & named and not _MACHINE_TO_MACHINE & named:
        return True
    return any(way <= named for way in _WAYS_THE_THRIFT_ROUTE_AUTHENTICATES)


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
        raise ConnectorError(
            'connecting to Databricks needs a credential and none was given: '
            'pass access_token=..., set DATABRICKS_TOKEN, or name '
            'credentials_provider, an auth_type, an oauth_client_id, or '
            '_use_cert_as_auth with _tls_client_cert_file. Half of a pair is '
            'not a credential, and '
            'neither is a machine-to-machine one on this route: the driver '
            'resolves what it does not recognise to its browser flow, which '
            'waits for a human and never returns',
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
