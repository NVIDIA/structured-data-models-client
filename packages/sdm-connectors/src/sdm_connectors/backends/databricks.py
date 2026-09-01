# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from typing import Any, TypeAlias

from sdm_connectors.sql import (
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
        'credentials_provider',
        'experimental_oauth_persistence',
        'oauth_client_id',
        'oauth_client_secret',
        'oauth_redirect_port_range',
        'oauth_scopes',
        'password',
        'session_configuration',
        'tls_verify',
        'use_cloud_fetch',
        'user_agent_entry',
        'username',
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
    check_connect_args('databricks', kwargs, _CONNECT_ARGS)
    kwargs = merge_driver_options('databricks', kwargs, driver_options)
    for arg, env in _ENV_BY_ARG.items():
        if kwargs.get(arg) is None:
            kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return databricks_sql.connect(**kwargs)
