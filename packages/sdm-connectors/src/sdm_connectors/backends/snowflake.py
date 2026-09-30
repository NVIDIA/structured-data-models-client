# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeAlias

from sdm_connectors.sql import (
    ConnectorError,
    check_connect_args,
    merge_driver_options,
    require_driver,
)

# A type checker reads the driver's own types; at runtime its absence is
# reported as MissingBackendError rather than an ImportError.
if TYPE_CHECKING:
    import snowflake.connector as snowflake_connector
else:
    snowflake_connector = require_driver(
        'snowflake',
        'snowflake-connector-python',
        'snowflake.connector',
        absent=('snowflake', 'snowflake.connector'),
    )

Connection: TypeAlias = snowflake_connector.SnowflakeConnection

_AUTH_ARGS = frozenset(
    {
        'account',
        'user',
        'password',
        'passcode',
        'private_key',
        'private_key_file',
        'private_key_path',
        'token',
        'authenticator',
        'auth_class',
        'connection_name',
        'oauth_client_id',
        'oauth_client_secret',
    }
)


def _declared_connect_args() -> frozenset[str]:
    r"""The connection keywords the Snowflake driver genuinely honours.

    ``DEFAULT_CONFIGURATION`` is the driver's *config-parameter* table, which
    omits the constructor-only parameters ``connection_name`` and
    ``connections_file_path`` -- the named-connection auth path, and the one
    way to authenticate without inlining credentials. Taking the union with
    ``SnowflakeConnection.__init__`` restores them, and folding in
    ``_AUTH_ARGS`` keeps the allow-list from ever contradicting the
    borrow logic that runs after it.
    """
    declared = set(snowflake_connector.connection.DEFAULT_CONFIGURATION)
    signature = inspect.signature(Connection.__init__)
    declared.update(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and name != 'self'
    )
    return frozenset(declared | _AUTH_ARGS)


_CONNECT_ARGS = _declared_connect_args()


def _active_snowpark_connection() -> Connection | None:
    try:
        from snowflake.snowpark.context import get_active_session
    except ModuleNotFoundError:
        return None
    try:
        session = get_active_session()
    except Exception:
        return None
    return session.connection


def connect(
    *,
    driver_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Connect to Snowflake, reusing an active Snowpark session when possible.

    An active Snowpark session is borrowed only when no authentication
    arguments are supplied, so explicit credentials are never silently ignored.
    Session-scoped arguments such as ``schema`` cannot be applied to a borrowed
    session without mutating a connection the caller owns, so that combination
    is rejected with an explanation instead of falling through to a
    credential-less connect that fails with "User is empty".

    ``driver_options`` cannot restate an argument given at the top level, so
    the escape hatch can never quietly replace a validated credential. What it
    does supply still counts towards the borrow decision, so credentials passed
    that way are honoured rather than silently ignored.
    """
    check_connect_args('snowflake', kwargs, _CONNECT_ARGS)
    kwargs = merge_driver_options('snowflake', kwargs, driver_options)
    if not _AUTH_ARGS & set(kwargs):
        borrowed = _active_snowpark_connection()
        if borrowed is not None:
            if kwargs:
                raise ConnectorError(
                    f'snowflake connector found an active Snowpark session but '
                    f'was also given {sorted(kwargs)}; a borrowed session '
                    f'cannot be reconfigured. Run the equivalent USE statement '
                    f'on your session, or pass full connection credentials to '
                    f'open a separate connection.',
                    code='INVALID_CONNECTOR_ARGS',
                    details={'arguments': sorted(kwargs)},
                )
            return borrowed
        if not kwargs:
            return _connect_without_arguments()
    return snowflake_connector.connect(**kwargs)


def _connect_without_arguments() -> Connection:
    r"""Connect with no arguments, explaining what this client expects on failure.

    With no arguments the driver falls back to its own default-connection file,
    whose absence it reports as "Default connection with name 'default' cannot
    be found", a feature this package never mentions. The fallback still works
    where it is configured; only the failure is re-stated in the client's terms.
    """
    try:
        connection = snowflake_connector.connect()
    except Exception as error:
        raise ConnectorError(
            f'snowflake connector requires connection arguments (at least '
            f"'account', 'user' and an authentication method such as "
            f"'password', 'private_key' or 'token'), an active Snowpark "
            f'session, or a configured default connection; got none: {error}',
            code='CONNECT_FAILED',
            details={'driver_error': type(error).__name__},
        ) from error
    return connection
