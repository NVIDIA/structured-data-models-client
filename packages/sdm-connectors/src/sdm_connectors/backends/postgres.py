# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, TypeAlias

from sdm_connectors.sql import (
    check_connect_args,
    merge_driver_options,
    require_driver,
)

psycopg = require_driver('postgres', 'psycopg[binary]', 'psycopg')

Connection: TypeAlias = psycopg.Connection

_ENV_BY_ARG = {
    'host': 'PGHOST',
    'port': 'PGPORT',
    'dbname': 'PGDATABASE',
    'user': 'PGUSER',
    'password': 'PGPASSWORD',
    'sslmode': 'PGSSLMODE',
}

# libpq connection parameters plus psycopg's Python-side connection options.
_CONNECT_ARGS = frozenset(
    {
        'application_name',
        'autocommit',
        'channel_binding',
        'client_encoding',
        'connect_timeout',
        'context',
        'cursor_factory',
        'dbname',
        'fallback_application_name',
        'gssencmode',
        'host',
        'hostaddr',
        'keepalives',
        'keepalives_count',
        'keepalives_idle',
        'keepalives_interval',
        'load_balance_hosts',
        'options',
        'passfile',
        'password',
        'port',
        'prepare_threshold',
        'replication',
        'require_auth',
        'row_factory',
        'service',
        'servicefile',
        'sslcert',
        'sslcrl',
        'sslcrldir',
        'sslkey',
        'sslmode',
        'sslnegotiation',
        'sslpassword',
        'sslrootcert',
        'sslsni',
        'target_session_attrs',
        'tcp_user_timeout',
        'user',
    }
)


def connect(
    conninfo: str = '',
    *,
    driver_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Connect to a PostgreSQL database.

    Standard ``PG*`` environment variables are used when their corresponding
    keyword arguments are absent. A PostgreSQL URI or libpq conninfo string can
    be supplied as ``conninfo``.
    """
    check_connect_args('postgres', kwargs, _CONNECT_ARGS)
    kwargs = merge_driver_options('postgres', kwargs, driver_options)
    if not conninfo:
        for arg, env in _ENV_BY_ARG.items():
            if kwargs.get(arg) is None:
                kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if conninfo:
        return psycopg.connect(conninfo, **kwargs)
    return psycopg.connect(**kwargs)
