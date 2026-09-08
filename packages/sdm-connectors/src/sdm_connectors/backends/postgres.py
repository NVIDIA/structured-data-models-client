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

# psycopg's own Python-side connection options. These are not libpq
# parameters and no libpq query reports them, so they are named here.
_DRIVER_ARGS = frozenset(
    {
        'autocommit',
        'context',
        'cursor_factory',
        'prepare_threshold',
        'row_factory',
    }
)

# libpq's parameters, taken from the libpq that is actually loaded rather than
# written out. A hand-kept list silently refuses whatever it has not caught up
# with, and the ones it was missing were not obscure: `sslcertmode` and
# `requirepeer` are how a caller tightens how the connection authenticates.
# Falls back to the names libpq has had for years if the query is unavailable,
# so an unusual build refuses a little rather than everything.
_KNOWN_LIBPQ_ARGS = frozenset(
    {
        'application_name',
        'channel_binding',
        'client_encoding',
        'connect_timeout',
        'dbname',
        'fallback_application_name',
        'gssencmode',
        'host',
        'hostaddr',
        'keepalives',
        'keepalives_count',
        'keepalives_idle',
        'keepalives_interval',
        'options',
        'passfile',
        'password',
        'port',
        'replication',
        'require_auth',
        'service',
        'servicefile',
        'sslcert',
        'sslcrl',
        'sslcrldir',
        'sslkey',
        'sslmode',
        'sslpassword',
        'sslrootcert',
        'sslsni',
        'target_session_attrs',
        'tcp_user_timeout',
        'user',
    }
)


def _libpq_args() -> frozenset[str]:
    try:
        from psycopg import pq

        reported = {
            option.keyword.decode() for option in pq.Conninfo.get_defaults()
        }
    except Exception:
        return _KNOWN_LIBPQ_ARGS
    return frozenset(reported | _KNOWN_LIBPQ_ARGS)


_CONNECT_ARGS = _DRIVER_ARGS | _libpq_args()


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
