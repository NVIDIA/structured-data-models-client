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
# with, and what it was missing was not obscure: `sslcertmode` and
# `requirepeer` are how a caller tightens how the connection authenticates, so
# refusing them pushes someone toward a weaker setting.
#
# The names below are what libpq 180006 reports, kept only as the answer
# when the query is unavailable, so an unusual build refuses what a recent
# libpq would not have known either rather than refusing nearly everything.
_KNOWN_LIBPQ_ARGS = frozenset(
    {
        'application_name',
        'channel_binding',
        'client_encoding',
        'connect_timeout',
        'dbname',
        'fallback_application_name',
        'gssdelegation',
        'gssencmode',
        'gsslib',
        'host',
        'hostaddr',
        'keepalives',
        'keepalives_count',
        'keepalives_idle',
        'keepalives_interval',
        'krbsrvname',
        'load_balance_hosts',
        'max_protocol_version',
        'min_protocol_version',
        'oauth_client_id',
        'oauth_client_secret',
        'oauth_issuer',
        'oauth_scope',
        'options',
        'passfile',
        'password',
        'port',
        'replication',
        'require_auth',
        'requirepeer',
        'scram_client_key',
        'scram_server_key',
        'service',
        'ssl_max_protocol_version',
        'ssl_min_protocol_version',
        'sslcert',
        'sslcertmode',
        'sslcompression',
        'sslcrl',
        'sslcrldir',
        'sslkey',
        'sslkeylogfile',
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


def _libpq_args() -> frozenset[str]:
    r"""What the loaded libpq takes, or the written-down set if it cannot say.

    Only one or the other. Unioning them accepts names this libpq does not
    know, which is worse than not checking at all: the argument gets past the
    guard that exists to catch it and fails later as a connection error, from
    the driver, naming nothing the caller can act on.
    """
    try:
        from psycopg import pq

        reported = {
            option.keyword.decode() for option in pq.Conninfo.get_defaults()
        }
    except Exception:
        return _KNOWN_LIBPQ_ARGS
    return frozenset(reported)


_CONNECT_ARGS = _DRIVER_ARGS | _libpq_args()


def connect(
    conninfo: str = '',
    *,
    driver_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Connect to a PostgreSQL database.

    Standard ``PG*`` environment variables are used when their corresponding
    keyword arguments are absent. Passing one explicitly as ``None`` stops this
    function reading the variable for it; libpq then reads the same variable
    itself, so the shell's value still applies. To connect without one, give
    the value you want. A PostgreSQL URI or libpq conninfo string can be
    supplied as ``conninfo``.
    """
    check_connect_args('postgres', kwargs, _CONNECT_ARGS)
    given = set(kwargs)
    kwargs = merge_driver_options('postgres', kwargs, driver_options)
    if not conninfo:
        for arg, env in _ENV_BY_ARG.items():
            if arg not in given and arg not in kwargs:
                kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if conninfo:
        return psycopg.connect(conninfo, **kwargs)
    return psycopg.connect(**kwargs)
