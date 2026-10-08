# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, TypeAlias

import pyarrow as pa

from sdm_connectors.sql import (
    check_connect_args,
    merge_driver_options,
    require_driver,
    rows_to_arrow,
)

# A type checker reads the driver's own types; at runtime its absence is
# reported as MissingBackendError rather than an ImportError.
if TYPE_CHECKING:
    import psycopg

    Connection: TypeAlias = psycopg.Connection[Any]
else:
    psycopg = require_driver('postgres', 'psycopg[binary]', 'psycopg')
    # Unsubscripted, because the engine checks connections with isinstance,
    # which refuses a subscripted generic.
    Connection = psycopg.Connection

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


def own_transaction(connection: Connection) -> AbstractContextManager[Any]:
    r"""The transaction one read on ``connection`` runs in: none under
    autocommit, and a savepoint inside a transaction the caller opened.
    """
    # Otherwise psycopg holds the read's transaction, and its locks, until a
    # commit, and one failed statement refuses every later one.
    if connection.autocommit:
        return nullcontext()
    return connection.transaction()


#: The field metadata key holding a column's PostgreSQL type, by its OID.
TYPE_OID_KEY = b'postgres.type_oid'

# The type of a function that returns nothing, such as `setseed`.
_VOID = 2278


def _text(value: Any) -> str:
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(value, separators=(',', ':'))


def _texts(values: Any) -> list[str | None]:
    return [None if value is None else str(value) for value in values]


#: Each built-in type's Arrow type, by OID, and how a value psycopg read is
#: turned into one, if it needs to be. A type Arrow has no exact counterpart
#: for is text: ``numeric`` its exact decimal, ``json`` and ``jsonb`` the
#: document as compact JSON, and the rest as PostgreSQL writes them.
_ARROW_TYPES: dict[int, tuple[pa.DataType, Callable[[Any], Any] | None]] = {
    16: (pa.bool_(), None),  # bool
    17: (pa.binary(), None),  # bytea
    18: (pa.string(), None),  # char
    19: (pa.string(), None),  # name
    20: (pa.int64(), None),  # int8
    21: (pa.int16(), None),  # int2
    23: (pa.int32(), None),  # int4
    25: (pa.string(), None),  # text
    26: (pa.int64(), None),  # oid
    114: (pa.string(), _json_text),  # json
    142: (pa.string(), _text),  # xml
    650: (pa.string(), _text),  # cidr
    # A float4 is read as the double of its text, as psycopg reads it.
    700: (pa.float64(), None),  # float4
    701: (pa.float64(), None),  # float8
    774: (pa.string(), _text),  # macaddr8
    790: (pa.string(), _text),  # money
    829: (pa.string(), _text),  # macaddr
    869: (pa.string(), _text),  # inet
    1005: (pa.list_(pa.int64()), None),  # int2[]
    1007: (pa.list_(pa.int64()), None),  # int4[]
    1009: (pa.list_(pa.string()), None),  # text[]
    1014: (pa.list_(pa.string()), None),  # bpchar[]
    1015: (pa.list_(pa.string()), None),  # varchar[]
    1016: (pa.list_(pa.int64()), None),  # int8[]
    1021: (pa.list_(pa.float64()), None),  # float4[]
    1022: (pa.list_(pa.float64()), None),  # float8[]
    1042: (pa.string(), None),  # bpchar
    1043: (pa.string(), None),  # varchar
    1082: (pa.date32(), None),  # date
    1083: (pa.time64('us'), None),  # time
    1114: (pa.timestamp('us'), None),  # timestamp
    1184: (pa.timestamp('us', tz='UTC'), None),  # timestamptz
    1186: (pa.duration('us'), None),  # interval
    1266: (pa.string(), None),  # timetz, read as its text by `_TEXT_LOADED`
    1231: (pa.list_(pa.string()), _texts),  # numeric[]
    1700: (pa.string(), _text),  # numeric
    2950: (pa.string(), _text),  # uuid
    2951: (pa.list_(pa.string()), _texts),  # uuid[]
    3802: (pa.string(), _json_text),  # jsonb
}


#: Types read as the text PostgreSQL writes, rather than as psycopg's Python
#: value: Arrow has no time with a zone, and psycopg's ``datetime.time`` keeps
#: an offset Arrow's ``time64`` would drop.
_TEXT_LOADED = (1266,)  # timetz


def _oid(column: Any) -> int | None:
    oid = getattr(column, 'type_code', None)
    return oid if isinstance(oid, int) else None


def _name(column: Any) -> str:
    return column.name if hasattr(column, 'name') else column[0]


def run(
    connection: Connection, sql: str, params: Sequence[Any] | None = None
) -> pa.Table:
    r"""Run one statement and return what it reads, typed as PostgreSQL
    declares its columns.

    ``%s`` markers are bound positionally from ``params``. Each read is its
    own transaction, or a savepoint inside one the caller opened, so it holds
    no locks once it returns, commits nothing of the caller's, and a statement
    that fails leaves the caller's transaction usable.

    psycopg returns rows, so each column becomes the Arrow type its declared
    type maps to, whatever values it holds: ``bigint`` is ``int64``, ``double
    precision`` ``float64``, ``text`` ``string``, ``boolean`` ``bool``,
    ``timestamp`` a timestamp, ``timestamptz`` a UTC timestamp. A type with no
    exact Arrow counterpart is text: ``numeric`` its exact decimal, ``json``
    and ``jsonb`` compact JSON, others, ``timetz`` among them, as PostgreSQL
    writes them. A type not built in takes the type Arrow infers. A value its
    column's Arrow type can't hold, such as a two-dimensional array, is
    refused, naming the column. Each field's metadata holds its type's OID
    under ``postgres.type_oid``.

    A result whose every column is ``void``, as ``SELECT setseed(...)``
    returns, holds nothing to read: it is not fetched, and comes back as its
    columns, typed null, with no rows.

    Rows are read as tuples whatever ``row_factory`` the connection was made
    with, and the connection's own is left as it was.
    """
    from psycopg.rows import tuple_row
    from psycopg.types.string import TextLoader

    with (
        own_transaction(connection),
        connection.cursor(row_factory=tuple_row) as cursor,
    ):
        # The cursor's adapters are its own copy, so the connection keeps
        # reading these types as it did.
        for text_oid in _TEXT_LOADED:
            cursor.adapters.register_loader(text_oid, TextLoader)
        if params is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, tuple(params))
        description = cursor.description
        if description is None:
            return pa.table({})
        names = [_name(column) for column in description]
        oids = [_oid(column) for column in description]
        if all(oid == _VOID for oid in oids):
            return pa.Table.from_arrays(
                [pa.nulls(0) for _ in names], names=names
            )
        rows = cursor.fetchall()
    types: list[pa.DataType | None] = []
    converted: list[list[Any]] = [list(row) for row in rows]
    for index, oid in enumerate(oids):
        arrow_type, convert = _ARROW_TYPES.get(oid or -1, (None, None))
        types.append(arrow_type)
        if convert is not None:
            for row in converted:
                if row[index] is not None:
                    row[index] = convert(row[index])
    table = rows_to_arrow(converted, names, types)
    fields = [
        field.with_metadata({TYPE_OID_KEY: str(oid).encode()})
        if oid is not None
        else field
        for field, oid in zip(table.schema, oids, strict=True)
    ]
    return pa.Table.from_arrays(table.columns, schema=pa.schema(fields))
