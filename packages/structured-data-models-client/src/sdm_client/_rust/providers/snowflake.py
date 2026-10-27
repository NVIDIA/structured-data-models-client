# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Snowflake provider (T3.6): it answers a fetch step by running the
statements its plan carries, in order, on one cursor of the source's
connector connection (a Snowpark session's own, which ``SnowflakeSource``
takes), and hands back the last one's result.

Each statement runs with the connection's parameter style switched to
``qmark``, so a bound value reaches Snowflake as data, and switched back
after, so the caller's connection is left as it was, as the reference
binds (``SNOW/binding.py:15-38``).

A result is the connector's Arrow (``fetch_arrow_all(force_return_table=
True)``, an empty result keeping its schema). A result the connector gives
as rows only (``SHOW``, ``DESCRIBE``: ``fetch_arrow_all`` raises
``NotSupportedError``) is built here from ``cursor.description`` and
``fetchall()``: every column nullable text, in the description's order,
each non-null cell its ``str``, an empty result keeping its columns. The
core reads those columns by position, as the reference reads the rows.

The statements are the core's (``sdm-data`` renders Snowflake's SQL); this
module writes none, folds no name's case, retries nothing and says nothing
of what a failure means; a key set arrives as the JSON text the core
wrote for it, bound as it is. The connector's ``errno`` and ``sqlstate`` reach
the core as the provider error's facts, and the core reads them. A plan
this provider cannot run, one for a host provider or another dialect, or
one binding a parameter it does not take or a key set not handed over,
raises before anything runs, and
the run loop hands the core a provider error.

It imports nothing of pandas.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from sdm_connectors import providers
from sdm_connectors.providers import PlanNotRun

from sdm_client._rust import codec


class SnowflakeProvider:
    r"""Answers fetch steps on ``connection``, a Snowflake connector
    connection, or what stands in for one: an object whose ``cursor()`` has
    ``execute``, ``fetch_arrow_all``, ``fetchall``, ``description`` and
    ``close``, and which keeps its parameter style as the connector does."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        read: dict[int, str] = {}
        statements = [
            (statement['text'], _values(statement, keys, read))
            for statement in providers.statements(step, 'snowflake')
        ]
        cursor = self._connection.cursor()
        try:
            for text, values in statements:
                with _qmark(self._connection):
                    if values:
                        cursor.execute(text, tuple(values))
                    else:
                        cursor.execute(text)
            return _fetched(cursor)
        finally:
            cursor.close()


def _values(
    statement: dict[str, Any],
    keys: Mapping[int, Any] | None,
    read: dict[int, str],
) -> list[Any]:
    r"""A statement's parameters, in order: each scalar's Python value, and
    a key set's text as the core wrote it (T3.28, ``providers.text``),
    bound as it is. ``read`` holds each text read so far, by attachment."""
    values: list[Any] = []
    for param in statement['params']:
        if param['param'] == 'text':
            attachment = param['attachment']
            if attachment not in read:
                read[attachment] = providers.text(param, keys)[1]
            values.append(read[attachment])
            continue
        if param['param'] != 'value':
            raise PlanNotRun('a parameter this provider does not bind')
        try:
            value = codec.decode_scalar(param['value'])
        except codec.CodecError:
            raise PlanNotRun('a parameter the contract does not hold') from None
        if isinstance(value, codec.Temporal):
            raise PlanNotRun('a time parameter this provider does not bind')
        values.append(value)
    return values


def _style(connection: Any) -> str:
    r"""The attribute keeping ``connection``'s parameter style: the
    connector's public, settable ``paramstyle`` where its class has one
    (connector 5), else ``_paramstyle``, which its cursors read."""
    if isinstance(getattr(type(connection), 'paramstyle', None), property):
        return 'paramstyle'
    return '_paramstyle'


@contextmanager
def _qmark(connection: Any) -> Iterator[None]:
    r"""``connection`` in ``qmark`` style for the block, its own after."""
    name = _style(connection)
    previous = getattr(connection, name)
    setattr(connection, name, 'qmark')
    try:
        yield
    finally:
        setattr(connection, name, previous)


def _fetched(cursor: Any) -> Any:
    r"""The cursor's result: the connector's Arrow, or the rows it gives as
    rows only, as text (:func:`_as_text`)."""
    from snowflake.connector.errors import NotSupportedError

    try:
        return cursor.fetch_arrow_all(force_return_table=True)
    except NotSupportedError:
        return _as_text(cursor.description or (), cursor.fetchall())


def _as_text(description: Any, rows: list[Any]) -> Any:
    r"""``rows`` as a table of nullable text columns, one per column of
    ``description``, in its order, each non-null cell its ``str``."""
    import pyarrow as pa

    names = [str(column[0]) for column in description]
    columns = [
        pa.array(
            [None if row[at] is None else str(row[at]) for row in rows],
            pa.string(),
        )
        for at in range(len(names))
    ]
    schema = pa.schema([pa.field(name, pa.string()) for name in names])
    return pa.Table.from_arrays(columns, schema=schema)
