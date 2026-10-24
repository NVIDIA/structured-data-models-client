# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Postgres provider (T2.3): it answers a fetch step by running the
statements its plan carries, in order, on one cursor of the source's
psycopg connection, and hands back the last one's rows as an Arrow table,
built from the driver's values with no pandas, an empty result keeping its
schema.

The statements are the core's (``sdm-data`` renders Postgres's SQL); this
module writes none. Each request gets its own cursor, in a transaction of
its own as the reference's reads take one (``own_transaction``): none under
autocommit, a savepoint inside a transaction the caller opened, so a failed
read leaves the caller's connection usable.

A key set a plan names (the seeds of a ``neighbors`` or ``lookup``
request) is bound as the dialect binds it: the statement's one parameter,
``$1``, the JSON text the core wrote of it (T3.28: each seed as the old
client writes it, which the statement reads in the source's own types),
handed over as a one-cell attachment and bound as it was handed; the core
splits a key set past the reference's budget into one fetch per chunk, so
this provider neither writes nor splits one.
psycopg writes a parameter ``%s`` and a ``%`` as ``%%``, so such a
statement's text is given to it so written (:func:`_psycopg_text`), its
``$1`` the one outside quotes; one holding no such ``$1``, or several, is
not run. A plan this
provider cannot run, one for a host provider, one with a scalar parameter,
the core's Arrow key set or more than one text in a statement, or one
naming a text it was not handed, raises, and the run loop hands the core a
provider error.

Each result column's Arrow type is read from the column's Postgres type,
by the type's name in the driver's registry, never by its number (a type
the registry does not name, a user's composite or enum, is read as text,
which is how the driver reads it); the values are the driver's, moved into
Arrow by pyarrow, and how the reference reads each is the core's
(``sdm-data``'s Postgres value rules, T2.14d). The driver reads each value
as it reads the reference's (its own loaders, so the statements' results
are the reference's): a ``json`` or ``jsonb`` value is handed over as the
JSON text of the value the driver read, its field marked as JSON (Arrow's
``arrow.json`` metadata, :data:`JSON_TAG`); an ``interval`` (the driver's
``timedelta``, a month counted as 30 days as the reference's driver counts
it) as pyarrow's ``MonthDayNano`` of it, its field tagged ``sdm:interval``
= ``timedelta`` (:data:`INTERVAL_TAG`); a ``numeric`` no Arrow decimal
holds as each value's exact text, its declaration marked
(:func:`_numeric_field`). A ``timestamptz`` is built as the instant in UTC,
whatever the session's zone.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from sdm_connectors import providers
from sdm_connectors.providers import PlanNotRun

from sdm_client._rust.providers.postgres_arrow import BATCH_ROWS, rows_arrow


class PostgresProvider:
    r"""Answers fetch steps on ``connection``, a psycopg connection."""

    def __init__(
        self, connection: Any, *, batch_rows: int = BATCH_ROWS
    ) -> None:
        self._connection = connection
        self._batch_rows = batch_rows

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        statements = providers.statements(step, 'postgres')
        bound: list[str | None] = []
        for statement in statements:
            params = statement['params']
            if len(params) > 1:
                raise PlanNotRun('a statement binding more than one key set')
            if not params:
                bound.append(None)
                continue
            if params[0]['param'] == 'value':
                raise PlanNotRun('a plan with a scalar parameter')
            bound.append(providers.text(params[0], keys)[1])
        connection = self._connection
        with _own_transaction(connection), connection.cursor() as cursor:
            for statement, text in zip(statements, bound, strict=True):
                if text is None:
                    cursor.execute(statement['text'])
                else:
                    cursor.execute(_psycopg_text(statement['text']), (text,))
            return _result(cursor, connection, self._batch_rows)


def _own_transaction(connection: Any) -> Any:
    r"""The transaction one read runs in, as the reference's reads take
    one: none under autocommit, else a savepoint inside the caller's
    transaction or one of its own."""
    if connection.autocommit:
        return nullcontext()
    return connection.transaction()


def _psycopg_text(text: str) -> str:
    r"""A statement binding its key set as ``$1``, as psycopg reads a
    statement with a parameter: each ``%`` doubled, quoted or not (psycopg
    reads every ``%`` as a placeholder's start), and the parameter, the one
    ``$1`` outside a quoted identifier (``"..."``) and a text literal
    (``'...'``, or ``E'...'`` with its backslash escapes), the only quoting
    the dialect writes, as ``%s``, which psycopg sends to the server as
    ``$1`` again. A ``$1`` inside quotes, a relation's name holding it
    (``"users$1"``), is text and stays.

    Raises:
        PlanNotRun: The text holds no such ``$1``, or more than one, so its
            parameter is not told apart.
    """
    out: list[str] = []
    found = 0
    at = 0
    while at < len(text):
        char = text[at]
        if char == '"' or char == "'":
            escaped = char == "'" and at > 0 and text[at - 1] in 'Ee'
            end = at + 1
            while end < len(text):
                if escaped and text[end] == '\\':
                    end += 2
                    continue
                if text[end] == char:
                    if text[end + 1 : end + 2] == char:
                        end += 2
                        continue
                    break
                end += 1
            out.append(text[at : end + 1].replace('%', '%%'))
            at = end + 1
        elif text.startswith('$1', at) and not text[at + 2 : at + 3].isdigit():
            out.append('%s')
            found += 1
            at += 2
        else:
            out.append('%%' if char == '%' else char)
            at += 1
    if found != 1:
        raise PlanNotRun('a statement whose parameter is not told apart')
    return ''.join(out)


def _result(cursor: Any, connection: Any, batch_rows: int) -> Any:
    r"""The last statement's rows as a pyarrow table, in batches of at most
    ``batch_rows``, each column typed by its Postgres type
    (:func:`postgres_arrow.rows_arrow`)."""
    description = cursor.description
    if description is None:
        raise PlanNotRun('a plan whose last statement returns no rows')
    # One read of every row, as the reference reads a result
    # (``fetchall``), built into batches of at most ``batch_rows``.
    return rows_arrow(description, cursor.fetchall(), connection, batch_rows)
