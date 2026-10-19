# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The SQLite provider (T2.2): it answers a fetch step by running the
statements its plan carries, in order, on a cursor of the source's ADBC
connection, and hands back the last one's result as the driver's Arrow
table, every batch the driver made, an empty result keeping its schema.

The statements are the core's (``sdm-data`` renders SQLite's SQL); this
module writes none. Each request gets its own cursor, as the reference's
reads do. A key set a plan names (the seeds of a ``neighbors`` or
``lookup`` request) is the Arrow attachment the core sent with the step,
which the run loop hands over by id: the driver ingests it as a temporary
table of the name its parameter gives (``adbc_ingest(..., temporary=True)``),
so its rows reach the statement as the driver writes Arrow into SQLite, and
the source's database is left as it was (the reference ingests its key
tables into the source's own database and leaves them there). The table is
created (``mode='create'``), or replaced (``mode='replace'``) only where
this module made that name on the connection before: a relation the caller
made is never replaced or dropped, and one of the same name fails the
ingest by name. A plan this provider cannot
run, one for a host provider, one with a scalar parameter, which no request
carries, or one naming a key set it was not handed, raises, and the run
loop hands the core a provider error.

It imports nothing of pandas, nor of this client; it reads a key set with
pyarrow, the client's base dependency.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

#: The key-set tables this module made, by connection (temporary tables are
#: the connection's own): only these are ever replaced. A connection object
#: that takes no weak reference (a wrapper's) is known by its ``id``.
_MADE: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()
_MADE_BY_ID: dict[int, set[str]] = {}


def _made(connection: Any) -> set[str]:
    r"""The names of the key-set tables this module made on
    ``connection``."""
    try:
        return _MADE.setdefault(connection, set())
    except TypeError:
        return _MADE_BY_ID.setdefault(id(connection), set())


class PlanNotRun(Exception):
    r"""A plan this provider does not run. Its text names no value."""


class SqliteProvider:
    r"""Answers fetch steps on ``connection``, an ADBC SQLite connection or
    what stands in for one."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        plan = step['plan']
        if plan['plan'] != 'sql':
            raise PlanNotRun("a host plan is not a SQL provider's to run")
        statements = plan['statements']
        if not statements or any(
            statement['dialect'] != 'sqlite' for statement in statements
        ):
            raise PlanNotRun('a plan with no statement or another dialect')
        relations: dict[str, int] = {}
        for statement in statements:
            for param in statement['params']:
                if param['param'] != 'key_set':
                    raise PlanNotRun('a plan with a scalar parameter')
                attachment = param['keys']['attachment']
                if keys is None or attachment not in keys:
                    raise PlanNotRun('a plan naming a key set not handed over')
                bound = relations.setdefault(param['relation'], attachment)
                if bound != attachment:
                    raise PlanNotRun('a plan binding two key sets to one name')
        made = _made(self._connection)
        with self._connection.cursor() as cursor:
            for relation, attachment in relations.items():
                cursor.adbc_ingest(
                    relation,
                    _arrow_table((keys or {})[attachment]),
                    mode='replace' if relation in made else 'create',
                    temporary=True,
                )
                made.add(relation)
            for statement in statements:
                cursor.execute(statement['text'])
            return cursor.fetch_arrow_table()


def _arrow_table(stream: Any) -> Any:
    r"""A key set the core sent, an object exporting an Arrow stream, read
    whole as a pyarrow table, through the PyCapsule interface pyarrow 14.0.1
    reads (decision E16)."""
    import pyarrow as pa

    # The private importer, because pyarrow 14.0.1, the floor, has no public
    # reader of a stream capsule (RecordBatchReader.from_stream is 15's).
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
