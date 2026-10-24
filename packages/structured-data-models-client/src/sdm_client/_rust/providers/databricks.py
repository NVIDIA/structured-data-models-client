# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Databricks providers (T3.4): they answer a fetch step by running
the statements its plan carries, in order, on one cursor of the source's
connection, and hand back the last one's result as the driver's Arrow
table (``fetchall_arrow``), an empty result keeping its schema.

One implementation serves both connections a Databricks source takes: a
SQL warehouse's (the Databricks SQL connector's), and a notebook's Spark
session wrapped in ``sdm_connectors``' ``SparkConnection``, whose cursor
runs ``spark.sql`` and fetches with ``DataFrame.toArrow()``. Both bind
positional ``?`` markers from ``execute(sql, parameters=[...])``, as the
reference's sampler binds them (``DB/sampler.py``).

The statements are the core's (``sdm-data`` renders Databricks' SQL); this
module writes none. A statement's parameters are bound in order: a scalar
as the Python value the contract's scalar holds (:mod:`..codec`), a key set
(the seeds of a ``neighbors`` or ``lookup`` request) as the one JSON text the
core wrote of it (T3.28), handed over as a one-cell attachment the run loop
passes by id, bound as it was handed. The core writes that text as the
reference binds it (``_dumps``) and splits a key set past the reference's
budget of the warehouse's parameter size into one fetch per chunk
(``_chunk_rows``, ``_flatten``), so this provider neither writes nor splits
one. A plan this provider cannot run, one for a host provider or another
dialect, one naming a text it was not handed, or one binding the core's
Arrow key set, raises, and the run loop hands the core a provider error.

What a Databricks failure says (its SQLSTATE, and whether it refused an
expression) is read by :func:`error_fields`, as the reference reads it
(``DB/errors.py``), for the run loop to hand the core.

It imports nothing of pandas; it reads a key set's text with pyarrow, the
client's base dependency.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping
from typing import Any

from sdm_connectors import providers
from sdm_connectors.providers import PlanNotRun

from sdm_client._rust import codec


class DatabricksProvider:
    r"""Answers fetch steps on ``connection``, a Databricks SQL connection,
    a ``SparkConnection``, or what stands in for one: an object whose
    ``cursor()`` is a context manager with ``execute(sql, parameters=...)``
    and ``fetchall_arrow()``."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @staticmethod
    def error_fields(error: BaseException) -> dict[str, Any] | None:
        r"""What a Databricks failure tells the core (:func:`error_fields`),
        for the run loop (``sdm_connectors.providers.Provider``)."""
        return error_fields(error)

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        statements = providers.statements(step, 'databricks')
        # Each text read once, however many statements bind it.
        read: dict[int, str] = {}
        bound = [_bound(statement, keys, read) for statement in statements]
        with self._connection.cursor() as cursor:
            for statement, values in zip(statements, bound, strict=True):
                _execute(cursor, statement['text'], values)
            return cursor.fetchall_arrow()


class DatabricksSqlProvider(DatabricksProvider):
    r"""The provider over a SQL warehouse connection
    (``databricks.sql.connect``)."""


class DatabricksSparkProvider(DatabricksProvider):
    r"""The provider over a notebook's Spark session: the session itself,
    which is wrapped in ``sdm_connectors.backends.spark.SparkConnection``
    (its checks and its refusal of a session whose frames cannot return
    Arrow included), or a ``SparkConnection`` already."""

    def __init__(self, session: Any) -> None:
        from sdm_connectors.backends.spark import SparkConnection

        connection = (
            session
            if isinstance(session, SparkConnection)
            else SparkConnection(session)
        )
        super().__init__(connection)


def _bound(
    statement: dict[str, Any],
    keys: Mapping[int, Any] | None,
    read: dict[int, str],
) -> list[Any]:
    r"""A statement's parameters' values, in order: each scalar's, and each
    key set's text as it was handed (``providers.text``). ``read`` holds
    each text read so far, by attachment."""
    values: list[Any] = []
    for param in statement['params']:
        if param['param'] == 'value':
            try:
                values.append(_value(codec.decode_scalar(param['value'])))
            except codec.CodecError:
                raise PlanNotRun(
                    'a parameter the contract does not hold'
                ) from None
            continue
        if param['param'] != 'text':
            raise PlanNotRun('a parameter of an unknown kind')
        attachment = param['attachment']
        if attachment not in read:
            read[attachment] = providers.text(param, keys)[1]
        values.append(read[attachment])
    return values


def _execute(cursor: Any, text: str, values: list[Any]) -> None:
    r"""Runs one statement, its values bound to its ``?`` markers in order;
    a statement binding none is run as text alone, as the reference runs
    one."""
    if values:
        cursor.execute(text, parameters=values)
    else:
        cursor.execute(text)


_EPOCH = datetime.datetime(1970, 1, 1)
_OFFSET = re.compile(r'([+-])(\d{2}):?(\d{2})\Z')


def _value(value: Any) -> Any:
    r"""A decoded scalar as the value the drivers bind: a temporal one as
    the ``date``, ``datetime``, ``time`` or ``timedelta`` it counts (a time
    stamp with a zone in that zone, one without naive), anything else as
    decoded."""
    if not isinstance(value, codec.Temporal):
        return value
    if value.tag == 'date32':
        return _EPOCH.date() + datetime.timedelta(days=value.value)
    micros = _micros(value)
    if value.tag == 'duration':
        return datetime.timedelta(microseconds=micros)
    if value.tag == 'time64':
        return (_EPOCH + datetime.timedelta(microseconds=micros)).time()
    if value.zone is None:
        return _EPOCH + datetime.timedelta(microseconds=micros)
    utc = _EPOCH.replace(tzinfo=datetime.timezone.utc)
    return (utc + datetime.timedelta(microseconds=micros)).astimezone(
        _zone(value.zone)
    )


def _micros(value: Any) -> int:
    r"""A temporal scalar's count in microseconds, the finest the drivers
    bind; one finer is not run."""
    if value.unit == 'us':
        return int(value.value)
    whole, rest = divmod(value.value, 1000)
    if rest:
        raise PlanNotRun('a time parameter finer than a microsecond')
    return int(whole)


def _zone(name: str) -> datetime.tzinfo:
    r"""A time zone by its name, or a fixed offset (``+05:30``)."""
    if name in ('UTC', 'Etc/UTC', 'Z'):
        return datetime.timezone.utc
    if match := _OFFSET.match(name):
        sign, hours, minutes = match.groups()
        delta = datetime.timedelta(hours=int(hours), minutes=int(minutes))
        return datetime.timezone(-delta if sign == '-' else delta)
    import zoneinfo

    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise PlanNotRun('a time parameter in a zone not known here') from None


# The SQLSTATE a Databricks failure says, as the reference reads it
# (DB/errors.py): written last in the message, before the statement
# Databricks appends, which may quote anything.
_SQL_STATE_RE = re.compile(r'SQLSTATE:\s*([0-9A-Z]{5})')
_SQL_CONTEXT_RE = re.compile(r'\n\s*== SQL\b')
# The error class a Databricks message opens with (DB/errors.py).
_ERROR_CLASS_RE = re.compile(r'\s*\[([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]')
#: The modules whose exceptions are a Databricks failure: the SQL
#: connector's, and Spark's (classic and Spark Connect).
_DRIVER_MODULES = ('databricks.sql', 'pyspark')


def sql_state(error: BaseException) -> str | None:
    r"""The SQLSTATE of a Databricks failure, if it says one, as the
    reference's ``sql_state`` reads it: what a Spark error answers
    (``getSqlState()``), else the last state the SQL connector's message
    writes before the statement it appends."""
    answer = getattr(error, 'getSqlState', None)
    state = answer() if callable(answer) else None
    if isinstance(state, str) and state:
        return state
    message = _SQL_CONTEXT_RE.split(str(error), maxsplit=1)[0]
    states = _SQL_STATE_RE.findall(message)
    return states[-1] if states else None


def _is_databricks(error: BaseException) -> bool:
    r"""Whether ``error`` is the SQL connector's or Spark's, by the module
    of its class or a base, so neither driver is imported to tell."""
    return any(
        module == prefix or module.startswith(prefix + '.')
        for module in (cls.__module__ or '' for cls in type(error).__mro__)
        for prefix in _DRIVER_MODULES
    )


def error_class(error: BaseException) -> str | None:
    r"""The error class of a Databricks failure, such as
    ``UNRESOLVED_COLUMN.WITH_SUGGESTION``, if it says one, as the
    reference's ``error_class`` reads it: what a Spark error answers
    (``getCondition()``, else ``getErrorClass()``), else the bracketed name
    the SQL connector's message opens with."""
    for method in ('getCondition', 'getErrorClass'):
        answer = getattr(error, method, None)
        name = answer() if callable(answer) else None
        if isinstance(name, str) and name:
            return name
    found = _ERROR_CLASS_RE.match(str(error))
    return None if found is None else found.group(1)


def error_fields(error: BaseException) -> dict[str, Any] | None:
    r"""What a Databricks failure says for the contract's provider error:
    its ``sqlstate`` and its ``error_class``, each where it says one, and
    ``None`` for a failure that is not Databricks'. What the failure means
    (a refused expression, a table that cannot be used) is the core's to
    say from these facts (T3.27, the data layer's ``failure``). Its message
    is never kept."""
    if not _is_databricks(error):
        return None
    fields: dict[str, Any] = {}
    state = sql_state(error)
    if state is not None:
        fields['sqlstate'] = state
    name = error_class(error)
    if name is not None:
        fields['error_class'] = name
    return fields
