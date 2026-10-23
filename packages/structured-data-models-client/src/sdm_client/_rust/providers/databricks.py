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
(the seeds of a ``neighbors`` or ``lookup`` request, an Arrow attachment the
run loop hands over by id) as one JSON text, as the reference binds it
(``_dumps``): a key set of one column a JSON array of its values, of
several a JSON array of objects keyed by column name in column order. A key
set whose JSON is past the reference's budget of the warehouse's parameter
size is split into chunks as the reference splits it (``_chunk_rows``), the
statement run once per chunk and the results stitched into one batch space
by :data:`BATCH_COLUMN` (``_flatten``). A plan this provider cannot run, one
for a host provider or another dialect, one naming a key set it was not
handed, or binding two in one statement, raises, and the run loop hands the
core a provider error.

What a Databricks failure says (its SQLSTATE, and whether it refused an
expression) is read by :func:`error_fields`, as the reference reads it
(``DB/errors.py``), for the run loop to hand the core.

It imports nothing of pandas; it reads a key set with pyarrow, the
client's base dependency.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping
from typing import Any

from sdm_connectors import providers
from sdm_connectors.providers import PlanNotRun, arrow_table

from sdm_client._rust import codec

#: The budget of a key set's JSON text, in bytes, below the warehouse's
#: native parameter limit of 1 MB (the reference's ``_chunk_rows``).
PARAMETER_BUDGET = 900_000
#: The result column numbering each row's seed within the key set bound
#: (the contract's ``BATCH_COLUMN``).
BATCH_COLUMN = '__kumo_batch__'


class DatabricksProvider:
    r"""Answers fetch steps on ``connection``, a Databricks SQL connection,
    a ``SparkConnection``, or what stands in for one: an object whose
    ``cursor()`` is a context manager with ``execute(sql, parameters=...)``
    and ``fetchall_arrow()``."""

    def __init__(
        self, connection: Any, *, budget: int = PARAMETER_BUDGET
    ) -> None:
        self._connection = connection
        self._budget = budget

    @staticmethod
    def error_fields(error: BaseException) -> dict[str, Any] | None:
        r"""What a Databricks failure tells the core (:func:`error_fields`),
        for the run loop (``sdm_connectors.providers.Provider``)."""
        return error_fields(error)

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        statements = providers.statements(step, 'databricks')
        # Each key set read once, however many statements bind it.
        read: dict[int, list[Any]] = {}
        *before, (text, chunks) = [
            _bound(statement, keys, read, self._budget)
            for statement in statements
        ]
        if any(len(earlier) != 1 for _, earlier in before):
            raise PlanNotRun(
                'a key set past the parameter budget in a statement whose '
                'rows are not the answer'
            )
        with self._connection.cursor() as cursor:
            for earlier_text, earlier in before:
                _execute(cursor, earlier_text, earlier[0][1])
            if len(chunks) == 1:
                _execute(cursor, text, chunks[0][1])
                return cursor.fetchall_arrow()
            tables = []
            for offset, values in chunks:
                _execute(cursor, text, values)
                tables.append(_shifted(cursor.fetchall_arrow(), offset))
        import pyarrow as pa

        return pa.concat_tables(tables)


class DatabricksSqlProvider(DatabricksProvider):
    r"""The provider over a SQL warehouse connection
    (``databricks.sql.connect``)."""


class DatabricksSparkProvider(DatabricksProvider):
    r"""The provider over a notebook's Spark session: the session itself,
    which is wrapped in ``sdm_connectors.backends.spark.SparkConnection``
    (its checks and its refusal of a session whose frames cannot return
    Arrow included), or a ``SparkConnection`` already."""

    def __init__(self, session: Any, *, budget: int = PARAMETER_BUDGET) -> None:
        from sdm_connectors.backends.spark import SparkConnection

        connection = (
            session
            if isinstance(session, SparkConnection)
            else SparkConnection(session)
        )
        super().__init__(connection, budget=budget)


#: One statement's parameters, per chunk: the offset of the chunk's first
#: seed in the key set, and the values bound.
_Chunks = list[tuple[int, list[Any]]]


def _bound(
    statement: dict[str, Any],
    keys: Mapping[int, Any] | None,
    read: dict[int, list[Any]],
    budget: int,
) -> tuple[str, _Chunks]:
    r"""A statement's text, and its parameters' values per chunk: one chunk
    unless it binds a key set past ``budget``. ``read`` holds the rows of
    each key set read so far, by attachment."""
    values: list[Any] = []
    at: int | None = None
    rows: list[Any] = []
    for param in statement['params']:
        if param['param'] == 'value':
            try:
                values.append(_value(codec.decode_scalar(param['value'])))
            except codec.CodecError:
                raise PlanNotRun(
                    'a parameter the contract does not hold'
                ) from None
            continue
        if param['param'] != 'key_set':
            raise PlanNotRun('a parameter of an unknown kind')
        if at is not None:
            raise PlanNotRun('a statement binding more than one key set')
        attachment, stream = providers.key_set(param, keys)
        at = len(values)
        if attachment not in read:
            read[attachment] = _rows(arrow_table(stream))
        rows = read[attachment]
        values.append(None)
    if at is None:
        return statement['text'], [(0, values)]
    chunks: _Chunks = []
    offset = 0
    for chunk in _chunk_rows(rows, budget):
        chunk_values = list(values)
        chunk_values[at] = _dumps(chunk)
        chunks.append((offset, chunk_values))
        offset += len(chunk)
    return statement['text'], chunks


def _execute(cursor: Any, text: str, values: list[Any]) -> None:
    r"""Runs one statement, its values bound to its ``?`` markers in order;
    a statement binding none is run as text alone, as the reference runs
    one."""
    if values:
        cursor.execute(text, parameters=values)
    else:
        cursor.execute(text)


def _json_default(obj: Any) -> Any:
    r"""A value JSON has no form of, as the reference writes it: a NumPy
    scalar as its Python value, anything else as its ``str``."""
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def _dumps(obj: Any) -> str:
    r"""Compact JSON text, as the reference's ``_dumps`` writes it, so the
    chunks' size accounting is the text's length exactly."""
    return json.dumps(obj, separators=(',', ':'), default=_json_default)


def _chunk_rows(rows: list[Any], budget: int) -> list[list[Any]]:
    r"""``rows`` split into chunks whose JSON text stays within ``budget``
    bytes, as the reference's ``_chunk_rows`` splits them: each row counted
    as its JSON text and one separator, the enclosing brackets as two, and
    never an empty chunk unless there is no row."""
    chunks: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 2
    for row in rows:
        size = len(_dumps(row)) + 1
        if current and current_bytes + size > budget:
            chunks.append(current)
            current, current_bytes = [], 2
        current.append(row)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks or [[]]


def _rows(table: Any) -> list[Any]:
    r"""A key set's rows as the reference binds them: a key set of one
    column its values, of several an object per row keyed by column name in
    column order; each value as Python reads it from Arrow."""
    if table.num_columns == 1:
        return list(table.column(0).to_pylist())
    return list(table.to_pylist())


def _shifted(table: Any, offset: int) -> Any:
    r"""A chunk's result, its :data:`BATCH_COLUMN` moved by ``offset``, the
    position of the chunk's first seed in the whole key set, as an
    ``int64`` column, as the reference's ``_flatten`` moves it."""
    import pyarrow as pa
    import pyarrow.compute as pc

    index = table.schema.get_field_index(BATCH_COLUMN)
    if index < 0:
        raise PlanNotRun('a chunked statement whose rows carry no batch')
    batch = pc.add(table.column(index).cast(pa.int64()), offset)
    return table.set_column(index, BATCH_COLUMN, batch)


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


def error_fields(error: BaseException) -> dict[str, Any] | None:
    r"""What a Databricks failure says for the contract's provider error:
    its ``sqlstate``, and ``refuses_expression`` when the state's class is
    ``22`` (a data error) or ``42`` (a syntax or analysis error), as the
    reference's ``refuses_expression`` reads it; nothing where it says no
    state, and ``None`` for a failure that is not Databricks'. Its message
    is never kept."""
    if not _is_databricks(error):
        return None
    state = sql_state(error)
    if state is None:
        return {}
    fields: dict[str, Any] = {'sqlstate': state}
    if state[:2] in ('22', '42'):
        fields['refuses_expression'] = True
    return fields
