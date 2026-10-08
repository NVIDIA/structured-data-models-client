# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The database steps of one run: written while recording, served on replay.

A run talks to the database through cursors (:class:`ParityCursor`) and a
connection facade (see :mod:`drivers`). Both send every call to a tape. A
:class:`RecordingTape` makes the call on the real connection and writes a
step; a :class:`ReplayingTape` checks the call against the next recorded step,
byte for byte, and answers with what was recorded, touching no database.

What a step holds is documented in ``tests/parity/FORMAT.md``.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa

from _parity.values import (
    Unrepresentable,
    as_arrow,
    canonical,
    decode,
    encode,
    encode_table,
    rows_to_table,
    same,
    table_to_rows,
)

#: How a recorded result was fetched: as rows, as an Arrow table, or as an
#: Arrow fetch the driver answered with nothing.
FETCH_ROWS = 'rows'
FETCH_ARROW = 'arrow'
FETCH_NONE = 'none'

#: The keyword names a driver's ``execute`` takes its parameters under:
#: databricks-sql, ADBC and DuckDB say ``parameters``, psycopg and the
#: Snowflake connector ``params``.
PARAMETER_KEYWORDS = ('parameters', 'params')


class Divergence(BaseException):
    r"""The run asked the database something the recording did not.

    A ``BaseException``, so that engine code catching ``Exception`` around a
    probe cannot swallow it and carry on down another path.
    """

    def __init__(self, index: int, message: str) -> None:
        super().__init__(f'step {index}: {message}')
        self.index = index


class Unsupported(AttributeError):
    r"""A driver call the parity connections do not record.

    An ``AttributeError``, so ``hasattr`` on a facade answers no for it.
    """


_INEXACT = object()


def _exact(value: Any) -> Any:
    r"""``value`` encoded, or ``_INEXACT`` if the encoding would not give it
    back with its type."""
    try:
        encoded = encode(value)
    except Unrepresentable:
        return _INEXACT
    return encoded if same(decode(encoded), value) else _INEXACT


# Column descriptions.

_DESCRIPTION_FIELDS = (
    'name',
    'type_code',
    'display_size',
    'internal_size',
    'precision',
    'scale',
    'null_ok',
)


class _DescriptionFields(NamedTuple):
    name: str
    type_code: Any = None
    display_size: Any = None
    internal_size: Any = None
    precision: Any = None
    scale: Any = None
    null_ok: Any = None


class Description(_DescriptionFields):
    r"""A DB-API column description, served on replay.

    A tuple of the seven DB-API fields, which are also attributes, as
    psycopg's are; ``type_display`` is set only where the recorded driver
    gave one.
    """


def describe_columns(description: Any) -> list[dict[str, Any]] | None:
    r"""A driver's ``cursor.description``, as the transcript stores it.

    Each column keeps its name and every other field that is set and that
    JSON holds exactly. A ``type_code`` it cannot hold, such as an Arrow
    type, is kept as text under ``type_text``.
    """
    if description is None:
        return None
    columns = []
    for entry in description:
        values = tuple(entry)
        column: dict[str, Any] = {'name': values[0]}
        for key, value in zip(
            _DESCRIPTION_FIELDS[1:], values[1:], strict=False
        ):
            if value is None:
                continue
            encoded = _exact(value)
            if encoded is not _INEXACT:
                column[key] = encoded
            elif key == 'type_code':
                column['type_text'] = str(value)
        display = getattr(entry, 'type_display', None)
        if isinstance(display, str):
            column['type_display'] = display
        columns.append(column)
    return columns


def served_description(
    recorded: list[dict[str, Any]] | None,
) -> list[Description] | None:
    r"""What a replay answers for ``cursor.description``."""
    if recorded is None:
        return None
    out = []
    for column in recorded:
        entry = Description(
            *(decode(column.get(key)) for key in _DESCRIPTION_FIELDS)
        )
        if 'type_display' in column:
            entry.type_display = column['type_display']  # type: ignore[attr-defined]
        out.append(entry)
    return out


# Errors.


def describe_error(error: BaseException) -> dict[str, Any]:
    r"""An error a driver raised, as the transcript stores it.

    Its class and text, its arguments where they are not just the text, and
    the public attributes JSON holds exactly, such as the Snowflake
    connector's ``errno`` and ``sqlstate``, which engine code reads.
    """
    cls = type(error)
    described: dict[str, Any] = {
        'type': f'{cls.__module__}.{cls.__qualname__}',
        'message': str(error),
    }
    args = _exact(list(error.args))
    if args is not _INEXACT and args != [described['message']]:
        described['args'] = args
    attributes = {}
    for key, value in sorted(getattr(error, '__dict__', {}).items()):
        if key.startswith('_'):
            continue
        encoded = _exact(value)
        if encoded is not _INEXACT:
            attributes[key] = encoded
    if attributes:
        described['attributes'] = attributes
    return described


def _error_class(qualified: str) -> type | None:
    parts = qualified.split('.')
    for cut in range(len(parts) - 1, 0, -1):
        try:
            found: Any = importlib.import_module('.'.join(parts[:cut]))
        except ImportError:
            continue
        try:
            for name in parts[cut:]:
                found = getattr(found, name)
        except AttributeError:
            return None
        return found if isinstance(found, type) else None
    return None


def rebuild_error(recorded: dict[str, Any]) -> BaseException:
    r"""The exception a recorded step raised, raised again on replay.

    The same class, since engine code catches driver errors by class, built
    from its recorded arguments, with its recorded attributes set again. A
    class that cannot be built so is built without its constructor. Where
    the class is gone, a ``RuntimeError`` holding the recorded text.
    """
    message = recorded.get('message', '')
    fallback = RuntimeError(f'{recorded["type"]}: {message}')
    cls = _error_class(recorded['type'])
    if cls is None or not issubclass(cls, BaseException):
        return fallback
    args = decode(recorded['args']) if 'args' in recorded else [message]
    try:
        error = cls(*args)
    except Exception:  # noqa: BLE001 - a constructor wanting more than args
        try:
            error = cls.__new__(cls, *args)
            BaseException.__init__(error, *args)
        except Exception:  # noqa: BLE001 - falls back to the message
            return fallback
    for key, value in recorded.get('attributes', {}).items():
        try:
            setattr(error, key, decode(value))
        except Exception:  # noqa: BLE001, S112 - a read-only attribute
            continue
    return error


def _excerpt(text: str, at: int, width: int = 30) -> str:
    start = max(0, at - width)
    return json.dumps(text[start : at + width])


def _first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


# Fetches.


@dataclass(frozen=True)
class Fetch:
    r"""One way of fetching a result: as rows, or one Arrow fetch call.

    ``method`` is ``None`` for rows (``fetchall``, ``fetchone`` and
    ``fetchmany`` all read the same recorded rows), else the driver's name for
    the Arrow fetch, with the arguments it was called with.
    """

    method: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def recorded(self) -> dict[str, Any]:
        r"""How the transcript names this fetch."""
        if self.method is None:
            return {'fetch': FETCH_ROWS}
        return {
            'fetch': FETCH_ARROW,
            'call': {
                'method': self.method,
                'args': encode(list(self.args)),
                'kwargs': encode(self.kwargs),
            },
        }


ROWS = Fetch()


class Statement:
    r"""One executed step's result, fetched the way the run asks for it."""

    def __init__(self, tape: Tape, index: int, real: Any | None = None) -> None:
        self._tape = tape
        self._index = index
        self._real = real
        self._loaded = False
        self._table: pa.Table | None = None
        self._rows: list[tuple[Any, ...]] | None = None
        self._position = 0

    def _load(self, fetch: Fetch) -> None:
        if self._loaded:
            return
        self._table = self._tape.result(self._index, self._real, fetch)
        self._loaded = True

    def rows(self) -> list[tuple[Any, ...]]:
        self._load(ROWS)
        if self._rows is None:
            self._rows = (
                [] if self._table is None else table_to_rows(self._table)
            )
        return self._rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self.rows()[self._position :]
        self._position += len(rows)
        return rows

    def fetchone(self) -> tuple[Any, ...] | None:
        rows = self.rows()
        if self._position >= len(rows):
            return None
        self._position += 1
        return rows[self._position - 1]

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        rows = self.rows()[self._position : self._position + size]
        self._position += len(rows)
        return rows

    def arrow(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> pa.Table | None:
        self._load(Fetch(method, args, kwargs))
        return self._table

    def description(self) -> Any:
        return self._tape.description(self._index, self._real)


class Tape:
    r"""The steps of one run."""

    recording: bool

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.divergence: Divergence | None = None

    def execute(
        self, real: Any, sql: Any, params: Any, keyword: str | None
    ) -> Statement:
        raise NotImplementedError

    def result(self, index: int, real: Any, fetch: Fetch) -> pa.Table | None:
        r"""The result of step ``index``, fetched as ``fetch`` asks; ``None``
        for an Arrow fetch the driver answered with nothing."""
        raise NotImplementedError

    def description(self, index: int, real: Any) -> Any:
        raise NotImplementedError

    def ingest(self, real: Any, table: str, data: Any, mode: str) -> Any:
        raise NotImplementedError

    def register(self, real: Any, name: str, data: Any) -> None:
        raise NotImplementedError

    def commit(self, real: Any) -> None:
        raise NotImplementedError

    def paramstyle(self, real: Any, style: Any) -> None:
        raise NotImplementedError

    def _diverge(self, index: int, message: str) -> Divergence:
        divergence = Divergence(index, message)
        if self.divergence is None:
            self.divergence = divergence
        return divergence


def _sql_text(sql: Any) -> str:
    if isinstance(sql, str):
        return sql
    raise Unsupported(
        f'SQL given as {type(sql).__name__} rather than text cannot be recorded'
    )


class RecordingTape(Tape):
    r"""Makes each call on the real connection, and writes it down."""

    recording = True

    def __init__(self) -> None:
        super().__init__()
        #: Results by step index, written out as Arrow files on save.
        self.results: dict[int, pa.Table] = {}

    def _append(self, step: dict[str, Any]) -> int:
        self.steps.append(step)
        return len(self.steps) - 1

    def execute(
        self, real: Any, sql: Any, params: Any, keyword: str | None
    ) -> Statement:
        text = _sql_text(sql)
        step: dict[str, Any] = {
            'op': 'execute',
            'sql': text,
            'params': encode(params),
            'result': None,
            'error': None,
        }
        index = self._append(step)
        try:
            # As the run called it: the driver may tell a keyword from a
            # position, or an absent argument from None.
            if keyword is not None:
                real.execute(text, **{keyword: params})
            elif params is None:
                real.execute(text)
            else:
                real.execute(text, params)
        except Exception as error:
            step['error'] = describe_error(error)
            raise
        step['description'] = describe_columns(real.description)
        return Statement(self, index, real)

    def result(self, index: int, real: Any, fetch: Fetch) -> pa.Table | None:
        step = self.steps[index]
        try:
            if fetch.method is None:
                fetched = real.fetchall()
            else:
                fetched = getattr(real, fetch.method)(
                    *fetch.args, **fetch.kwargs
                )
        except Exception as error:
            step.setdefault('fetch_errors', []).append(
                {**fetch.recorded(), 'error': describe_error(error)}
            )
            raise
        recorded = fetch.recorded()
        if fetch.method is None:
            names = [column['name'] for column in step['description'] or ()]
            table = rows_to_table([tuple(row) for row in fetched], names)
            step['fetch'] = FETCH_ROWS
        else:
            step['fetch_call'] = recorded['call']
            if fetched is None:
                step['fetch'] = FETCH_NONE
                return None
            table = as_arrow(fetched)
            step['fetch'] = FETCH_ARROW
        self.results[index] = table
        return table

    def description(self, index: int, real: Any) -> Any:
        return real.description

    def ingest(self, real: Any, table: str, data: Any, mode: str) -> Any:
        step = {
            'op': 'ingest',
            'table': table,
            'mode': mode,
            'data': encode_table(data),
            'error': None,
        }
        self._append(step)
        try:
            return real.adbc_ingest(table, data, mode=mode)
        except Exception as error:
            step['error'] = describe_error(error)
            raise

    def register(self, real: Any, name: str, data: Any) -> None:
        step = {
            'op': 'register',
            'name': name,
            'data': encode_table(data),
            'error': None,
        }
        self._append(step)
        try:
            real.register(name, data)
        except Exception as error:
            step['error'] = describe_error(error)
            raise

    def commit(self, real: Any) -> None:
        self._append({'op': 'commit'})
        real.commit()

    def paramstyle(self, real: Any, style: Any) -> None:
        self._append({'op': 'paramstyle', 'style': style})
        real._paramstyle = style


class ReplayingTape(Tape):
    r"""Checks each call against the recording, and answers from it."""

    recording = False

    def __init__(self, steps: list[dict[str, Any]], root: Path) -> None:
        super().__init__()
        self.steps = steps
        self.root = root
        self.position = 0
        #: Per step, how many of its recorded failed fetches the run made.
        self._failed_fetches: dict[int, int] = {}

    def _next(self, op: str, what: str) -> tuple[int, dict[str, Any]]:
        index = self.position
        if index >= len(self.steps):
            raise self._diverge(
                index,
                f'the run {what}, and the recording ends after '
                f'{len(self.steps)} steps',
            )
        step = self.steps[index]
        if step.get('op') != op:
            raise self._diverge(
                index,
                f'the run {what}, and the recording has '
                f'{step.get("op")!r} here',
            )
        self.position += 1
        return index, step

    def _compare(self, index: int, field: str, recorded: Any, ran: Any) -> None:
        if canonical(recorded) != canonical(ran):
            raise self._diverge(
                index,
                f'the {field} differ: recorded {canonical(recorded)[:200]}, '
                f'ran {canonical(ran)[:200]}',
            )

    def _raise_recorded(self, step: dict[str, Any]) -> None:
        if step.get('error'):
            raise rebuild_error(step['error'])

    def execute(
        self, real: Any, sql: Any, params: Any, keyword: str | None
    ) -> Statement:
        text = _sql_text(sql)
        index, step = self._next('execute', f'executes {json.dumps(text[:60])}')
        recorded = step.get('sql')
        if recorded != text:
            at = _first_difference(recorded or '', text)
            raise self._diverge(
                index,
                f'the SQL differs at character {at}: recorded '
                f'{_excerpt(recorded or "", at)}, ran {_excerpt(text, at)}',
            )
        self._compare(index, 'parameters', step.get('params'), encode(params))
        self._raise_recorded(step)
        return Statement(self, index)

    def result(self, index: int, real: Any, fetch: Fetch) -> pa.Table | None:
        step = self.steps[index]
        ran = fetch.recorded()
        failed = step.get('fetch_errors') or []
        used = self._failed_fetches.get(index, 0)
        if used < len(failed):
            self._failed_fetches[index] = used + 1
            attempt = dict(failed[used])
            error = attempt.pop('error')
            self._compare(index, 'fetches', attempt, ran)
            raise rebuild_error(error)
        fetched = step.get('fetch')
        if fetched is None:
            raise self._diverge(
                index, 'the run fetches a result the recording does not hold'
            )
        if fetched == FETCH_ROWS:
            recorded = {'fetch': FETCH_ROWS}
        else:
            recorded = {'fetch': FETCH_ARROW, 'call': step.get('fetch_call')}
        self._compare(index, 'fetches', recorded, ran)
        if fetched == FETCH_NONE:
            return None
        ref = step.get('result')
        if not isinstance(ref, str):
            raise self._diverge(
                index, 'the run fetches a result the recording does not hold'
            )
        from _parity.transcript import read_arrow

        return read_arrow(self.root / ref)

    def description(self, index: int, real: Any) -> Any:
        return served_description(self.steps[index].get('description'))

    def ingest(self, real: Any, table: str, data: Any, mode: str) -> Any:
        index, step = self._next('ingest', f'ingests into {table!r}')
        self._compare(index, 'ingested table names', step.get('table'), table)
        self._compare(index, 'ingest modes', step.get('mode'), mode)
        self._compare(
            index, 'ingested rows', step.get('data'), encode_table(data)
        )
        self._raise_recorded(step)
        return as_arrow(data).num_rows

    def register(self, real: Any, name: str, data: Any) -> None:
        index, step = self._next('register', f'registers {name!r}')
        self._compare(index, 'registered names', step.get('name'), name)
        self._compare(
            index, 'registered rows', step.get('data'), encode_table(data)
        )
        self._raise_recorded(step)

    def commit(self, real: Any) -> None:
        self._next('commit', 'commits')

    def paramstyle(self, real: Any, style: Any) -> None:
        index, step = self._next('paramstyle', f'sets paramstyle {style!r}')
        self._compare(index, 'paramstyles', step.get('style'), style)

    def leftover(self) -> Divergence | None:
        r"""Steps the recording holds that the run never reached."""
        if self.position < len(self.steps):
            step = self.steps[self.position]
            return self._diverge(
                self.position,
                f'the run ended, and the recording goes on with '
                f'{len(self.steps) - self.position} more steps, the next '
                f'{step.get("op")!r}',
            )
        return None


def _parameters(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, str | None]:
    r"""The parameters an ``execute`` call passed, and the keyword it named
    them with, if it did."""
    unknown = set(kwargs) - set(PARAMETER_KEYWORDS)
    if len(args) > 1 or unknown or len(kwargs) > 1 or (args and kwargs):
        raise Unsupported(
            f'execute() with more than the SQL and its parameters '
            f'({args!r}, {kwargs!r}) is not recorded'
        )
    if kwargs:
        ((keyword, params),) = kwargs.items()
        return params, keyword
    return (args[0] if args else None), None


class ParityCursor:
    r"""A cursor whose every call goes through the tape.

    On record it wraps the driver's own cursor. The fetch methods cover the
    drivers the engine samples through; any other call fails loudly rather
    than reaching a database unrecorded.
    """

    def __init__(
        self,
        tape: Tape,
        real: Any | None = None,
        arrow_methods: tuple[str, ...] = (),
    ) -> None:
        self._tape = tape
        self._real = real
        self._arrow_methods = arrow_methods
        self._statement: Statement | None = None

    # The statement

    def execute(self, sql: Any, *args: Any, **kwargs: Any) -> Any:
        params, keyword = _parameters(args, kwargs)
        self._statement = self._tape.execute(self._real, sql, params, keyword)
        return self

    def _current(self) -> Statement:
        if self._statement is None:
            raise Unsupported('a fetch before any execute')
        return self._statement

    # Rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._current().fetchall()

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._current().fetchone()

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        return self._current().fetchmany(size)

    @property
    def description(self) -> Any:
        if self._statement is None:
            return None
        return self._statement.description()

    # Writes

    def adbc_ingest(
        self, table_name: str, data: Any, mode: str = 'create', **kw: Any
    ) -> Any:
        if kw:
            raise Unsupported(f'adbc_ingest() options {kw!r} are not recorded')
        return self._tape.ingest(self._real, table_name, data, mode)

    def register(self, view_name: str, python_object: Any) -> Any:
        self._tape.register(self._real, view_name, python_object)
        return self

    # Lifetime

    def close(self) -> None:
        if self._real is not None:
            self._real.close()

    def __enter__(self) -> ParityCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        # The driver's own name for an Arrow fetch, and only that driver's,
        # so code choosing a fetch by what the cursor has chooses as it would.
        if name in self.__dict__.get('_arrow_methods', ()):

            def fetch(*args: Any, **kwargs: Any) -> pa.Table | None:
                return self._current().arrow(name, args, kwargs)

            return fetch
        raise Unsupported(f'cursor.{name} is not recorded by the parity tape')
