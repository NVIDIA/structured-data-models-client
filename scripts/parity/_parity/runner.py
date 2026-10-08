# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Running one scenario: on a real database to record it, or on its
recording to replay it.

Both run the same calls through the same public API. The only differences
are where the database steps go (:mod:`tape`) and where the data comes from:
a recording loads the generated data into the database first, outside the
tape, and a replay needs no database at all.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from _parity import data as datagen
from _parity.drivers import engine_hooks, facade, facts_of
from _parity.nim import ReplayingSession, SyntheticSession, normalized
from _parity.tape import RecordingTape, Tape
from _parity.transcript import frame_to_table
from _parity.values import decode, encode

#: Where the PostgreSQL scenarios find a server to record against.
POSTGRES_DSN = 'SDM_POSTGRES_TEST_DSN'


#: Backends recorded against a warehouse, with credentials.
LIVE = ('databricks', 'lakebase', 'snowflake')
#: Backends whose data is in memory: no database, so no steps.
IN_MEMORY = ('memory', 'tabular')


class CannotRecord(Exception):
    r"""A scenario cannot be recorded here: a missing credential or driver."""


@dataclass
class Outcome:
    r"""What one run produced.

    ``error`` is what the run's last call raised, described (see
    :func:`describe_error`), and ``answer`` what the decoded result holds
    that its frame does not show (see :func:`beyond_the_frame`).
    """

    tape: Tape
    payloads: list[dict[str, Any]] = field(default_factory=list)
    responses: list[dict[str, Any]] = field(default_factory=list)
    result: pa.Table | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    answer: dict[str, Any] = field(default_factory=dict)


def _qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def describe_error(error: BaseException) -> dict[str, Any]:
    r"""What a run that ended in ``error`` records of it: its class, its
    ``code`` where it has one, its message, and its ``details``, encoded."""
    code = getattr(error, 'code', None)
    message = getattr(error, 'message', None)
    details = getattr(error, 'details', None)
    described: dict[str, Any] = {
        'class': _qualname(type(error)),
        'code': code if isinstance(code, str) and code else None,
        'message': message if isinstance(message, str) else str(error),
    }
    if isinstance(details, dict) and details:
        described['details'] = encode(details)
    status = getattr(error, 'status_code', None)
    if isinstance(status, int):
        described['status_code'] = status
    return described


def beyond_the_frame(result: Any) -> dict[str, Any]:
    r"""What a result holds that ``to_pandas()`` does not show: the
    candidates a ranking left unscored, and the explanation. Each only when
    there is one."""
    out: dict[str, Any] = {}
    unscored = list(getattr(result, 'unscored_candidates', ()) or ())
    if unscored:
        out['unscored_candidates'] = encode(unscored)
    explanation = getattr(result, 'explanation', None)
    if explanation is not None:
        out['explanation'] = encode(_text_keys(explanation))
    return out


def _text_keys(value: Any) -> Any:
    r"""``value`` with every mapping key written as text, as JSON keys are:
    a forecast's explanation is keyed by its integer steps."""
    if isinstance(value, dict):
        return {str(key): _text_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_text_keys(item) for item in value]
    return value


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    return normalized(payload)


# Loading the generated data, before anything is recorded.


def _sqlite(frames: datagen.Frames, workdir: Path, spec: dict[str, Any]) -> Any:
    import adbc_driver_sqlite.dbapi as dbapi

    connection = dbapi.connect(str(workdir / 'shop.sqlite'))
    with connection.cursor() as cursor:
        for name, frame in frames.items():
            cursor.adbc_ingest(name, frame, mode='create')
            typed = _column_types(spec, name)
            if typed:
                _sqlite_declare(cursor, name, typed)
    connection.commit()
    return connection


def _sqlite_declare(cursor: Any, table: str, typed: dict[str, str]) -> None:
    r"""Make ``table`` again with the columns ``typed`` names declared as
    it says, an empty type declaring none, and every other column as
    ``adbc_ingest`` declared it; the rows keep their order."""
    cursor.execute(f'PRAGMA table_info({table})')
    declared = {str(row[1]): str(row[2]) for row in cursor.fetchall()}
    unknown = sorted(set(typed) - set(declared))
    if unknown:
        raise ValueError(f'column_types names no column {unknown} of {table}')
    columns = ', '.join(
        f'{column} {typed.get(column, kind)}'.rstrip()
        for column, kind in declared.items()
    )
    cursor.execute(f'ALTER TABLE {table} RENAME TO {table}_ingested')
    cursor.execute(f'CREATE TABLE {table} ({columns})')
    cursor.execute(
        f'INSERT INTO {table} SELECT * FROM {table}_ingested ORDER BY rowid'
    )
    cursor.execute(f'DROP TABLE {table}_ingested')


def _column_types(spec: dict[str, Any], table: str) -> dict[str, str]:
    r"""The types ``data.column_types`` gives columns of ``table``."""
    prefix = f'{table}.'
    return {
        key[len(prefix) :]: value
        for key, value in (spec.get('column_types') or {}).items()
        if key.startswith(prefix)
    }


def _duckdb(frames: datagen.Frames, spec: dict[str, Any]) -> Any:
    import duckdb

    # The session's settings, such as its `TimeZone`, are the connection's
    # own, which every cursor the engine opens on it inherits.
    connection = duckdb.connect(config=dict(spec.get('session') or {}))
    # One thread, so a query without ORDER BY returns its rows in the same
    # order every time and a re-recording does not churn the stored results.
    # The engine's DuckDB key lookups order nothing and reorder by batch
    # afterwards, so only the recorded files would differ, not the payloads.
    connection.execute('SET threads TO 1')
    for name, frame in frames.items():
        typed = _column_types(spec, name)
        columns = ', '.join(
            _duckdb_column(column, typed.get(column)) for column in frame
        )
        connection.register(f'{name}_frame', frame)
        connection.execute(
            f'CREATE TABLE {name} AS SELECT {columns} FROM {name}_frame'
        )
        connection.unregister(f'{name}_frame')
    return connection


def _duckdb_column(column: str, column_type: str | None) -> str:
    r"""A generated column as stored: as generated, or as ``column_type``.

    A naive time is UTC, so a ``TIMESTAMPTZ`` column holds the instant it
    names, whatever the session's zone.
    """
    if column_type is None or column_type.upper() == 'TIMESTAMP':
        return column
    if column_type.upper() in ('TIMESTAMPTZ', 'TIMESTAMP WITH TIME ZONE'):
        return f"timezone('UTC', {column}) AS {column}"
    return f'CAST({column} AS {column_type}) AS {column}'


_PG_TYPES = {
    'int64': 'BIGINT',
    'float64': 'DOUBLE PRECISION',
    'datetime64[us]': 'TIMESTAMP',
    'datetime64[ns]': 'TIMESTAMP',
}


def _pg_type(dtype: Any) -> str:
    return _PG_TYPES.get(str(dtype), 'TEXT')


def _python_rows(frame: Any) -> list[tuple[Any, ...]]:
    import pandas as pd

    return [
        tuple(
            v.to_pydatetime() if isinstance(v, pd.Timestamp) else v
            for v in (
                value.item() if hasattr(value, 'item') else value
                for value in row
            )
        )
        for row in frame.itertuples(index=False, name=None)
    ]


def load_postgres(
    connection: Any,
    schema: str,
    frames: datagen.Frames,
    column_types: dict[str, dict[str, str]] | None = None,
) -> None:
    r"""The frames as tables of ``schema``, which is made afresh, with the
    types ``column_types`` gives a table's columns by name.

    The session is in UTC while it loads, so a naive time, which is UTC,
    lands in a ``timestamptz`` column as the instant it names; its zone is
    put back afterwards.
    """
    (zone,) = connection.execute(
        "SELECT current_setting('TimeZone')"
    ).fetchone()
    connection.execute("SELECT set_config('TimeZone', 'UTC', false)")
    try:
        _load_postgres(connection, schema, frames, column_types)
    finally:
        connection.execute("SELECT set_config('TimeZone', %s, false)", (zone,))


def _load_postgres(
    connection: Any,
    schema: str,
    frames: datagen.Frames,
    column_types: dict[str, dict[str, str]] | None,
) -> None:
    connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
    connection.execute(f'CREATE SCHEMA {schema}')
    for name, frame in frames.items():
        typed = (column_types or {}).get(name, {})
        columns = ', '.join(
            f'{column} {typed.get(column, _pg_type(dtype))}'
            for column, dtype in frame.dtypes.items()
        )
        connection.execute(f'CREATE TABLE {schema}.{name} ({columns})')
        markers = ', '.join(['%s'] * len(frame.columns))
        with connection.cursor() as cursor:
            cursor.executemany(
                f'INSERT INTO {schema}.{name} VALUES ({markers})',
                _python_rows(frame),
            )


@contextlib.contextmanager
def _postgres(frames: datagen.Frames, spec: dict[str, Any]) -> Iterator[Any]:
    schema = spec['namespace']['schema']
    dsn = os.environ.get(POSTGRES_DSN)
    if not dsn:
        raise CannotRecord(
            f'recording a PostgreSQL scenario needs a server: set {POSTGRES_DSN}'
        )
    import psycopg

    with contextlib.ExitStack() as stack:
        try:
            # The schema's name is in the recorded SQL, so two recordings
            # into one database would drop each other's tables. A lock held
            # on its own connection, outside the tape, makes the second wait;
            # closing that connection releases it.
            lock = stack.enter_context(psycopg.connect(dsn, autocommit=True))
            lock.execute(
                'SELECT pg_advisory_lock(hashtext(%s))',
                (f'sdm-parity:{schema}',),
            )
            connection = psycopg.connect(dsn, autocommit=True)
        except psycopg.Error as error:
            raise CannotRecord(
                f'cannot reach {POSTGRES_DSN}: {error}'
            ) from error
        try:
            load_postgres(
                connection,
                schema,
                frames,
                {name: _column_types(spec, name) for name in frames},
            )
            # The session's settings, such as its `TimeZone`, once the data
            # is in.
            for setting, value in (spec.get('session') or {}).items():
                connection.execute(
                    'SELECT set_config(%s, %s, false)', (setting, value)
                )
            yield connection
        finally:
            with contextlib.suppress(Exception):
                connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
            connection.close()


@contextlib.contextmanager
def real_connection(transcript: dict[str, Any]) -> Iterator[Any]:
    r"""A real connection holding the scenario's data, for recording."""
    backend = transcript['backend']
    spec = transcript['data']
    frames = datagen.generate(spec['generator'], spec['seed'])
    if backend == 'sqlite':
        with tempfile.TemporaryDirectory(prefix='sdm-parity-') as workdir:
            connection = _sqlite(frames, Path(workdir), spec)
            try:
                yield connection
            finally:
                connection.close()
    elif backend == 'duckdb':
        connection = _duckdb(frames, spec)
        try:
            yield connection
        finally:
            connection.close()
    elif backend == 'postgres':
        with _postgres(frames, spec) as connection:
            yield connection
    else:
        from _parity import live

        with live.connection(transcript, frames) as connection:
            yield connection


def _source(transcript: dict[str, Any], connection: Any, real: Any) -> Any:
    from sdm_connectors import (
        DatabricksSource,
        DuckDBSource,
        PostgresSource,
        SnowflakeSource,
        SQLiteSource,
    )

    backend = transcript['backend']
    namespace = transcript['data'].get('namespace') or {}
    if backend == 'sqlite':
        return SQLiteSource(connection)
    if backend == 'duckdb':
        # A source is told its backend by the connection's class, and the
        # DuckDB facade is not DuckDB's class: it is bound to a real (on
        # replay, empty) connection, then given the facade.
        import duckdb

        source = DuckDBSource(real if real is not None else duckdb.connect())
        source._connection = connection
        return source
    if backend in ('postgres', 'lakebase'):
        return PostgresSource(connection, schema=namespace.get('schema'))
    if backend == 'databricks':
        return DatabricksSource(connection, **namespace)
    if backend == 'snowflake':
        return SnowflakeSource(connection, **namespace)
    raise ValueError(f'no source for backend {backend!r}')


# The calls.


class _Interpreter:
    r"""Makes a transcript's calls, in order, through the public API."""

    def __init__(
        self,
        transcript: dict[str, Any],
        source: Any,
        session: SyntheticSession | ReplayingSession,
    ) -> None:
        spec = transcript['data']
        self.frames = datagen.generate(spec['generator'], spec['seed'])
        self.source = source
        self.schema: Any = None
        self.query: Any = None
        self.request: Any = None
        self.result: Any = None
        self.results: list[Any] = []
        self.fitted: Any = None
        self.frame: Any = None
        self.session = session
        self.client: Any = None
        #: The position of the call being made.
        self.at = -1

    def value(self, raw: Any) -> Any:
        import pandas as pd

        if isinstance(raw, dict) and len(raw) == 1:
            ((tag, body),) = raw.items()
            if tag == '$frame':
                frame = self.frames[body['table']]
                if 'rows' in body:
                    start, stop = body['rows']
                    frame = frame.iloc[start:stop]
                if 'head' in body:
                    frame = frame.head(body['head'])
                if body.get('drop'):
                    frame = frame.drop(columns=body['drop'])
                return frame
            if tag == '$frames':
                return {name: self.frames[name] for name in body}
            if tag == '$timestamp':
                return pd.Timestamp(body)
            if tag == '$relationship':
                from sdm_client import Relationship

                return Relationship(*body)
        if isinstance(raw, dict):
            return {key: self.value(item) for key, item in raw.items()}
        if isinstance(raw, list):
            return [self.value(item) for item in raw]
        return decode(raw)

    def run(self, calls: list[dict[str, Any]]) -> None:
        for self.at, call in enumerate(calls):
            args = {
                key: self.value(v) for key, v in call.get('args', {}).items()
            }
            getattr(self, '_' + call['call'].replace('.', '_'))(**args)

    def _client(self) -> Any:
        r"""The one client every model call of the run goes through, sending
        through the stand-in session; closed when the run ends."""
        if self.client is None:
            from sdm_client import SDMClient

            self.client = SDMClient(url='http://localhost:8000')
            self.client._transport._session.close()
            self.client._transport._session = self.session
        return self.client

    def close(self) -> None:
        r"""Close the client, releasing what it pinned."""
        if self.client is not None:
            client, self.client = self.client, None
            client.close()

    def _Schema_from_source(self, **args: Any) -> None:
        from sdm_client import Schema

        model = args.pop('semantic_model', None)
        source = self.source
        if model is not None:
            source = source.semantic_model(model)
        self.schema = Schema.from_source(source, **args)

    def _Schema_from_data(self, tables: list[str], **args: Any) -> None:
        from sdm_client import Schema

        self.schema = Schema.from_data(
            {name: self.frames[name] for name in tables}, **args
        )

    def _PredictiveQuery(self, query: str, schema: Any = None) -> None:
        from sdm_client import PredictiveQuery

        self.query = PredictiveQuery(
            query, self.schema if schema is None else schema
        )

    def _PredictiveQuery_materialize(self, **args: Any) -> None:
        self.request = self.query.materialize(**args)

    def _Request(self, **args: Any) -> None:
        from sdm_client import Request

        self.request = Request(**args)

    def _SDMClient_predict(self, model: str, outputs: Any = None) -> None:
        self.result = self._client().predict(
            self.request, model=model, outputs=outputs
        )
        self.results.append(self.result)

    def _SDMClient_fit(
        self, model: str, outputs: Any = None, holdout: Any = None
    ) -> None:
        self.fitted = self._client().fit(
            self.request, model=model, outputs=outputs, holdout=holdout
        )

    def _FittedModel_predict(self) -> None:
        self.result = self.fitted.predict(self.request)
        self.results.append(self.result)

    def _FittedModel_close(self) -> None:
        self.fitted.close()

    def _PredictionResult_merge(self) -> None:
        from sdm_client import PredictionResult

        self.result = PredictionResult.merge(self.results)

    def _PredictionResult_to_pandas(self) -> None:
        self.frame = self.result.to_pandas()


def _warning(caught: warnings.WarningMessage) -> dict[str, str]:
    category = caught.category
    return {
        'category': f'{category.__module__}.{category.__qualname__}',
        'message': str(caught.message),
    }


def run(
    transcript: dict[str, Any],
    tape: Tape,
    real: Any = None,
    session: SyntheticSession | ReplayingSession | None = None,
) -> Outcome:
    r"""Make a transcript's calls, with database steps going to ``tape``.

    ``real`` is the real connection when recording. ``session`` answers the
    model calls: by default the stand-in, with the transcript's
    ``stand_in`` quirks. Warnings the client and the engine give the caller
    (``UserWarning`` and its subclasses) are kept, every one, in order.

    An error the last call raises is the run's outcome, recorded and
    compared like its result; one any earlier call raises fails the run.
    """
    backend = transcript['backend']
    if session is None:
        session = SyntheticSession(transcript.get('stand_in'))
    outcome = Outcome(tape=tape)
    with engine_hooks(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        source = None
        if backend not in IN_MEMORY:
            driver = backend
            facts = (
                facts_of(driver, real)
                if real is not None
                else transcript['data'].get('connection', {})
            )
            outcome.data = {'connection': facts} if facts else {}
            connection = facade(driver, tape, real, facts)
            source = _source(transcript, connection, real)
        interpreter = _Interpreter(transcript, source, session)
        calls = transcript['calls']
        try:
            try:
                interpreter.run(calls)
            except Exception as error:
                if interpreter.at != len(calls) - 1:
                    raise
                outcome.error = describe_error(error)
            interpreter.close()
        finally:
            outcome.payloads = [
                normalize(payload) for payload in session.payloads
            ]
            outcome.responses = list(session.responses)
        if interpreter.frame is not None:
            outcome.result = frame_to_table(interpreter.frame)
        if interpreter.result is not None and outcome.error is None:
            outcome.answer = beyond_the_frame(interpreter.result)
    outcome.warnings = [
        _warning(w) for w in caught if issubclass(w.category, UserWarning)
    ]
    return outcome


def record(transcript: dict[str, Any]) -> tuple[dict[str, Any], Outcome]:
    r"""Record a scenario; the transcript, and the tables to write beside it."""
    tape = RecordingTape()
    if transcript['backend'] in LIVE:
        from _parity import live

        transcript = live.prepare(transcript)
    if transcript['backend'] in IN_MEMORY:
        outcome = run(transcript, tape)
    else:
        with real_connection(transcript) as real:
            outcome = run(transcript, tape, real)
    recorded = dict(transcript)
    recorded['data'] = {**transcript['data'], **outcome.data}
    recorded['steps'] = tape.steps
    recorded['payloads'] = outcome.payloads
    recorded['responses'] = outcome.responses
    recorded['result'] = None
    recorded['warnings'] = outcome.warnings
    if outcome.error is not None:
        recorded['error'] = outcome.error
    if outcome.answer:
        recorded['answer'] = outcome.answer
    return recorded, outcome
