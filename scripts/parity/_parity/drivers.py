# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Connections the engine accepts, whose every call goes through a tape.

The engine type-checks the connection a source holds, so each facade is a
subclass of its driver's connection class, built without running the driver's
constructor: on record it wraps the real connection, on replay there is none.
DuckDB's connection class is a native one whose instances cannot be built
that way, so its facade is a plain class, and the engine's DuckDB backend is
told to accept it for the length of the run (:func:`engine_hooks`).

Nothing under ``packages/*/src`` is changed; the hooks are undone after every
run.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from typing import Any

from _parity.tape import ParityCursor, Tape, Unsupported

#: Each driver's names for fetching a result as Arrow.
ARROW_METHODS = {
    'sqlite': ('fetch_arrow_table',),
    'duckdb': ('fetch_arrow_table', 'to_arrow_table', 'arrow'),
    'postgres': (),
    'lakebase': (),
    'databricks': ('fetchall_arrow',),
    'snowflake': ('fetch_arrow_all',),
}


class _Facade:
    r"""What every facade does, whatever driver class it stands in for."""

    _parity_tape: Tape
    _parity_real: Any
    _parity_driver: str
    _parity_facts: dict[str, Any]

    def _parity_bind(
        self, tape: Tape, real: Any, driver: str, facts: dict[str, Any]
    ) -> None:
        # Set through object's own setattr: a driver class may route its
        # attribute writes through properties that need a live connection.
        object.__setattr__(self, '_parity_tape', tape)
        object.__setattr__(self, '_parity_real', real)
        object.__setattr__(self, '_parity_driver', driver)
        object.__setattr__(self, '_parity_facts', dict(facts))

    def cursor(self, *args: Any, **kwargs: Any) -> ParityCursor:
        if args or kwargs:
            raise Unsupported(f'cursor({args!r}, {kwargs!r}) is not recorded')
        tape = self._parity_tape
        real = self._parity_real.cursor() if tape.recording else None
        return ParityCursor(tape, real, ARROW_METHODS[self._parity_driver])

    def execute(self, sql: Any, params: Any = None) -> ParityCursor:
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor

    def commit(self) -> None:
        self._parity_tape.commit(self._parity_real)

    def close(self) -> None:
        r"""Nothing: the recorder owns the real connection and closes it."""

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __del__(self) -> None:
        return None

    def __repr__(self) -> str:
        mode = 'recording' if self._parity_tape.recording else 'replaying'
        return f'<parity {self._parity_driver} connection, {mode}>'


def _bare(cls: type) -> Any:
    r"""An instance of ``cls`` whose driver constructor never ran."""
    return object.__new__(cls)


@contextlib.contextmanager
def _setattr(owner: Any, name: str, value: Any) -> Iterator[None]:
    missing = object()
    previous = owner.__dict__.get(name, missing)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if previous is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, previous)


# SQLite through ADBC.


def _sqlite_class() -> type:
    from adbc_driver_sqlite.dbapi import AdbcSqliteConnection

    class ParitySQLiteConnection(_Facade, AdbcSqliteConnection):
        pass

    return ParitySQLiteConnection


# PostgreSQL and Lakebase through psycopg.


class _ReplayedAdapters:
    r"""A replayed cursor's adapters: what they load is in the recording."""

    def register_loader(self, oid: Any, loader: Any) -> None:
        return None


class ParityPostgresCursor(ParityCursor):
    r"""A psycopg cursor through the tape. The source reads some types as
    text by registering loaders on the cursor's own adapters: on record
    those of the real cursor, on replay none, as the rows are recorded."""

    @property
    def adapters(self) -> Any:
        if self._real is not None:
            return self._real.adapters
        return _ReplayedAdapters()


def _postgres_class() -> type:
    import psycopg
    from psycopg.rows import tuple_row

    class ParityPostgresConnection(_Facade, psycopg.Connection):
        def cursor(self, *args: Any, **kwargs: Any) -> ParityCursor:
            # The source reads every result as tuples, whatever the
            # connection's own row factory, which is how the tape holds rows.
            row_factory = kwargs.pop('row_factory', tuple_row)
            if args or kwargs or row_factory is not tuple_row:
                raise Unsupported(
                    f'cursor({args!r}, {kwargs!r}, row_factory='
                    f'{row_factory!r}) is not recorded'
                )
            tape = self._parity_tape
            real = (
                self._parity_real.cursor(row_factory=tuple_row)
                if tape.recording
                else None
            )
            return ParityPostgresCursor(
                tape, real, ARROW_METHODS[self._parity_driver]
            )

        @property
        def autocommit(self) -> bool:
            value = self._parity_facts['autocommit']
            if self._parity_tape.recording:
                assert self._parity_real.autocommit == value
            return value

        def transaction(self, *args: Any, **kwargs: Any) -> Any:
            raise Unsupported(
                'the parity connections record autocommit sessions, where '
                'the engine opens no transaction'
            )

    return ParityPostgresConnection


# DuckDB: a plain class, accepted by the engine through engine_hooks().


class ParityDuckDBConnection(_Facade):
    def register(self, view_name: str, python_object: Any) -> Any:
        self._parity_tape.register(self._parity_real, view_name, python_object)
        return self


# Databricks, through a SQL warehouse connection.


def _databricks_class() -> type:
    from databricks.sql.client import Connection

    class ParityDatabricksConnection(_Facade, Connection):
        pass

    return ParityDatabricksConnection


# Snowflake, whose parameter style the connector switches per statement.


def _snowflake_class() -> type:
    from snowflake.connector import SnowflakeConnection

    class ParitySnowflakeConnection(_Facade, SnowflakeConnection):
        @property
        def _paramstyle(self) -> Any:
            return self._parity_facts['paramstyle']

        @_paramstyle.setter
        def _paramstyle(self, style: Any) -> None:
            self._parity_tape.paramstyle(self._parity_real, style)
            self._parity_facts['paramstyle'] = style

    return ParitySnowflakeConnection


_CLASSES = {
    'sqlite': _sqlite_class,
    'postgres': _postgres_class,
    'lakebase': _postgres_class,
    'duckdb': lambda: ParityDuckDBConnection,
    'databricks': _databricks_class,
    'snowflake': _snowflake_class,
}


def facade(
    driver: str,
    tape: Tape,
    real: Any = None,
    facts: dict[str, Any] | None = None,
) -> Any:
    r"""A connection for ``driver`` that runs through ``tape``.

    ``facts`` are what the engine reads off a connection without a query,
    such as psycopg's ``autocommit``. They are stored in the transcript so a
    replay answers the same.
    """
    connection = _bare(_CLASSES[driver]())
    connection._parity_bind(tape, real, driver, facts or {})
    return connection


def facts_of(driver: str, real: Any) -> dict[str, Any]:
    r"""What the engine reads off ``real`` without a query."""
    if driver in ('postgres', 'lakebase'):
        return {'autocommit': bool(real.autocommit)}
    if driver == 'snowflake':
        return {'paramstyle': real._paramstyle}
    return {}


@contextlib.contextmanager
def engine_hooks() -> Iterator[None]:
    r"""What the engine needs to run through the facades, undone on exit.

    * The DuckDB backend checks a connection it is given, rather than a
      source, against its module's ``Connection``, the driver's native class;
      it also accepts :class:`ParityDuckDBConnection` here.

    Without the engine or DuckDB there is nothing to adjust: a run that needs
    neither, such as a tabular case, runs as it is.
    """
    try:
        import duckdb

        backend = importlib.import_module(
            'sdm_relational_engine.rfm.backend.duckdb'
        )
    except ImportError:
        yield
        return
    with _setattr(
        backend,
        'Connection',
        (duckdb.DuckDBPyConnection, ParityDuckDBConnection),
    ):
        yield
