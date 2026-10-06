# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A constructor that opens its own connection must not leak it.

``Graph.__del__`` closes the connection the graph owns, so the only window in
which one can escape is a failure *before* the graph exists to own it, an
unknown table name, an edge to a table that was not selected, a database with
nothing in it. A caller who passed their own connection keeps it: the
constructor closes only what it opened itself.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sdm_relational_engine.rfm import Graph

_FAILURES: list[tuple[str, dict[str, Any]]] = [
    ('unknown-table', dict(tables=['nope'])),
    ('edge-to-unselected-table', dict(tables=['t'], edges=[('t', 'a', 'u')])),
    ('empty-database', dict()),
]
_IDS = [case for case, _ in _FAILURES]


def _is_closed(connection: Any) -> bool:
    r"""Both drivers refuse to serve a statement once closed."""
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
    except Exception:
        return True
    return False


def _recording(module: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    r"""Capture what the backend's ``connect`` hands back, unchanged.

    The constructor asserts the object is a real ``Connection`` before it does
    anything with it, so this records rather than wraps.
    """
    opened: list[Any] = []
    real: Callable[..., Any] = module.connect

    def connect(*args: Any, **kwargs: Any) -> Any:
        opened.append(real(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(module, 'connect', connect)
    return opened


@pytest.fixture
def sqlite_backend() -> Any:
    pytest.importorskip(
        'adbc_driver_sqlite', reason="'sqlite' extension not installed"
    )
    from sdm_relational_engine.rfm.backend import sqlite as backend

    return backend


@pytest.fixture
def duckdb_backend() -> Any:
    pytest.importorskip('duckdb', reason="'duckdb' extension not installed")
    from sdm_relational_engine.rfm.backend import duckdb as backend

    return backend


@pytest.mark.parametrize(('case', 'kwargs'), _FAILURES, ids=_IDS)
def test_sqlite_closes_the_connection_it_opened(
    case: str,
    kwargs: dict[str, Any],
    sqlite_backend: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'x.db'
    setup = sqlite3.connect(path)
    if case != 'empty-database':
        setup.execute('CREATE TABLE t (a INTEGER)')
        setup.execute('INSERT INTO t VALUES (1)')
    setup.commit()
    setup.close()

    opened = _recording(sqlite_backend, monkeypatch)

    with pytest.raises(ValueError):
        Graph.from_sqlite(str(path), verbose=False, **kwargs)

    assert len(opened) == 1
    assert _is_closed(opened[0])


@pytest.mark.parametrize(('case', 'kwargs'), _FAILURES, ids=_IDS)
def test_duckdb_closes_the_connection_it_opened(
    case: str,
    kwargs: dict[str, Any],
    duckdb_backend: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duckdb

    path = tmp_path / 'x.duckdb'
    setup = duckdb.connect(str(path))
    if case != 'empty-database':
        setup.execute('CREATE TABLE t (a INTEGER)')
        setup.execute('INSERT INTO t VALUES (1)')
    setup.close()

    opened = _recording(duckdb_backend, monkeypatch)

    with pytest.raises(ValueError):
        Graph.from_duckdb(str(path), verbose=False, **kwargs)

    assert len(opened) == 1
    assert _is_closed(opened[0])


def test_a_caller_supplied_connection_survives_a_failure(
    sqlite_backend: Any, tmp_path: Path
) -> None:
    path = tmp_path / 'x.db'
    setup = sqlite3.connect(path)
    setup.execute('CREATE TABLE t (a INTEGER)')
    setup.commit()
    setup.close()

    connection = sqlite_backend.connect(str(path))
    try:
        with pytest.raises(ValueError):
            Graph.from_sqlite(connection, tables=['nope'], verbose=False)

        assert not _is_closed(connection)
    finally:
        connection.close()
