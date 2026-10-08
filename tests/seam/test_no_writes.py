# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.nowrite: a push-down prediction leaves the user's database as it was.

The SQLite and DuckDB samplers used to hand key lists to the database as
tables: ``adbc_ingest`` into the user's SQLite file, ``register`` on the
user's DuckDB connection. The SQLite ones were never dropped, so a
connection in autocommit mode, or one its owner later committed, kept a
``tmp_*`` table per lookup in the file. Keys now travel as a JSON parameter,
and the database is only read.

Each check builds a database file holding a small shop, opens it the way a
user would, predicts through the public API, and then requires:

* the file, and every file beside it, byte-identical to before;
* the same tables, views and indexes, temporary ones included, as seen
  through the user's own connection, with no ``tmp_*`` among them.

A read-only database must be enough to predict from.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from _seam import duckdb_database, predict, sqlite_database


def _files(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def _sqlite_objects(connection: Any) -> list[tuple[Any, ...]]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT type, name, tbl_name FROM sqlite_master '
            'UNION ALL SELECT type, name, tbl_name FROM sqlite_temp_master '
            'ORDER BY 1, 2, 3'
        )
        return [tuple(row) for row in cursor.fetchall()]


def _duckdb_objects(connection: Any) -> list[tuple[Any, ...]]:
    rows = connection.execute(
        "SELECT 'table', database_name, schema_name, table_name, temporary "
        'FROM duckdb_tables() '
        "UNION ALL SELECT 'view', database_name, schema_name, view_name, "
        'temporary FROM duckdb_views() WHERE NOT internal '
        "UNION ALL SELECT 'index', database_name, schema_name, index_name, "
        'NULL FROM duckdb_indexes() '
        'ORDER BY ALL'
    ).fetchall()
    return [tuple(row) for row in rows]


def _no_tmp(objects: list[tuple[Any, ...]]) -> None:
    leaked = [o for o in objects if any(str(v).startswith('tmp_') for v in o)]
    assert not leaked, f'tmp_* objects left in the database: {leaked}'


@pytest.fixture
def sqlite_driver() -> None:
    pytest.importorskip('sdm_relational_engine.relationallib')
    pytest.importorskip('adbc_driver_sqlite')


@pytest.fixture
def duckdb_driver() -> None:
    pytest.importorskip('sdm_relational_engine.relationallib')
    pytest.importorskip('duckdb')


@pytest.mark.parametrize(
    'mode', ['autocommit', 'committed-afterwards', 'closed-uncommitted']
)
def test_sqlite_prediction_writes_nothing(
    sqlite_driver: None, tmp_path: Path, mode: str
) -> None:
    from sdm_connectors import SQLiteSource

    path = tmp_path / 'shop.sqlite'
    connection = sqlite_database(path, autocommit=mode == 'autocommit')
    before_files = _files(tmp_path)
    before_objects = _sqlite_objects(connection)

    predict(SQLiteSource(connection))

    after_objects = _sqlite_objects(connection)
    _no_tmp(after_objects)
    assert after_objects == before_objects, (
        'the prediction changed what the database holds'
    )
    if mode == 'committed-afterwards':
        # The owner commits their own work later, which would keep anything
        # the prediction wrote:
        connection.commit()
    connection.close()
    assert _files(tmp_path) == before_files, (
        'the SQLite file, or a file beside it, changed'
    )


def test_sqlite_prediction_reads_a_read_only_database(
    sqlite_driver: None, tmp_path: Path
) -> None:
    from adbc_driver_sqlite import dbapi
    from sdm_connectors import SQLiteSource

    path = tmp_path / 'shop.sqlite'
    sqlite_database(path).close()
    before_files = _files(tmp_path)
    connection = dbapi.connect(f'file:{path}?mode=ro')
    try:
        predict(SQLiteSource(connection))
        _no_tmp(_sqlite_objects(connection))
    finally:
        connection.close()
    assert _files(tmp_path) == before_files


@pytest.mark.parametrize('read_only', [False, True])
def test_duckdb_prediction_writes_nothing(
    duckdb_driver: None, tmp_path: Path, read_only: bool
) -> None:
    from sdm_connectors import DuckDBSource

    path = tmp_path / 'shop.duckdb'
    connection = duckdb_database(path, read_only=read_only)
    before_files = _files(tmp_path)
    before_objects = _duckdb_objects(connection)

    predict(DuckDBSource(connection))

    after_objects = _duckdb_objects(connection)
    _no_tmp(after_objects)
    assert after_objects == before_objects, (
        'the prediction changed what the database holds'
    )
    connection.close()
    assert _files(tmp_path) == before_files, (
        'the DuckDB file, or a file beside it, changed'
    )
