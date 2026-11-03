# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent source descriptions must not share active native run state."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


def test_paused_duckdb_description_does_not_change_sqlite_description() -> None:
    statements = json.loads(
        (Path(__file__).parent / 'data/shop.json').read_text()
    )['statements']
    script = f"""
    import json
    import threading
    import adbc_driver_sqlite.dbapi
    import duckdb
    from sdm_client import Schema
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_connectors import DuckDBSource, SQLiteSource

    def sqlite_schema():
        with adbc_driver_sqlite.dbapi.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute('CREATE TABLE users (user_id INTEGER PRIMARY KEY, age INTEGER)')
                cursor.executemany('INSERT INTO users VALUES (?, ?)', [(i, 20 + i) for i in range(8)])
            schema = Schema.from_source(SQLiteSource(connection))
            return {{name: {{'columns': [(c, col.dtype, col.stype) for c, col in table.columns.items()],
                            'primary_key': table.primary_key, 'time_column': table.time_column}}
                    for name, table in schema.tables.items()}}

    paused, resume = threading.Event(), threading.Event()
    original = DuckDBProvider.__call__
    def pausing(self, step, keys=None):
        if not paused.is_set():
            paused.set()
            assert resume.wait(30), 'SQLite description did not finish'
        return original(self, step, keys)
    DuckDBProvider.__call__ = pausing
    results, failures = {{}}, []
    def describe_duckdb():
        try:
            with duckdb.connect() as connection:
                for statement in {statements!r}:
                    connection.execute(statement)
                results['tables'] = sorted(Schema.from_source(DuckDBSource(connection)).tables)
        except BaseException as error:
            failures.append(repr(error))
    thread = threading.Thread(target=describe_duckdb)
    thread.start()
    try:
        assert paused.wait(30), 'DuckDB description never reached provider'
        during = sqlite_schema()
    finally:
        resume.set()
        thread.join(30)
    assert not thread.is_alive()
    assert not failures, failures
    alone = sqlite_schema()
    assert during == alone
    assert alone == {{'users': {{'columns': [('user_id', 'int64', 'ID'), ('age', 'int64', 'numerical')],
                               'primary_key': 'user_id', 'time_column': None}}}}
    assert results['tables'] == ['orders', 'users']
    print('passed')
    """
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'passed'
