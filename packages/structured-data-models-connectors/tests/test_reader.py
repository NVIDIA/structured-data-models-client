# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
import warnings
from pathlib import PurePosixPath

import pandas as pd
import pyarrow as pa
import pytest

from sdm_connectors import read, read_table
from sdm_connectors.sql import ConnectorError

LARGE_ID = 9007199254740993


class _FakeCursor:
    def __init__(self, arrow_table, pandas_frame=None):
        self.arrow_table = arrow_table
        self.pandas_frame = pandas_frame
        self.pandas_calls = 0

    def execute(self, sql):
        pass

    def fetch_arrow_all(self):
        return self.arrow_table

    def fetch_pandas_all(self):
        self.pandas_calls += 1
        return self.pandas_frame

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _adbc_sqlite(database: str):
    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    return dbapi.connect(database)


def test_read_local_from_csv(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(path, index=False)
    assert read(str(path)).shape == (2, 2)


def test_read_local_from_parquet(tmp_path):
    path = tmp_path / 'table.parquet'
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    assert read(str(path)).shape == (3, 1)


def test_read_local_rejects_storage_options():
    """Valid for s3, not for local."""
    with pytest.raises(ConnectorError) as excinfo:
        read('table.csv', storage_options={})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


@pytest.mark.parametrize('name', ['table.json', 'table.tsv', 'table.orc'])
def test_read_local_rejects_unsupported_format(tmp_path, name):
    path = tmp_path / name
    pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']}).to_json(path)
    with pytest.raises(ConnectorError) as excinfo:
        read(str(path))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['suffix'] == PurePosixPath(name).suffix


def test_read_local_accepts_parquet_aliases(tmp_path):
    """'.pq' is Parquet."""
    path = tmp_path / 'table.pq'
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    assert read(str(path)).shape == (3, 1)


def test_read_local_accepts_compressed_csv(tmp_path):
    path = tmp_path / 'table.csv.gz'
    pd.DataFrame({'a': [1, 2]}).to_csv(path, index=False)
    assert read(str(path)).shape == (2, 1)


def test_read_local_format_override_reads_suffixless_file(tmp_path):
    """Escape hatch."""
    path = tmp_path / 'table'
    pd.DataFrame({'a': [1, 2]}).to_csv(path, index=False)
    assert read(str(path), format='csv').shape == (2, 1)
    with pytest.raises(ConnectorError) as excinfo:
        read(str(path), format='avro')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_parquet_directory_without_trailing_slash(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(dataset / 'part-0.parquet')
    assert read(str(dataset)).shape == (3, 1)


def test_the_sqlite_backend_refuses_a_missing_database(tmp_path):
    """The engine opens its own connection from a path, and a read must not
    create the database it was pointed at."""
    from sdm_connectors.backends import sqlite

    database = tmp_path / 'typo.sqlite'
    with pytest.raises(ConnectorError) as excinfo:
        sqlite.connect(str(database))
    assert excinfo.value.code == 'NOT_FOUND'
    assert not database.exists()


def test_the_sqlite_backend_refuses_database_and_uri_together(tmp_path):
    from sdm_connectors.backends import sqlite

    first, second = str(tmp_path / 'a.sqlite'), str(tmp_path / 'b.sqlite')
    for database in (first, second):
        sqlite3.connect(database).close()
    with pytest.raises(ConnectorError) as excinfo:
        sqlite.connect(first, database=second)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_table_runs_in_the_caller_duckdb_session():
    import duckdb

    connection = duckdb.connect()
    try:
        connection.register('mydf', pd.DataFrame({'a': [1, 2, 3]}))
        connection.execute('CREATE TEMP TABLE tmp_t AS SELECT 1 AS a')

        assert read_table(connection, table='mydf').shape == (3, 1)
        assert read_table(connection, query='SELECT * FROM tmp_t').shape == (
            1,
            1,
        )
        assert connection.execute('SELECT 1').fetchall() == [(1,)]
    finally:
        connection.close()


def test_read_table_over_sqlite_by_table_and_query(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1, 2, 3]}).to_sql('items', connection, index=False)
    connection = _adbc_sqlite(database)
    try:
        assert len(read_table(connection, table='items')) == 3
        assert (
            len(read_table(connection, query='SELECT * FROM items WHERE a > 1'))
            == 2
        )
    finally:
        connection.close()


def test_read_table_over_duckdb_by_table(tmp_path):
    import duckdb

    database = str(tmp_path / 'db.duckdb')
    connection = duckdb.connect(database)
    connection.execute('CREATE TABLE items AS SELECT * FROM range(5) AS t(a)')
    connection.close()
    connection = duckdb.connect(database, read_only=True)
    try:
        assert len(read_table(connection, table='items')) == 5
    finally:
        connection.close()


def test_read_table_over_duckdb_does_not_warn():
    """DuckDB 2.0 warns on Arrow from a DB-API result, so the read must not
    take one."""
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute(
            'CREATE TABLE items AS SELECT * FROM range(5) AS t(a)'
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert len(read_table(connection, table='items')) == 5
    finally:
        connection.close()


def test_an_integer_column_has_one_dtype_however_it_is_fetched():
    """Plain rows (sqlite3) and Arrow (DuckDB) give the same dtypes: `int64`
    without a null, `Int64` with one."""
    duckdb = pytest.importorskip('duckdb')
    by_rows = sqlite3.connect(':memory:')
    by_rows.execute('CREATE TABLE t (a INTEGER, b INTEGER)')
    by_rows.execute('INSERT INTO t VALUES (1, 1), (2, NULL)')
    by_arrow = duckdb.connect()
    by_arrow.execute('CREATE TABLE t (a BIGINT, b BIGINT)')
    by_arrow.execute('INSERT INTO t VALUES (1, 1), (2, NULL)')
    try:
        rows = read_table(by_rows, table='t')
        arrow = read_table(by_arrow, table='t')
    finally:
        by_rows.close()
        by_arrow.close()

    assert [str(dtype) for dtype in rows.dtypes] == ['int64', 'Int64']
    assert rows.dtypes.equals(arrow.dtypes)


def test_read_table_over_sqlite_keeps_nullable_integers_exact(tmp_path):
    database = str(tmp_path / 'ids.sqlite')
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE users (user_id BIGINT, age INTEGER)')
        connection.execute('INSERT INTO users VALUES (?, 20)', (LARGE_ID,))
        connection.execute('INSERT INTO users VALUES (NULL, 30)')

    connection = _adbc_sqlite(database)
    try:
        frame = read_table(connection, table='users')
    finally:
        connection.close()

    assert frame['user_id'].dtype == 'Int64'
    assert frame['user_id'][0] == LARGE_ID
    assert frame['user_id'].isna().tolist() == [False, True]
    assert frame['age'].dtype == 'int64'
    assert frame['age'].tolist() == [20, 30]


def test_read_table_prefers_arrow_over_pandas_fetch():
    cursor = _FakeCursor(
        pa.table({'user_id': pa.array([LARGE_ID, None], type=pa.int64())})
    )

    frame = read_table(_FakeConnection(cursor), table='users')

    assert cursor.pandas_calls == 0
    assert frame['user_id'].dtype == 'Int64'
    assert frame['user_id'][0] == LARGE_ID


def test_read_table_falls_back_to_pandas_fetch_without_arrow_result():
    cursor = _FakeCursor(None, pd.DataFrame({'user_id': []}))

    frame = read_table(_FakeConnection(cursor), table='users')

    assert cursor.pandas_calls == 1
    assert frame.empty


def test_read_local_accepts_a_path_object(tmp_path):
    """``_require_local_path`` ran ``urlparse`` on the raw argument, so a
    ``pathlib.Path``, which ``pandas`` reads, raised ``AttributeError``
    from inside ``urllib``, escaping the connector's error contract entirely.
    """
    path = tmp_path / 'rows.csv'
    pd.DataFrame({'a': [1, 2]}).to_csv(path, index=False)

    frame = read(path)
    assert frame.shape == (2, 1)
    assert read(str(path)).equals(frame)


def test_read_local_rejects_a_non_path_argument():
    """Coercion must not turn the contract break into a different one: an
    argument that is neither ``str`` nor ``os.PathLike`` still raises the
    client's own error with a stable code.
    """
    with pytest.raises(ConnectorError) as excinfo:
        read(object())
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_still_rejects_a_remote_uri_given_as_a_path(tmp_path):
    """The coercion runs before the scheme guard, so the guard must still see a
    URI as a URI.
    """
    with pytest.raises(ConnectorError) as excinfo:
        read('https://example.com/rows.csv')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_the_duckdb_backend_refuses_a_missing_database(tmp_path):
    """The "a read must not write" guard landed on sqlite only; duckdb
    still created an empty database for a mistyped path and then blamed the
    table.
    """
    pytest.importorskip('duckdb')
    from sdm_connectors.backends import duckdb

    database = tmp_path / 'typo.duckdb'
    with pytest.raises(ConnectorError) as excinfo:
        duckdb.connect(str(database))
    assert excinfo.value.code == 'NOT_FOUND'
    assert not database.exists()


def test_the_duckdb_backend_opens_an_in_memory_database():
    pytest.importorskip('duckdb')
    from sdm_connectors.backends import duckdb

    connection = duckdb.connect(database=':memory:')
    try:
        assert read_table(connection, query='SELECT 1 AS a').shape == (1, 1)
    finally:
        connection.close()
