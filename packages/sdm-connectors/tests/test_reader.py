# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import sqlite3
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


def test_read_unknown_connector_raises():
    with pytest.raises(ConnectorError) as excinfo:
        read('mysql')
    assert excinfo.value.code == 'UNKNOWN_CONNECTOR'


def test_read_local_from_dataframe():
    frame = pd.DataFrame({'a': [1, 2]})
    assert read('local', data=frame) is frame


def test_read_local_from_dict():
    assert read('local', data={'a': [1, 2, 3]}).shape == (3, 1)


def test_read_local_from_csv(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(path, index=False)
    assert read('local', path=str(path)).shape == (2, 2)


def test_read_local_from_parquet(tmp_path):
    path = tmp_path / 'table.parquet'
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    assert read('local', path=str(path)).shape == (3, 1)


def test_read_local_without_data_or_path_raises():
    with pytest.raises(ConnectorError) as excinfo:
        read('local')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_rejects_both_data_and_path(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1]}).to_csv(path, index=False)
    with pytest.raises(ConnectorError) as excinfo:
        read('local', data={'a': [1]}, path=str(path))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_rejects_unknown_kwarg():
    with pytest.raises(ConnectorError) as excinfo:
        read('local', data={'a': [1]}, bogus=1)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['arguments'] == ['bogus']


def test_read_local_rejects_storage_options():
    """Valid for s3, not for local."""
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path='table.csv', storage_options={})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


@pytest.mark.parametrize('name', ['table.json', 'table.tsv', 'table.orc'])
def test_read_local_rejects_unsupported_format(tmp_path, name):
    path = tmp_path / name
    pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']}).to_json(path)
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path=str(path))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['suffix'] == PurePosixPath(name).suffix


def test_read_local_accepts_parquet_aliases(tmp_path):
    """'.pq' is Parquet."""
    path = tmp_path / 'table.pq'
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    assert read('local', path=str(path)).shape == (3, 1)


def test_read_local_accepts_compressed_csv(tmp_path):
    path = tmp_path / 'table.csv.gz'
    pd.DataFrame({'a': [1, 2]}).to_csv(path, index=False)
    assert read('local', path=str(path)).shape == (2, 1)


def test_read_local_format_override_reads_suffixless_file(tmp_path):
    """Escape hatch."""
    path = tmp_path / 'table'
    pd.DataFrame({'a': [1, 2]}).to_csv(path, index=False)
    assert read('local', path=str(path), format='csv').shape == (2, 1)
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path=str(path), format='avro')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_parquet_directory_without_trailing_slash(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(dataset / 'part-0.parquet')
    assert read('local', path=str(dataset)).shape == (3, 1)


def test_read_sqlite_missing_database_is_not_found(tmp_path):
    database = tmp_path / 'typo.sqlite'
    with pytest.raises(ConnectorError) as excinfo:
        read('sqlite', database=str(database), table='items')
    assert excinfo.value.code == 'NOT_FOUND'
    assert not database.exists()


def test_read_sqlite_rejects_database_and_uri_together(tmp_path):
    first, second = str(tmp_path / 'a.sqlite'), str(tmp_path / 'b.sqlite')
    for database in (first, second):
        with sqlite3.connect(database) as connection:
            pd.DataFrame({'a': [1]}).to_sql('items', connection, index=False)
    with pytest.raises(ConnectorError) as excinfo:
        read('sqlite', database=first, uri=second, table='items')
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


def test_read_sqlite_by_table_and_query(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1, 2, 3]}).to_sql('items', connection, index=False)
    assert len(read('sqlite', database=database, table='items')) == 3
    assert (
        len(
            read(
                'sqlite',
                database=database,
                query='SELECT * FROM items WHERE a > 1',
            )
        )
        == 2
    )


def test_read_duckdb_in_memory_query():
    frame = read('duckdb', query='SELECT 1 AS a, 2 AS b')
    assert frame.shape == (1, 2)
    assert int(frame['a'][0]) == 1


def test_read_duckdb_by_table(tmp_path):
    database = str(tmp_path / 'db.duckdb')
    import duckdb

    connection = duckdb.connect(database)
    connection.execute('CREATE TABLE items AS SELECT * FROM range(5) AS t(a)')
    connection.close()
    assert len(read('duckdb', database=database, table='items')) == 5


def test_read_sqlite_keeps_nullable_integers_exact(tmp_path):
    database = str(tmp_path / 'ids.sqlite')
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE users (user_id BIGINT, age INTEGER)')
        connection.execute('INSERT INTO users VALUES (?, 20)', (LARGE_ID,))
        connection.execute('INSERT INTO users VALUES (NULL, 30)')

    frame = read('sqlite', database=database, table='users')

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
    ``pathlib.Path``, which the sibling ``sqlite``/``duckdb`` connectors
    accept, and which ``pandas`` reads, raised ``AttributeError`` from
    inside ``urllib``, escaping the connector's error contract entirely.
    """
    path = tmp_path / 'rows.csv'
    pd.DataFrame({'a': [1, 2]}).to_csv(path, index=False)

    frame = read('local', path=path)
    assert frame.shape == (2, 1)
    assert read('local', path=str(path)).equals(frame)


def test_read_local_rejects_a_non_path_argument():
    """Coercion must not turn the contract break into a different one: an
    argument that is neither ``str`` nor ``os.PathLike`` still raises the
    client's own error with a stable code.
    """
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path=object())
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_still_rejects_a_remote_uri_given_as_a_path(tmp_path):
    """The coercion runs before the scheme guard, so the guard must still see a
    URI as a URI. See .
    """
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path='https://example.com/rows.csv')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_duckdb_missing_database_is_not_found(tmp_path):
    """The "a read must not write" guard landed on sqlite only; duckdb
    still created an empty database for a mistyped path and then blamed the
    table.
    """
    pytest.importorskip('duckdb')
    database = tmp_path / 'typo.duckdb'
    with pytest.raises(ConnectorError) as excinfo:
        read('duckdb', database=str(database), table='items')
    assert excinfo.value.code == 'NOT_FOUND'
    assert not database.exists()


def test_duckdb_in_memory_database_is_unaffected():
    pytest.importorskip('duckdb')
    frame = read('duckdb', database=':memory:', query='SELECT 1 AS a')
    assert frame.shape == (1, 1)
