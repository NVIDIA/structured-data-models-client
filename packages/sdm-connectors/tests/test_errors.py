# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sqlite3

import pandas as pd
import pytest

from sdm_connectors import connect, read, read_table
from sdm_connectors.sql import (
    ConnectorError,
    MissingBackendError,
    driver_guard,
    require_driver,
)


@pytest.fixture()
def sqlite_db(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1, 2, 3]}).to_sql('items', connection, index=False)
    return database


def test_connect_unknown_backend_raises_connector_error():
    with pytest.raises(ConnectorError) as excinfo:
        connect('mysql')
    assert excinfo.value.code == 'UNKNOWN_CONNECTOR'


def test_connect_failure_maps_to_connect_failed(tmp_path):
    database = tmp_path / 'not-a-database.duckdb'
    database.write_text('this is not a duckdb database')
    with pytest.raises(ConnectorError) as excinfo:
        read('duckdb', database=str(database), table='items')
    assert excinfo.value.code == 'CONNECT_FAILED'
    assert excinfo.value.details['driver_error']
    assert excinfo.value.__cause__ is not None


def test_query_failure_maps_to_query_failed(sqlite_db):
    with pytest.raises(ConnectorError) as excinfo:
        read('sqlite', database=sqlite_db, query='SELECT * FROM missing')
    assert excinfo.value.code == 'QUERY_FAILED'
    assert excinfo.value.details['sql'] == 'SELECT * FROM missing'
    assert excinfo.value.__cause__ is not None


def test_read_table_direct_query_failure(sqlite_db):
    connection = connect('sqlite', sqlite_db)
    try:
        with pytest.raises(ConnectorError) as excinfo:
            read_table(connection, query='SELECT nope FROM items')
        assert excinfo.value.code == 'QUERY_FAILED'
    finally:
        connection.close()


def test_invalid_table_identifier_is_not_rewrapped(sqlite_db):
    with pytest.raises(ConnectorError) as excinfo:
        read('sqlite', database=sqlite_db, table='bad ident!')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_local_missing_file_maps_to_not_found(tmp_path):
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path=str(tmp_path / 'absent.csv'))
    assert excinfo.value.code == 'NOT_FOUND'
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_local_malformed_parquet_maps_to_read_failed(tmp_path):
    path = tmp_path / 'table.parquet'
    path.write_text('this is not parquet')
    with pytest.raises(ConnectorError) as excinfo:
        read('local', path=str(path))
    assert excinfo.value.code == 'READ_FAILED'
    assert excinfo.value.details['path'] == str(path)


def test_local_bad_data_maps_to_invalid_args():
    with pytest.raises(ConnectorError) as excinfo:
        read('local', data={'a': [1, 2], 'b': [1]})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def _driver_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(
    _driver_installed('snowflake.connector'),
    reason='snowflake driver installed',
)
def test_missing_snowflake_driver_raises_missing_backend():
    with pytest.raises(MissingBackendError) as excinfo:
        read('snowflake', table='items')
    assert excinfo.value.backend == 'snowflake'


@pytest.mark.skipif(
    _driver_installed('databricks.sql'),
    reason='databricks driver installed',
)
def test_missing_databricks_driver_raises_missing_backend():
    with pytest.raises(MissingBackendError) as excinfo:
        read('databricks', table='items')
    assert excinfo.value.backend == 'databricks'


def test_require_driver_returns_module():
    module = require_driver('duckdb', 'duckdb', 'duckdb')
    assert module.__name__ == 'duckdb'


def test_require_driver_missing_module_raises_missing_backend():
    with pytest.raises(MissingBackendError) as excinfo:
        require_driver('fake', 'fake-driver', 'sdm_no_such_module')
    assert excinfo.value.driver == 'fake-driver'


def test_connect_time_missing_file_stays_connect_failed():
    with pytest.raises(ConnectorError) as excinfo:
        with driver_guard('CONNECT_FAILED', 'connecting'):
            raise FileNotFoundError('missing private key file')
    assert excinfo.value.code == 'CONNECT_FAILED'


def test_read_guard_maps_missing_file_to_not_found():
    with pytest.raises(ConnectorError) as excinfo:
        with driver_guard('READ_FAILED', 'reading', missing_as_not_found=True):
            raise FileNotFoundError('no such object')
    assert excinfo.value.code == 'NOT_FOUND'


def test_driver_guard_propagates_import_error():
    with pytest.raises(ImportError) as excinfo:
        with driver_guard('CONNECT_FAILED', 'connecting'):
            raise ImportError('broken driver install')
    assert not isinstance(excinfo.value, ConnectorError)


def test_unexpected_connect_kwarg_maps_to_invalid_args(tmp_path):
    import duckdb as duckdb_driver

    database = tmp_path / 'db.duckdb'
    duckdb_driver.connect(str(database)).close()
    with pytest.raises(ConnectorError) as excinfo:
        read(
            'duckdb',
            database=str(database),
            table='items',
            bogus_option=1,
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_close_failure_does_not_mask_result(monkeypatch):
    import sdm_connectors.reader as reader_module

    class _Cursor:
        description = [('a',)]

        def execute(self, sql):
            pass

        def fetchall(self):
            return [(1,), (2,)]

        def close(self):
            raise RuntimeError('cursor close exploded')

    class _Connection:
        def cursor(self):
            return _Cursor()

        def close(self):
            raise RuntimeError('close exploded')

    monkeypatch.setattr(
        reader_module,
        'connect',
        lambda source, *a, **k: _Connection(),
    )
    frame = read('sqlite', database='ignored', table='items')
    assert list(frame['a']) == [1, 2]


def test_driver_guard_passes_connector_error_through():
    original = ConnectorError('kept', code='INVALID_CONNECTOR_ARGS')
    with pytest.raises(ConnectorError) as excinfo:
        with driver_guard('QUERY_FAILED', 'context'):
            raise original
    assert excinfo.value is original


def test_driver_guard_uses_type_name_for_empty_message():
    with pytest.raises(ConnectorError) as excinfo:
        with driver_guard('READ_FAILED', 'reading'):
            raise RuntimeError()
    assert excinfo.value.message == 'reading: RuntimeError'
    assert excinfo.value.details['driver_error'] == 'RuntimeError'
