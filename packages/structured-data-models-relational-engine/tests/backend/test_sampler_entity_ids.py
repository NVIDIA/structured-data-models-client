# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Entity ids must reach the warehouse as bound parameters, never as SQL
text.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from predictive_query.schema.typing import Dtype, Stype

# Ordinary keys that Python's `repr()` used to mangle, plus a payload that used
# to close the string literal and append attacker-chosen SQL:
_PLAIN = 'u1'
_BACKSLASH = 'a\\b'
_QUOTES = 'x\'"y'
_APOSTROPHE = "O'Brien"
_INJECTION = 'a\') OR 1=1 --"'
_ROWS = [_PLAIN, _BACKSLASH, _QUOTES, _APOSTROPHE]
_COLUMNS = {'user_id', 'ts'}


def _insert_sql() -> str:
    values = ', '.join(
        f"('{key.replace(chr(39), chr(39) * 2)}', '2024-01-0{i + 1}')"
        for i, key in enumerate(_ROWS)
    )
    return f'INSERT INTO users VALUES {values}'


@pytest.fixture(scope='module')
def sqlite_sampler(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Any]:
    pytest.importorskip('adbc_driver_sqlite')

    import adbc_driver_sqlite.dbapi as adbc
    import sdm_relational_engine.rfm as rfm
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler

    path = Path(tmp_path_factory.mktemp('sqlite')) / 'entity_ids.db'
    connection = adbc.connect(str(path))
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE users (user_id TEXT PRIMARY KEY, ts TEXT NOT NULL)'
        )
        cursor.execute(_insert_sql())
        cursor.execute('CREATE TABLE secrets (user_id TEXT, ts TEXT)')
        cursor.execute("INSERT INTO secrets VALUES ('TOPSECRET', '2099-01-01')")
    connection.commit()

    graph = rfm.Graph.from_sqlite(
        connection,
        tables=[dict(name='users', primary_key='user_id', time_column='ts')],
        verbose=False,
    )
    graph.validate()
    yield SQLiteSampler(graph, verbose=False)
    connection.close()


@pytest.fixture(scope='module')
def duckdb_sampler(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Any]:
    duckdb = pytest.importorskip('duckdb')

    import sdm_relational_engine.rfm as rfm
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler

    path = Path(tmp_path_factory.mktemp('duckdb')) / 'entity_ids.duckdb'
    connection = duckdb.connect(str(path))
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE users (user_id VARCHAR PRIMARY KEY, '
            'ts TIMESTAMP NOT NULL)'
        )
        cursor.execute(_insert_sql())
        cursor.execute('CREATE TABLE secrets (user_id VARCHAR, ts TIMESTAMP)')
        cursor.execute("INSERT INTO secrets VALUES ('TOPSECRET', '2099-01-01')")
    connection.commit()

    graph = rfm.Graph.from_duckdb(
        connection,
        tables=[dict(name='users', primary_key='user_id', time_column='ts')],
        verbose=False,
    )
    graph.validate()
    yield DuckDBSampler(graph, verbose=False)
    connection.close()


@pytest.fixture(scope='module')
def postgres_sampler(postgres_dsn: str) -> Iterator[Any]:
    dsn = postgres_dsn
    import psycopg
    import sdm_relational_engine.rfm as rfm
    from sdm_relational_engine.rfm.backend.postgres import PostgresSampler

    schema = f'sdm_ids_{uuid.uuid4().hex}'
    connection = psycopg.connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA "{schema}"')
            cursor.execute(
                f'CREATE TABLE "{schema}".users ('
                'user_id TEXT PRIMARY KEY, ts TIMESTAMP NOT NULL)'
            )
            cursor.executemany(
                f'INSERT INTO "{schema}".users VALUES (%s, %s)',
                [
                    (key, f'2024-01-0{index + 1}')
                    for index, key in enumerate(_ROWS)
                ],
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".secrets (user_id TEXT, ts TIMESTAMP)'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".secrets VALUES '
                "('TOPSECRET', '2099-01-01')"
            )
        connection.commit()
    finally:
        connection.close()

    graph = rfm.Graph.from_postgres(
        dsn,
        schema=schema,
        tables=[dict(name='users', primary_key='user_id', time_column='ts')],
        verbose=False,
    )
    graph.validate()
    try:
        yield PostgresSampler(graph, verbose=False)
    finally:
        assert graph._connection is not None
        graph._connection.close()
        graph._connection = None
        connection = psycopg.connect(dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP SCHEMA "{schema}" CASCADE')
            connection.commit()
        finally:
            connection.close()


@pytest.fixture(params=['sqlite', 'duckdb', 'postgres'])
def sampler(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue(f'{request.param}_sampler')


def _sample(sampler: Any, entity_ids: list) -> pd.DataFrame:
    return sampler._sample_entity_table(
        table_name='users',
        columns=_COLUMNS,
        num_rows=10,
        entity_ids=entity_ids,
    )


@pytest.mark.parametrize(
    'entity_id', [_PLAIN, _BACKSLASH, _QUOTES, _APOSTROPHE]
)
def test_quotable_entity_id_round_trips(sampler: Any, entity_id: str) -> None:
    df = _sample(sampler, [entity_id])
    assert df['user_id'].tolist() == [entity_id]


def test_all_entity_ids_round_trip_at_once(sampler: Any) -> None:
    df = _sample(sampler, list(_ROWS))
    assert sorted(df['user_id'].tolist()) == sorted(_ROWS)


def test_injected_entity_id_matches_nothing(sampler: Any) -> None:
    assert len(_sample(sampler, [_INJECTION])) == 0


def test_injected_entity_id_cannot_reach_a_table_outside_the_graph(
    sampler: Any,
) -> None:
    payload = "zz') UNION ALL SELECT user_id, ts FROM secrets --"
    df = _sample(sampler, [payload])
    assert 'TOPSECRET' not in df['user_id'].tolist()
    assert len(df) == 0


def test_unmatched_entity_id_still_returns_nothing(sampler: Any) -> None:
    assert len(_sample(sampler, ['does-not-exist'])) == 0


class _SnowCursor:
    def __init__(self, connection: _SnowConnection) -> None:
        self._connection = connection

    def __enter__(self) -> _SnowCursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append(
            (sql, parameters, self._connection._paramstyle)
        )

    def fetch_arrow_all(self, force_return_table: bool = True) -> pa.Table:
        rows = self._connection.rows
        if rows:
            names = [f'c{index}' for index in range(len(rows[0]))]
            return pa.table(
                {name: [row[i] for row in rows] for i, name in enumerate(names)}
            )
        columns = {
            'user_id': pa.array([], type=pa.string()),
            'ts': pa.array([], type=pa.string()),
        }
        if '__KUMO_BATCH__' in self._connection.calls[-1][0]:
            columns['__KUMO_BATCH__'] = pa.array([], type=pa.int64())
        return pa.table(columns)

    def fetchall(self) -> list:
        return self._connection.rows


class _SnowConnection:
    def __init__(self) -> None:
        self._paramstyle = 'pyformat'
        self.calls: list[tuple[str, Any, str]] = []
        self.rows: list = []

    def cursor(self) -> _SnowCursor:
        return _SnowCursor(self)


@pytest.fixture
def snow_sampler() -> Any:
    pytest.importorskip('snowflake.connector')

    from sdm_relational_engine.rfm.backend.snow import SnowSampler
    from sdm_relational_engine.rfm.base import SourceColumn

    sampler = SnowSampler.__new__(SnowSampler)
    sampler._connection = _SnowConnection()
    sampler._num_rows_dict = {'users': len(_ROWS)}
    sampler._primary_key_dict = {'users': 'user_id'}
    sampler._enforced_key_tables = frozenset()
    sampler._time_column_dict = {'users': 'ts'}
    sampler._source_name_dict = {'users': '"DB"."SCHEMA"."USERS"'}
    sampler._source_table_dict = {
        'users': {
            'user_id': SourceColumn(
                name='user_id',
                dtype=Dtype.string,
                is_primary_key=True,
                is_unique_key=True,
                is_nullable=False,
            ),
            'ts': SourceColumn(
                name='ts',
                dtype=Dtype.date,
                is_primary_key=False,
                is_unique_key=False,
                is_nullable=False,
            ),
        },
    }
    sampler._table_dtype_dict = {
        'users': {'user_id': Dtype.string, 'ts': Dtype.date},
    }
    sampler._table_stype_dict = {'users': {'ts': Stype.timestamp}}
    sampler._snowflake_type_dict = {
        'users': {'user_id': 'VARCHAR(16777216)', 'ts': 'TIMESTAMP_NTZ(9)'},
    }
    # A table too small for its micro-partitions to be sampled:
    sampler._table_dict = {
        'users': SimpleNamespace(_prefilter=lambda rows, seed: None)
    }
    sampler._table_column_ref_dict = {
        'users': {'user_id': '"user_id"', 'ts': '"ts"'},
    }
    sampler._table_column_proj_dict = sampler._table_column_ref_dict
    return sampler


def test_snow_binds_entity_ids_instead_of_interpolating(
    snow_sampler: Any,
) -> None:
    snow_sampler._sample_entity_table(
        table_name='users',
        columns=_COLUMNS,
        num_rows=10,
        entity_ids=[_QUOTES, _INJECTION],
    )

    sql, parameters, paramstyle = snow_sampler._connection.calls[-1]
    assert '"user_id" IN (?, ?)' in sql
    assert parameters == [_QUOTES, _INJECTION]
    assert paramstyle == 'qmark'
    assert _QUOTES not in sql
    assert _INJECTION not in sql
    assert 'OR 1=1' not in sql
    assert snow_sampler._connection._paramstyle == 'pyformat'


def test_snow_random_sample_still_binds_nothing(snow_sampler: Any) -> None:
    snow_sampler._sample_entity_table(
        table_name='users',
        columns=_COLUMNS,
        num_rows=10,
    )

    sql, parameters, _ = snow_sampler._connection.calls[-1]
    assert 'SAMPLE ROW (10 ROWS)' in sql
    assert parameters is None


def test_snow_seeded_sample_keeps_the_rows_a_seeded_hash_picks(
    snow_sampler: Any,
) -> None:
    r"""Snowflake seeds no fixed-size sample, so a seeded one drew different
    rows on every call. The rows whose key hashes lowest with the seed are
    kept instead, in that order.
    """
    snow_sampler._sample_entity_table(
        table_name='users',
        columns=_COLUMNS,
        num_rows=10,
        random_seed=7,
    )

    sql, parameters, _ = snow_sampler._connection.calls[-1]
    assert 'SAMPLE' not in sql
    assert sql.endswith(
        'ORDER BY MD5_NUMBER_LOWER64(\'7:\' || "user_id"), '
        '"user_id", "ts"\nLIMIT 10'
    )
    assert parameters is None


@pytest.mark.parametrize(
    ('dtype', 'source_type', 'text'),
    [
        (Dtype.int, 'NUMBER(38,0)', 'TO_VARCHAR("user_id")'),
        (Dtype.float, 'FLOAT', 'TO_VARCHAR("user_id")'),
        (
            Dtype.date,
            'TIMESTAMP_LTZ(9)',
            'TO_VARCHAR(DATE_PART(EPOCH_NANOSECOND, "user_id"))',
        ),
        # `DATE_PART` gives no `EPOCH_NANOSECOND` of a `DATE`:
        (Dtype.date, 'DATE', 'TO_VARCHAR("user_id", \'YYYY-MM-DD\')'),
        (Dtype.time, 'TIME(9)', 'TO_VARCHAR("user_id", \'HH24:MI:SS.FF9\')'),
        (Dtype.binary, 'BINARY(8388608)', 'TO_VARCHAR("user_id", \'HEX\')'),
    ],
)
def test_snow_seeded_sample_hashes_a_key_no_session_setting_formats(
    snow_sampler: Any,
    dtype: Dtype,
    source_type: str,
    text: str,
) -> None:
    r"""``TO_VARCHAR`` of a timestamp, date, time or binary value follows the
    session's output formats and time zone, so two sessions drew different
    rows from one seed. Each key type is hashed in a form no setting changes.
    """
    snow_sampler._source_table_dict['users']['user_id'].dtype = dtype
    snow_sampler._table_dtype_dict['users']['user_id'] = dtype
    snow_sampler._snowflake_type_dict['users']['user_id'] = source_type

    snow_sampler._sample_entity_table(
        table_name='users',
        columns=_COLUMNS,
        num_rows=10,
        random_seed=7,
    )

    sql, _, _ = snow_sampler._connection.calls[-1]
    assert f"ORDER BY MD5_NUMBER_LOWER64('7:' || {text}), " in sql


def test_snow_min_max_probe_keeps_table_names_out_of_the_sql(
    snow_sampler: Any,
) -> None:
    r"""The per-branch label used to be the graph table name in a string
    literal. Discovery adopts whatever names the schema holds, so that name is
    not necessarily one the caller chose, and quote-doubling does not
    neutralise ``\'`` on Snowflake. The branch is identified by position now.
    """
    hostile = "users\\' UNION ALL SELECT 'pwned', NULL, NULL --"
    for attribute in (
        '_time_column_dict',
        '_source_name_dict',
        '_table_column_ref_dict',
    ):
        mapping = getattr(snow_sampler, attribute)
        mapping[hostile] = mapping['users']
    snow_sampler._connection.rows = [(0, None, None)]

    out = snow_sampler._get_min_max_time_dict([hostile])

    sql, _, _ = snow_sampler._connection.calls[-1]
    assert '0 as table_index' in sql
    assert hostile not in sql
    assert "'" not in sql
    assert set(out) == {hostile}


def test_snow_binds_the_time_bounds_of_a_window(snow_sampler: Any) -> None:
    snow_sampler._foreign_key_dict = {'users': [('user_id', 'users')]}
    snow_sampler._min_time_dict = {'users': pd.Timestamp('2024-01-01')}
    snow_sampler._max_time_dict = {'users': pd.Timestamp('2024-01-04')}
    anchor = pd.Series(pd.to_datetime(['2024-02-01', '2024-03-01']))

    snow_sampler._by_time(
        'users',
        'user_id',
        pd.Series(['u1', 'u2']),
        anchor_time=anchor,
        min_offset=pd.DateOffset(days=-30),
        max_offset=pd.DateOffset(days=0),
        columns=_COLUMNS,
    )
    by_time = snow_sampler._connection.calls[-1]
    snow_sampler._by_fkey(
        'users',
        'user_id',
        pd.Series(['u1', 'u2']),
        num_neighbors=2,
        anchor_time=anchor,
        columns=_COLUMNS,
    )
    by_fkey = snow_sampler._connection.calls[-1]

    assert by_time[1][1:] == ['2024-03-01 00:00:00', '2024-01-02 00:00:00']
    assert by_fkey[1][1] == '2024-03-01 00:00:00'
    for sql, parameters, paramstyle in (by_time, by_fkey):
        assert paramstyle == 'qmark'
        assert len(parameters) == 3
        assert sql.count('?') == 3
        assert "'2024" not in sql
