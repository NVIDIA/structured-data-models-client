# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import warnings
from pathlib import Path

import pytest
from sdm_relational_engine.rfm import Graph

pytest.importorskip(
    'adbc_driver_sqlite', reason="'sqlite' extension not installed"
)

from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler


def _create_database(path: Path, foreign_key_type: str) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE users (  user_id INTEGER PRIMARY KEY,  age INTEGER)'
    )
    connection.execute(
        f'CREATE TABLE orders ('
        f'  order_id INTEGER PRIMARY KEY,'
        f'  user_id {foreign_key_type},'
        f'  ts TEXT)'
    )
    connection.executemany(
        'INSERT INTO users VALUES (?, ?)',
        [(i, 20 + i % 40) for i in range(50)],
    )
    connection.executemany(
        'INSERT INTO orders VALUES (?, ?, ?)',
        [
            (
                i,
                str(i % 50) if foreign_key_type == 'TEXT' else i % 50,
                f'2024-01-{i % 28 + 1:02d}',
            )
            for i in range(200)
        ],
    )
    connection.execute('CREATE UNIQUE INDEX users_pkey ON users (user_id)')
    connection.execute('CREATE UNIQUE INDEX orders_pkey ON orders (order_id)')
    connection.execute('CREATE INDEX orders_fkey ON orders (user_id)')
    connection.commit()
    connection.close()
    return path


def _graph(path: Path) -> Graph:
    return Graph.from_sqlite(
        path,
        tables=[
            dict(name='users', primary_key='user_id'),
            dict(name='orders', primary_key='order_id', time_column='ts'),
        ],
        edges=[('orders', 'user_id', 'users')],
        verbose=False,
    )


def test_validate_mismatched_data_type_families(tmp_path: Path) -> None:
    # Regression test: the
    # foreign key/primary key data type check used to run on the local backend
    # only, so SQL-backed graphs passed validation and failed inside the
    # sampler with a bare `TypeError` instead.
    graph = _graph(_create_database(tmp_path / 'mismatch.db', 'TEXT'))

    assert str(graph['orders']['user_id'].dtype) == 'string'
    assert str(graph['users']['user_id'].dtype) == 'int'

    with pytest.raises(ValueError, match='have incompatible data types'):
        graph.validate()


def test_validate_compatible_data_type_families(tmp_path: Path) -> None:
    graph = _graph(_create_database(tmp_path / 'match.db', 'INTEGER'))

    assert graph.validate() is graph


def test_random_seed_warns_once(tmp_path: Path) -> None:
    # Regression test:
    # SQLite cannot seed `ORDER BY RANDOM()`, which must be surfaced to the
    # caller rather than dropped silently.
    graph = _graph(_create_database(tmp_path / 'seed.db', 'INTEGER'))
    sampler = SQLiteSampler(graph, verbose=False)

    with pytest.warns(UserWarning, match='seeded random sampling'):
        sampler._sample_entity_table('users', {'user_id'}, 5, random_seed=42)

    with warnings.catch_warnings():  # Only warn once per sampler:
        warnings.simplefilter('error')
        sampler._sample_entity_table('users', {'user_id'}, 5, random_seed=42)


def test_random_seed_does_not_warn_without_seed(tmp_path: Path) -> None:
    graph = _graph(_create_database(tmp_path / 'no_seed.db', 'INTEGER'))
    sampler = SQLiteSampler(graph, verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sampler._sample_entity_table('users', {'user_id'}, 5)
        sampler._sample_entity_table(
            'users', {'user_id'}, 5, random_seed=42, entity_ids=[1, 2, 3]
        )


def test_discovery_on_an_empty_database_names_what_it_searched(
    tmp_path: Path,  #
) -> None:
    """A database with nothing in it, what a mistyped path used to produce,
    since sqlite creates on connect, returned a valid-looking empty graph
    that failed much later as "At least one table needs to be added to the
    graph", pointing at the caller's table list rather than the path.
    """
    path = tmp_path / 'empty.db'
    sqlite3.connect(path).close()

    with pytest.raises(ValueError, match='No tables found'):
        Graph.from_sqlite(str(path), verbose=False)
