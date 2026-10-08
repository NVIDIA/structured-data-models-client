# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import warnings
from pathlib import Path

import pytest
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.base.utils import random_state

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


def _hash_order(values: list[int], seed: int) -> list[int]:
    r"""The order ``_seeded_hash`` puts integers in, computed here."""
    p = 2**31 - 1
    c0, c1, c2, c3 = (int(c) for c in random_state(seed).randint(0, p, size=4))

    def hashed(value: int) -> int:
        x = value % p
        return (((c3 * x + c2) % p * x + c1) % p * x + c0) % p

    return sorted(values, key=lambda value: (hashed(value), value))


def test_a_seed_draws_the_rows_that_hash_lowest(tmp_path: Path) -> None:
    # Regression test:
    # SQLite cannot seed `ORDER BY RANDOM()`, so a seeded request drew a
    # different sample on every call. It orders by a seeded hash instead.
    graph = _graph(_create_database(tmp_path / 'seed.db', 'INTEGER'))
    sampler = SQLiteSampler(graph, verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        first = sampler._sample_entity_table(
            'users', {'user_id'}, 5, random_seed=42
        )
        again = sampler._sample_entity_table(
            'users', {'user_id'}, 5, random_seed=42
        )
        other = sampler._sample_entity_table(
            'users', {'user_id'}, 5, random_seed=43
        )

    # `user_id` is the rowid, since it is declared `INTEGER PRIMARY KEY`.
    assert first['user_id'].tolist() == _hash_order(list(range(50)), 42)[:5]
    assert again['user_id'].tolist() == first['user_id'].tolist()
    assert other['user_id'].tolist() == _hash_order(list(range(50)), 43)[:5]
    assert other['user_id'].tolist() != first['user_id'].tolist()


def test_a_seed_hashes_an_integer_key_not_where_its_row_is_stored(
    tmp_path: Path,
) -> None:
    # Regression test: the seeded draw hashed the rowid, which `VACUUM` may
    # renumber when the key is not the table's `INTEGER PRIMARY KEY`, so the
    # same rows drew a different sample once stored differently.
    keys = [(7 * i + 3) % 50 for i in range(50)]
    draws = []
    for order in (keys, keys[::-1]):
        path = tmp_path / f'stored_{len(draws)}.db'
        connection = sqlite3.connect(path)
        connection.execute('CREATE TABLE users (user_id INTEGER, age INTEGER)')
        connection.executemany(
            'INSERT INTO users VALUES (?, ?)', [(k, 20 + k) for k in order]
        )
        connection.execute('CREATE UNIQUE INDEX users_pkey ON users (user_id)')
        connection.commit()
        connection.close()
        graph = Graph.from_sqlite(
            path,
            tables=[dict(name='users', primary_key='user_id')],
            verbose=False,
        )
        sampler = SQLiteSampler(graph, verbose=False)
        draws.append(
            sampler._sample_entity_table(
                'users', {'user_id'}, 5, random_seed=42
            )['user_id'].tolist()
        )

    assert draws[0] == draws[1] == _hash_order(keys, 42)[:5]


def test_a_seed_hashes_the_rowid_of_a_table_keyed_by_text(
    tmp_path: Path,
) -> None:
    # SQLite has no hash of a string, so the draw follows the rowids.
    path = tmp_path / 'text_key.db'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE users (user_id TEXT, age INTEGER)')
    connection.executemany(
        'INSERT INTO users VALUES (?, ?)',
        [(f'u{i}', 20 + i) for i in range(50)],
    )
    connection.execute('CREATE UNIQUE INDEX users_pkey ON users (user_id)')
    connection.commit()
    connection.close()
    graph = Graph.from_sqlite(
        path, tables=[dict(name='users', primary_key='user_id')], verbose=False
    )
    sampler = SQLiteSampler(graph, verbose=False)

    drawn = sampler._sample_entity_table(
        'users', {'user_id'}, 5, random_seed=42
    )['user_id'].tolist()

    # The rowids run from 1, in the order the rows were inserted:
    assert drawn == [f'u{r - 1}' for r in _hash_order(range(1, 51), 42)[:5]]


def test_a_seeded_sample_is_spread_over_the_table(tmp_path: Path) -> None:
    graph = _graph(_create_database(tmp_path / 'spread.db', 'INTEGER'))
    sampler = SQLiteSampler(graph, verbose=False)

    drawn = sampler._sample_entity_table(
        'users', {'user_id'}, 25, random_seed=7
    )['user_id']
    # Neither the first nor the last rows, as an order by rowid would give:
    assert drawn.is_unique
    assert 0 < (drawn < 25).sum() < 25


def test_random_seed_does_not_warn(tmp_path: Path) -> None:
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
