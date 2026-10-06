# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path

import pytest
from sdm_relational_engine.rfm import Graph

duckdb = pytest.importorskip(
    'duckdb', reason="'duckdb' extension not installed"
)

from sdm_relational_engine.rfm.backend.duckdb import (  # noqa: E402
    DuckDBSampler,
)


@pytest.fixture()
def graph(tmp_path: Path) -> Graph:
    path = tmp_path / 'users.duckdb'
    connection = duckdb.connect(path)
    connection.execute(
        'CREATE TABLE users AS '
        'SELECT i AS user_id, i % 40 AS age '
        'FROM range(200) t(i)'
    )
    connection.execute('CREATE UNIQUE INDEX users_pkey ON users (user_id)')
    connection.close()

    return Graph.from_duckdb(
        path,
        tables=[dict(name='users', primary_key='user_id')],
        verbose=False,
    )


def _sample(sampler: DuckDBSampler, random_seed: int | None) -> list[int]:
    df = sampler._sample_entity_table(
        'users', {'user_id'}, 5, random_seed=random_seed
    )
    return sorted(df['user_id'].tolist())


def test_random_seed_is_reproducible(graph: Graph) -> None:
    # Regression test:
    # DuckDB supports `USING SAMPLE ... REPEATABLE (seed)`, so reusing a seed
    # must draw the same in-context examples.
    sampler = DuckDBSampler(graph, verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _sample(sampler, 42) == _sample(sampler, 42)
        assert _sample(sampler, 42) != _sample(sampler, 7)


def test_unseeded_sample_is_random(graph: Graph) -> None:
    sampler = DuckDBSampler(graph, verbose=False)

    samples = {tuple(_sample(sampler, None)) for _ in range(5)}
    assert len(samples) > 1


def test_discovery_on_an_empty_database_names_what_it_searched(
    tmp_path: Path,  #
) -> None:
    """``_require_discovered_tables`` was wired into three of the four discovery
    constructors. DuckDB, which needs it most because it creates a database
    rather than failing on a mistyped path, was the one left out.
    """
    path = tmp_path / 'empty.duckdb'
    duckdb.connect(str(path)).close()

    with pytest.raises(ValueError, match='No tables found in the DuckDB'):
        Graph.from_duckdb(str(path), verbose=False)


def test_missing_database_is_reported_rather_than_created(
    tmp_path: Path,  #
) -> None:
    """DuckDB creates a database for a missing path by default, so the shared
    connector's existence guard must run before opening it. A read must not
    write.
    """
    path = tmp_path / 'typo.duckdb'

    with pytest.raises(Exception, match='does not exist'):
        Graph.from_duckdb(str(path), verbose=False)
    assert not path.exists()


def test_a_decimal_identifier_keeps_every_digit(tmp_path: Path) -> None:
    """A `DECIMAL(38, 0)` column declared a string, as an identifier is, was
    read as a float, so 123456789012345678 and 123456789012345679 became one
    1.2345678901234568e+17. Only an unset, integer or float data type makes
    a decimal a number; a string one keeps the exact digits.
    """
    import pandas as pd
    import pyarrow as pa

    path = tmp_path / 'ids.duckdb'
    connection = duckdb.connect(path)
    connection.execute(
        'CREATE TABLE u '
        '(id DECIMAL(38, 0), code DECIMAL(38, 0), price DECIMAL(10, 2))'
    )
    connection.execute(
        'INSERT INTO u VALUES '
        '(123456789012345678, 123456789012345678, 1.25), '
        '(123456789012345679, 123456789012345679, 2.50)'
    )
    connection.close()

    graph = Graph.from_duckdb(
        path,
        tables=[
            dict(
                name='u',
                primary_key='id',
                columns=[
                    'id',
                    dict(name='code', dtype='string', stype='ID'),
                    'price',
                ],
            )
        ],
        edges=[],
        verbose=False,
    )
    ids = [123456789012345678, 123456789012345679]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        subgraph = DuckDBSampler(graph, verbose=False).sample_subgraph(
            ('u',),
            pd.Series(ids),
            pd.Series(pd.to_datetime(['2025-01-01'] * 2)),
            [4],
        )
    df = subgraph.tables['u'].df
    assert df['id'].tolist() == ids
    assert df['code'].dtype == pd.ArrowDtype(pa.string())
    assert df['code'].tolist() == [str(i) for i in ids]
    assert df['price'].tolist() == [1.25, 2.5]
