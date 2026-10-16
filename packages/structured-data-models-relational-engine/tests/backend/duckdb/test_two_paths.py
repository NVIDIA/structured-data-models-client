# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A table reached by two paths, and a column linking to two tables, as a
metric view's snowflake and role-playing joins make them.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip('duckdb')

import sdm_relational_engine.rfm as rfm  # noqa: E402
from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler  # noqa: E402

_ENTITIES = [0, 1, 2]


@pytest.fixture
def connection() -> Any:
    connection = duckdb.connect()
    connection.execute('CREATE TABLE customers (cid INT, seg VARCHAR)')
    connection.execute(
        "INSERT INTO customers SELECT i, 'x' FROM range(10) t(i)"
    )
    connection.execute('CREATE VIEW buyers AS SELECT * FROM customers')
    connection.execute('CREATE TABLE headers (hid INT, cid INT, ts TIMESTAMP)')
    connection.execute(
        'INSERT INTO headers SELECT i, i % 10, '
        "TIMESTAMP '2024-01-01' + INTERVAL (i) DAY FROM range(40) t(i)"
    )
    # Every line's customer is its order's, and its id modulo 10:
    connection.execute(
        'CREATE TABLE details (did INT, hid INT, cid INT, ts TIMESTAMP)'
    )
    connection.execute(
        'INSERT INTO details SELECT i, i % 40, i % 10, '
        "TIMESTAMP '2024-01-01' + INTERVAL (i % 40) DAY FROM range(120) t(i)"
    )
    yield connection
    connection.close()


def _sample(connection: Any, tables: list, edges: list) -> Any:
    graph = rfm.Graph.from_duckdb(
        connection,
        tables=tables,
        edges=edges,
        infer_metadata=False,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # Missing indices.
        sampler = DuckDBSampler(graph, verbose=False)
    return sampler.sample_subgraph(
        ('customers',),
        pd.Series(_ENTITIES),
        pd.Series(pd.to_datetime(['2025-01-01'] * len(_ENTITIES))),
        [16, 16, 16],
    )


def _customer_of_each_line(subgraph: Any) -> None:
    lines = subgraph.tables['details']
    assert lines.num_rows > 0
    assert sum(lines.num_sampled_nodes) == len(lines.batch)
    entities = np.asarray(_ENTITIES)[lines.batch]
    # A line's customer is its id modulo 10:
    assert (lines.df['did'].to_numpy() % 10 == entities).all()


def test_a_table_reached_by_two_paths_is_sampled_once_per_hop(
    connection: Any,
) -> None:
    r"""The lines are reached from a customer directly and through orders.
    Those reached through orders arrived while the lines of the hop before
    were being read, and were read with them, so the lines and the count of
    them disagreed and sampling failed with an `IndexError`.
    """
    subgraph = _sample(
        connection,
        [
            dict(name='customers', primary_key='cid', columns=['cid', 'seg']),
            dict(
                name='headers',
                primary_key='hid',
                time_column='ts',
                columns=['hid', 'cid', 'ts'],
            ),
            dict(name='details', time_column='ts'),
        ],
        [
            ('headers', 'cid', 'customers'),
            ('details', 'hid', 'headers'),
            ('details', 'cid', 'customers'),
        ],
    )

    _customer_of_each_line(subgraph)


def test_a_column_linking_to_two_tables_is_read_once(connection: Any) -> None:
    r"""A role-playing join links one column to two tables, and selecting it
    once per link made it two columns, which no key look-up takes.
    """
    subgraph = _sample(
        connection,
        [
            dict(name='customers', primary_key='cid', columns=['cid', 'seg']),
            dict(name='buyers', primary_key='cid', columns=['cid', 'seg']),
            dict(name='details', time_column='ts'),
        ],
        [('details', 'cid', 'customers'), ('details', 'cid', 'buyers')],
    )

    _customer_of_each_line(subgraph)
    assert subgraph.tables['buyers'].num_rows > 0


def test_two_paths_to_a_keyed_table_sample_as_in_memory(
    connection: Any,
) -> None:
    r"""With every table keyed, two paths to the lines did not fail but
    sampled a hop too deep: lines the orders reached for the next hop were
    read with this one, so their neighbours arrived a hop early, and
    predictions differed from the in-memory sampler's over the same data.
    """
    from sdm_relational_engine.rfm.backend.local.sampler import LocalSampler

    # A line's own customer is not its order's, so the orders reach lines
    # the customers do not.
    connection.execute('UPDATE details SET cid = (did + 1) % 10')

    tables = [
        dict(name='customers', primary_key='cid', columns=['cid', 'seg']),
        dict(
            name='headers',
            primary_key='hid',
            time_column='ts',
            columns=['hid', 'cid', 'ts'],
        ),
        dict(
            name='details',
            primary_key='did',
            time_column='ts',
            columns=['did', 'hid', 'cid', 'ts'],
        ),
    ]
    edges = [
        ('headers', 'cid', 'customers'),
        ('details', 'hid', 'headers'),
        ('details', 'cid', 'customers'),
    ]
    subgraph = _sample(connection, tables, edges)

    graph = rfm.Graph.from_data(
        {
            spec['name']: connection.execute(
                f'SELECT * FROM {spec["name"]}'
            ).df()
            for spec in tables
        },
        edges=edges,
        infer_metadata=False,
        verbose=False,
    )
    for spec in tables:
        graph[spec['name']].primary_key = spec['primary_key']
        if 'time_column' in spec:
            graph[spec['name']].time_column = spec['time_column']
    in_memory = LocalSampler(graph, verbose=False).sample_subgraph(
        ('customers',),
        pd.Series(_ENTITIES),
        pd.Series(pd.to_datetime(['2025-01-01'] * len(_ENTITIES))),
        [16, 16, 16],
    )

    def _hops(sampled: Any) -> dict[str, list[int]]:
        # The SQL sampler stops at the first hop that reaches nothing.
        return {
            name: [
                int(n)
                for n in np.trim_zeros(np.asarray(table.num_sampled_nodes), 'b')
            ]
            for name, table in sorted(sampled.tables.items())
        }

    assert _hops(subgraph) == _hops(in_memory)
