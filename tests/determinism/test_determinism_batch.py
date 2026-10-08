# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A repeated batch index is refused, not sorted into some order.

The SQL samplers put rows back into request order by sorting on the batch
index each row came back with, using an unstable sort. That is only a
well-defined order when every index occurs once. A repeated index would put
two rows in an order the platform's sort decides, silently, so the three
places that sort by batch index must raise instead.

``_by_pkey`` is wrapped to return a repeated index: no source can produce one
through it today, which is why this is an internal error and not a message to
the caller.
"""

from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sampler(tmp_path: Path) -> Iterator[Any]:
    pytest.importorskip('sdm_relational_engine')
    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler

    path = tmp_path / 'shop.sqlite'
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE users (user_id INTEGER PRIMARY KEY, age INTEGER)'
    )
    connection.execute(
        'CREATE TABLE orders (order_id INTEGER PRIMARY KEY, '
        'user_id INTEGER REFERENCES users(user_id), price REAL, '
        'ordered_at TEXT)'
    )
    connection.executemany(
        'INSERT INTO users VALUES (?, ?)',
        [(user, 20 + user) for user in range(1, 6)],
    )
    connection.executemany(
        'INSERT INTO orders VALUES (?, ?, ?, ?)',
        [
            (
                order,
                1 + order % 5,
                10.0 * order,
                f'2026-01-{order:02d} 00:00:00',
            )
            for order in range(1, 21)
        ],
    )
    connection.commit()
    connection.close()

    adbc = dbapi.connect(str(path))
    graph = Graph.from_sqlite(adbc, verbose=False)
    edges = {(e.src_table, e.fkey, e.dst_table) for e in graph.edges}
    if ('orders', 'user_id', 'users') not in edges:
        graph.link('orders', 'user_id', 'users')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # No indices: irrelevant here.
        yield SQLiteSampler(graph, verbose=False)
    adbc.close()


def _repeat_an_index(
    monkeypatch: pytest.MonkeyPatch, sampler: Any, table: str
) -> list[np.ndarray]:
    r"""Make ``sampler._by_pkey`` repeat a batch index for ``table``."""
    real = sampler._by_pkey
    returned: list[np.ndarray] = []

    def by_pkey(**kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        df, batch = real(**kwargs)
        if kwargs['table_name'] == table and len(batch) >= 2:
            batch = np.array(batch, copy=True)
            batch[-1] = batch[0]
            returned.append(batch)
        return df, batch

    monkeypatch.setattr(sampler, '_by_pkey', by_pkey)
    return returned


def _assert_internal_error(error: pytest.ExceptionInfo[BaseException]) -> None:
    # Not an `assert`: `python -O` strips those and the reordering returns.
    assert not isinstance(error.value, AssertionError), error.value
    # Not a ValueError: the client reports those as the caller's mistake.
    assert not isinstance(error.value, ValueError), error.value
    assert 'duplicate' in str(error.value).lower(), str(error.value)


def test_the_entity_rows_of_a_subgraph(
    monkeypatch: pytest.MonkeyPatch, sampler: Any
) -> None:
    returned = _repeat_an_index(monkeypatch, sampler, 'users')
    with pytest.raises(Exception) as error:
        sampler.sample_subgraph(
            entity_table_names=('users',),
            entity_pkey=pd.Series([1, 2, 3]),
            anchor_time=pd.Series(pd.to_datetime(['2026-02-01'] * 3)),
            num_neighbors=[2],
        )
    assert returned, 'the repeated index was never handed to the sampler'
    _assert_internal_error(error)


def test_the_rows_a_foreign_key_points_at(
    monkeypatch: pytest.MonkeyPatch, sampler: Any
) -> None:
    returned = _repeat_an_index(monkeypatch, sampler, 'users')
    with pytest.raises(Exception) as error:
        sampler.sample_subgraph(
            entity_table_names=('orders',),
            entity_pkey=pd.Series([1, 2, 3]),
            anchor_time=pd.Series(pd.to_datetime(['2026-02-01'] * 3)),
            num_neighbors=[2],
        )
    assert returned, 'the repeated index was never handed to the sampler'
    _assert_internal_error(error)


def test_the_entity_rows_of_a_query(
    monkeypatch: pytest.MonkeyPatch, sampler: Any
) -> None:
    returned = _repeat_an_index(monkeypatch, sampler, 'users')
    with pytest.raises(Exception) as error:
        sampler._sample_query_data(
            entity_table_name='users',
            entity_pkey=pd.Series([1, 2, 3]),
            anchor_time=pd.Series(pd.to_datetime(['2026-02-01'] * 3)),
            columns_dict={'users': {'user_id', 'age'}},
            time_offset_dict={},
        )
    assert returned, 'the repeated index was never handed to the sampler'
    _assert_internal_error(error)


def test_unique_indices_still_sample(sampler: Any) -> None:
    subgraph = sampler.sample_subgraph(
        entity_table_names=('users',),
        entity_pkey=pd.Series([3, 1, 2]),
        anchor_time=pd.Series(pd.to_datetime(['2026-02-01'] * 3)),
        num_neighbors=[2],
    )
    assert subgraph is not None
