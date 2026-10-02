# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The Databricks backend sampling through a notebook's Spark session.

The session here is a double that answers each statement the backend issues
from two small tables, the way Unity Catalog and the tables themselves would.
What it proves is the path: that a Spark session is accepted where a SQL
warehouse connection was, that every statement reaches ``spark.sql`` with its
values bound rather than formatted in, and that what comes back is read the
way the warehouse's answers are. Whether Databricks answers these statements
the same way through Spark as through a warehouse is for a live run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sdm_relational_engine.rfm import Graph, KumoRelational
from sdm_relational_engine.rfm.base import KeyCount, count_keys

try:
    from sdm_relational_engine.rfm.backend.databricks import (
        DatabricksSampler,
        SparkConnection,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)

pytest.importorskip('sdm_relational_engine.relationallib')

_USERS = pa.table(
    {
        'user_id': pa.array([1, 2, 3], pa.int64()),
        'age': pa.array([31.0, 45.0, 27.0], pa.float64()),
    }
)
_ORDERS = pa.table(
    {
        'order_id': pa.array([10, 11, 12, 13], pa.int64()),
        'user_id': pa.array([1, 1, 2, 3], pa.int64()),
        'amount': pa.array([5.0, 7.5, 3.0, 9.0], pa.float64()),
        'ts': pa.array(
            pd.to_datetime(
                ['2024-01-01', '2024-02-01', '2024-01-15', '2024-03-01']
            ),
            pa.timestamp('us'),
        ),
    }
)
_TABLES = {'users': _USERS, 'orders': _ORDERS}
_UC_TYPES = {
    pa.int64(): 'bigint',
    pa.float64(): 'double',
    pa.timestamp('us'): 'timestamp_ntz',
}
_PRIMARY_KEYS = {'users': 'user_id', 'orders': 'order_id'}


class _Row(tuple):
    pass


class _Frame:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def first(self) -> _Row | None:
        rows = self.collect()
        return rows[0] if rows else None

    def collect(self) -> list[_Row]:
        columns = [column.to_pylist() for column in self._table.columns]
        return [_Row(values) for values in zip(*columns, strict=True)]

    def toArrow(self) -> pa.Table:  # noqa: N802 - pyspark's name
        return self._table


def _rows(names: list[str], rows: list[tuple]) -> pa.Table:
    columns = list(zip(*rows)) if rows else [[] for _ in names]
    return pa.table(
        {name: pa.array(list(values)) for name, values in zip(names, columns)}
    )


class _Session:
    r"""Answers the statements the Databricks backend issues, and nothing
    else: an unrecognised statement fails the test rather than returning
    something plausible.
    """

    version = '4.0.0'

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def sql(self, query: str, args: Any = None) -> _Frame:
        self.calls.append((query, args))
        return _Frame(self._answer(query, args))

    def _answer(self, query: str, args: Any) -> pa.Table:
        if 'information_schema.columns' in query:
            schema, name = args
            table = _TABLES[name]
            return _rows(
                [
                    'column_name',
                    'full_data_type',
                    'is_nullable',
                    'table_catalog',
                    'table_schema',
                    'table_name',
                ],
                [
                    (
                        field.name,
                        _UC_TYPES[field.type],
                        'YES',
                        'cat',
                        schema,
                        name,
                    )
                    for field in table.schema
                ],
            )
        if "IN ('PRIMARY KEY', 'UNIQUE')" in query:
            _, name = args
            return _rows(
                ['constraint_type', 'constraint_name', 'column_name'],
                [('PRIMARY KEY', f'{name}_pk', _PRIMARY_KEYS[name])],
            )
        if "constraint_type = 'FOREIGN KEY'" in query:
            _, name = args
            links = (
                [
                    (
                        'orders_users_fk',
                        'user_id',
                        'cat',
                        'sch',
                        'users',
                        'user_id',
                    )
                ]
                if name == 'orders'
                else []
            )
            return _rows(
                [
                    'constraint_name',
                    'column_name',
                    'table_catalog',
                    'table_schema',
                    'table_name',
                    'pk_column',
                ],
                links,
            )
        if ' AS k, COUNT(*) AS n, ' in query:
            return _rows(
                ['k', 'n', 'd', 'z'],
                [_count_key(select) for select in query.split('\nUNION ALL\n')],
            )
        if 'HAVING COUNT(*) > 1' in query:
            return _repeated_key(query)
        if 'COUNT(*)' in query:
            return _rows(['n'], [(_TABLES[_source(query)].num_rows,)])
        if 'MIN(' in query and 'MAX(' in query:
            ts = _ORDERS['ts'].to_pylist()
            return _rows(
                ['table_index', 'min_date', 'max_date'], [(0, min(ts), max(ts))]
            )
        if 'posexplode(from_json(?' in query:
            return self._by_primary_key(query, args)
        if re.search(r'\bLIMIT \d+$', query):
            table = _TABLES[_source(query)]
            return table.select(
                re.findall(r'`(\w+)`', query.split(' FROM ')[0])
            )
        raise AssertionError(f'unexpected statement:\n{query}')

    @staticmethod
    def _by_primary_key(query: str, args: Any) -> pa.Table:
        # A window's time bounds follow the payload.
        payload = args[0]
        ids = json.loads(payload)
        table = _TABLES[_source(query)]
        key = _PRIMARY_KEYS[_source(query)]
        batch, rows = [], []
        for position, value in enumerate(ids):
            for row in table.to_pylist():
                if row[key] == value:
                    batch.append(position)
                    rows.append(row)
        out = pa.Table.from_pylist(rows, schema=table.schema)
        return out.add_column(0, '__KUMO_BATCH__', pa.array(batch, pa.int32()))


def _source(query: str) -> str:
    return re.search(r'`sch`\.`(\w+)`', query).group(1)


def _key(query: str, around: str) -> list[tuple]:
    table = _TABLES[_source(query)]
    names = re.findall(r'`(\w+)`', re.search(around, query).group(1))
    return list(zip(*(table[name].to_pylist() for name in names), strict=True))


def _count_key(select: str) -> tuple[int, int, int, int]:
    r"""Answer one ``SELECT`` of a key count from the tables above."""
    keys = _key(select, r'COUNT\(DISTINCT ([^)]*)\)')
    present = [key for key in keys if None not in key]
    label = int(re.match(r'SELECT (\d+) AS k', select).group(1))
    return label, len(keys), len(set(present)), len(keys) - len(present)


def _repeated_key(query: str) -> pa.Table:
    keys = _key(query, r'GROUP BY (.*)\n')
    names = re.findall(r'`(\w+)`', re.search(r'GROUP BY (.*)\n', query)[1])
    repeated = [key for key in dict.fromkeys(keys) if keys.count(key) > 1]
    return _rows(names, repeated[:1])


@pytest.fixture
def session() -> _Session:
    return _Session()


@pytest.fixture
def graph(session: _Session) -> Graph:
    return Graph.from_databricks(
        connection=SparkConnection(session),
        tables=['users', 'orders'],
        catalog='cat',
        schema='sch',
        verbose=False,
    )


def test_a_spark_session_builds_the_graph_from_unity_catalog(
    graph: Graph,
) -> None:
    assert graph['users'].primary_key.name == 'user_id'
    assert graph['orders'].primary_key.name == 'order_id'
    assert graph['orders'].time_column.name == 'ts'
    assert [tuple(edge) for edge in graph.edges] == [
        ('orders', 'user_id', 'users')
    ]


def test_every_name_reaches_spark_as_a_bound_value(
    session: _Session, graph: Graph
) -> None:
    looked_up = [
        args for query, args in session.calls if 'information_schema' in query
    ]
    assert looked_up
    assert all(args[0] == 'sch' for args in looked_up)
    for query, _ in session.calls:
        assert "'sch'" not in query
        assert "'users'" not in query


def test_the_sampler_runs_through_the_session(
    session: _Session, graph: Graph
) -> None:
    model = KumoRelational(graph, verbose=False)
    sampler = model._sampler
    assert isinstance(sampler, DatabricksSampler)
    assert sampler.num_rows_dict == {'users': 3, 'orders': 4}

    before = len(session.calls)
    df, batch = sampler._by_pkey(
        'users', pd.Series([3, 1]), columns={'user_id', 'age'}
    )
    _, args = session.calls[before]

    assert args == ['[3,1]']
    assert df['user_id'].tolist() == [3, 1]
    assert df['age'].tolist() == [27.0, 31.0]
    assert batch.tolist() == [0, 1]


def test_neighbours_come_back_in_the_order_they_were_chosen(
    session: _Session, graph: Graph
) -> None:
    """Spark and a warehouse read rows in different orders, so neighbours keep
    the order that chose them.
    """
    graph['orders'].primary_key = None
    sampler = KumoRelational(graph, verbose=False)._sampler

    before = len(session.calls)
    sampler._by_fkey(
        'orders',
        'user_id',
        pd.Series([1]),
        num_neighbors=2,
        anchor_time=None,
        columns={'user_id', 'ts'},
    )
    query, _ = session.calls[before]

    refs = sampler.table_column_ref_dict['orders']
    order_by = sampler._neighbor_order_by('orders', refs['ts'], refs['user_id'])
    assert query.rstrip().endswith(f'ORDER BY TMP.__KUMO_BATCH__, {order_by}')
    assert all(ref in order_by for ref in refs.values())


def test_the_time_bounds_of_a_window_are_bound_values(
    session: _Session, graph: Graph
) -> None:
    sampler = KumoRelational(graph, verbose=False)._sampler
    anchor = pd.Series(pd.to_datetime(['2024-02-01', '2024-03-01']))

    before = len(session.calls)
    sampler._by_time(
        'orders',
        'user_id',
        pd.Series([1, 2]),
        anchor_time=anchor,
        min_offset=pd.DateOffset(days=-30),
        max_offset=pd.DateOffset(days=0),
        columns={'user_id', 'ts'},
    )
    sampler._by_fkey(
        'orders',
        'user_id',
        pd.Series([1, 2]),
        num_neighbors=2,
        anchor_time=anchor,
        columns={'user_id', 'ts'},
    )
    windows = [
        (query, args)
        for query, args in session.calls[before:]
        if 'posexplode' in query
    ]

    assert len(windows) == 2
    by_time, by_fkey = windows
    assert by_time[1][1:] == ['2024-03-01 00:00:00', '2024-01-02 00:00:00']
    assert by_fkey[1][1] == '2024-03-01 00:00:00'
    for query, args in windows:
        assert len(args) == 3
        assert "'2024" not in query
        assert query.count('?') == 3


def test_every_key_is_counted_in_one_statement(
    session: _Session, graph: Graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        _TABLES, 'users', pa.concat_tables([_USERS, _USERS.slice(1, 1)])
    )
    before = len(session.calls)

    counts = count_keys(
        [(graph['users'], ['user_id']), (graph['orders'], ['user_id', 'ts'])]
    )

    counted, shown = (query for query, _ in session.calls[before:])
    assert counted == (
        'SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT `user_id`) AS d, '
        'COUNT(CASE WHEN `user_id` IS NULL THEN 1 END) AS z\n'
        'FROM `cat`.`sch`.`users`\n'
        'UNION ALL\n'
        'SELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT `user_id`, `ts`) AS d, '
        'COUNT(CASE WHEN `user_id` IS NULL OR `ts` IS NULL THEN 1 END) AS z\n'
        'FROM `cat`.`sch`.`orders`'
    )
    assert shown == (
        'SELECT `user_id`\nFROM `cat`.`sch`.`users`\n'
        'WHERE `user_id` IS NOT NULL\nGROUP BY `user_id`\n'
        'HAVING COUNT(*) > 1\nLIMIT 1'
    )
    # Databricks enforces no key, so a declared one is counted too.
    assert counts == [
        KeyCount(rows=4, distinct=3, nulls=0, enforced=False, example=2),
        KeyCount(rows=4, distinct=4, nulls=0, enforced=False),
    ]


def test_an_id_two_rows_hold_is_refused_where_it_is_scored(
    graph: Graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        _TABLES, 'users', pa.concat_tables([_USERS, _USERS.slice(1, 1)])
    )
    sampler = KumoRelational(graph, verbose=False)._sampler
    columns = {'user_id', 'age'}

    with pytest.raises(ValueError, match=r'Cannot score 2: more than one row'):
        sampler._by_pkey(
            'users', pd.Series([1, 2]), columns, scored=np.array([1])
        )
    # Reached only through a link, it takes one of them, as it always has.
    df, batch = sampler._by_pkey(
        'users', pd.Series([1, 2]), columns, scored=np.array([0])
    )
    assert df['user_id'].tolist() == [1, 2]
    assert batch.tolist() == [0, 1]


def test_the_session_route_needs_no_sql_connector() -> None:
    """A notebook samples through its own session, which needs no
    databricks-sql-connector, so every test above runs again where the
    connector cannot be imported.
    """
    script = (
        'import sys\n'
        "sys.modules['databricks.sql'] = None\n"
        'import pytest\n'
        'sys.exit(pytest.main(sys.argv[1:]))\n'
    )
    here = Path(__file__)
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            script,
            str(here),
            '-q',
            '-p',
            'no:cacheprovider',
            '--color=no',
            '-k',
            'not needs_no_sql_connector',
        ],
        capture_output=True,
        text=True,
        cwd=here.parents[3],
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert re.search(r'\b[1-9][0-9]* passed\b', output), output
    assert 'skipped' not in output, output
