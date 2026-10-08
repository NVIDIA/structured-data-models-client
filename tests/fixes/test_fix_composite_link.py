# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.link: a multi-column link is checked by asking the parent.

``DESIGN-composite-link.md`` A1, A2 and the guards, through the engine's
``Graph.link`` as a user calls it (``sdm_client`` refuses composite
relationships, so only the engine API reaches the check):

* A1: on DuckDB, SQLite and PostgreSQL, a 3,000-row parent whose child
  references only parents outside the parent's own 1,000-row sample links.
  Today it is refused, every time, because the check compared two samples.
* A2: on all five warehouse backends, linking sends one lookup of the parent:
  one statement naming the parent, with ``LIMIT 1``, the parent's fold
  expression (the one its derived key column holds, which the prediction
  joins on), no sampling clause, and one parameter, a JSON array of at most
  1,000 distinct values, sorted, each the fold of a child row of the child's
  sample with no null part; two builds send the same statement. Databricks
  and Snowflake run over the stub drivers of ``_warehouse_stub.py``.
* Guards, which hold before and after: on all three local databases, with
  parents over 1,000 rows, columns named in the wrong order and values the
  parent doesn't hold are refused; a null part in the child's reference is
  refused, as Adi decided it keeps being.

A4, the matrix over large PostgreSQL tables, is the opt-in script
``qa/composite_link_matrix.py`` (``CONTRACT.md``); this only checks it is
there and compiles.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from _fixes import REPO, postgres_dsn, statements

REGIONS = ('east', 'north', 'south')
KEY = ('region', 'code')


def _parent() -> pd.DataFrame:
    r"""3,000 parents, keyed by ``(region, code)``, stored shuffled."""
    rows = [
        (region, f'c{number:04d}', f'seg{number % 7}')
        for number in range(1000)
        for region in REGIONS
    ]
    order = sorted(range(len(rows)), key=lambda i: (i * 7919) % len(rows))
    frame = pd.DataFrame(
        [rows[i] for i in order], columns=['region', 'code', 'segment']
    )
    return frame


def _child(pairs: list[tuple[Any, Any]]) -> pd.DataFrame:
    r"""Two rows per referenced pair."""
    rows = []
    for index, (region, code) in enumerate(pairs * 2):
        rows.append((index + 1, region, code, float(index % 11)))
    return pd.DataFrame(rows, columns=['child_id', 'region', 'code', 'amount'])


def _fold(frame: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    from sdm_relational_engine.rfm.base import composite_key

    return list(composite_key.encode_frame(frame, list(names)))


# Databases ##################################################################


class Local:
    r"""One local database holding a parent and, later, a child."""

    def __init__(self, backend: str, root: Path) -> None:
        self.backend = backend
        self.root = root
        self.schema = f'fix_link_{uuid.uuid4().hex[:10]}'
        self.path = root / f'{backend}.db'
        if backend == 'postgres':
            import psycopg

            self.dsn = postgres_dsn()
            with psycopg.connect(self.dsn, autocommit=True) as connection:
                connection.execute(f'CREATE SCHEMA "{self.schema}"')

    def write(self, name: str, frame: pd.DataFrame) -> None:
        if self.backend == 'duckdb':
            import duckdb

            connection = duckdb.connect(str(self.path))
            connection.register('frame', frame)
            connection.execute(f'CREATE TABLE {name} AS SELECT * FROM frame')
            connection.close()
        elif self.backend == 'sqlite':
            connection = sqlite3.connect(self.path)
            frame.to_sql(name, connection, index=False)
            connection.commit()
            connection.close()
        else:
            import psycopg

            types = {
                'int64': 'BIGINT',
                'float64': 'DOUBLE PRECISION',
            }
            columns = ', '.join(
                f'{c} {types.get(str(t), "TEXT")}'
                for c, t in frame.dtypes.items()
            )
            with psycopg.connect(self.dsn, autocommit=True) as connection:
                connection.execute(
                    f'CREATE TABLE "{self.schema}".{name} ({columns})'
                )
                with connection.cursor() as cursor:
                    cursor.executemany(
                        f'INSERT INTO "{self.schema}".{name} VALUES '
                        f'({", ".join(["%s"] * frame.shape[1])})',
                        [
                            tuple(None if pd.isna(v) else v for v in row)
                            for row in frame.astype(object).itertuples(
                                index=False, name=None
                            )
                        ],
                    )

    def graph(self, tables: list[str]) -> Any:
        import sdm_relational_engine.rfm as rfm

        if self.backend == 'duckdb':
            return rfm.Graph.from_duckdb(
                str(self.path),
                tables=tables,
                infer_metadata=False,
                verbose=False,
            )
        if self.backend == 'sqlite':
            return rfm.Graph.from_sqlite(
                str(self.path),
                tables=tables,
                infer_metadata=False,
                verbose=False,
            )
        return rfm.Graph.from_postgres(
            self.dsn,
            tables=tables,
            schema=self.schema,
            infer_metadata=False,
            verbose=False,
        )

    def close(self) -> None:
        if self.backend == 'postgres':
            import psycopg

            with psycopg.connect(self.dsn, autocommit=True) as connection:
                connection.execute(
                    f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE'
                )


LOCAL = ('duckdb', 'sqlite', 'postgres')


@pytest.fixture(params=LOCAL)
def local(request: pytest.FixtureRequest) -> Iterator[Local]:
    with tempfile.TemporaryDirectory() as tmp:
        database = Local(request.param, Path(tmp))
        try:
            yield database
        finally:
            database.close()


def _keyed(graph: Any) -> Any:
    graph['parent'].primary_key = KEY
    graph['child'].primary_key = 'child_id'
    return graph


def _parent_sample(database: Local) -> set[str]:
    r"""The folded identities in the parent's own inference sample."""
    graph = database.graph(['parent'])
    graph['parent'].primary_key = KEY
    values = graph['parent']._sample_values(graph['parent'].primary_key.name)
    assert values is not None
    sample = set(values.dropna())
    assert len(sample) == 1000, len(sample)
    return sample


def _outside(database: Local, count: int) -> list[tuple[str, str]]:
    r"""``count`` parents the parent's sample doesn't hold."""
    sample = _parent_sample(database)
    parent = _parent()
    folds = _fold(parent, KEY)
    pairs = [
        (row.region, row.code)
        for row, fold in zip(parent.itertuples(), folds, strict=True)
        if fold not in sample
    ]
    assert len(pairs) == 2000
    return pairs[:count]


# A1 ##########################################################################


def test_a1_a_link_to_parents_outside_the_parents_sample_holds(
    local: Local,
) -> None:
    local.write('parent', _parent())
    local.write('child', _child(_outside(local, 600)))
    graph = _keyed(local.graph(['parent', 'child']))
    graph.link(src_table='child', fkey=KEY, dst_table='parent')
    assert [(e.src_table, e.dst_table) for e in graph.edges] == [
        ('child', 'parent')
    ]


# Guards ######################################################################


def test_guard_columns_in_the_wrong_order_are_refused(local: Local) -> None:
    local.write('parent', _parent())
    local.write('child', _child(_outside(local, 600)))
    graph = _keyed(local.graph(['parent', 'child']))
    with pytest.raises(ValueError, match='matches no rows'):
        graph.link(
            src_table='child', fkey=('code', 'region'), dst_table='parent'
        )
    assert not graph.edges


def test_guard_values_the_parent_does_not_hold_are_refused(
    local: Local,
) -> None:
    local.write('parent', _parent())
    local.write('child', _child([('west', f'c{n:04d}') for n in range(600)]))
    graph = _keyed(local.graph(['parent', 'child']))
    with pytest.raises(ValueError, match='matches no rows'):
        graph.link(src_table='child', fkey=KEY, dst_table='parent')
    assert not graph.edges


def test_guard_a_null_part_in_the_reference_is_refused(local: Local) -> None:
    local.write('parent', _parent())
    pairs: list[tuple[Any, Any]] = list(_outside(local, 600))
    pairs = [
        (region, None if index % 10 == 0 else code)
        for index, (region, code) in enumerate(pairs)
    ]
    local.write('child', _child(pairs))
    graph = _keyed(local.graph(['parent', 'child']))
    with pytest.raises(ValueError, match='null'):
        graph.link(src_table='child', fkey=KEY, dst_table='parent')
    assert not graph.edges


# A2 ##########################################################################


def _lookup(
    graph: Any, parent_name: str, child_name: str
) -> tuple[list[tuple[str, Any]], str, set[str]]:
    r"""The statements linking sends that name the parent, the parent's fold
    expression, and the folds of the child's sample without a null part."""
    child = graph[child_name]
    sample = child._source_sample_df
    names = [c for c in sample.columns if c.lower() in KEY]
    by_lower = {c.lower(): c for c in names}
    ordered = tuple(by_lower[k] for k in KEY)
    complete = sample.dropna(subset=list(ordered))
    folds = set(_fold(complete, ordered))
    parent = graph[parent_name]
    expression = parent[parent.primary_key.name].expr
    fold = getattr(expression, 'value', expression)
    assert isinstance(fold, str) and fold, (
        'the parent keys on no folded expression'
    )
    source = parent._quoted_source_name
    with statements() as log:
        graph.link(src_table=child_name, fkey=ordered, dst_table=parent_name)
    sent = [
        (sql, params)
        for sql, params in log
        if source in sql and '__kumo_order_' not in sql
    ]
    return sent, fold, folds


def _values(param: Any) -> list[Any]:
    assert isinstance(param, str), (
        f'the values are not one JSON text: {param!r}'
    )
    items = json.loads(param)
    assert isinstance(items, list), param
    out = []
    for item in items:
        if isinstance(item, dict):
            assert len(item) == 1, item
            item = next(iter(item.values()))
        elif isinstance(item, list):
            assert len(item) == 1, item
            item = item[0]
        out.append(item)
    return out


def _assert_one_lookup(
    sent: list[tuple[str, Any]], fold: str, folds: set[str]
) -> tuple[str, Any]:
    assert len(sent) == 1, [s for s, _ in sent]
    sql, params = sent[0]
    flat = ' '.join(sql.split())
    assert flat.upper().endswith('LIMIT 1'), sql
    assert ' '.join(fold.split()) in flat, f'no {fold!r} in {sql}'
    assert 'TABLESAMPLE' not in flat.upper(), sql
    assert ' SAMPLE ' not in flat.upper(), sql
    assert params is not None and len(params) == 1, params
    values = _values(params[0])
    assert 0 < len(values) <= 1000
    assert all(isinstance(v, str) for v in values), values[:5]
    assert values == sorted(set(values)), 'not sorted and distinct'
    assert set(values) <= folds, sorted(set(values) - folds)[:5]
    assert len(values) == min(1000, len(folds))
    return sql, params


@pytest.mark.parametrize('backend', LOCAL)
def test_a2_a_local_link_sends_one_lookup(backend: str) -> None:
    seen = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            database = Local(backend, Path(tmp))
            try:
                database.write('parent', _parent())
                database.write('child', _child(_outside(database, 700)))
                graph = _keyed(database.graph(['parent', 'child']))
                sent, fold, folds = _lookup(graph, 'parent', 'child')
                sql, params = _assert_one_lookup(sent, fold, folds)
                # Each build has a schema of its own.
                seen.append((sql.replace(database.schema, '<schema>'), params))
            finally:
                database.close()
    assert seen[0] == seen[1]


def _warehouse(kind: str) -> Any:
    import _warehouse_stub as stub
    import pyarrow as pa

    upper = kind == 'snowflake'

    def name(text: str) -> str:
        return text.upper() if upper else text

    parent = _parent()
    pairs = [(row.region, row.code) for row in parent.itertuples()][:700]
    child = _child(pairs)
    tables = {
        'parent': pa.Table.from_pandas(parent, preserve_index=False),
        'child': pa.Table.from_pandas(child, preserve_index=False),
    }
    warehouse = stub.Warehouse(
        kind=kind,
        tables=tables,
        keys={'child': 'child_id'},
        time={},
        links={},
    )
    import sdm_relational_engine.rfm as rfm

    source = stub.source(warehouse)
    if upper:
        graph = rfm.Graph.from_snowflake(
            source,
            tables=['PARENT', 'CHILD'],
            database='DB',
            schema='SHOP',
            infer_metadata=False,
            verbose=False,
        )
    else:
        graph = rfm.Graph.from_databricks(
            source,
            tables=['parent', 'child'],
            catalog='main',
            schema='shop',
            infer_metadata=False,
            verbose=False,
        )
    graph[name('parent')].primary_key = tuple(name(k) for k in KEY)
    graph[name('child')].primary_key = name('child_id')
    return graph, name('parent'), name('child')


@pytest.mark.parametrize('kind', ('databricks', 'snowflake'))
def test_a2_a_warehouse_link_sends_one_lookup(kind: str) -> None:
    seen = []
    for _ in range(2):
        graph, parent, child = _warehouse(kind)
        sent, fold, folds = _lookup(graph, parent, child)
        seen.append(_assert_one_lookup(sent, fold, folds))
        assert len(graph.edges) == 1
    assert seen[0] == seen[1]


# A4 ##########################################################################


def test_a4_the_matrix_is_an_opt_in_script() -> None:
    script = REPO / 'qa' / 'composite_link_matrix.py'
    assert script.is_file(), f'{script} is missing'
    compile(script.read_bytes(), str(script), 'exec')
