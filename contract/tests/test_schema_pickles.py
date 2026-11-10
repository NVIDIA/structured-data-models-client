# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen schema-pickle compatibility and current schema state round trips."""

from __future__ import annotations

import io
import pickle
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PICKLES = ROOT / 'contract/fixtures/pickles'


def projection(schema: Any) -> dict[str, Any]:
    r"""What a schema says, as the reference and the candidate both read it."""
    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'kind': type(table).__name__,
                'columns': [
                    [
                        column,
                        c.dtype,
                        c.stype,
                        c._stype_before_key,
                        c.stype_declared,
                    ]
                    for column, c in table.columns.items()
                ],
                'primary_key': key
                if key is None or isinstance(key, str)
                else list(key),
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
                'rows': None
                if getattr(table, 'data', None) is None
                else len(table.data),
            }
        )
    return {
        'tables': tables,
        'links': [
            [
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
            ]
            for link in schema.relationships
        ],
        'source': schema.source is not None,
    }


NAMES = ('one-table', 'two-tables', 'edited')


def assert_saved_schema(schema: Any, name: str) -> None:
    """Check authored declaration/edit semantics, without a projection snapshot."""
    assert schema.source is None
    assert list(schema.tables) == (
        ['orders'] if name == 'one-table' else ['users', 'orders']
    )
    for table_name, table in schema.tables.items():
        assert type(table).__name__ == 'Table'
        assert table.duplicate_keys == 0
        assert len(table.data) == (6 if table_name == 'users' else 12)
        assert table.time_column == (
            'ts' if table_name == 'orders' and name != 'edited' else None
        )
        assert table.primary_key == (
            'order_id'
            if table_name == 'orders'
            else 'age'
            if name == 'edited'
            else 'user_id'
        )
        columns = (
            [('user_id', 'int64'), ('age', 'int64'), ('score', 'float64')]
            if table_name == 'users'
            else [
                ('order_id', 'int64'),
                ('user_id', 'int64'),
                ('price', 'float64'),
                ('ts', 'timestamp[us]'),
            ]
        )
        assert list(table.columns) == [column for column, _ in columns]
        for column, dtype in columns:
            c = table.columns[column]
            assert c.dtype == dtype
            key = column == table.primary_key or (
                table_name == 'orders'
                and column == 'user_id'
                and name == 'two-tables'
            )
            assert c.stype == (
                'ID' if key else 'timestamp' if column == 'ts' else 'numerical'
            )
            assert c._stype_before_key == (
                'categorical'
                if name == 'edited' and column == 'age'
                else 'numerical'
                if key
                else None
            )
            assert c.stype_declared is (
                column == 'ts' or (name == 'edited' and column == 'age')
            )
    links = [
        (
            link.source_table,
            link.source_column,
            link.target_table,
            link.target_column,
        )
        for link in schema.relationships
    ]
    assert links == (
        [('orders', 'user_id', 'users', 'user_id')]
        if name == 'two-tables'
        else []
    )


def test_historical_schema_pickle_inventory():
    assert {path.stem for path in PICKLES.glob('*.pkl')} == set(NAMES)
    assert all(
        (PICKLES / f'{name}.pkl').read_bytes().startswith(b'\x80\x04')
        for name in NAMES
    )


@pytest.mark.parametrize('name', NAMES)
def test_historical_schema_preserves_declarations_and_edits(name: str) -> None:
    # Historical frames were written with pandas 3.
    assert int(pd.__version__.split('.')[0]) >= 3
    from sdm_client import Relationship, Schema, StructuredDataError

    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        loaded = pickle.loads((PICKLES / f'{name}.pkl').read_bytes())
    assert isinstance(loaded, Schema)
    assert [
        w for w in shown if not issubclass(w.category, ResourceWarning)
    ] == []
    assert_saved_schema(loaded, name)
    assert not _described_on_the_core(loaded)
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('orders', 'nothing', 'users', 'user_id')
        )
    assert_saved_schema(pickle.loads(pickle.dumps(loaded)), name)


def _described_on_the_core(schema: Any) -> bool:
    from sdm_client.schema import _described_on_the_core as described

    return described(schema)


def _facts(schema: Any) -> dict[tuple[str, str], Any]:
    r"""What the source said of each column, as the schema keeps it on the
    column."""
    return {
        (name, column_name): column._source
        for name, table in schema.tables.items()
        for column_name, column in table.columns.items()
    }


def _shop() -> Any:
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT)'
    )
    connection.execute('INSERT INTO users SELECT i, i % 50 FROM range(9) t(i)')
    connection.execute(
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
        'price DOUBLE, ts TIMESTAMP)'
    )
    connection.execute(
        "INSERT INTO orders SELECT i, i % 9, i * 1.5, TIMESTAMP '2024-01-01' "
        '+ to_days(CAST(i AS INTEGER)) FROM range(30) t(i)'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            DuckDBSource(connection), {'buyers': 'users', 'orders': 'orders'}
        )


def test_source_backed_schema_refuses_pickle_but_detached_state_round_trips() -> (
    None
):
    from sdm_client import Relationship, Schema, StructuredDataError

    core = _shop()
    assert _described_on_the_core(core)
    with pytest.raises(
        TypeError,
        match='a Source holds an open connection, which cannot be pickled',
    ):
        pickle.dumps(core)

    # The rest of the core-built schema's state, its source left out as a
    # schema pickled before it could have one was written, pickles and
    # reloads: the core's description and the store of engine tables
    # among it.
    state = {
        key: value for key, value in vars(core).items() if key != '_source'
    }

    class _WithoutTheSource(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if obj is core:
                return object.__new__, (Schema,), state
            return NotImplemented

    written = io.BytesIO()
    _WithoutTheSource(written).dump(core)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        loaded = pickle.loads(written.getvalue())
    assert [
        w for w in shown if not issubclass(w.category, ResourceWarning)
    ] == []
    assert _facts(loaded) == _facts(core)
    assert type(loaded._engine_tables) is type(core._engine_tables)
    assert projection(loaded) == {
        **projection(core),
        'source': False,
    }
    assert loaded['buyers'].source_name == 'users'
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('orders', 'nothing', 'buyers', 'user_id')
        )


def test_an_edited_core_built_schema_round_trips_its_state() -> None:
    #: the edits a core-built schema took are its
    # state, the core's last answer among it; reloaded, the schema is the
    # edited one, and its next edit is the core's again.
    from sdm_client import Relationship, Schema, StructuredDataError

    core = _shop()
    core['buyers']['age'].stype = 'categorical'
    core['orders'].time_column = None
    core['orders'].primary_key = ('order_id',)
    del core['orders']['price']
    core.relationships = [
        Relationship('orders', 'user_id', 'buyers', 'user_id')
    ]
    state = {
        key: value for key, value in vars(core).items() if key != '_source'
    }

    class _WithoutTheSource(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if obj is core:
                return object.__new__, (Schema,), state
            return NotImplemented

    written = io.BytesIO()
    _WithoutTheSource(written).dump(core)
    loaded = pickle.loads(written.getvalue())
    assert _facts(loaded) == _facts(core)
    assert projection(loaded) == {
        **projection(core),
        'source': False,
    }
    assert loaded['buyers']['age'].stype_declared is True
    assert loaded['orders'].primary_key == ('order_id',)
    assert 'price' not in loaded['orders'].columns
    with pytest.raises(StructuredDataError) as refused:
        loaded['orders'].time_column = 'user_id'
    assert refused.value.details['field'] == 'time_column'
    loaded['orders'].time_column = 'ts'
    assert loaded['orders'].time_column == 'ts'
