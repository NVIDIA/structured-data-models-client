# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen schema-pickle compatibility and current schema state round trips."""

from __future__ import annotations

import io
import json
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


EXPECTED = json.loads((PICKLES / 'expected.json').read_text(encoding='utf-8'))


def test_every_pickle_the_reference_wrote_is_listed() -> None:
    written = {path.stem for path in PICKLES.glob('*.pkl')}
    assert (
        written
        == set(EXPECTED['schemas'])
        == {
            'one-table',
            'two-tables',
            'edited',
        }
    )


@pytest.mark.parametrize('name', sorted(EXPECTED['schemas']))
def test_a_schema_the_reference_pickled_loads_as_the_reference_reads_it(
    name: str,
) -> None:
    # A pickled frame loads in its own pandas or a later one: the
    # reference wrote these with pandas 3, which the contract tests run on.
    assert int(pd.__version__.split('.')[0]) >= int(
        EXPECTED['pandas'].split('.')[0]
    )
    from sdm_client import Relationship, Schema, StructuredDataError

    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        loaded = pickle.loads((PICKLES / f'{name}.pkl').read_bytes())
    assert isinstance(loaded, Schema)
    # A copy is not a new declaration: loading says nothing.
    assert [
        w for w in shown if not issubclass(w.category, ResourceWarning)
    ] == []
    assert projection(loaded) == EXPECTED['schemas'][name]
    assert not _described_on_the_core(loaded)
    assert loaded.source is None
    # Its relationships are still checked when edited.
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('orders', 'nothing', 'users', 'user_id')
        )
    # And it pickles again to what it was.
    again = pickle.loads(pickle.dumps(loaded))
    assert projection(again) == EXPECTED['schemas'][name]


def _described_on_the_core(schema: Any) -> bool:
    from sdm_client.schema import _described_on_the_core as described

    return described(schema)


def _facts(schema: Any) -> dict[tuple[str, str], Any]:
    r"""What the source said of each column, as the schema keeps it on the
    column (T3.16)."""
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
    # T1.24b: the edits a core-built schema took (decision E67) are its
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
