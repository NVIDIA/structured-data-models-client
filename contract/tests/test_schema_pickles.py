# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Schema`` pickles across the reference and the core route (T1.24).

Run where the contract tests run (CI's ``engine`` job, Python 3.12 with
pandas 3, the reference environment's):

    python -m pytest contract/tests -q

A ``Schema`` the reference client pickled (``contract/fixtures/pickles``,
written by its ``write.py`` from the reference commit) loads in this client
as the reference reads it back: its tables, each column's type, meaning,
what it meant before the schema joined on it and whether it was declared,
the keys, time columns and relationships, and an edit to it is still
checked. A schema of tables in their source holds its source, which no
client pickles: the core route's refuses as the old route's does, with the
Source's own refusal, and everything the core route adds to a schema's
state (the core's description of the source, the store of engine tables)
pickles and reloads with the rest of it.
"""

from __future__ import annotations

import importlib.util
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


def _writer() -> Any:
    spec = importlib.util.spec_from_file_location(
        'write_pickles', PICKLES / 'write.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert _writer().projection(loaded) == EXPECTED['schemas'][name]
    assert loaded._core_description is None
    assert loaded.source is None
    # Its relationships are still checked when edited.
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('orders', 'nothing', 'users', 'user_id')
        )
    # And it pickles again to what it was.
    again = pickle.loads(pickle.dumps(loaded))
    assert _writer().projection(again) == EXPECTED['schemas'][name]


def _shop(routes: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    from sdm_client import Schema, _routing
    from sdm_connectors import DuckDBSource

    if routes:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
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


def test_a_core_built_schema_pickles_as_the_old_route_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client import Relationship, Schema, StructuredDataError, _routing

    core = _shop(','.join(_routing.FROM_SOURCE_DUCKDB_TABLES), monkeypatch)
    assert core._core_description is not None
    refusals = {}
    routes = {'core': core}
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        routes['old'] = _shop('', monkeypatch)
        assert routes['old']._core_description is None
    for route, schema in routes.items():
        with pytest.raises(TypeError) as refused:
            pickle.dumps(schema)
        refusals[route] = str(refused.value)
    assert refusals['core'] == (
        'a Source holds an open connection, which cannot be pickled'
    )
    assert len(set(refusals.values())) == 1, refusals

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
    assert loaded._core_description == core._core_description
    assert type(loaded._engine_tables) is type(core._engine_tables)
    assert _writer().projection(loaded) == {
        **_writer().projection(core),
        'source': False,
    }
    assert loaded['buyers'].source_name == 'users'
    with pytest.raises(StructuredDataError):
        loaded.relationships.append(
            Relationship('orders', 'nothing', 'buyers', 'user_id')
        )
