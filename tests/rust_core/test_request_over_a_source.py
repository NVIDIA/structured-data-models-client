# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The fixture cases over a DuckDB source that the Rust runner cannot run,
run through a host instead.

The cases build a DuckDB database, which the Rust fixture runner cannot
read rows from, so they are excluded there
(``scripts/parity/request_fixture_exclusions.json``). Here the core runs them
through ``scripts/parity/request_host.py`` over a real DuckDB database:

- ``PredictiveQuery.materialize`` refuses rows given as ``predict=`` and the
  tabular model over a source schema before anything is read; the core's
  copies of the two refusals (``rust/crates/request/src/materialize.rs``)
  must give today's class, code and message.
- A column left out of a table in its source (``del schema[t][c]``) is not
  read or sent; the host leaves it out of the schema it holds, as a wrapper
  will, and the core must post today's bodies and give today's frame.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_relational_engine')
pytest.importorskip('sdm_core._native')
duckdb = pytest.importorskip('duckdb')

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts' / 'parity'

# Today's Python, which these hold the core to, runs public calls here.
if str(Path(__file__).resolve().parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _old_path import python_path  # noqa: E402,F401 - the fixture

pytestmark = pytest.mark.usefixtures('python_path')
COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'


def _host() -> Any:
    spec = importlib.util.spec_from_file_location(
        'request_host', SCRIPTS / 'request_host.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('request_host', module)
    spec.loader.exec_module(module)
    return module


def _shop() -> dict[str, Any]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from _fixture_messages import shop

    return shop()


@pytest.fixture
def source() -> Iterator[Any]:
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute('SET threads TO 1')
    for name, frame in _shop().items():
        connection.register(f'{name}_frame', frame)
        connection.execute(f'CREATE TABLE {name} AS SELECT * FROM {name}_frame')
        connection.unregister(f'{name}_frame')
    yield DuckDBSource(connection)
    connection.close()


def _client(session: Any) -> Any:
    from sdm_client import SDMClient

    client = SDMClient(url='http://localhost:8000')
    client._transport._session.close()
    client._transport._session = session
    return client


def _refusal(error: BaseException) -> tuple[str, Any, str]:
    cls = type(error)
    return (
        f'{cls.__module__}.{cls.__qualname__}',
        getattr(error, 'code', None),
        getattr(error, 'message', str(error)),
    )


@pytest.mark.parametrize(
    ('args', 'said'),
    [
        pytest.param(
            {'predict': 'users', 'model': 'kumo-relational'},
            'predict= carries rows',
            id='rows-given-as-predict',
        ),
        pytest.param(
            {'indices': [1], 'model': 'kumo-tabular'},
            'sampled with SQL',
            id='the-tabular-model',
        ),
    ],
)
def test_the_core_refuses_as_today(
    source: Any, args: dict[str, Any], said: str
) -> None:
    from sdm_client import PredictiveQuery, Schema

    given = dict(args)
    if 'predict' in given:
        given['predict'] = _shop()[given['predict']]

    with pytest.raises(Exception) as today:
        PredictiveQuery(COUNT, Schema.from_source(source)).materialize(**given)

    host = _host()
    schema = host.schema_from_source(source)
    with pytest.raises(Exception) as core:
        host.materialize(host.predictive_query(COUNT, schema), **given)

    assert said in _refusal(today.value)[2]
    assert _refusal(core.value) == _refusal(today.value)


def test_a_column_left_out_of_a_table_in_its_source(source: Any) -> None:
    import pandas as pd
    from _parity.nim import SyntheticSession, normalized
    from sdm_client import PredictiveQuery, Schema

    tables = {'users': 'users', 'orders': 'orders'}
    gather = {'indices': [1, 2], 'num_context_samples': 6, 'random_seed': 7}

    schema = Schema.from_source(source, tables=tables)
    del schema['users']['country']
    today = SyntheticSession()
    frame = (
        _client(today)
        .predict(
            PredictiveQuery(COUNT, schema).materialize(**gather),
            model='kumo-relational',
        )
        .to_pandas()
    )

    host = _host()
    held = host.schema_from_source(source, tables=tables)
    for table in held.state['schema']['tables']:
        if table['name'] == 'users':
            table['columns'] = [
                c for c in table['columns'] if c['name'] != 'country'
            ]
    core = SyntheticSession()
    answer = host.Client(_client(core)._transport).predict(
        host.materialize(host.predictive_query(COUNT, held), **gather),
        model='kumo-relational',
    )

    sent = [normalized(body) for body in today.payloads]
    assert sent and 'country' not in str(sent)
    assert [normalized(body) for body in core.payloads] == sent
    pd.testing.assert_frame_equal(host.frame(host.flatten(answer)), frame)


#: The column `composite_key` derives for the key ['order_id', 'user_id'].
DERIVED = '__kumo_key_6f726465725f6964_757365725f6964'


@pytest.fixture
def holding_the_derived_name() -> Iterator[Any]:
    r"""``shop()`` in DuckDB, its ``orders`` holding a text column named
    ``DERIVED``: ``'x'`` and the order's id."""
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute('SET threads TO 1')
    for name, frame in _shop().items():
        connection.register(f'{name}_frame', frame)
        connection.execute(f'CREATE TABLE {name} AS SELECT * FROM {name}_frame')
        connection.unregister(f'{name}_frame')
    connection.execute(f'ALTER TABLE orders ADD COLUMN "{DERIVED}" VARCHAR')
    connection.execute(f'UPDATE orders SET "{DERIVED}" = \'x\' || order_id')
    yield DuckDBSource(connection)
    connection.close()


def test_a_composite_key_whose_derived_name_its_source_holds(
    holding_the_derived_name: Any,
) -> None:
    r"""A source table holding a column named as the engine names the column
    it derives for a composite key, left out of the schema
    (``del schema[t][c]``), and the table then keyed on the two columns that
    name derives from: the engine folds the key to that name, finds the
    source already holds it, and the gather is refused for the identity the
    warehouse produced (``rfm/base/table.py`` ``_derive_composite_key``).

    The two columns are typed ``ID`` already, so the host sets the key on
    the schema it holds as ``Table.primary_key`` does."""
    from sdm_client import PredictiveQuery, Schema

    source = holding_the_derived_name
    key = ['order_id', 'user_id']

    def today() -> Any:
        schema = Schema.from_source(source)
        del schema['orders'][DERIVED]
        schema['orders'].primary_key = key
        assert all(schema['orders'].columns[n].stype == 'ID' for n in key)
        return PredictiveQuery(COUNT, schema).materialize(indices=[1])

    host = _host()

    def core() -> Any:
        schema = host.schema_from_source(source)
        for table in schema.state['schema']['tables']:
            if table['name'] == 'orders':
                table['columns'] = [
                    c for c in table['columns'] if c['name'] != DERIVED
                ]
                assert all(
                    c['stype'] == 'ID'
                    for c in table['columns']
                    if c['name'] in key
                )
                table['primary_key'] = list(key)
        return host.materialize(
            host.predictive_query(COUNT, schema), indices=[1]
        )

    def outcome(make: Any) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                make()
            except Exception as error:  # noqa: BLE001 - compared
                got: Any = (
                    *_refusal(error),
                    dict(getattr(error, 'details', {}) or {}),
                )
            else:
                got = None
        return got, [(w.category.__name__, str(w.message)) for w in caught]

    want, said = outcome(today)
    got, given = outcome(core)
    assert want is not None and 'folds' in want[2] and "['x1']" in want[2], want
    assert (got, given) == (want, said)
