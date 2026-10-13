# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``PredictiveQuery.materialize`` on the core, over a DuckDB source (T1.11c).

Forced onto the core (``SDM_CORE_ROUTES``, with the schema's route), the
call's gathering runs on the core's ``materialize``: the request it builds
for ``base.regression`` is the one the old route builds in the same
process, payload for payload; a seed of ``2**70`` crosses from Python into
the core as that exact integer, which DuckDB's data layer refuses by name
(register entry RS-7); an option or a schema the core does not take yet is
refused by name; nothing of the relational engine is imported on the
route, which a test-only import hook in a process of its own shows. The
reference comparison of the payload is ``contract/tests``'s.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    PredictiveQuery,
    Request,
    Schema,
    StructuredDataError,
    _routing,
)
from sdm_client._rust import codec
from sdm_client._rust import materialize as _rust_materialize
from sdm_client._rust import run as _rust_run
from sdm_client._rust import schema as _rust_schema
from sdm_client._rust.run import run
from sdm_client.errors import MissingExtraError

ROOT = Path(__file__).resolve().parents[3]
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
ANCHOR = pd.Timestamp('2024-03-01')


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _shop() -> list[str]:
    r"""The ``shop`` base of ``contract/bases.yaml``, read without a YAML
    library: its statements are the plain list items under its name."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def _connection(*setup: str) -> Any:
    connection = duckdb.connect()
    for statement in [*_shop(), *setup]:
        connection.execute(statement)
    return connection


def _materialize(
    *setup: str,
    query: str = QUERY,
    indices: list[Any] | None = None,
    after: tuple[str, ...] = (),
    **options: Any,
) -> Request:
    r"""``query`` materialized over the ``shop`` base and ``setup``; the
    statements ``after`` run once the schema is described."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection(*setup)
        schema = Schema.from_source(DuckDBSource(connection))
        for statement in after:
            connection.execute(statement)
        return PredictiveQuery(query, schema).materialize(
            indices=[0, 1] if indices is None else indices,
            anchor_time=ANCHOR,
            **options,
        )


def _both_routes(
    monkeypatch: pytest.MonkeyPatch, *setup: str, **arguments: Any
) -> dict[str, Any]:
    r"""The core's payload for the call, held equal to the old route's in
    the same process where the engine is installed (CI's ``engine`` job);
    where it is not, the old route's refusal is held instead."""
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _materialize(*setup, **arguments)._payloads[0]
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is None:
        with pytest.raises(MissingExtraError):
            _materialize(*setup, **arguments)
    else:
        old = _materialize(*setup, **arguments)._payloads[0]
        assert json.loads(json.dumps(core)) == json.loads(
            json.dumps(old, default=str)
        )
    return core


def _context_row(payload: dict[str, Any], instance: int) -> list[Any]:
    rows = payload['context']['instance_table']['rows']
    [row] = [row for row in rows if row[0] == instance]
    return row


def test_the_route_takes_a_duckdb_source_s_materialize_paths_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(_routing.MATERIALIZE_DUCKDB) <= _routing.CORE_ROUTED
    assert len(_routing.MATERIALIZE_DUCKDB) == 6
    # Forcing some of the call's paths and not the others fails the call.
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_DUCKDB)
        + ','
        + _routing.MATERIALIZE_DUCKDB[2],
    )
    with pytest.raises(_routing.RouteError):
        _materialize()


def _old_route_needs_the_engine() -> bool:
    r"""Whether the relational engine, which the old route gathers with, is
    missing. CI's ``test`` job installs none; its ``engine`` job, which runs
    this suite again, does. Without it the old route refuses, which is held
    instead."""
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        return False
    with pytest.raises(MissingExtraError):
        _materialize()
    return True


def test_the_core_builds_the_request_the_old_route_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _old_route_needs_the_engine():
        return
    old = _materialize()
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _materialize()
    assert 'sdm_client._native' in sys.modules
    assert len(core._payloads) == len(old._payloads) == 1
    # Through JSON, as each is sent: the engine's numbers are numpy's.
    assert json.loads(json.dumps(core._payloads[0])) == json.loads(
        json.dumps(old._payloads[0], default=str)
    )
    assert core.task == old.task
    assert core.entity_table == old.entity_table == 'users'
    assert core.indices == old.indices == [0, 1]
    # The same seed gives the same request, another seed another order.
    again = _materialize(random_seed=42)
    assert again._payloads == core._payloads
    other = _materialize(random_seed=7)
    assert other._payloads != core._payloads


def test_the_regression_s_statistic_is_recorded_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _old_route_needs_the_engine():
        return
    old = _materialize(prediction_statistic='mean')
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _materialize(prediction_statistic='mean')
    assert core._payloads[0]['output'] == old._payloads[0]['output']


@pytest.mark.parametrize(
    ('options', 'rule'),
    [
        ({'num_neighbors': [8, 8]}, 'num_neighbors'),
        # A sequence JSON does not know, written as its items (fix round 1,
        # the subagent's P2-1), so refused by name as a list is.
        ({'num_neighbors': range(2)}, 'num_neighbors'),
        ({'context_anchor_time': ANCHOR}, 'context_anchor_time'),
        ({'anchor_time': 'entity'}, 'anchor_entity'),
        # A seed given as None, which the reference reads as unseeded.
        ({'random_seed': None}, 'unseeded'),
        # The first in name order, before the seed, as the bridge refused
        # them until T1.24.
        (
            {'top_k': 3, 'num_hops': 2, 'random_seed': None},
            'num_hops',
        ),
    ],
)
def test_what_the_core_does_not_take_yet_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], rule: str
) -> None:
    # Every option passes through to the core, which refuses by name what
    # it does not take (T1.24, decision E61): the bridge refuses none of
    # its own, so the refusal is the core's, mapped by the host.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        arguments = {'indices': [0, 1], 'anchor_time': ANCHOR, **options}
        with pytest.raises(StructuredDataError) as refused:
            PredictiveQuery(QUERY, schema).materialize(**arguments)
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert type(refused.value).__name__ == 'UnsupportedRequestError'
    assert refused.value.details['operation'] == 'materialize'
    assert refused.value.details['rule'] == rule


@pytest.mark.parametrize(
    'given',
    [['num_estimators'], object()],
    ids=['a-list', 'not-json'],
)
def test_inference_options_not_a_mapping_are_refused(
    monkeypatch: pytest.MonkeyPatch, given: Any
) -> None:
    # The core reads the call's values as the caller gave them (T1.28a):
    # inference options that are not a mapping are refused, INVALID_REQUEST,
    # by the core where JSON holds them and by the host's encoder where it
    # does not, as the client's transport refuses such a value. The
    # reference takes its engine's own config object here (register P1B-3).
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as refused:
        _materialize(inference_config=given)
    assert refused.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize(
    'depth', [500, 100_000], ids=['past-the-core', 'past-json']
)
def test_inference_options_nested_deep_are_refused_never_a_crash(
    monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    # E49's depth probe of the inference options' reader (T1.28a): nested
    # past the core's reader, the core refuses them by name; past what
    # Python's encoder writes, the host does, as its transport refuses a
    # value it cannot send; neither a crash nor a RecursionError.
    nested: Any = 1
    for _ in range(depth):
        nested = [nested]
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as refused:
        _materialize(inference_config={'num_estimators': nested})
    assert refused.value.code == 'INVALID_REQUEST'
    if depth == 500:
        assert refused.value.details['field'] == 'values'
        assert refused.value.details['category'] == 'depth'


#: Tables named otherwise than the source names them (T1.24): the query
#: and the payload use the schema's names, the reads the source's.
NAMED = {'buyers': 'users', 'purchases': 'orders'}


def _named(**options: Any) -> tuple[Request, Schema]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # The key's guess is by the schema's name of the table, as the
        # engine's, so both are named.
        schema = Schema.from_source(
            DuckDBSource(_connection()),
            NAMED,
            primary_keys={'buyers': 'user_id', 'purchases': 'order_id'},
        )
        request = PredictiveQuery(
            'PREDICT SUM(purchases.price, 0, 30, days) FOR EACH buyers.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=ANCHOR, **options)
    return request, schema


def test_tables_named_are_sampled_as_the_old_route_samples_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            _routing.FROM_SOURCE_DUCKDB_TABLES + _routing.MATERIALIZE_DUCKDB
        ),
    )
    core, schema = _named()
    assert schema._core_description is not None
    assert {
        t['name']: t['source'] for t in schema._core_description['tables']
    } == NAMED
    payload = core._payloads[0]
    assert set(payload['context']['related_tables']) == {'buyers', 'purchases'}
    assert core.entity_table == 'buyers'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    old, _ = _named()
    assert json.loads(json.dumps(payload)) == json.loads(
        json.dumps(old._payloads[0], default=str)
    )
    assert core.task == old.task


def test_a_schema_not_described_on_the_core_has_no_core_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    # As a schema the old route described: no description of the core's.
    schema._core_description = None
    with pytest.raises(_routing.RouteError):
        PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )


def test_a_seed_of_2_to_the_70_reaches_the_core_as_that_exact_integer() -> None:
    seed = 2**70
    assert codec.seed(seed) == '0x400000000000000000'
    # A host provider, so the seed reaches the sample request unrefused,
    # and the request carries it back: the core read it exactly.
    description = {
        'tables': [
            {
                'name': 'users',
                'source': 'users',
                'columns': [
                    {
                        'name': 'user_id',
                        'received': {'type': 'int64'},
                        'dtype': 'int',
                        'stype': 'ID',
                        'nullable': False,
                        'unique': True,
                    }
                ],
                'primary_key': ['user_id'],
                'time_column': None,
                'duplicate_keys': None,
            },
            {
                'name': 'orders',
                'source': 'orders',
                'columns': [
                    {
                        'name': 'order_id',
                        'received': {'type': 'int64'},
                        'dtype': 'int',
                        'stype': 'ID',
                        'nullable': False,
                        'unique': True,
                    },
                    {
                        'name': 'user_id',
                        'received': {'type': 'int64'},
                        'dtype': 'int',
                        'stype': 'ID',
                        'nullable': False,
                        'unique': True,
                    },
                    {
                        'name': 'price',
                        'received': {'type': 'float64'},
                        'dtype': 'float',
                        'stype': 'numerical',
                        'nullable': True,
                        'unique': False,
                    },
                    {
                        'name': 'ts',
                        'received': {
                            'type': 'timestamp',
                            'unit': 'us',
                            'zone': None,
                        },
                        'dtype': 'date',
                        'stype': 'timestamp',
                        'nullable': True,
                        'unique': False,
                    },
                ],
                'primary_key': ['order_id'],
                'time_column': 'ts',
                'duplicate_keys': None,
            },
        ],
        'links': [
            [
                {'table': 'orders', 'column': 'user_id'},
                {'table': 'users', 'column': 'user_id'},
            ]
        ],
    }
    import pyarrow as pa

    asked: list[dict[str, Any]] = []

    class Stop(Exception):
        pass

    def provider(step: dict[str, Any], keys: Any) -> Any:
        request = step['request']
        asked.append(request)
        if request['request'] == 'time_bounds':
            return pa.table(
                {
                    'table_index': pa.array([0], pa.int32()),
                    'min_date': pa.array([0], pa.timestamp('us')),
                    'max_date': pa.array([2**50], pa.timestamp('us')),
                }
            )
        raise Stop

    operation = {
        'op': 'materialize',
        'provider': {
            'provider': 'host',
            'capabilities': {
                'optional': [
                    'time_bounds',
                    'sample_entities',
                    'neighbors',
                    'lookup',
                ],
                'max_keys_per_request': None,
                'sampling': 'seeded_stable',
                'max_seed': None,
                # Declared, so the sampler is not refused before it fetches
                # (T1.26b).
                'null_order': 'last',
            },
        },
        'schema': 1,
        'query': 2,
        'indices': 3,
        'anchor_time': codec.timestamp(int(ANCHOR.value), 'ns', None),
        'seed': codec.seed(seed),
    }
    with pytest.raises(StructuredDataError) as stopped:
        run(
            operation,
            {
                1: codec.attachment(description),
                2: codec.attachment(QUERY),
                3: pa.table({'indices': pa.array([0, 1], pa.int64())}),
            },
            provider=provider,
        )
    assert stopped.value.code == 'PROVIDER_ERROR'
    sample = asked[-1]
    assert sample['request'] == 'sample_entities'
    assert codec.decode_scalar(sample['seed']) == seed


def test_duckdb_refuses_a_seed_its_parser_refuses_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RS-7: the core refuses a seed past the largest DuckDB's dialect
    # declares, reading the capability, before the sample is planned.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    for seed in (2**31, 2**70):
        with pytest.raises(StructuredDataError) as refused:
            _materialize(random_seed=seed)
        assert refused.value.code == 'INVALID_REQUEST'
        assert refused.value.details['field'] == 'seed'
        assert refused.value.details['max_seed'] == 2**31 - 1
    # The largest DuckDB takes is taken.
    _materialize(random_seed=2**31 - 1)


def test_the_route_imports_nothing_of_the_engine(tmp_path: Path) -> None:
    # A stand-in engine any import would find first; the hook is this
    # test's own, in a process of its own (production installs none).
    fake = tmp_path / 'sdm_relational_engine'
    fake.mkdir()
    (fake / '__init__.py').write_text('IMPORTED = True\n', encoding='utf-8')
    script = f"""
    import sys, warnings
    sys.path.insert(0, {str(tmp_path)!r})
    import duckdb, pandas as pd
    from sdm_client import PredictiveQuery, Schema
    from sdm_client._routing import RouteError
    from sdm_connectors import DuckDBSource

    ENGINE = 'sdm_relational_engine'

    class NoEngine:
        def find_spec(self, name, path=None, target=None):
            if name == ENGINE or name.startswith(ENGINE + '.'):
                raise RouteError(f'{{name}} was imported after route entry')

    connection = duckdb.connect()
    for statement in {_shop()!r}:
        connection.execute(statement)
    hook = NoEngine()
    sys.meta_path.insert(0, hook)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery({QUERY!r}, schema).materialize(
                indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
            )
    finally:
        sys.meta_path.remove(hook)
    print(ENGINE in sys.modules, len(request._payloads))
    """
    import os

    environment = {
        key: value
        for key, value in os.environ.items()
        if key != _routing.ENVIRONMENT
    }
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**environment, _routing.ENVIRONMENT: CORE},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['False', '1']


# --- review round 1 (T1.11c) --------------------------------------------------

#: Order 50 is user 2's one order in instance 22's window (anchor
#: 2023-12-02): its price made NaN, the window holds only NaN.
NAN = "UPDATE orders SET price = 'NaN'::DOUBLE WHERE order_id = 50"
#: Beside it, a finite price at the same time: the window holds both.
FINITE = (
    'INSERT INTO orders SELECT 1000, 2, 5.0, ts FROM orders WHERE order_id = 50'
)


def test_a_nan_in_a_label_s_window_is_skipped_as_the_reference_skips_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The data change once the schema is described, which reads no NaN.
    only = _both_routes(monkeypatch, after=(NAN,))
    row = _context_row(only, 22)
    assert row[1:3] == [2, '2023-12-02T00:00:00.000000Z']
    assert row[3] == 0.0
    mixed = _both_routes(monkeypatch, after=(NAN, FINITE))
    assert _context_row(mixed, 22)[3] == 5.0


def test_a_table_mapped_to_another_source_table_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection()
        schema = Schema.from_source(DuckDBSource(connection))
        connection.execute(
            'CREATE TABLE other_orders AS SELECT order_id, user_id, '
            'price * 2 AS price, ts FROM orders'
        )
        schema.tables['orders'].source_name = 'other_orders'
        with pytest.raises(StructuredDataError) as refused:
            PredictiveQuery(QUERY, schema).materialize(
                indices=[0, 1], anchor_time=ANCHOR
            )
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert refused.value.details['rule'] == 'schema_edited'


def test_a_table_named_as_the_core_s_seeds_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Its key declared, as its name tells the inference nothing.
    payload = _both_routes(
        monkeypatch,
        'CREATE TABLE __sdm_seeds__ (order_id BIGINT PRIMARY KEY, '
        'user_id BIGINT, price DOUBLE, ts TIMESTAMP)',
        'INSERT INTO __sdm_seeds__ SELECT * FROM orders',
        'DROP TABLE orders',
        query=QUERY.replace('orders.', '__sdm_seeds__.'),
    )
    assert len(payload['context']['instance_table']['rows']) == 112
    assert '__sdm_seeds__' in payload['context']['related_tables']


@pytest.mark.parametrize(
    ('setup', 'indices', 'cast'),
    [
        ((), ['0', '1'], [0, 1]),
        ((), [True, False], [1, 0]),
        (
            (
                'ALTER TABLE orders ALTER user_id TYPE VARCHAR',
                'ALTER TABLE users ALTER user_id TYPE VARCHAR',
            ),
            [0, 1],
            ['0', '1'],
        ),
    ],
)
def test_entities_are_cast_to_the_key_s_type_as_the_reference_casts_them(
    monkeypatch: pytest.MonkeyPatch,
    setup: tuple[str, ...],
    indices: list[Any],
    cast: list[Any],
) -> None:
    payload = _both_routes(monkeypatch, *setup, indices=indices)
    predicted = payload['predict']['instance_table']['rows']
    assert [row[1] for row in predicted] == cast


def test_missing_indices_are_warned_of_once_per_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    query = PredictiveQuery(QUERY, schema)
    shown = []
    for _ in range(3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            query.materialize(indices=[0, 1], anchor_time=ANCHOR)
        shown.append(
            sum('MISSING_INDICES' in str(warning.message) for warning in caught)
        )
    assert shown == [1, 0, 0]


def test_the_provider_refuses_two_key_sets_bound_to_one_name() -> None:
    import pyarrow as pa

    from sdm_client._rust.providers.duckdb import DuckDBProvider, PlanNotRun

    def key_set(attachment: int) -> dict[str, Any]:
        return {
            'param': 'key_set',
            'keys': {'attachment': attachment, 'columns': []},
            'relation': 'seeds',
        }

    step = {
        'plan': {
            'plan': 'sql',
            'statements': [
                {
                    'dialect': 'duckdb',
                    'text': 'SELECT 1',
                    'params': [key_set(1), key_set(2)],
                }
            ],
        }
    }
    keys = {1: pa.table({'key': [1]}), 2: pa.table({'key': [2]})}
    with pytest.raises(PlanNotRun):
        DuckDBProvider(duckdb.connect())(step, keys)


# --- review round 2 (T1.11c) --------------------------------------------------


@pytest.mark.parametrize(
    'indices',
    [[2**100], [2**63, 0], [2**64, 0], [datetime.date(2024, 1, 1), 0]],
)
def test_entities_that_do_not_cast_are_the_caller_s_error_on_the_core(
    monkeypatch: pytest.MonkeyPatch, indices: list[Any]
) -> None:
    # Past the key's width, or of no castable type: the engine's own
    # handling lets these escape, and the old route reports INTERNAL_ERROR;
    # the core's cast (T1.32a) refuses them as the caller's error,
    # INVALID_REQUEST, naming the entity's position (register P1B-27).
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as refused:
        _materialize(indices=indices)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == 'indices'
    assert refused.value.details['position'] == 0
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        with pytest.raises(StructuredDataError) as old:
            _materialize(indices=indices)
        assert old.value.code == 'INTERNAL_ERROR'


def _warned(query: PredictiveQuery, calls: list[dict[str, Any]]) -> list[int]:
    r"""How many missing-index warnings each call of ``calls`` on ``query``
    showed, a refusal of the call held as its outcome."""
    shown = []
    for arguments in calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                query.materialize(anchor_time=ANCHOR, **arguments)
            except StructuredDataError:
                pass
        shown.append(
            sum('Missing' in str(warning.message) for warning in caught)
        )
    return shown


@pytest.mark.parametrize(
    'first',
    [
        # Refused by the core after its index read, as the reference
        # refuses them after building its sampler.
        {'indices': [None, 0]},
        {'indices': [0, 1], 'random_seed': 10**40},
    ],
)
def test_a_refused_call_does_not_lose_the_index_warning(
    monkeypatch: pytest.MonkeyPatch, first: dict[str, Any]
) -> None:
    calls = [first, {'indices': [0, 1]}, {'indices': [0, 1]}]
    shown = {}
    for route in ('core', 'old'):
        if route == 'core':
            monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
        else:
            monkeypatch.delenv(_routing.ENVIRONMENT)
            if importlib.util.find_spec('sdm_relational_engine') is None:
                break
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(_connection()))
        shown[route] = _warned(PredictiveQuery(QUERY, schema), calls)
    assert shown['core'] == [1, 0, 0]
    if 'old' in shown:
        assert shown['old'] == shown['core']


def test_an_entity_that_does_not_cast_is_refused_after_the_index_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # From T1.32a the core casts the entities where the reference casts
    # them, after its sample and so after its index read: the refused call
    # gives the warning, as the reference's does (decision E41(j)), and the
    # next calls do not.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    query = PredictiveQuery(QUERY, schema)
    calls = [{'indices': ['abc']}, {'indices': [0, 1]}, {'indices': [0, 1]}]
    assert _warned(query, calls) == [1, 0, 0]


def test_a_source_of_5000_columns_is_described_and_materialized_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T1.23's exit: describe_source and materialize on a generated source
    # of 5,000 columns run to Done through HostRun on this core runner
    # (sdm-testkit's duckdb_provider.rs runs them on rust-host): the shop
    # base with its users widened to 5,000 columns, each a feature.
    features = ', '.join(
        f'(user_id * {i}) % 7 AS f{i}' for i in range(3, 5_000)
    )
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    taken: list[str] = []
    real = _rust_run.run

    def counted(operation: dict[str, Any], *rest: Any, **named: Any) -> Any:
        finished = real(operation, *rest, **named)
        taken.append(operation['op'])
        return finished

    for module in (_rust_schema, _rust_materialize):
        monkeypatch.setattr(module, 'run', counted)
    request = _materialize(
        f'CREATE OR REPLACE TABLE users AS SELECT *, {features} FROM users'
    )
    # Each core run finished with its result: Done.
    assert taken == ['describe_source', 'materialize']
    users = request._payloads[0]['context']['related_tables']['users']
    assert len(users['columns']) == 5_001


@pytest.mark.parametrize(
    'count',
    [np.int64(4), np.uint64(4), np.int8(4)],
    ids=['int64', 'uint64', 'int8'],
)
def test_a_numpy_context_size_reaches_the_core(
    monkeypatch: pytest.MonkeyPatch, count: Any
) -> None:
    # T1.24 fix 1 (review P2-1): the client takes a NumPy integer as a whole
    # number, and the codec writes it as the integer it is; from T1.28a the
    # core takes it, narrowing the context to it as the old route does.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _materialize(num_context_samples=count)
    assert core._payloads[0]['metadata']['num_context_examples'] == 4
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    old = _materialize(num_context_samples=count)
    assert core._payloads == old._payloads


@pytest.mark.parametrize(
    ('tables', 'key'),
    [
        (None, ['user_id']),
        ({'users': 'users', 'orders': 'orders'}, ('user_id',)),
    ],
    ids=['every-table-list', 'tables-named-tuple'],
)
def test_a_key_of_one_column_as_a_sequence_is_pinned_on_both_routes(
    monkeypatch: pytest.MonkeyPatch, tables: Any, key: Any
) -> None:
    # Register entries P1B-2 and P1B-3 (proposed; T1.24's review, P2-2):
    # the reference keeps a key of one column given as a sequence as a
    # tuple, infers no link to it and refuses the question's entity,
    # INVALID_REQUEST; the core's description cannot say the key's form, so
    # on the core route the key is the column's name, the link is inferred
    # and materialize builds the request. Pinned, naming both outcomes.
    def built() -> tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(
                DuckDBSource(_connection()), tables, primary_keys={'users': key}
            )
            try:
                PredictiveQuery(QUERY, schema).materialize(
                    indices=[0, 1], anchor_time=ANCHOR
                )
            except StructuredDataError as refused:
                return schema, refused.code
            return schema, 'success'

    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            dict.fromkeys(
                _routing.FROM_SOURCE_DUCKDB
                + _routing.FROM_SOURCE_DUCKDB_TABLES
                + _routing.MATERIALIZE_DUCKDB
            )
        ),
    )
    schema, outcome = built()
    assert schema['users'].primary_key == 'user_id'
    assert len(schema.relationships) == 1
    assert outcome == 'success'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    schema, outcome = built()
    assert schema['users'].primary_key == ('user_id',)
    assert schema.relationships == []
    assert outcome == 'INVALID_REQUEST'


#: Orders' time column as each type the reference reads (T1.25): a date, a
#: zoned timestamp under a session zone, and text dates; and a users'
#: feature of text dates.
TIME_TYPES = {
    'date': ['ALTER TABLE orders ALTER ts TYPE DATE'],
    'timestamptz': [
        "SET TimeZone = 'America/New_York'",
        'ALTER TABLE orders ALTER ts TYPE TIMESTAMPTZ',
    ],
    'text': [
        "ALTER TABLE orders ALTER ts TYPE VARCHAR USING strftime(ts, '%Y-%m-%d')"
    ],
    'text feature': [
        'CREATE OR REPLACE TABLE users AS SELECT *, '
        "strftime(DATE '2020-01-01' + user_id::INTEGER, '%Y-%m-%d') "
        'AS signup_date FROM users'
    ],
}


@pytest.mark.parametrize('kind', sorted(TIME_TYPES))
def test_a_time_of_any_type_the_reference_reads_materializes_on_the_core(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # Until T1.25 the core read a zone-less timestamp time column only
    # (time_type) and refused text that could hold dates at describe
    # (text_time); it reads each as the reference's engine does, the
    # payload the old route's.
    payload = _both_routes(monkeypatch, *TIME_TYPES[kind])
    assert payload['context']['instance_table']['rows']


def _anchored(anchor: pd.Timestamp) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=anchor
        )
    return json.loads(json.dumps(request._payloads[0], default=str))


def test_a_zoned_anchor_time_is_read_at_its_utc_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ledger's row moved into the core at T1.25: the bridge sends the
    # zoned anchor's instant with its zone, and the core reads it at its
    # UTC wall clock, as the engine reads one (to_naive_utc).
    zoned = pd.Timestamp('2024-03-01 05:00', tz='America/New_York')
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _anchored(zoned)
    assert core == _anchored(pd.Timestamp('2024-03-01 10:00'))
    assert codec.timestamp(1, 'ns', 'UTC')['zone'] == 'UTC'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        assert core == _anchored(zoned)


#: Time values past the years a nanosecond count holds (1677 to 2262), on
#: both routes: neither takes them, the old route refusing with its own
#: code, the core with VALUE_OUT_OF_RANGE (register DT-14, proposed).
OUT_OF_RANGE = {
    'date 0001': (
        'CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, '
        "DATE '0001-01-01' + (order_id * 7)::INTEGER AS ts FROM orders",
        'INTERNAL_ERROR',
    ),
    'date 3000': (
        'CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, '
        "DATE '3000-01-01' + (order_id * 7)::INTEGER AS ts FROM orders",
        'INVALID_REQUEST',
    ),
    'timestamptz 3000': (
        'CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, '
        "CASE WHEN order_id = 3 THEN TIMESTAMPTZ '3000-01-01 00:00:00+00' "
        'ELSE ts::TIMESTAMPTZ END AS ts FROM orders',
        'INTERNAL_ERROR',
    ),
    'text 0001': (
        'CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, '
        "strftime(DATE '0001-01-01' + (order_id * 7)::INTEGER, '%Y-%m-%d') "
        'AS ts FROM orders',
        'INTERNAL_ERROR',
    ),
}


@pytest.mark.parametrize('kind', sorted(OUT_OF_RANGE))
def test_a_time_past_nanoseconds_is_refused_on_both_routes_under_its_code(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # Diagnostic, measured at T1.25's review: pinned so the day either
    # route's code changes, this fails.
    setup, old_code = OUT_OF_RANGE[kind]
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as core:
        _materialize(setup)
    assert core.value.code == 'VALUE_OUT_OF_RANGE'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        with pytest.raises(StructuredDataError) as old:
            _materialize(setup)
        assert old.value.code == old_code


# --- T1.26b: the sampler's walk and its state ----------------------------------


def _logged(connection: Any) -> int:
    r"""How many statements DuckDB logged since its query log was turned
    on, the log's own reads left out."""
    rows = connection.execute(
        "SELECT message FROM duckdb_logs WHERE type = 'QueryLog'"
    ).fetchall()
    return sum(1 for (text,) in rows if text and 'duckdb_logs' not in text)


def _calls(
    queries: list[PredictiveQuery], connection: Any, **arguments: Any
) -> list[tuple[int, int]]:
    r"""Each query of ``queries`` materialized in turn: the statements each
    call ran and the missing-index warnings it showed."""
    made = []
    for query in queries:
        before = _logged(connection)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                query.materialize(anchor_time=ANCHOR, **arguments)
            except StructuredDataError:
                pass
        warned = sum('Missing' in str(w.message) for w in caught)
        made.append((_logged(connection) - before, warned))
    return made


def _on_both_routes(
    monkeypatch: pytest.MonkeyPatch, made: Any
) -> list[list[tuple[int, int]]]:
    r"""What ``made`` gives on the core, then on the old route where the
    engine is installed (CI's ``engine`` job); where it is not, the old
    route's refusal is held instead."""
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    found = [made()]
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is None:
        with pytest.raises(MissingExtraError):
            made()
    else:
        found.append(made())
    return found


def _shop_logged() -> tuple[Any, Schema]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection()
        schema = Schema.from_source(DuckDBSource(connection))
    connection.execute("CALL enable_logging('QueryLog')")
    return connection, schema


def test_a_second_call_runs_the_reference_s_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # H9's counts on a repeated call (decision E52): the reference checks
    # the indexes once, building its sampler, and keeps the time bounds it
    # read, so its second call runs 4 statements where its first ran 6; the
    # core is handed the state its first run left and skips the same two.
    def made() -> list[tuple[int, int]]:
        connection, schema = _shop_logged()
        query = PredictiveQuery(QUERY, schema)
        return _calls([query] * 3, connection, indices=[0, 1])

    for found in _on_both_routes(monkeypatch, made):
        assert found == [(6, 1), (4, 0), (4, 0)]


def test_two_materializers_each_check_their_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each query holds its own materializer, and each materializer its own
    # state: each checks the indexes, and warns, once.
    def made() -> list[tuple[int, int]]:
        connection, schema = _shop_logged()
        one = PredictiveQuery(QUERY, schema)
        two = PredictiveQuery(QUERY, schema)
        return _calls([one, two, one, two], connection, indices=[0, 1])

    for found in _on_both_routes(monkeypatch, made):
        assert found == [(6, 1), (6, 1), (4, 0), (4, 0)]


def test_a_call_the_core_refuses_before_its_check_leaves_it_to_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An option the core does not take is refused before anything is read
    # (P1B-3): the state that run leaves says nothing was read, so the next
    # call checks the indexes and warns; one refused after the check (an
    # entity missing, refused before the time bounds are read) keeps it,
    # and the next call skips it, reading the bounds.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    connection, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    assert _calls([query], connection, indices=[0, 1], num_hops=1) == [(0, 0)]
    assert _calls([query], connection, indices=[0, 1]) == [(6, 1)]
    other = PredictiveQuery(QUERY, schema)
    assert _calls([other], connection, indices=[None, 1]) == [(1, 1)]
    assert _calls([other], connection, indices=[0, 1]) == [(5, 0)]


#: A source whose order lines refer to items as well as users, the items
#: with a text feature: the walk's second hop and the sampler's words.
_DEEPER = (
    "CREATE TABLE items AS SELECT i::BIGINT AS item_id, 'Item ' || i || "
    "' (the BEST<br>one)!' AS title, (i * 2.5)::DOUBLE AS cost FROM range(5) t(i)",
    'ALTER TABLE orders ADD COLUMN item_id BIGINT',
    'UPDATE orders SET item_id = order_id % 5',
)


def _deeper(*setup: str, edit: bool = True) -> Request:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection(*_DEEPER, *setup)
        schema = Schema.from_source(
            DuckDBSource(connection),
            primary_keys={
                'users': 'user_id',
                'orders': 'order_id',
                'items': 'item_id',
            },
            check_keys=False,
        )
        if edit:
            schema['items']['title'].stype = 'text'
        return PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )


def test_a_second_hop_and_a_text_feature_are_sampled_as_the_old_route_samples_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The order lines' items are reached at the second hop, looked up by
    # key for each example, and their titles sent as the reference's
    # words; the request is the old route's, payload for payload.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _deeper()._payloads[0]
    items = core['context']['related_tables']['items']
    title = items['columns'].index('title')
    assert items['rows'][0][title][:3] == ['item', '0', 'the']
    assert core['schema']['related_tables']['items']['columns']['title'] == {
        'dtype': 'stringlist',
        'stype': 'text',
    }
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        old = _deeper()._payloads[0]
        assert json.loads(json.dumps(core)) == json.loads(
            json.dumps(old, default=str)
        )


def _crossing(payload: dict[str, Any]) -> int:
    r"""How many foreign keys of ``payload`` name no row of their own
    example in the table they refer to."""
    crossing = 0
    for half in ('context', 'predict'):
        tables = payload[half]['related_tables']
        held = set()
        for name, table in tables.items():
            columns = table['columns']
            key = '__node_id' if '__node_id' in columns else columns[1]
            at = columns.index(key)
            held.update((name, row[at], row[0]) for row in table['rows'])
        for link in payload['schema']['relationships']:
            if 'source_table' not in link or link['source_table'] not in tables:
                continue
            table = tables[link['source_table']]
            at = table['columns'].index(link['source_columns'][0])
            for row in table['rows']:
                if row[at] is None:
                    continue
                value = int(row[at]) if isinstance(row[at], float) else row[at]
                crossing += (link['target_table'], value, row[0]) not in held
    return crossing


def _unreferenced(payload: dict[str, Any]) -> int:
    r"""How many rows of ``payload``'s items no order of their own example
    refers to."""
    unreferenced = 0
    for half in ('context', 'predict'):
        tables = payload[half]['related_tables']
        if 'items' not in tables:
            continue
        orders = tables['orders']
        at = orders['columns'].index('item_id')
        referred = {
            (row[0], int(row[at]))
            for row in orders['rows']
            if row[at] is not None
        }
        items = tables['items']
        key = items['columns'].index('__node_id')
        unreferenced += sum(
            (row[0], row[key]) not in referred for row in items['rows']
        )
    return unreferenced


@pytest.mark.parametrize(
    'setup',
    [
        # A foreign key the walk looks up, null in some rows.
        ('UPDATE orders SET item_id = NULL WHERE order_id % 3 = 0',),
        # A key of a table the walk samples, null in some rows.
        ('UPDATE orders SET order_id = NULL WHERE order_id % 13 = 6',),
    ],
)
def test_a_null_key_the_walk_reaches_links_nothing(
    monkeypatch: pytest.MonkeyPatch, setup: tuple[str, ...]
) -> None:
    # Register P1B-13, approved by the owner (2026-10-11): a null is no
    # link. The core adds no row for a null foreign key and sends it as
    # null, and takes a row whose key is null as no other row; every link
    # it sends names a row of its own example, and every row it adds is
    # referred to. The reference numbers such rows out of step, so its
    # request links rows across examples (held on the old route where the
    # engine is installed).
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _deeper(*setup, edit=False)._payloads[0]
    assert _crossing(core) == 0
    assert _unreferenced(core) == 0
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        old = _deeper(*setup, edit=False)._payloads[0]
        assert _crossing(old) > 0


def test_every_foreign_key_looked_up_null_is_the_reference_s_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Where the reference is clean, the core is it: every item key null,
    # the reference looks up no key (one statement, no row) and sends each
    # null; so does the core, statement for statement.
    def made() -> list[tuple[int, int]]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            connection = _connection(
                *_DEEPER, 'UPDATE orders SET item_id = NULL'
            )
            schema = Schema.from_source(
                DuckDBSource(connection),
                primary_keys={
                    'users': 'user_id',
                    'orders': 'order_id',
                    'items': 'item_id',
                },
            )
        connection.execute("CALL enable_logging('QueryLog')")
        query = PredictiveQuery(QUERY, schema)
        return _calls([query, query], connection, indices=[0, 1])

    found = _on_both_routes(monkeypatch, made)
    assert found[0] == [(7, 1), (5, 0)]
    assert all(each == found[0] for each in found)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _deeper('UPDATE orders SET item_id = NULL', edit=False)._payloads[0]
    monkeypatch.delenv(_routing.ENVIRONMENT)
    assert 'items' not in core['context']['related_tables']
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        old = _deeper('UPDATE orders SET item_id = NULL', edit=False)._payloads[
            0
        ]
        assert json.loads(json.dumps(core)) == json.loads(
            json.dumps(old, default=str)
        )


#: The shop with a second time column, a second key and a second table
#: referring to the users, so a schema can be edited between two calls.
_EDITABLE = (
    'ALTER TABLE orders ADD COLUMN ts2 TIMESTAMP',
    "UPDATE orders SET ts2 = ts + INTERVAL '10 days'",
    'ALTER TABLE orders ADD COLUMN alt BIGINT',
    'UPDATE orders SET alt = order_id + 1000',
    'CREATE TABLE extra AS SELECT i::BIGINT AS extra_id, (i % 8)::BIGINT AS '
    'user_id, i::DOUBLE AS w FROM range(20) t(i)',
)


def _edit(schema: Schema, edit: str) -> None:
    if edit == 'time_column':
        schema['orders'].time_column = 'ts2'
    elif edit == 'key':
        schema['orders'].primary_key = 'alt'
    else:
        schema.relationships = [
            link
            for link in schema.relationships
            if link.source_table != 'extra'
        ]


def _editable() -> tuple[Any, Schema]:
    from sdm_client import Relationship

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection(*_EDITABLE)
        schema = Schema.from_source(
            DuckDBSource(connection),
            primary_keys={
                'users': 'user_id',
                'orders': 'order_id',
                'extra': 'extra_id',
            },
            time_columns={'orders': 'ts'},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id'),
                Relationship('extra', 'user_id', 'users', 'user_id'),
            ],
        )
    connection.execute("CALL enable_logging('QueryLog')")
    return connection, schema


def _edited_between_calls(edit: str) -> list[tuple[int, int, str]]:
    r"""Two calls of one query, ``edit`` made to its schema between them:
    each call's statements, missing-index warnings and payload."""
    connection, schema = _editable()
    query = PredictiveQuery(QUERY, schema)
    made = []
    for call in range(2):
        if call == 1:
            _edit(schema, edit)
        before = _logged(connection)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            payload = query.materialize(
                indices=[0, 1], anchor_time=ANCHOR
            )._payloads[0]
        made.append(
            (
                _logged(connection) - before,
                sum('Missing' in str(w.message) for w in caught),
                json.dumps(payload, sort_keys=True, default=str),
            )
        )
    return made


@pytest.mark.parametrize('edit', ['time_column', 'key', 'link'])
def test_a_schema_edited_between_two_calls_is_sampled_as_edited(
    monkeypatch: pytest.MonkeyPatch, edit: str
) -> None:
    # Register P1B-14, diagnostic: the reference builds its model once per
    # materializer, so a key, link or time column edited between two calls
    # is not followed (the second call reads nothing again, warns of
    # nothing, and sends the first call's request). The core's state is
    # made under the keys, links and time columns (E64), so it is not read
    # across the edit: the second call checks and reads afresh, warns
    # again, and sends the request a new materializer of the edited schema
    # sends.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _edited_between_calls(edit)
    assert [made[:2] for made in core] == [
        (7, 1),
        (6, 1) if edit == 'link' else (7, 1),
    ]
    assert core[0][2] != core[1][2]
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is None:
        with pytest.raises(MissingExtraError):
            _edited_between_calls(edit)
        return
    old = _edited_between_calls(edit)
    assert old[0] == core[0]
    assert old[1][:2] == (5, 0)
    assert old[1][2] == old[0][2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, schema = _editable()
        _edit(schema, edit)
        fresh = (
            PredictiveQuery(QUERY, schema)
            .materialize(indices=[0, 1], anchor_time=ANCHOR)
            ._payloads[0]
        )
    assert json.dumps(fresh, sort_keys=True, default=str) == core[1][2]


@pytest.mark.parametrize(
    'held',
    [
        {'version': 99, 'memo': b'{}'},
        {'version': 1, 'memo': b'not a memo'},
    ],
)
def test_a_state_the_core_refuses_is_not_handed_back_again(
    monkeypatch: pytest.MonkeyPatch, held: dict[str, Any]
) -> None:
    # A state of another version, or a memo that does not read, is refused
    # by name at the start, and dropped: the next call starts afresh, as a
    # materializer's first does, rather than being refused the same way.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    connection, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    query._relational_materializer._core_sampler_state['held'] = held
    with pytest.raises(StructuredDataError) as refused:
        query.materialize(indices=[0, 1], anchor_time=ANCHOR)
    assert refused.value.code in ('INVALID_REQUEST', 'INVALID_MESSAGE')
    assert 'held' not in query._relational_materializer._core_sampler_state
    assert _calls([query, query], connection, indices=[0, 1]) == [
        (6, 1),
        (4, 0),
    ]


@pytest.mark.parametrize('said', [True, False])
def test_the_host_drops_a_state_on_the_core_s_word_not_the_field_it_names(
    monkeypatch: pytest.MonkeyPatch, said: bool
) -> None:
    # The core's refusal of a held state names its field and says to drop
    # it (drop_held, decision E74); seen with the two told apart, the host
    # follows drop_held alone: kept where the refusal names the field but
    # does not say to drop it, dropped where it says so naming no field.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    read = codec.read

    def apart(control: bytes, attachments: dict[int, Any]) -> dict[str, Any]:
        message = read(control, attachments)
        error = message.get('error')
        if message['message'] == 'error' and error is not None:
            assert error.get('drop_held') == ['sampler_state']
            assert error['fields']['field']['v'] == 'sampler_state.version'
            if said:
                del error['fields']['field']
            else:
                del error['drop_held']
        return message

    monkeypatch.setattr(codec, 'read', apart)
    _, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    held = {'version': 99, 'memo': b'{}'}
    query._relational_materializer._core_sampler_state['held'] = held
    with pytest.raises(StructuredDataError):
        query.materialize(indices=[0, 1], anchor_time=ANCHOR)
    kept = query._relational_materializer._core_sampler_state
    assert ('held' in kept) is not said


#: A decimal average whose sum would not fit DECIMAL(38, 18) (T1.27's fix
#: round 2): two prices of 6e19 for user 1 in the window, one earlier order
#: bounding the data. The reference averages them; the sum it refuses.
DECIMAL_AVERAGE = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY)',
    'INSERT INTO users SELECT i FROM range(4) t(i)',
    'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
    'price DECIMAL(38, 18), ts TIMESTAMP)',
    "INSERT INTO orders VALUES (1, 1, 60000000000000000000, '2024-02-15'), "
    "(2, 1, 60000000000000000000, '2024-02-15'), (3, 2, 1, '2024-01-01')",
)
#: The reference's context for the average, recorded from the reference
#: checkout's environment (b413e9a; pandas 3.0.6, pyarrow 25.0.1) through
#: the same public calls: one example, user 1 at 2024-02-01, its label the
#: mean read at float32.
DECIMAL_AVERAGE_REFERENCE = [
    [0, 1, '2024-02-01T00:00:00.000000Z', 6.000000208206194e19]
]


def _decimal_average(query: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = duckdb.connect()
        for statement in DECIMAL_AVERAGE:
            connection.execute(statement)
        schema = Schema.from_source(DuckDBSource(connection))
        return PredictiveQuery(query, schema).materialize(
            indices=[1], anchor_time=ANCHOR
        )


def test_a_decimal_average_past_38_digits_is_not_the_caller_s_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    average = 'PREDICT AVG(orders.price, 0, 1, months) FOR EACH users.user_id'
    total = 'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id'
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        # The old route is the reference's: the average taken, as recorded,
        # and the sum refused as the caller's error.
        request = _decimal_average(average)
        rows = request._payloads[0]['context']['instance_table']['rows']
        assert json.loads(json.dumps(rows)) == DECIMAL_AVERAGE_REFERENCE
        with pytest.raises(StructuredDataError) as refused:
            _decimal_average(total)
        assert refused.value.code == 'INVALID_REQUEST'
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    # On the core a decimal's average takes a rule it does not port yet,
    # refused by name (P1B-3), never as the caller's error.
    with pytest.raises(StructuredDataError) as refused:
        _decimal_average(average)
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert refused.value.details['rule'] == 'target_type'
    # The sum is the caller's error on both routes.
    with pytest.raises(StructuredDataError) as refused:
        _decimal_average(total)
    assert refused.value.code == 'INVALID_REQUEST'


#: Decimal sums whose running total passes 38 digits (or ``i128``) and
#: comes back (T1.27's fix round 3), user 1's prices in time order, and the
#: label the reference gives user 1 at 2024-02-01, recorded from the
#: reference checkout's environment (b413e9a; pandas 3.0.6, pyarrow
#: 25.0.1) through the same public calls; every other example's label is
#: 0.0. ``tests_test_ops/test_the_labels.py`` holds the core's labels to
#: the same values.
SIX = '60000000000000000000'
DECIMAL_SUMS = {
    'running-past-38-digits': (
        'DECIMAL(38, 18)',
        [SIX, SIX, '-' + SIX],
        6.000000208206194e19,
    ),
    'back-to-zero': ('DECIMAL(38, 18)', [SIX, SIX, '-' + SIX, '-' + SIX], 0.0),
    'another-order': (
        'DECIMAL(38, 18)',
        [SIX, '-' + SIX, SIX],
        6.000000208206194e19,
    ),
    'scale-0': (
        'DECIMAL(38, 0)',
        ['6' + '0' * 37] * 2 + ['-6' + '0' * 37],
        6.000000213819608e37,
    ),
    'running-past-i128': (
        'DECIMAL(38, 0)',
        ['9' + '0' * 37] * 2 + ['-9' + '0' * 37],
        8.999999813669171e37,
    ),
}
#: The reference's context over those sources: the examples' instance id,
#: entity and anchor; user 1 at 2024-02-01 is the one example with a sum.
DECIMAL_SUM_EXAMPLES = [
    (0, 0, '2024-02-01'),
    (1, 1, '2024-02-01'),
    (2, 2, '2024-02-01'),
    (3, 3, '2024-02-01'),
    (4, 0, '2024-01-01'),
    (5, 1, '2024-01-01'),
    (6, 3, '2024-01-01'),
    (7, 2, '2024-01-01'),
]


def _decimal_sum_rows(label: float) -> list[list[Any]]:
    return [
        [
            i,
            entity,
            f'{day}T00:00:00.000000Z',
            label if (entity, day) == (1, '2024-02-01') else 0.0,
        ]
        for i, entity, day in DECIMAL_SUM_EXAMPLES
    ]


def _decimal_sum(dtype: str, prices: list[str]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = duckdb.connect()
        connection.execute('CREATE TABLE users (user_id BIGINT PRIMARY KEY)')
        connection.execute('INSERT INTO users SELECT i FROM range(4) t(i)')
        connection.execute(
            'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
            f'price {dtype}, ts TIMESTAMP)'
        )
        connection.execute("INSERT INTO orders VALUES (0, 2, 1, '2024-01-01')")
        for i, price in enumerate(prices):
            connection.execute(
                f'INSERT INTO orders VALUES ({i + 1}, 1, {price}, '
                f"TIMESTAMP '2024-02-10' + to_days({i}))"
            )
        schema = Schema.from_source(DuckDBSource(connection))
        return PredictiveQuery(
            'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id',
            schema,
        ).materialize(indices=[1], anchor_time=ANCHOR)


@pytest.mark.parametrize('name', sorted(DECIMAL_SUMS))
def test_a_decimal_sum_s_total_alone_is_checked(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dtype, prices, label = DECIMAL_SUMS[name]
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        # The old route is the reference's: the labels as recorded.
        request = _decimal_sum(dtype, prices)
        rows = request._payloads[0]['context']['instance_table']['rows']
        assert json.loads(json.dumps(rows)) == _decimal_sum_rows(label)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    # On the core the sum is labelled, never the caller's error, and from
    # T1.28a the request sends the decimal feature as the reference does
    # (its wire dtype): the context is the reference's.
    request = _decimal_sum(dtype, prices)
    rows = request._payloads[0]['context']['instance_table']['rows']
    assert rows == _decimal_sum_rows(label)


def _notices(*setup: str, text: str | None = None) -> list[str]:
    r"""The client's notices (``SDMWarning``) a materialize of ``QUERY``
    over the shop base and ``setup`` gives, each without the core's code;
    ``text`` names a column of ``users`` typed text."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        connection = _connection(*setup)
        schema = Schema.from_source(DuckDBSource(connection))
        if text is not None:
            schema['users'][text].stype = 'text'
        PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )
    return sorted(
        re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))
        for w in caught
        if type(w.message).__name__ == 'SDMWarning'
    )


@pytest.mark.parametrize(
    'setup',
    [
        # A table no relationship connects, and one linked and reached by
        # no row; a text feature of a table the request sends.
        (
            'CREATE TABLE notes (note_id BIGINT PRIMARY KEY, body VARCHAR)',
            "INSERT INTO notes VALUES (1, 'a')",
        ),
        (
            'CREATE TABLE tags AS SELECT i::BIGINT AS tag_id, (100 + i)::BIGINT '
            "AS user_id, 'x' AS tag FROM range(3) t(i)",
        ),
        (
            'CREATE TABLE notes (note_id BIGINT PRIMARY KEY, body VARCHAR)',
            "INSERT INTO notes VALUES (1, 'a')",
            'CREATE TABLE more AS SELECT 1::BIGINT AS more_id, 2.0 AS x',
            "CREATE OR REPLACE TABLE users AS SELECT *, repeat('a long piece "
            "of free text that says something different each time ', 4) || "
            'user_id AS bio FROM users',
        ),
    ],
    ids=['unlinked', 'unreached', 'several-and-text'],
)
def test_the_notices_of_what_the_request_leaves_out_are_the_old_route_s(
    monkeypatch: pytest.MonkeyPatch, setup: tuple[str, ...]
) -> None:
    # T1.28a: the core gives the client's notices of the tables the request
    # does not carry and the columns the model does not read, as codes and
    # fields this host words; their words are the old route's.
    text = 'bio' if any(' AS bio ' in s for s in setup) else None
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _notices(*setup, text=text)
    assert core, 'a notice was given'
    if text is not None:
        assert any('users.bio is typed text' in n for n in core), core
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    assert core == _notices(*setup, text=text)


def test_the_default_mode_s_cap_is_noticed_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T1.28a: a context the default run mode capped at its examples is said,
    # by the core's code, in the old route's words; a size asked for is not.
    statements = [
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i % 50)'
        '::INTEGER AS age FROM range(1001) t(i)',
        'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 1001)'
        '::BIGINT AS user_id, (10 + (i % 97) * 1.5)::DOUBLE AS price, '
        "TIMESTAMP '2023-06-01' + to_days((i % 270)::INTEGER) AS ts "
        'FROM range(900) t(i)',
    ]

    def notices(**options: Any) -> list[str]:
        connection = duckdb.connect()
        for statement in statements:
            connection.execute(statement)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            schema = Schema.from_source(DuckDBSource(connection))
            PredictiveQuery(QUERY, schema).materialize(
                indices=[0, 1], anchor_time=ANCHOR, **options
            )
        return [
            re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))
            for w in caught
            if type(w.message).__name__ == 'SDMWarning'
        ]

    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = notices()
    assert any('context examples were gathered' in n for n in core), core
    assert not any(
        'context examples were gathered' in n
        for n in notices(num_context_samples=1_000)
    )
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    assert core == notices()


def _refusal_of(**options: Any) -> StructuredDataError:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(**options)
    return refused.value


def test_a_limit_is_raised_as_the_subclass_on_the_core_register_p1b_20(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register P1B-20 (proposed): a limit the reference refuses as
    # StructuredDataError(UNSUPPORTED_REQUEST) the core refuses with the
    # same code, raised as UnsupportedRequestError, its subclass.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    for options in (
        {'indices': list(range(1_001))},
        {'num_context_samples': 10_001},
        {'num_hops': 7},
    ):
        core = _refusal_of(**options)
        assert core.code == 'UNSUPPORTED_REQUEST', options
        assert type(core).__name__ == 'UnsupportedRequestError', options
        assert 'rule' not in core.details, options
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    for options in ({'indices': list(range(1_001))}, {'num_hops': 7}):
        old = _refusal_of(**options)
        assert (type(old), old.code) == (
            StructuredDataError,
            'UNSUPPORTED_REQUEST',
        ), options


def test_an_average_compared_with_null_is_refused_by_name_register_p1b_21(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register P1B-21 (proposed, E66(b)'s class): the reference fails
    # INTERNAL_ERROR; the core refuses the shape by name.
    query = (
        'PREDICT AVG(orders.price, 0, 30, days) != NULL FOR EACH users.user_id'
    )
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _refusal_of(query=query)
    assert (core.code, core.details['rule']) == (
        'UNSUPPORTED_REQUEST',
        'average_null',
    )
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    assert _refusal_of(query=query).code == 'INTERNAL_ERROR'


def test_a_time_of_day_as_the_time_column_is_the_source_s_failure_register_dt_18(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register DT-18 (proposed, E66(b)'s class): the source's CAST of a
    # TIME column fails on both routes; the reference reports it as
    # INTERNAL_ERROR, the core as the provider's error.
    def refused() -> StructuredDataError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            connection = _connection(
                'CREATE OR REPLACE TABLE users AS SELECT *, '
                'make_time(user_id::BIGINT, 30, 0) AS tod FROM users'
            )
            schema = Schema.from_source(DuckDBSource(connection))
            schema['users'].time_column = 'tod'
            with pytest.raises(StructuredDataError) as error:
                PredictiveQuery(
                    'PREDICT users.age > 30 FOR EACH users.user_id', schema
                ).materialize(indices=[0, 1])
        return error.value

    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    assert refused().code == 'PROVIDER_ERROR'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    assert refused().code == 'INTERNAL_ERROR'


@pytest.mark.parametrize(
    ('query', 'config'),
    [
        (QUERY, {'num_estimators': np.float32(2)}),
        (
            QUERY,
            {'num_estimators': np.int64(3), 'column_shuffle': np.bool_(True)},
        ),
        (
            QUERY,
            {
                'hop_shuffle': np.bool_(False),
                'category_shuffle': np.float64(1.0),
            },
        ),
        (BINARY, {'class_shuffle': np.bool_(False)}),
    ],
    ids=['float32', 'int64-bool', 'bool-float64', 'binary-class-shuffle'],
)
def test_numpy_inference_options_are_read_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch, query: str, config: dict[str, Any]
) -> None:
    # Fix round 1 (Codex P2-2, the subagent's P2-1): the call's values carry
    # a NumPy scalar as the Python value it holds, its type kept, so the
    # core reads the options as the engine's validation reads them.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    core = _materialize(query=query, inference_config=config)
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if _old_route_needs_the_engine():
        return
    old = _materialize(query=query, inference_config=config)
    assert core._payloads[0]['inference'] == old._payloads[0]['inference']
