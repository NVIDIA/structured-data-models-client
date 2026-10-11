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
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import duckdb
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
from sdm_client._rust.run import run
from sdm_client.errors import MissingExtraError

ROOT = Path(__file__).resolve().parents[3]
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
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
        ({'context_anchor_time': ANCHOR}, 'context_anchor_time'),
        ({'num_context_samples': 50}, 'num_context_samples'),
        ({'anchor_time': 'entity'}, 'anchor_entity'),
        # A seed given as None, which the reference reads as unseeded.
        ({'random_seed': None}, 'unseeded'),
    ],
)
def test_what_the_core_does_not_take_yet_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], rule: str
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        arguments = {'indices': [0, 1], 'anchor_time': ANCHOR, **options}
        with pytest.raises(StructuredDataError) as refused:
            PredictiveQuery(QUERY, schema).materialize(**arguments)
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert refused.value.details['rule'] == rule


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
def test_entities_that_do_not_cast_fail_as_the_reference_fails_them(
    monkeypatch: pytest.MonkeyPatch, indices: list[Any]
) -> None:
    # Past the key's width, or of no castable type: the engine's own
    # handling lets these escape, and the client reports INTERNAL_ERROR.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as refused:
        _materialize(indices=indices)
    assert refused.value.code == 'INTERNAL_ERROR'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        with pytest.raises(StructuredDataError) as old:
            _materialize(indices=indices)
        assert old.value.code == refused.value.code


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


def test_a_call_refused_before_the_run_leaves_the_warning_to_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An entity that does not cast is refused in the host, before the core
    # reads the indexes; the reference casts after its index read, so it
    # warns on that call (decision E41(j)). The warning is not lost: the
    # next call gives it.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    query = PredictiveQuery(QUERY, schema)
    calls = [{'indices': ['abc']}, {'indices': [0, 1]}, {'indices': [0, 1]}]
    assert _warned(query, calls) == [0, 1, 0]
