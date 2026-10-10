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


def _connection() -> Any:
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    return connection


def _materialize(**options: Any) -> Request:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        return PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR, **options
        )


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
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    for seed in (2**31, 2**70):
        with pytest.raises(StructuredDataError) as refused:
            _materialize(random_seed=seed)
        assert refused.value.code == 'INVALID_REQUEST'
        assert refused.value.details['field'] == 'seed'
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
