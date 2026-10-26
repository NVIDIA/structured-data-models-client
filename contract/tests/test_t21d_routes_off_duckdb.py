# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d's runner change for the routes off DuckDB (T2.2's test
author's): python-core forces, off DuckDB, every route of each public call
a case makes through its ``until``, so no call of a core leg takes the old
route there; on DuckDB a case's routes are its own.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sdm_client import _routing

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


RUNNER = _module(ROOT / 'contract/runners/run_python.py', 't21d_run_python')


def _case(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def _on(paths: tuple[str, ...], database: str) -> set[str]:
    return {RUNNER.on_database(path, database) for path in paths}


def test_off_duckdb_every_call_a_case_makes_is_routed() -> None:
    r"""An S0 case (validation alone routed, ``until`` materialize) forces
    ``from_source`` and ``materialize`` on SQLite too, not ``predict``; a
    case DuckDB stops at ``from_source`` (``nested.map``, ``until``
    predict) forces all three; every forced path is one the client routes;
    and on DuckDB each case's routes are its own."""
    s0 = _case('contract/cases/s0/s0.accept.binary-classification.json')
    nested = _case('contract/cases/s1/nested.map.json')
    from_source = _on(_routing.FROM_SOURCE_DUCKDB, 'sqlite')
    materialize = _on(_routing.MATERIALIZE_DUCKDB, 'sqlite')
    predict = _on(_routing.PREDICT_DUCKDB, 'sqlite')
    forced = set(RUNNER.routes_on(s0, 'sqlite'))
    assert set(s0['routes']) <= forced
    assert from_source | materialize <= forced
    assert not predict & forced
    assert from_source | materialize | predict <= set(
        RUNNER.routes_on(nested, 'sqlite')
    )
    for case in (s0, nested):
        assert RUNNER.routes_on(case, 'duckdb') == case['routes']
        assert set(RUNNER.routes_on(case, 'sqlite')) <= set(_routing.PATHS)


@pytest.mark.parametrize(
    'path',
    [
        'contract/cases/s1/assuming.binary.json',
        'contract/cases/s2/lakebase.binary.json',
    ],
)
def test_lakebase_forces_the_actual_postgres_source_routes(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    case = _case(path)
    routes = RUNNER.routes_on(case, 'lakebase')
    assert routes and not any(route.endswith('/lakebase') for route in routes)
    expected = {
        *(
            p.rsplit('/', 1)[0] + '/postgres'
            for p in _routing.FROM_SOURCE_DUCKDB
        ),
        *(
            p.rsplit('/', 1)[0] + '/postgres'
            for p in _routing.MATERIALIZE_DUCKDB
        ),
        *(p.rsplit('/', 1)[0] + '/postgres' for p in _routing.PREDICT_DUCKDB),
    }
    assert expected <= set(routes)
    monkeypatch.setenv('SDM_CORE_ROUTES', ','.join(routes))
    assert _routing.forced() == frozenset(routes)
    assert RUNNER.case_key(case['id'], 'lakebase') == case['id'] + '@lakebase'


@pytest.mark.parametrize('leg', ['snowflake', 'lakebase'])
@pytest.mark.parametrize(
    'name',
    [
        'adventureworks.notebook.demand-3-timeframes',
        'adventureworks.notebook.demand-30d-historical',
        'adventureworks.notebook.recommend-top5',
        'adventureworks.quickstart.large-order-90d',
        'adventureworks.quickstart.no-order-180d',
        'adventureworks.quickstart.no-order-90d',
        'adventureworks.quickstart.product-line',
    ],
)
def test_cross_warehouse_cases_have_no_unimplemented_lakebase_alias(
    monkeypatch: pytest.MonkeyPatch, leg: str, name: str
) -> None:
    case = _case('contract/cases/s2/' + name + '.json')
    assert any(route.endswith('/lakebase') for route in case['routes'])
    routes = RUNNER.routes_on(case, leg)
    assert routes and not any(route.endswith('/lakebase') for route in routes)
    monkeypatch.setenv('SDM_CORE_ROUTES', ','.join(routes))
    assert _routing.forced() == frozenset(routes)
    assert RUNNER.case_key(case['id'], leg) == case['id'] + '@' + leg


@pytest.mark.parametrize(
    'name,leg',
    [
        ('databricks.connect.warehouse', 'databricks-sql'),
        ('databricks.connect.spark', 'databricks-spark'),
        ('lakebase.connect', 'lakebase'),
        ('snowflake.connect.snowpark', 'snowflake'),
    ],
)
def test_existing_connection_journeys_force_core_calls_but_not_host_sources(
    monkeypatch: pytest.MonkeyPatch, name: str, leg: str
) -> None:
    case = _case('contract/cases/s2/' + name + '.json')
    declared = list(case['routes'])
    routes = RUNNER.routes_on(case, leg)
    # Opening and connecting a source are host operations in scope-python;
    # their catalog paths remain coverage facts, never forced core routes.
    assert not any(route.startswith('source.') for route in routes)
    assert case['routes'] == declared
    assert _on(_routing.FROM_SOURCE_DUCKDB, leg) <= set(routes)
    assert _on(_routing.MATERIALIZE_DUCKDB, leg) <= set(routes)
    assert _on(_routing.PREDICT_DUCKDB, leg) <= set(routes)
    assert set(routes) <= set(_routing.PATHS)
    monkeypatch.setenv('SDM_CORE_ROUTES', ','.join(routes))
    assert _routing.forced() == frozenset(routes)


def test_an_unknown_core_route_is_not_silently_filtered_with_host_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case('contract/cases/s2/databricks.connect.warehouse.json')
    unknown = 'query.validate/not-a-real-mode/any'
    case['routes'].append(unknown)
    routes = RUNNER.routes_on(case, 'databricks-sql')
    assert unknown in routes
    monkeypatch.setenv('SDM_CORE_ROUTES', ','.join(routes))
    with pytest.raises(_routing.RouteError, match='unknown path'):
        _routing.forced()


@pytest.mark.parametrize(
    'backend',
    [
        'postgres',
        'databricks-sql',
        'databricks-spark',
        'databricks-metric-view',
        'snowflake',
        'snowflake-semantic-view',
    ],
)
def test_other_explicit_source_kinds_keep_their_route(backend: str) -> None:
    path = 'schema.from-source/all-tables/' + backend
    assert RUNNER.on_database(path, 'lakebase') == path


@pytest.mark.parametrize(
    'leg,given,parts,kind,keywords',
    [
        ('lakebase', {}, ['sch'], 'PostgresSource', {'schema': 'sch'}),
        (
            'snowflake',
            {},
            ['cat', 'sch'],
            'SnowflakeSource',
            {'database': 'cat', 'schema': 'sch'},
        ),
        (
            'databricks-sql',
            {},
            ['cat', 'sch'],
            'DatabricksSource',
            {'catalog': 'cat', 'schema': 'sch'},
        ),
        (
            'databricks-spark',
            {'named_part': 'schema'},
            [None, 'sch'],
            'DatabricksSource',
            {'schema': 'sch'},
        ),
        (
            'snowflake',
            {'named_part': 'catalog'},
            ['cat', None],
            'SnowflakeSource',
            {'database': 'cat'},
        ),
        ('lakebase', {'connect': 'lakebase'}, [], 'Source', {}),
    ],
)
def test_warehouse_namespace_facts_are_the_real_source_arguments(
    monkeypatch: pytest.MonkeyPatch,
    leg: str,
    given: dict,
    parts: list,
    kind: str,
    keywords: dict,
) -> None:
    import sdm_connectors

    connection = object()
    received = []

    def construct(value, **named):
        received.append((value, named))
        return connection

    monkeypatch.setattr(sdm_connectors, kind, construct)
    names = {'catalog': 'cat', 'schema': 'sch'}
    assert RUNNER.warehouse_namespace(leg, names, given) == parts
    assert RUNNER.warehouse_source(leg, connection, names, given) is connection
    assert received == [(connection, keywords)]
