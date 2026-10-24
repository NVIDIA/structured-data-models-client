# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Phase 3 switch uses the core without a test routing override."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sdm_client import _routing

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    'paths',
    [
        _routing.S0,
        _routing.CAPABILITIES,
        _routing.FROM_SOURCE_DUCKDB,
        _routing.MATERIALIZE_DUCKDB,
        _routing.PREDICT_DUCKDB,
        _routing.FROM_SOURCE_SQLITE,
        _routing.MATERIALIZE_SQLITE,
        _routing.PREDICT_SQLITE,
        _routing.FROM_SOURCE_POSTGRES,
        _routing.MATERIALIZE_POSTGRES,
        _routing.PREDICT_POSTGRES,
        _routing.FROM_SOURCE_DATABRICKS_SQL,
        _routing.MATERIALIZE_DATABRICKS_SPARK,
        _routing.PREDICT_DATABRICKS_SPARK,
        _routing.FROM_SOURCE_DATABRICKS_METRIC_VIEW,
        _routing.MATERIALIZE_DATABRICKS_METRIC_VIEW,
        _routing.PREDICT_DATABRICKS_METRIC_VIEW,
        _routing.FROM_SOURCE_SNOWFLAKE,
        _routing.MATERIALIZE_SNOWFLAKE,
        _routing.PREDICT_SNOWFLAKE,
        _routing.FROM_SOURCE_SNOWFLAKE_SEMANTIC_VIEW,
        _routing.MATERIALIZE_SNOWFLAKE_SEMANTIC_VIEW,
        _routing.PREDICT_SNOWFLAKE_SEMANTIC_VIEW,
    ],
)
def test_covered_calls_default_to_the_core(
    paths: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    assert _routing.route(paths) == 'core'


def test_an_explicit_override_can_select_old_then_restore_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    assert _routing.route(_routing.S0) == 'core'
    monkeypatch.setenv(_routing.ENVIRONMENT, '')
    assert _routing.route(_routing.S0) == 'old'
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(_routing.S0))
    assert _routing.route(_routing.S0) == 'core'
    assert _routing.route(_routing.FROM_SOURCE_DUCKDB) == 'old'
    monkeypatch.delenv(_routing.ENVIRONMENT)
    assert _routing.route(_routing.FROM_SOURCE_DUCKDB) == 'core'


def test_unimplemented_training_still_uses_its_existing_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    assert _routing.route(('client.fit/regression/duckdb',)) == 'old'


@pytest.mark.parametrize('backend', ['duckdb', 'sqlite'])
def test_public_source_flow_defaults_to_core_without_old_packages(
    backend: str, tmp_path: Path
) -> None:
    script = f"""
    import importlib.util
    import sys
    import warnings

    OLD = ('predictive_query', 'sdm_relational_engine')

    class RefuseOld:
        def find_spec(self, name, path=None, target=None):
            if name in OLD or name.startswith(tuple(o + '.' for o in OLD)):
                raise ImportError(name + ' must not be imported')

    sys.meta_path.insert(0, RefuseOld())

    from sdm_client import KumoRelational, PredictiveQuery, SDMClient, Schema
    from sdm_connectors import DuckDBSource, SQLiteSource

    if {backend!r} == 'duckdb':
        import duckdb
        connection = duckdb.connect()
        source = DuckDBSource(connection)
    else:
        from adbc_driver_sqlite import dbapi
        connection = dbapi.connect()
        source = SQLiteSource(connection)

    cursor = connection.cursor()
    cursor.execute('CREATE TABLE users (user_id INTEGER PRIMARY KEY, age DOUBLE)')
    cursor.execute('INSERT INTO users VALUES (1,10.5),(2,20.5),(3,30.5),(4,40.5)')
    connection.commit()

    spec = importlib.util.spec_from_file_location(
        'scripted_endpoint', {str(ROOT / 'contract/runners/scripted_endpoint.py')!r}
    )
    endpoint = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = endpoint
    spec.loader.exec_module(endpoint)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(source, {{'users': 'users'}})
        schema['users']['age'].stype = 'numerical'
        with endpoint.serve() as (url, received):
            request = PredictiveQuery(
                'PREDICT users.age FOR EACH users.user_id', schema
            ).materialize(indices=[1,2])
            with SDMClient(url, max_retries=0) as client:
                result = client.predict(request, model=KumoRelational())
        assert len(result) == 2
        assert len(received) == 1
    assert 'sdm_client._native' in sys.modules
    assert not any(name in sys.modules for name in OLD)
    print('core-default-ok')
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != _routing.ENVIRONMENT
    }
    done = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'core-default-ok'
