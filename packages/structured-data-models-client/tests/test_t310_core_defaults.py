# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public warehouse journeys load the native client without retired packages."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


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
    cursor.close()

    spec = importlib.util.spec_from_file_location(
        'scripted_endpoint', {str(ROOT / 'tests/support/scripted_endpoint.py')!r}
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
        assert len(received.requests) == 1
    connection.close()
    assert 'sdm_client._native' in sys.modules
    assert not any(name in sys.modules for name in OLD)
    print('core-default-ok')
    """
    done = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'core-default-ok'
