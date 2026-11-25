# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted receipts cannot turn partial or mismatched runs into acceptance."""

import contextlib
import json

import pytest
from hosted_aggregate import require_models
from hosted_model_smoke import require_tests


def test_model_evidence_requires_same_artifacts_and_all_tests(tmp_path):
    identity = {'commit': 'candidate', 'ci_run_id': 123}
    (tmp_path / 'artifacts.json').write_text(json.dumps(identity))
    smoke = {
        'status': 'passed',
        'device': 'cpu',
        'python_tests': 4,
        'java_tests': 3,
        'gradient_combinations_per_language': 4,
        'java_execution': 'source-equivalent-with-verified-packaged-jni',
    }
    path = tmp_path / 'cpu-model-smoke.json'
    path.write_text(json.dumps(smoke))
    assert require_models(tmp_path, identity) == smoke
    with pytest.raises(ValueError, match='CI artifacts'):
        require_models(tmp_path, {**identity, 'ci_run_id': 456})
    path.write_text(json.dumps({**smoke, 'java_tests': 2}))
    with pytest.raises(ValueError, match='CI artifacts'):
        require_models(tmp_path, identity)


@pytest.mark.parametrize('tag', ['skipped', 'failure', 'error'])
def test_skipped_or_failed_cpu_test_cannot_pass(tmp_path, tag):
    path = tmp_path / 'tests.xml'
    path.write_text(f'<testsuite><testcase><{tag}/></testcase></testsuite>')
    with pytest.raises(ValueError, match='did not all pass'):
        require_tests(path, 1)


def test_cleanup_failure_is_recorded_as_incomplete(tmp_path, monkeypatch):
    import hosted_runner

    @contextlib.contextmanager
    def failed_cleanup(_):
        yield {}
        raise RuntimeError('cleanup failed')

    monkeypatch.setattr(hosted_runner, 'provision', failed_cleanup)
    monkeypatch.setattr(
        hosted_runner.subprocess,
        'run',
        lambda *args, **kwargs: type('Result', (), {'returncode': 0})(),
    )
    for name in ('PARITY_PROVENANCE', 'PARITY_NATIVE', 'PARITY_JAR'):
        monkeypatch.setenv(name, 'test-only')
    with pytest.raises(RuntimeError, match='cleanup failed'):
        hosted_runner.run('memory', tmp_path, 'candidate')
    receipt = json.loads((tmp_path / 'evidence/execution.json').read_text())
    assert receipt['completed'] is False
    assert not (tmp_path / 'private/fixture-config.json').exists()


@pytest.mark.parametrize('connector', ['memory', 'sqlite', 'databricks-spark'])
def test_only_spark_runner_requires_connect_runtime(
    tmp_path, monkeypatch, connector
):
    import hosted_runner

    monkeypatch.setattr(
        hosted_runner, 'provision', lambda _: contextlib.nullcontext({})
    )
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(hosted_runner.subprocess, 'run', execute)
    for name in ('PARITY_PROVENANCE', 'PARITY_NATIVE', 'PARITY_JAR'):
        monkeypatch.setenv(name, 'test-only')
    hosted_runner.run(connector, tmp_path, 'candidate')
    runtime = [
        arg
        for arg in commands[1]
        if arg.startswith('-Dsdm.parity.connect.classpath=')
    ]
    assert bool(runtime) == (connector == 'databricks-spark')


def test_shared_parquet_preserves_nanosecond_timezone_and_cleans_up():
    from pathlib import Path

    import pandas as pd
    import pyarrow.parquet as pq
    from hosted_fixtures import parquet

    with parquet() as config:
        paths = config['fixture_sources']['fixtures/shop-temporal.json'][
            'parquet'
        ]['paths']
        path = Path(paths['users'])
        data = pq.read_table(path)
        dtype = data.schema.field('observed_at').type
        assert dtype.unit == 'ns'
        assert dtype.tz == 'America/Los_Angeles'
        assert (
            data['observed_at'][1].value
            == pd.Timestamp('2024-03-10T03:00:00.987654321-07:00').value
        )
        assert data['observed_at'][0].as_py() is None
    assert not path.exists()
