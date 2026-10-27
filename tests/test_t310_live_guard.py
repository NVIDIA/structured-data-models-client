# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The existing runner installs the existing guard before any live call."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / 'contract/runners/run_python.py'
    spec = importlib.util.spec_from_file_location('t310_guard_runner', path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_live_guard_wraps_runner_calls_and_is_released_on_failure(
    tmp_path, monkeypatch
):
    run = _runner()
    events = []
    loader = object()
    guard = object()
    helper = SimpleNamespace(
        load_guard=lambda: guard,
        install_guard=lambda leg, seen, actual: (
            events.append(('install', leg, seen is guard, actual is loader))
            or (lambda: events.append('uninstall'))
        ),
    )
    monkeypatch.setattr(run, '_capture_run', lambda: helper)
    monkeypatch.setattr(run, '_sibling', lambda name: loader)
    monkeypatch.setattr(
        run, 'load_cases', lambda paths: [({'id': 'dummy.guard'}, 'digest')]
    )
    monkeypatch.setattr(run, 'inventory_cases', lambda ids: [])
    monkeypatch.setattr(run, 'environment_refusal', lambda env: None)

    def fail(*args, **kwargs):
        assert events == [('install', 'snowflake', True, True)]
        raise run.RunnerFailure('owned test refusal')

    monkeypatch.setattr(run, 'run', fail)
    assert (
        run.main(
            [
                '--mode',
                'python-core',
                '--database',
                'snowflake',
                '--guard-live',
                '--out',
                str(tmp_path / 'out.jsonl'),
                str(tmp_path / 'case.json'),
            ]
        )
        == 1
    )
    assert events[-1] == 'uninstall'


def test_live_guard_option_refuses_an_offline_leg_before_running(tmp_path):
    run = _runner()
    with pytest.raises(SystemExit) as refused:
        run.main(
            [
                '--mode',
                'python-core',
                '--guard-live',
                '--out',
                str(tmp_path / 'out.jsonl'),
            ]
        )
    assert refused.value.code == 2


def test_published_warehouse_observations_agree_without_changing_raw_replay(
    tmp_path, monkeypatch
):
    import copy
    import json
    import os
    import subprocess

    run = _runner()
    config = tmp_path / 'live.json'
    config.write_text(
        json.dumps(
            {'format': 'sdm-live/1', 'legs': {'databricks-sql': {'driver': {}}}}
        )
    )
    names = {
        'catalog': 'owner_catalog',
        'schema': 'owner_schema',
        'shop_metric_view': 'owner_metric_view',
    }
    environment = {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in run.PRIVATE_ENVIRONMENT
            and not key.startswith('SDM_CAPTURE_')
        },
        # A CI account named "runner" collides with every identity's key;
        # these synthetic private values belong only to this fixture.
        'USER': 'fixture_warehouse_operator_7afe1',
        'LOGNAME': 'fixture_warehouse_operator_7afe1',
        'SDM_LIVE_CONFIG': str(config),
        **{
            'SDM_CAPTURE_' + name.upper(): value
            for name, value in names.items()
        },
    }
    raw = {
        'case': 'dummy.publish@databricks-sql',
        'identity': {
            'runner': 'python-core',
            'operations': ['describe_source'],
            'contract': 'test',
        },
        'core': {
            'inputs': [
                {
                    'op': 'describe_source',
                    'args': {
                        'source': {
                            'namespace': ['owner_catalog', 'owner_schema'],
                            'semantic_view': 'owner_metric_view',
                        }
                    },
                }
            ],
            'fetches': [
                {
                    'statements': [
                        {
                            'sql': 'SELECT * FROM owner_catalog.owner_schema.owner_metric_view'
                        }
                    ]
                }
            ],
            'described': {'name': 'owner_metric_view'},
            'materialized': {'value': 0.1, 'integer': 9007199254740993},
            'predicted': None,
            'accepted': None,
            'error': None,
        },
        'public': {
            'payloads': [
                {
                    'parsed': {
                        'amount': 0.1,
                        'quantity': 9007199254740993,
                        'text': 'ordinary feature',
                    }
                }
            ]
        },
    }
    original = copy.deepcopy(raw)
    replay = tmp_path / 'fetches.private.jsonl'
    replay.write_text(
        json.dumps(
            {
                'core_inputs': raw['core']['inputs'],
                'fetches': raw['core']['fetches'],
            }
        )
        + '\n'
    )
    replay_bytes = replay.read_bytes()
    published = []
    for who in ('python-core', 'rust-host'):
        source = tmp_path / (who + '.raw.jsonl')
        target = tmp_path / (who + '.jsonl')
        entry = copy.deepcopy(raw)
        entry['identity']['runner'] = who
        source.write_text(json.dumps(entry) + '\n')
        before = source.read_bytes()
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(ROOT / 'contract/runners/run_python.py'),
                'publish',
                '--database',
                'databricks-sql',
                '--on',
                str(source),
                '--out',
                str(target),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert (
            source.read_bytes() == before
            and replay.read_bytes() == replay_bytes
        )
        placed = json.loads(target.read_text())
        assert placed['public'] == original['public']
        assert (
            placed['core']['materialized'] == original['core']['materialized']
        )
        published.append(placed)
    expected = run.Placeheld(names, names.values()).value(raw)
    expected['identity']['runner'] = (
        'java'  # the Java runner's existing publication contract
    )
    spec = importlib.util.spec_from_file_location(
        't310_publication_compare', ROOT / 'scripts/rust_core/compare.py'
    )
    assert spec is not None and spec.loader is not None
    comparator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparator
    spec.loader.exec_module(comparator)
    for value in published:
        report = comparator.Report()
        comparator._core_agree(
            report, {'id': raw['case']}, value, expected, host='java'
        )
        assert not report.failures
    changed = copy.deepcopy(published[1])
    changed['core']['materialized']['value'] = 0.2
    report = comparator.Report()
    comparator._core_agree(report, {'id': raw['case']}, published[0], changed)
    assert report.failures  # meaningful result differences are still compared

    leaked = copy.deepcopy(raw)
    leaked['public']['payloads'][0]['parsed']['text'] = environment['USER']
    source = tmp_path / 'leaked.raw.jsonl'
    source.write_text(json.dumps(leaked) + '\n')
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            'publish',
            '--database',
            'databricks-sql',
            '--on',
            str(source),
            '--out',
            str(tmp_path / 'leaked.jsonl'),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 1
    assert 'observation publication failed (RunnerFailure)' in done.stderr
