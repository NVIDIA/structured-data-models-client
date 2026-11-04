# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise API evidence collection through the real pytest lifecycle."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    'selection', [['--gate', 'native'], ['--group', 'unknown']]
)
def test_empty_selection_never_collects_unrelated_tests(tmp_path, selection):
    marker = tmp_path / 'collected'
    (tmp_path / 'test_unrelated.py').write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name('api_contracts.py').resolve()),
            *selection,
            '--run',
            '--out',
            str(tmp_path / 'result.json'),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'no tests selected' in result.stderr
    assert not marker.exists()
    assert not (tmp_path / 'result.json').exists()


@pytest.mark.parametrize(
    ('extra', 'complete', 'completed'),
    [([], True, 2), (['-k', 'one'], False, 1)],
)
def test_parameterized_results_and_deselection(
    tmp_path, extra, complete, completed
):
    suite = tmp_path / 'test_example.py'
    suite.write_text(
        'import pytest\n'
        '@pytest.mark.parametrize("value", [1, 2], ids=["one", "two"])\n'
        'def test_value(value):\n'
        '    assert value > 0\n'
    )
    runner = (
        'import json, pytest\n'
        'from api_contracts import Outcomes\n'
        'outcomes = Outcomes()\n'
        f'pytest.main({[str(suite), "-q", *extra]!r}, plugins=[outcomes])\n'
        'print(json.dumps({"complete": outcomes.complete, '
        '"expected": outcomes.expected, "completed": outcomes.completed, '
        '"deselected": outcomes.deselected}))\n'
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parent),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
    )
    result = subprocess.run(
        [sys.executable, '-c', runner],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence['complete'] is complete
    assert len(evidence['completed']) == completed
    assert all('test_value[' in node for node in evidence['completed'])
    assert bool(evidence['deselected']) is not complete


def test_missing_duplicate_and_unexpected_results_fail():
    from api_contracts import Outcomes

    outcome = Outcomes()
    outcome.expected = ['test.py::test_a', 'test.py::test_b']
    for results in (
        ['test.py::test_a'],
        ['test.py::test_a', 'test.py::test_a', 'test.py::test_b'],
        ['test.py::test_a', 'test.py::test_b', 'test.py::test_c'],
    ):
        outcome.completed = results
        assert not outcome.complete


@pytest.mark.parametrize(
    'reported',
    ['test held ... ignored\n', '', 'test held ... ok\ntest held ... ok\n'],
)
def test_rust_missing_ignored_and_duplicate_results_fail(
    tmp_path, monkeypatch, reported
):
    from api_contracts import rust_run

    def run(command, **kwargs):
        listing = '--list' in command
        return subprocess.CompletedProcess(
            command, 0, 'held: test\n' if listing else reported, ''
        )

    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ValueError, match='evidence incomplete'):
        rust_run(
            ['rust/crates/sdm-host/tests/admission.rs::held'],
            tmp_path / 'results.json',
        )


@pytest.mark.parametrize('changed', [False, True])
def test_api_ledger_checks_windows_checkout(monkeypatch, changed):
    from api_contracts import ROOT, validate

    from contract import manifest

    snapshot = ROOT / 'tests/golden/public_api.json'
    original = Path.read_bytes

    def checkout_bytes(path):
        value = original(path)
        if path == snapshot:
            if changed:
                value = value.replace(b'"sdm_client"', b'"changed_client"')
            return value.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        return value

    monkeypatch.setattr(Path, 'read_bytes', checkout_bytes)
    if changed:
        with pytest.raises(AssertionError, match='public surface changed'):
            validate(manifest())
    else:
        validate(manifest())


def test_runner_makes_repository_fixture_helpers_importable(tmp_path):
    suite = tmp_path / 'test_import.py'
    suite.write_text(
        'def test_import():\n    from tests.parity.routes import LegacyGuard\n    assert LegacyGuard\n'
    )
    runner = (
        'import sys, api_contracts as api\n'
        "sys.argv = ['api_contracts', '--run', '--out', sys.argv[1]]\n"
        f"api.groups = lambda *args: [{{'id': 'probe', 'tests': {{'python': [{str(suite)!r}]}}}}]\n"
        'raise SystemExit(api.main())\n'
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parent.resolve()),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
    )
    result = subprocess.run(
        [sys.executable, '-c', runner, str(tmp_path / 'result.json')],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / 'result.json').read_text())['passed'] == 1


@pytest.mark.parametrize('registered', [True, False])
def test_local_gate_excludes_only_explicit_live_ownership(tmp_path, registered):
    suite = tmp_path / 'test_owned.py'
    suite.write_text(
        'import pytest\n'
        'def test_local():\n    assert True\n'
        '@pytest.mark.parametrize("value", [1, 2])\n'
        'def test_live(value):\n    pytest.skip("endpoint required")\n'
    )
    owner = ['test_owned.py::test_live'] if registered else []
    runner = (
        'import json, pytest\n'
        'from api_contracts import Outcomes\n'
        f'outcomes = Outcomes(live_owned={owner!r})\n'
        f'pytest.main({[str(suite), "-q"]!r}, plugins=[outcomes])\n'
        'print(json.dumps({"complete": outcomes.complete, '
        '"expected": outcomes.expected, "completed": outcomes.completed, '
        '"live_owned": outcomes.live_owned_tests, "skipped": outcomes.skipped_tests}))\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', runner],
        cwd=tmp_path,
        env={
            **os.environ,
            'PYTHONPATH': str(Path(__file__).parent.resolve()),
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        },
        capture_output=True,
        text=True,
        check=True,
    )
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence['complete'] is registered
    assert evidence['completed'] == ['test_owned.py::test_local']
    live = ['test_owned.py::test_live[1]', 'test_owned.py::test_live[2]']
    assert evidence['live_owned'] == (live if registered else [])
    assert evidence['skipped'] == ([] if registered else live)
    assert evidence['expected'] == ['test_owned.py::test_local'] + (
        [] if registered else live
    )


def test_required_local_skip_remains_incomplete_with_live_ownership():
    from types import SimpleNamespace

    from api_contracts import Outcomes

    outcomes = Outcomes(live_owned=['test.py::test_live'])
    items = [
        SimpleNamespace(nodeid='test.py::test_local'),
        SimpleNamespace(nodeid='test.py::test_live[1]'),
    ]
    outcomes.pytest_collection_modifyitems(items)
    outcomes.pytest_collection_finish(SimpleNamespace(items=items))
    outcomes.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid='test.py::test_local', failed=False, skipped=True
        )
    )
    assert not outcomes.complete
    assert outcomes.expected == ['test.py::test_local']
    assert outcomes.skipped_tests == ['test.py::test_local']
