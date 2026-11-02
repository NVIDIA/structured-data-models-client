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
