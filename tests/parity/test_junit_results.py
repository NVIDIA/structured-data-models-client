# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required JUnit methods and every reported invocation must pass."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('gate', ['local', 'live'])
@pytest.mark.parametrize(
    'fault', ['missing', 'duplicate', 'skipped', 'failure', 'error', None]
)
def test_required_junit_results(tmp_path, fault, gate):
    coverage = runpy.run_path(str(ROOT / 'tests/parity/check_junit.py'))
    ledger = tmp_path / 'tests/parity/manifest.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                'live_api_tests': [
                    {
                        'api_contract': 'example',
                        'test': 'java/src/test/java/com/nvidia/sdm/ExampleTest.java#required',
                    }
                ],
                'api_contracts': [
                    {
                        'gate': 'local',
                        'tests': {
                            'java': [
                                'java/src/test/java/com/nvidia/sdm/ExampleTest.java#required',
                            ]
                        },
                    }
                ],
            }
        )
    )
    case = '<testcase classname="com.nvidia.sdm.ExampleTest" name="required(int)[1]">'
    if fault in ('skipped', 'failure', 'error'):
        case += f'<{fault}/>'
    case += '</testcase>'
    if fault == 'missing':
        case = case.replace('required(int)[1]', 'unrelated')
    if fault == 'duplicate':
        case *= 2
    reports = tmp_path / 'reports'
    reports.mkdir()
    (reports / 'TEST-ExampleTest.xml').write_text(
        f'<testsuite>{case}</testsuite>'
    )
    report = coverage['Report']()
    coverage['parity_junit_check'](tmp_path, reports, report, gate)
    assert bool(report.problems) is (fault is not None)


def test_missing_ledger_is_not_a_success(tmp_path):
    checker = runpy.run_path(str(ROOT / 'tests/parity/check_junit.py'))
    report = checker['Report']()
    checker['parity_junit_check'](tmp_path, tmp_path / 'reports', report)
    assert report.problems and 'missing ledger' in report.problems[0]


@pytest.mark.parametrize('exit_code', [0, 7])
def test_runner_uses_fresh_reports_and_preserves_natives(
    tmp_path, monkeypatch, exit_code
):
    import sys
    from types import SimpleNamespace

    checker = runpy.run_path(str(ROOT / 'tests/parity/check_junit.py'))
    reports = tmp_path / 'java/target/surefire-reports'
    reports.mkdir(parents=True)
    stale = reports / 'TEST-stale.xml'
    stale.write_text('<testsuite/>')
    native = tmp_path / 'java/target/natives/library.so'
    native.parent.mkdir()
    native.write_bytes(b'keep-native')
    ledger = tmp_path / 'tests/parity/manifest.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({'api_contracts': []}))
    calls = []

    def run(command, *, cwd, check):
        assert not reports.exists()
        assert native.read_bytes() == b'keep-native'
        assert cwd == tmp_path
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(checker['subprocess'], 'run', run)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'check_junit.py',
            '--repo',
            str(tmp_path),
            '--run',
            '--maven',
            'test-maven',
        ],
    )
    assert checker['main']() == exit_code
    assert calls == [['test-maven', '-B', '-f', 'java/pom.xml', 'verify']]
    assert native.read_bytes() == b'keep-native'


def test_live_missing_parameterized_invocation_fails(tmp_path):
    from check_junit import Report, parity_junit_check

    ledger = tmp_path / 'tests/parity/manifest.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                'live_api_tests': [
                    {
                        'api_contract': 'example',
                        'test': 'java/src/test/java/com/nvidia/sdm/ExampleTest.java#required',
                        'invocations': 2,
                    }
                ]
            }
        )
    )
    reports = tmp_path / 'reports'
    reports.mkdir()
    (reports / 'TEST-ExampleTest.xml').write_text(
        '<testsuite><testcase classname="com.nvidia.sdm.ExampleTest" name="required(int)[1]"/></testsuite>'
    )
    report = Report()
    parity_junit_check(tmp_path, reports, report, 'live')
    assert any(
        'expected 2 invocations, got 1' in problem
        for problem in report.problems
    )
