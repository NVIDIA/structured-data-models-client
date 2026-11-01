# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required JUnit methods and every reported invocation must pass."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    'fault', ['missing', 'duplicate', 'skipped', 'failure', None]
)
def test_required_junit_results(tmp_path, fault):
    coverage = runpy.run_path(str(ROOT / 'scripts/rust_core/coverage.py'))
    ledger = tmp_path / 'tests/parity/manifest.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                'api_contracts': [
                    {
                        'gate': 'local',
                        'tests': {
                            'java': [
                                'java/src/test/java/com/nvidia/sdm/ExampleTest.java#required',
                            ]
                        },
                    }
                ]
            }
        )
    )
    case = '<testcase classname="com.nvidia.sdm.ExampleTest" name="required(int)[1]">'
    if fault in ('skipped', 'failure'):
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
    coverage['parity_junit_check'](tmp_path, reports, report)
    assert bool(report.problems) is (fault is not None)
