# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional prerequisites may skip; suppressed failures remain forbidden."""

import runpy
from pathlib import Path

import pytest

SCAN = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / 'scripts/rust_core/architecture_policy.py'
    )
)


@pytest.mark.parametrize(
    'source',
    [
        "pytest.skip('PostgreSQL is not configured')",
        "@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX only')",
        "pytest.importorskip('optional_driver')",
    ],
)
def test_optional_python_prerequisites_may_skip(source):
    assert list(SCAN['find_skips'](source, python=True)) == []


@pytest.mark.parametrize(
    'source',
    [
        '@pytest.mark.xfail',
        '@unittest.expectedFailure',
        'pytest.xfail("broken")',
    ],
)
def test_python_expected_failures_remain_forbidden(source):
    assert list(SCAN['find_skips'](source, python=True))


@pytest.mark.parametrize(
    'source',
    ['#[ignore]', '#[cfg_attr(miri, ignore)]', 'continue-on-error: true'],
)
def test_rust_ignored_tests_and_ci_failure_suppression_remain_forbidden(source):
    assert list(SCAN['find_skips'](source))
