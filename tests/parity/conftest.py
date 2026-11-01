# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional whole-test guard wraps fixture setup through teardown."""

import pytest
from routes import LegacyGuard


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if item.get_closest_marker('parity_default_routes') is None:
        yield
        return
    with LegacyGuard(raise_caught=False) as guard:
        item._parity_guard = guard
        item.user_properties.append(('parity_route', guard.report))
        yield


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'parity_default_routes: trap legacy policy for the whole public journey',
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    guard = getattr(item, '_parity_guard', None)
    if (
        guard is not None
        and (
            guard.report['legacy_hits']
            or guard.report['unobserved_subprocesses']
        )
        and report.passed
    ):
        report.outcome = 'failed'
        report.longrepr = 'Legacy policy was invoked: ' + ', '.join(
            guard.report['legacy_hits']
        )
