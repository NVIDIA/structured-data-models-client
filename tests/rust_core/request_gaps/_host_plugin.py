# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A pytest plugin that measures each test it runs (``CONTRACT.md``, "Host
tests"): its outcome, how many calls it made into ``sdm_core._native``, and
the branches it took, under branch coverage of today's Python, in the
modules ``SDM_P86_HOST_MODULES`` names. It writes them, by node id, to
``SDM_P86_HOST_OUT`` when the session ends.

Loaded with ``-p _host_plugin``, this directory on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ``scripts/parity/coverage.py`` would shadow the package: import it first.
import coverage
import pytest

PACKAGES = (
    'sdm_relational_engine',
    'sdm_client',
    'sdm_connectors',
    'predictive_query',
)
REPO = Path(__file__).resolve().parents[3]

_RESULTS: dict[str, dict[str, Any]] = {}
_CALLS = {'n': 0}
_STATE: dict[str, Any] = {}


def _native_call(code: Any, offset: int, callable_: Any, arg0: Any) -> Any:
    owner = getattr(callable_, '__self__', None)
    if getattr(callable_, '__module__', None) == 'sdm_core._native' or (
        owner is not None
        and getattr(type(owner), '__module__', None) == 'sdm_core._native'
    ):
        _CALLS['n'] += 1
    return None


def _watch_native() -> None:
    monitoring = sys.monitoring
    for tool in range(6):
        if monitoring.get_tool(tool) is None:
            monitoring.use_tool_id(tool, 'sdm-p86-native-calls')
            break
    else:
        raise RuntimeError('no sys.monitoring tool id is free')
    monitoring.register_callback(tool, monitoring.events.CALL, _native_call)
    monitoring.set_events(tool, monitoring.events.CALL)
    _STATE['tool'] = tool


def pytest_configure(config: pytest.Config) -> None:
    _STATE['coverage'] = coverage.Coverage(
        branch=True,
        data_file=None,
        source_pkgs=list(PACKAGES),
        omit=['*/predictive_query/grammar/PQLGrammar*'],
    )
    _STATE['modules'] = json.loads(os.environ.get('SDM_P86_HOST_MODULES', '[]'))
    _watch_native()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Any) -> Iterator[None]:
    cov = _STATE['coverage']
    cov.erase()
    _CALLS['n'] = 0
    _RESULTS[item.nodeid] = {'outcomes': [], 'native_calls': 0, 'executed': {}}
    cov.start()
    try:
        yield
    finally:
        cov.stop()
    found = _RESULTS[item.nodeid]
    found['native_calls'] = _CALLS['n']
    with tempfile.TemporaryDirectory(prefix='sdm-p86-item-') as tmp:
        out = Path(tmp) / 'item.json'
        try:
            cov.json_report(
                outfile=str(out),
                include=[str(REPO / module) for module in _STATE['modules']],
            )
        except coverage.exceptions.NoDataError:
            return
        report = json.loads(out.read_text('utf-8'))
    found['executed'] = {
        module: body.get('executed_branches', [])
        for module, body in report.get('files', {}).items()
    }


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    found = _RESULTS.get(report.nodeid)
    if found is not None:
        found['outcomes'].append([report.when, report.outcome])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get('SDM_P86_HOST_OUT')
    if out:
        Path(out).write_text(json.dumps(_RESULTS), 'utf-8')
