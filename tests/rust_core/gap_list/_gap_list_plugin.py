# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The pytest plugin the Phase 7.8 checks run the host tests with
(``CONTRACT.md``, "Measuring tests").

Loaded with ``-p _gap_list_plugin`` in a pytest process that runs under
``coverage run``. It names coverage's dynamic context after the test that
is running, ``<node id>|setup``, ``|call`` or ``|teardown``, and writes,
under the directory ``$SDM_P78_MEASURE_OUT`` names:

- ``outcomes.json``: each test's node id, ``passed`` when its setup, call
  and teardown all passed, otherwise ``failed`` or ``skipped``;
- ``core_calls.json``: of the tests ``$SDM_P78_WATCH_CORE`` names (a JSON
  list of node ids), those that called into ``sdm_core._native`` in this
  process while they ran.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_OUT = Path(os.environ.get('SDM_P78_MEASURE_OUT', '.'))
_WATCH: set[str] = set(json.loads(os.environ.get('SDM_P78_WATCH_CORE', '[]')))
_OUTCOMES: dict[str, str] = {}
_CORE_CALLS: set[str] = set()
_NATIVE = 'sdm_core._native'


def _switch(label: str) -> None:
    import coverage

    current = coverage.Coverage.current()
    if current is not None:
        current.switch_context(label)


def _is_core(callable_: Any) -> bool:
    r"""Whether a call is into the binding: one of its functions, a method
    of one of its objects, or one of its classes."""
    if getattr(callable_, '__module__', None) == _NATIVE:
        return True
    owner = getattr(callable_, '__self__', None)
    if owner is not None:
        if isinstance(owner, type):
            return owner.__module__ == _NATIVE
        return type(owner).__module__ == _NATIVE
    return isinstance(callable_, type) and callable_.__module__ == _NATIVE


class _CoreWatch:
    r"""Notes, with ``sys.monitoring``'s ``CALL`` events, whether the test
    called into the binding."""

    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.tool: int | None = None

    def _call(self, code: Any, offset: int, callable_: Any, arg0: Any) -> Any:
        if _is_core(callable_):
            _CORE_CALLS.add(self.nodeid)
        return None

    def __enter__(self) -> _CoreWatch:
        monitoring = sys.monitoring
        for tool in (4, 3, 1):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, 'sdm-p78-core-calls')
                self.tool = tool
                break
        if self.tool is not None:
            monitoring.register_callback(
                self.tool, monitoring.events.CALL, self._call
            )
            monitoring.set_events(self.tool, monitoring.events.CALL)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.tool is None:
            return
        monitoring = sys.monitoring
        monitoring.set_events(self.tool, 0)
        monitoring.register_callback(self.tool, monitoring.events.CALL, None)
        monitoring.free_tool_id(self.tool)


def _phase(item: pytest.Item, when: str) -> Iterator[None]:
    _switch(f'{item.nodeid}|{when}')
    if item.nodeid in _WATCH:
        with _CoreWatch(item.nodeid):
            yield
    else:
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[None]:
    yield from _phase(item, 'setup')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    yield from _phase(item, 'call')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Iterator[None]:
    yield from _phase(item, 'teardown')
    _switch('')


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    before = _OUTCOMES.get(report.nodeid, 'passed')
    if report.passed:
        now = before
    elif report.skipped:
        now = 'skipped' if before == 'passed' else before
    else:
        now = 'failed'
    _OUTCOMES[report.nodeid] = now


def pytest_sessionfinish(session: pytest.Session) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / 'outcomes.json').write_text(
        json.dumps(_OUTCOMES, indent=1, sort_keys=True), 'utf-8'
    )
    (_OUT / 'core_calls.json').write_text(
        json.dumps(sorted(_CORE_CALLS), indent=1), 'utf-8'
    )
