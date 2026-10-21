# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI's cross-host step fails closed on evidence missing from both hosts
at once (T1.29 tests; the standing rule that a gate rule fails when its
evidence is missing, empty or retyped, not only when it differs).

``contract/tests/test_cross_host.py`` (the implementer's) seeds each fault
on one host's side. A fault on both sides is one the ``core`` rule, which
compares the hosts with each other, cannot see by difference; the pin the
step replaced held both to one recording. Probed on T1.29's first
candidate: dropping, nulling or retyping a whole section on both hosts, or
an acceptance case on either, fails the step by another rule; an acceptance
case reaching ``predict`` with no fetch on both hosts passes it. Seeded
here, through ``test_cross_host.py``'s own step runner and stand-in:

- every acceptance case reaching ``predict`` (it sends a request built
  from fetched rows) with its step trace emptied on both hosts fails the
  step;
- the controls that already fail: the case absent from both, and its
  ``core`` section empty on both.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

HERE = Path(__file__).resolve().parent
if 'cross_host_tests' in sys.modules:
    cross = sys.modules['cross_host_tests']
else:
    spec = importlib.util.spec_from_file_location(
        'cross_host_tests', HERE / 'test_cross_host.py'
    )
    assert spec is not None and spec.loader is not None
    cross = importlib.util.module_from_spec(spec)
    sys.modules['cross_host_tests'] = cross
    spec.loader.exec_module(cross)

observed = cross.observed

#: The acceptance cases whose request the core builds from fetched rows.
PREDICTING = (
    'base.regression',
    'base.binary',
    'base.multiclass',
    'base.adventureworks',
)


def _both(
    observed: dict[str, list[Any]], case: str, change: Any
) -> dict[str, str | None]:
    changed = copy.deepcopy(observed)
    for side in ('python-core', 'rust-host'):
        lines = changed[side]
        found = [line for line in lines if line['case'] == case]
        assert len(found) == 1, (side, case)
        change(lines, found[0])
    return cross._files(changed)


def _no_fetch(lines: list[Any], line: dict[str, Any]) -> None:
    assert line['core']['fetches'], 'the case fetched nothing to empty'
    line['core']['fetches'] = []


@pytest.mark.parametrize('case', PREDICTING)
def test_no_fetch_on_both_hosts_fails_the_step(
    tmp_path: Path, observed: dict[str, list[Any]], case: str
) -> None:
    r"""Both hosts claim a request built from rows with no fetch recorded:
    the evidence of what the core fetched is empty, which no rule may
    take as agreement."""
    done = cross._run(tmp_path, _both(observed, case, _no_fetch))
    assert done.returncode != 0, done.stdout
    assert 'DIFFERS ' in done.stdout and case in done.stdout, done.stdout


def test_the_case_absent_from_both_fails_the_step(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    def drop(lines: list[Any], line: dict[str, Any]) -> None:
        lines.remove(line)

    done = cross._run(tmp_path, _both(observed, 'base.regression', drop))
    assert done.returncode != 0, done.stdout
    assert 'base.regression: rust-host: runner failure: no observation' in (
        done.stdout
    )


def test_an_empty_core_section_on_both_fails_the_step(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    def empty(lines: list[Any], line: dict[str, Any]) -> None:
        line['core'] = {}

    done = cross._run(tmp_path, _both(observed, 'base.regression', empty))
    assert done.returncode != 0, done.stdout
