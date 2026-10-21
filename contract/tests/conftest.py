# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How pytest-xdist shares the contract suite out under ``--dist loadfile``.

Each test file goes to one worker, as ``loadfile`` sends it, except the
files in :data:`ONE_AT_A_TIME`, whose module fixtures hold a whole family
run parsed (about 3.5 GB and 4 GB on Linux). They go to one worker together,
which runs them one after the other, so no two of them are in memory at
once: side by side on CI's two workers and about 8 GB they ran the runner
out of memory, and the worker killed mid-file left xdist waiting until the
job was killed. Every test still runs; only which worker runs it changes.

The gate's own slow files go out first (:data:`SLOW_FIRST`), in that order,
and then the rest as ``loadfile`` orders them, largest first: given last,
they ran on alone for 20 to 40 minutes after the others were done (T2.1q).
The slowest, ``test_gate_modes.py``, goes out as two units (:data:`SPLIT`).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import pytest

#: The files run on one worker, one after the other.
ONE_AT_A_TIME = frozenset(
    {
        'test_t132e_multicategorical.py',
        'test_t132e_time_feature_family.py',
    }
)

#: The scope the files in :data:`ONE_AT_A_TIME` share.
SHARED_SCOPE = 'contract/tests: a family run in memory'

#: The files given out before any other, slowest first as measured: most
#: run the gate or the ratchet over a temporary repository, minutes a test.
SLOW_FIRST = (
    'test_gate_modes.py',
    'test_cross_host.py',
    'test_t20f_gate.py',
    'test_gate_controls.py',
    'test_t132e_time_feature_family.py',
    'test_gate_evidence_paths.py',
    'test_t136_forecast.py',
    'test_cross_host_evidence.py',
    'test_t20_locked_set.py',
    'test_t20_checkpoints.py',
    'test_t20f_main_seeds.py',
)

#: A file given out as two units, by test name: its tests starting with the
#: prefix, and the rest (each unit builds the file's module fixtures).
SPLIT = {'test_gate_modes.py': 'test_locked_1a_'}


def scope_of(nodeid: str) -> str:
    r"""The unit a worker takes: the test's file, as ``loadfile`` has it,
    :data:`SHARED_SCOPE` for a file in :data:`ONE_AT_A_TIME`, or one of
    the two halves of a file in :data:`SPLIT`."""
    path, _, test = nodeid.partition('::')
    name = PurePosixPath(path).name
    if name in ONE_AT_A_TIME:
        return SHARED_SCOPE
    if name in SPLIT and test.startswith(SPLIT[name]):
        return f'{path}::{SPLIT[name]}'
    return path


def rank(scope: str) -> int:
    r"""Where a unit goes out: by :data:`SLOW_FIRST`, every other unit after
    (:data:`SHARED_SCOPE` where the first of its files is)."""
    names = (
        ONE_AT_A_TIME
        if scope == SHARED_SCOPE
        else {PurePosixPath(scope.partition('::')[0]).name}
    )
    return min(
        (SLOW_FIRST.index(name) for name in names if name in SLOW_FIRST),
        default=len(SLOW_FIRST),
    )


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config: pytest.Config, log: Any) -> Any:
    if config.getvalue('dist') != 'loadfile':
        return None
    from xdist.scheduler import LoadScopeScheduling

    class _Scheduling(LoadScopeScheduling):  # type: ignore[misc]
        def _split_scope(self, nodeid: str) -> str:
            return scope_of(nodeid)

        def _assign_work_unit(self, node: Any) -> None:
            # A stable sort keeps loadfile's order within each rank.
            for unit in sorted(self.workqueue, key=rank):
                self.workqueue.move_to_end(unit)
            super()._assign_work_unit(node)

    return _Scheduling(config, log)
