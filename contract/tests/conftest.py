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


def scope_of(nodeid: str) -> str:
    r"""The unit a worker takes: the test's file, as ``loadfile`` has it,
    or :data:`SHARED_SCOPE` for a file in :data:`ONE_AT_A_TIME`."""
    path = nodeid.split('::', 1)[0]
    if PurePosixPath(path).name in ONE_AT_A_TIME:
        return SHARED_SCOPE
    return path


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config: pytest.Config, log: Any) -> Any:
    if config.getvalue('dist') != 'loadfile':
        return None
    from xdist.scheduler import LoadScopeScheduling

    class _Scheduling(LoadScopeScheduling):  # type: ignore[misc]
        def _split_scope(self, nodeid: str) -> str:
            return scope_of(nodeid)

    return _Scheduling(config, log)
