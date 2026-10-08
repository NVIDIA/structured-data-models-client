# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The ``test_fix_*.py`` regression tests here read through the relational
engine, so without it they skip, as the root suite's other engine tests do.

CI's ``test`` job installs no engine; its ``engine`` job does, and runs
``tests/`` again, so these tests still run there.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope='module')
def _needs_the_engine() -> None:
    pytest.importorskip(
        'sdm_relational_engine',
        reason='the phase 5.5 regression tests read through the engine',
    )
